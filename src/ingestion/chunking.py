
from __future__ import annotations

from src.core.interfaces import Document


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


class RecursiveCharacterTextSplitter:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def split_text(self, text: str) -> list[str]:
        pieces = self._split(text, self.separators)
        return self._merge(pieces)


    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        separator, remaining_separators = separators[0], separators[1:]
        if separator == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = [p for p in text.split(separator) if p]
        pieces: list[str] = []
        for part_idx, part in enumerate(parts):
            if len(part) <= self.chunk_size:
                sub_pieces = [part]
            elif remaining_separators:
                sub_pieces = self._split(part, remaining_separators)
            else:
                sub_pieces = [part[i : i + self.chunk_size] for i in range(0, len(part), self.chunk_size)]

            if sub_pieces and part_idx < len(parts) - 1:
                sub_pieces[-1] = sub_pieces[-1] + separator
            pieces.extend(sub_pieces)
        return pieces

    def _merge(self, pieces: list[str]) -> list[str]:
        if not pieces:
            return []

        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = current + piece
            if len(candidate) <= self.chunk_size or not current:
                current = candidate
            else:
                chunks.append(current.strip())
                overlap_budget = max(0, min(self.chunk_overlap, self.chunk_size - len(piece)))
                overlap_tail = current[-overlap_budget:] if overlap_budget else ""
                current = overlap_tail + piece
        if current.strip():
            chunks.append(current.strip())
        return chunks


def chunk_document(document: Document, chunk_size: int = 1000, overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    texts = splitter.split_text(document.text)

    chunks: list[Document] = []
    for chunk_idx, chunk_text in enumerate(texts):
        chunks.append(
            Document(
                id=f"{document.id}::chunk_{chunk_idx}",
                text=chunk_text,
                metadata={**document.metadata, "parent_id": document.id, "chunk_index": chunk_idx},
            )
        )
    return chunks


def chunk_documents(documents: list[Document], chunk_size: int = 1000, overlap: int = 150) -> list[Document]:
    all_chunks: list[Document] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
