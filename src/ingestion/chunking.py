"""Fixed-size word chunking with overlap -- the same baseline strategy used
in rag-from-scratch, kept deliberately simple so the interesting complexity
in this project stays in the agent loop, not in yet another chunker."""

from __future__ import annotations

from src.core.interfaces import Document


def chunk_document(document: Document, chunk_size: int = 500, overlap: int = 50) -> list[Document]:
    words = document.text.split()
    if not words:
        return []

    chunks: list[Document] = []
    start = 0
    chunk_idx = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(
            Document(
                id=f"{document.id}::chunk_{chunk_idx}",
                text=chunk_text,
                metadata={**document.metadata, "parent_id": document.id, "chunk_index": chunk_idx},
            )
        )
        chunk_idx += 1
        if end == len(words):
            break
        start += step

    return chunks


def chunk_documents(documents: list[Document], chunk_size: int = 500, overlap: int = 50) -> list[Document]:
    all_chunks: list[Document] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
