"""
Document loaders.

The agent is designed to be handed ANY file you drop into `data/` --
`AI Engineering.pdf`, a `.txt` export, or a `.md` note. Each loader only
knows how to turn one file format into plain-text Document objects; nothing
downstream (chunking, indexing, tools) needs to know which loader was used.
"""

from __future__ import annotations

import os

from src.core.interfaces import Document, DocumentLoader


class TextLoader(DocumentLoader):
    """Loads .txt / .md files as a single Document."""

    def load(self, path: str) -> list[Document]:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        doc_id = os.path.basename(path)
        return [Document(id=doc_id, text=text, metadata={"source": path})]


class PDFLoader(DocumentLoader):
    """
    Loads a PDF, one Document per page, using pypdf.

    Page-level documents (rather than one giant blob) keep chunk provenance
    meaningful -- a retrieved chunk can point back to "page 12" instead of
    just "the file".
    """

    def load(self, path: str) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is required to read PDFs. Install with `pip install pypdf`."
            ) from exc

        reader = PdfReader(path)
        doc_id_base = os.path.basename(path)
        documents = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        id=f"{doc_id_base}::page_{page_num}",
                        text=text,
                        metadata={"source": path, "page": page_num},
                    )
                )
        return documents


def load_any(path: str) -> list[Document]:
    """Dispatch to the right loader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PDFLoader().load(path)
    if ext in (".txt", ".md"):
        return TextLoader().load(path)
    raise ValueError(f"Unsupported file type: {ext} (supported: .pdf, .txt, .md)")


def load_directory(directory: str) -> list[Document]:
    """Load every supported file found directly inside `directory`."""
    all_docs: list[Document] = []
    if not os.path.isdir(directory):
        return all_docs
    for name in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, name)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".pdf", ".txt", ".md"):
            continue
        all_docs.extend(load_any(full_path))
    return all_docs
