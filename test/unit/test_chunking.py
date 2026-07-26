from __future__ import annotations

import pytest

from src.core.interfaces import Document
from src.ingestion.chunking import RecursiveCharacterTextSplitter, chunk_document, chunk_documents


def test_splitter_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)


def test_split_text_on_empty_string_returns_empty_list():
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    assert splitter.split_text("") == []


def test_short_text_returns_single_chunk():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text = "This is a short paragraph that fits in one chunk easily."
    assert splitter.split_text(text) == [text]


def test_no_chunk_exceeds_chunk_size_even_with_no_natural_boundaries():
    """A single run-on 'word' with no spaces/punctuation at all must still
    terminate and never produce a chunk bigger than chunk_size -- the raw
    character fallback (the final '' entry in DEFAULT_SEPARATORS)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    text = "x" * 500
    chunks = splitter.split_text(text)
    assert all(len(c) <= 50 for c in chunks)
    assert len(chunks) > 1


def test_prefers_paragraph_boundary_over_mid_sentence_cut():
    splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
    text = "First paragraph is short.\n\nSecond paragraph is also fairly short."
    chunks = splitter.split_text(text)
    # The first paragraph should end up whole in some chunk rather than
    # being cut mid-word/mid-sentence.
    assert any(c.strip() == "First paragraph is short." for c in chunks)


def test_consecutive_chunks_share_overlap_characters():
    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
    text = "Sentence one is here. " * 20
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    # The tail of chunk i should reappear at the start of chunk i+1.
    tail = chunks[0][-15:]
    assert tail in chunks[1]


def test_reattached_separators_do_not_duplicate_across_recursion_levels():
    """Regression test: an earlier version of _split re-inserted the
    CURRENT level's separator between every sub-piece produced by a
    recursive call, corrupting text with repeated separators that were
    never in the original (e.g. spurious '\\n\\n\\n' runs)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    text = "Paragraph one.\n\nParagraph two.\n\n" + "Sentence A. " * 40
    chunks = splitter.split_text(text)
    joined = "".join(chunks)
    assert "\n\n\n" not in joined


def test_chunk_document_preserves_metadata_and_ids():
    doc = Document(id="doc1", text="Sentence one. " * 100, metadata={"source": "sample.txt"})
    chunks = chunk_document(doc, chunk_size=200, overlap=30)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.id == f"doc1::chunk_{i}"
        assert chunk.metadata["source"] == "sample.txt"
        assert chunk.metadata["parent_id"] == "doc1"
        assert chunk.metadata["chunk_index"] == i


def test_chunk_document_empty_text_returns_no_chunks():
    doc = Document(id="doc1", text="", metadata={})
    assert chunk_document(doc) == []


def test_chunk_documents_handles_multiple_documents():
    docs = [
        Document(id="a", text="Short text A.", metadata={}),
        Document(id="b", text="Short text B.", metadata={}),
    ]
    chunks = chunk_documents(docs, chunk_size=1000, overlap=100)
    assert [c.id for c in chunks] == ["a::chunk_0", "b::chunk_0"]
