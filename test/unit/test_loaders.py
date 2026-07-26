from __future__ import annotations

import pytest

from src.ingestion.loaders import load_any, load_directory


def test_text_loader_reads_normal_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "Retrieval augmented generation combines a retriever with a language model to ground answers."
    )
    docs = load_any(str(path))
    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(path)


def test_text_loader_rejects_garbled_content(tmp_path):
    path = tmp_path / "corrupt.txt"
    path.write_text("\x00\x01\x02\x03" * 50)
    with pytest.raises(RuntimeError, match="not enough readable text"):
        load_any(str(path))


def test_text_loader_rejects_near_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("ok")
    with pytest.raises(RuntimeError):
        load_any(str(path))


def test_load_any_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "file.docx"
    path.write_text("whatever")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_any(str(path))


def test_load_directory_skips_bad_file_but_keeps_good_ones(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text(
        "An AI agent perceives its environment through tools and takes actions to achieve a goal."
    )
    bad = tmp_path / "bad.txt"
    bad.write_text("\x00\x01\x02\x03" * 50)

    docs = load_directory(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(good)


def test_load_directory_raises_when_every_file_is_unreadable(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("\x00\x01\x02\x03" * 50)

    with pytest.raises(RuntimeError, match="None of the file"):
        load_directory(str(tmp_path))


def test_load_directory_on_missing_directory_returns_empty_list(tmp_path):
    assert load_directory(str(tmp_path / "does_not_exist")) == []


def test_load_directory_on_empty_directory_returns_empty_list(tmp_path):
    assert load_directory(str(tmp_path)) == []


def test_pdf_loader_rejects_page_with_no_text_layer(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)  # simulates a scanned/image-only page
    path = tmp_path / "scanned.pdf"
    with open(path, "wb") as f:
        writer.write(f)

    with pytest.raises(RuntimeError, match="scanned or photographed PDF"):
        load_any(str(path))
