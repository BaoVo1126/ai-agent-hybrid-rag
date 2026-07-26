from __future__ import annotations

from src.ingestion.quality import assess_text_quality


def test_normal_prose_passes():
    text = (
        "Retrieval augmented generation combines a retriever with a language model. "
        "The retriever finds relevant passages and the model conditions its answer on them, "
        "which improves factual grounding significantly."
    )
    assert assess_text_quality(text).ok


def test_empty_text_fails():
    result = assess_text_quality("")
    assert not result.ok
    assert "little extracted text" in result.reason


def test_too_short_text_fails():
    result = assess_text_quality("ok.")
    assert not result.ok


def test_garbled_symbol_noise_fails():
    # Typical of a badly font-mapped/OCR'd scan: lots of non-alphabetic
    # symbol noise, few real words.
    garbled = "▯▯▯ ¤¤¤ §§§ ___===+++ %%% &&& ###@@@ ***(((" * 3
    result = assess_text_quality(garbled)
    assert not result.ok


def test_repeated_single_character_fails():
    result = assess_text_quality("a" * 200)
    assert not result.ok


def test_repeated_character_with_enough_words_hits_repeated_char_check():
    # Pad with real short words first so the "too few words" check doesn't
    # fire before the repeated-character check gets a chance to.
    text = "the cat sat on mat " + "a" * 300
    result = assess_text_quality(text)
    assert not result.ok
    assert "repeated character" in result.reason


def test_too_few_words_fails():
    result = assess_text_quality("42 17 99 3.14 " * 10)
    assert not result.ok


def test_vietnamese_prose_passes():
    text = (
        "Học sâu là một nhánh của học máy sử dụng mạng nơ-ron nhiều lớp để học biểu diễn dữ liệu. "
        "Các mô hình này được huấn luyện trên tập dữ liệu lớn để nhận diện các mẫu phức tạp."
    )
    assert assess_text_quality(text).ok
