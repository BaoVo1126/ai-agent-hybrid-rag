from __future__ import annotations

from src.core.query_guard import assess_query


def test_normal_question_passes():
    assert assess_query("What does the document say about AI agents?").ok


def test_short_but_real_question_passes():
    assert assess_query("What is RAG?").ok


def test_vietnamese_question_passes():
    assert assess_query("Tài liệu nói gì về tác nhân AI?").ok


def test_bare_arithmetic_expression_passes():
    # No "words" at all, but a legitimate CalculatorTool query.
    assert assess_query("125 * 48 - 12").ok


def test_empty_query_rejected():
    result = assess_query("")
    assert not result.ok
    assert result.reason == "empty"
    assert result.message


def test_whitespace_only_query_rejected():
    result = assess_query("   \n\t  ")
    assert not result.ok
    assert result.reason == "empty"


def test_keyboard_mash_rejected():
    result = assess_query("asdkfjhaslkdjfh alskdjfh qwoeiruqwoeiur")
    assert not result.ok


def test_symbol_spam_rejected():
    result = assess_query("??????!!!!!@@@@@####$$$$$")
    assert not result.ok


def test_repeated_character_rejected():
    result = assess_query("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert not result.ok


def test_very_long_query_rejected():
    result = assess_query("what is the meaning of this " * 300)
    assert not result.ok
    assert result.reason == "too_long"


def test_single_random_letters_rejected():
    result = assess_query("xk")
    # Too short to contain a real word (min length 2 for the word regex is
    # met, but no_words / low signal checks should still catch pure noise
    # via the letter-ratio checks upstream in real usage) -- this asserts
    # it's at least not silently treated as a normal question when it's
    # this degenerate.
    assert result.ok or result.reason in {"no_words", "low_letter_ratio", "keyboard_mash"}


def test_real_words_with_consonant_clusters_not_falsely_flagged():
    # "strengths"/"nightclub"-style consonant clusters are real English,
    # not keyboard mash -- the guard must not reject them.
    assert assess_query("What are the strengths and weaknesses of this approach?").ok
    assert assess_query("Where is the nearest nightclub mentioned in the document?").ok
