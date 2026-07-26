"""
Guards every agent strategy against queries that aren't worth spending an
LLM/retrieval call on: empty input, keyboard-mash gibberish, or a wall of
repeated symbols. Wired in centrally via GuardedAgentStrategy
(agents/guarded_agent.py), which agents/factory.py::get_agent() wraps every
strategy in -- the same "wrap the interface once, not four times" shape as
RerankedRetriever wrapping a base retriever (retrieval/reranker.py).

Deliberately conservative and fully offline (no LLM call to decide whether
to even make an LLM call): the goal is only to catch input that's clearly
not a real question -- blank submissions, a cat walking across the
keyboard, a wall of the same punctuation mark -- not to second-guess
anything that could plausibly be a genuine (if terse, non-English, or
oddly phrased) question. False positives here are much more costly than
false negatives: rejecting a real question reads as the product being
broken, while letting a borderline case through just costs one normal
agent turn that will likely come back short on retrieved evidence anyway
(and self_correcting_rag_agent.py's groundedness check already catches
that case downstream).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A "word" here just means "a run of 2+ letters" -- Unicode-aware, not
# restricted to ASCII, so real questions in Vietnamese, French, etc. still
# count, not just English.
_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

# A query that's ONLY digits/operators/parentheses is legitimate input for
# this project's CalculatorTool ("125*48-12", "(3+4)*2") even though it has
# no "words" at all -- it must never be flagged as nonsensical just for
# lacking letters.
_MATH_EXPR_RE = re.compile(r"^[\d\s+\-*/().,%^=?]+$")

MAX_CHARS = 4000
MIN_LETTER_RATIO = 0.3
MAX_REPEATED_CHAR_RATIO = 0.5
MIN_REPEATED_CHAR_LEN = 6
MIN_MASH_WORD_LEN = 6
MAX_CONSONANT_RUN = 6  # real English/Vietnamese words essentially never have
# 6+ consecutive consonant letters in a row ("strengths" tops out at 5) --
# this is a much sturdier keyboard-mash signal than "the word has zero
# vowels", which real short forms (acronyms, "rhythm"-style words) can
# also hit.
_VOWELS = set("aeiouy")


def _max_consonant_run(word: str) -> int:
    longest = current = 0
    for ch in word.lower():
        if ch.isalpha() and ch not in _VOWELS:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


@dataclass
class QueryAssessment:
    ok: bool
    reason: str | None = None
    message: str | None = None  # shown to the user in place of a real answer


def assess_query(query: str) -> QueryAssessment:
    text = (query or "").strip()

    if not text:
        return QueryAssessment(
            ok=False,
            reason="empty",
            message="I didn't receive a question -- what would you like to know about the document?",
        )

    if len(text) > MAX_CHARS:
        return QueryAssessment(
            ok=False,
            reason="too_long",
            message=(
                f"That's a lot of text ({len(text)} characters) for one question -- "
                "could you break it down into a shorter, more specific question?"
            ),
        )

    if _MATH_EXPR_RE.match(text):
        return QueryAssessment(ok=True)  # legitimate calculator-only input

    words = _WORD_RE.findall(text)
    if not words:
        return QueryAssessment(
            ok=False,
            reason="no_words",
            message=(
                'I couldn\'t find an actual question in that -- could you rephrase it, '
                'e.g. "What does the document say about X?"'
            ),
        )

    letter_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    if letter_ratio < MIN_LETTER_RATIO:
        return QueryAssessment(
            ok=False,
            reason="low_letter_ratio",
            message="That doesn't look like a question I can work with -- could you type it out in words?",
        )

    most_common_ratio = max(text.count(c) for c in set(text)) / len(text)
    if most_common_ratio > MAX_REPEATED_CHAR_RATIO and len(text) >= MIN_REPEATED_CHAR_LEN:
        return QueryAssessment(
            ok=False,
            reason="repeated_character",
            message="That looks like it might be an accidental submission -- could you type your actual question?",
        )

    # A long run of consecutive consonant letters is a strong keyboard-mash
    # signal ("sdkfjhsdkjfh", "qwrtyplkjhg") that real words essentially
    # never produce. Restricted to pure-ASCII words specifically so this
    # never fires on Vietnamese or other accented text.
    for word in words:
        if word.isascii() and len(word) >= MIN_MASH_WORD_LEN and _max_consonant_run(word) >= MAX_CONSONANT_RUN:
            return QueryAssessment(
                ok=False,
                reason="keyboard_mash",
                message="That doesn't look like a real word or question -- could you try rephrasing it?",
            )

    return QueryAssessment(ok=True)
