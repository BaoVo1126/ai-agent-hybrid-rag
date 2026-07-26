"""
Lightweight, dependency-free text-quality heuristics used right after text
extraction (loaders.py) to catch the case where extraction "succeeds"
(produces *some* text) but what came out isn't actually readable -- most
commonly a scanned/photographed page pypdf can only read as a jumble of
font-mapping artifacts, or genuinely blurry/low-resolution source material a
scanner's OCR layer got mostly wrong.

This is deliberately NOT an OCR-quality classifier or a language model --
just fast, offline signal so a garbled page gets excluded from the index
(and reported clearly) instead of silently poisoning retrieval with noise
that can score as a relevant BM25/embedding match while reading as nonsense
to the person asking the question. That gap -- a page "loads" but its text
is unusable -- was a real source of the wrong/garbled answers this check is
meant to catch before they ever reach an LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A "word" here just means "a run of 2+ letters" -- Unicode-aware (not
# restricted to ASCII) so this also works for Vietnamese/accented text, not
# just English.
_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

MIN_CHARS = 40
MIN_PRINTABLE_RATIO = 0.85
MIN_ALPHA_RATIO = 0.35
MIN_WORDS = 5
MAX_REPEATED_CHAR_RATIO = 0.4


@dataclass
class TextQuality:
    ok: bool
    reason: str | None = None


def assess_text_quality(text: str) -> TextQuality:
    """Cheap pass/fail signal for "is this actually readable extracted
    text", not a graded score -- callers (loaders.py) just need to decide
    whether to keep a page/file or skip it and say why."""
    stripped = text.strip()

    if len(stripped) < MIN_CHARS:
        return TextQuality(
            ok=False,
            reason="too little extracted text (page is likely a scanned image with no embedded text layer)",
        )

    printable_ratio = sum(1 for c in stripped if c.isprintable() or c in "\n\t") / len(stripped)
    if printable_ratio < MIN_PRINTABLE_RATIO:
        return TextQuality(
            ok=False,
            reason="mostly non-printable/garbled characters (likely a font-encoding or scan artifact)",
        )

    alpha_ratio = sum(1 for c in stripped if c.isalpha()) / len(stripped)
    if alpha_ratio < MIN_ALPHA_RATIO:
        return TextQuality(
            ok=False,
            reason="too few alphabetic characters relative to symbols/noise",
        )

    words = _WORD_RE.findall(stripped)
    if len(words) < MIN_WORDS:
        return TextQuality(ok=False, reason="too few recognizable words")

    # OCR/font-mapping artifacts on a badly scanned page sometimes degrade
    # into long runs of the same glyph -- a real page of prose never has
    # one character dominate like that.
    most_common_char_ratio = max(stripped.count(c) for c in set(stripped)) / len(stripped)
    if most_common_char_ratio > MAX_REPEATED_CHAR_RATIO:
        return TextQuality(
            ok=False,
            reason="dominated by one repeated character (likely a scan/rendering artifact)",
        )

    return TextQuality(ok=True)
