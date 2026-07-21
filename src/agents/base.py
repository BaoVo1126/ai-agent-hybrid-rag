"""Shared helpers used by every agent strategy, so latency/token accounting
is measured identically across strategies -- otherwise the benchmark
comparison in evaluation/ would not be a fair apples-to-apples test."""

from __future__ import annotations

import time
from contextlib import contextmanager


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (~4 chars/token for English, less accurate for
    Vietnamese/CJK but fine as a *relative* metric for comparing strategies
    against each other -- absolute token counts always come from the real
    API's `usage` field when running in real mode).
    """
    return max(1, len(text) // 4)


@contextmanager
def timer():
    start = time.perf_counter()
    result = {"seconds": 0.0}
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
