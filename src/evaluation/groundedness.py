"""
LLM-as-judge groundedness scoring.

Applied post-hoc to any strategy's run that touched document_search --
lets the benchmark report not just "did it call the right tool" (the
existing shallow metric in metrics.py) but "was the final answer actually
supported by what it retrieved". This is the exact number
self_correcting_rag_agent.py is built to improve, so putting it in the
benchmark is what makes a head-to-head comparison against the other three
strategies meaningful rather than just measuring speed/tool-call overhead.
"""

from __future__ import annotations

import re

from src.core.interfaces import AgentRunResult, LLMClient

_SYSTEM = (
    "You judge whether an answer is fully supported by the given passages "
    "-- i.e. it does not state anything the passages don't back up. "
    "Reply with exactly one word on its own line: yes or no."
)
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def score_groundedness(result: AgentRunResult, llm: LLMClient) -> bool | None:
    """Returns None when the strategy never called document_search for this
    query -- there's no retrieved context to check the answer against, so
    "grounded" isn't a meaningful question (e.g. a pure calculator query)."""
    context_chunks = [
        step.tool_result.output
        for step in result.steps
        if step.step_type == "observation" and step.tool_result is not None and step.tool_result.tool_name == "document_search"
    ]
    if not context_chunks:
        return None

    context = "\n\n".join(context_chunks)[:4000]
    response = llm.complete(
        [{"role": "user", "content": f"Passages:\n{context}\n\nAnswer:\n{result.final_answer}"}],
        tools=[],
        system=_SYSTEM,
        mode="react_text",
    )
    text = " ".join(b["text"] for b in response["content"] if b.get("type") == "text")
    match = _YES_NO_RE.search(text)
    if not match:
        return None
    return match.group(1).lower() == "yes"
