from __future__ import annotations

import json
import re
import uuid
from typing import Iterator

from src.agents.base import estimate_tokens, timer
from src.agents.prompts import (
    GENERATE_SYSTEM as _GENERATE_SYSTEM,
    GRADE_RELEVANCE_SYSTEM as _GRADE_RELEVANCE_SYSTEM,
    GROUNDED_SYSTEM as _GROUNDED_SYSTEM,
    REWRITE_SYSTEM as _REWRITE_SYSTEM,
    USEFUL_SYSTEM as _USEFUL_SYSTEM,
)
from src.config import SETTINGS
from src.core.interfaces import AgentRunResult, AgentStep, AgentStrategy, LLMClient, ToolCall, ToolResult
from src.tools.registry import ToolRegistry

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _ask(llm: LLMClient, system: str, user: str) -> str:
    """One-shot free-text call -- graders/rewriter don't need tool-calling,
    so `react_text` mode (plain text in, plain text out) is the cheapest fit
    against the existing LLMClient contract."""
    response = llm.complete([{"role": "user", "content": user}], tools=[], system=system, mode="react_text")
    return " ".join(b["text"] for b in response["content"] if b.get("type") == "text").strip()


def _parse_verdict(text: str, field: str, default: bool) -> bool:
    """Parses the structured `{"<field>": true|false, "reason": "..."}`
    contract the grader/verifier prompts now require (see agents/prompts.py
    for why this replaced a bare yes/no regex).

    Fail-closed on a real model response: a genuinely malformed or evasive
    response is treated as the check FAILING (`False`), not passing --
    the previous version defaulted unparseable output to `True`, which
    silently waved through anything the model didn't answer cleanly and is
    the likely source of the low measured accuracy this replaces.

    `default` is only used for MockLLMClient's canned, deliberately
    non-JSON phrasing in structural/unit tests -- real grading paths always
    hit the JSON branch or the loose "yes/no" fallback below it.
    """
    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            value = parsed.get(field)
            if isinstance(value, bool):
                return value
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback for MockLLMClient / models that ignore the JSON contract but
    # still say "yes"/"no" somewhere -- still better than nothing, but any
    # response that matches neither format fails closed, not open.
    loose = _YES_NO_RE.search(text)
    if loose:
        return loose.group(1).lower() == "yes"

    return default


class SelfCorrectingRAGAgent(AgentStrategy):
    name = "self_correcting_rag"

    def __init__(self, llm: LLMClient, registry: ToolRegistry, max_retries: int | None = None) -> None:
        self.llm = llm
        self.registry = registry
        # Reaches into the registered document_search tool for its retriever
        # rather than going through tool.run() -- this strategy needs the
        # individual (Document, score) pairs to grade one at a time, not the
        # single pre-formatted text blob DocumentSearchTool.run() returns.
        self.retriever = registry.get("document_search").retriever
        self.max_retries = SETTINGS.self_correction_max_retries if max_retries is None else max_retries

    def _run_iter(self, query: str, stats: dict) -> Iterator[AgentStep]:
        stats["llm_calls"] = 0
        stats["tool_calls"] = 0
        stats["output_tokens"] = 0
        stats["retries_used"] = 0
        stats["final_answer"] = "(no answer produced)"

        current_query = query
        fetch_k = max(SETTINGS.top_k * 3, 6)

        for attempt in range(self.max_retries + 1):
            # ---- retrieve ----
            call = ToolCall(tool_name="document_search", arguments={"query": current_query}, call_id=str(uuid.uuid4()))
            candidates = self.retriever.search(current_query, top_k=fetch_k)
            observation_text = (
                "\n\n".join(f"[{doc.metadata.get('source', '?')}] {doc.text[:500]}" for doc, _score in candidates)
                or "No results."
            )
            yield AgentStep(step_type="tool_call", content=f"search: {current_query}", tool_call=call)
            stats["tool_calls"] += 1
            result = ToolResult(call_id=call.call_id, tool_name="document_search", output=observation_text)
            yield AgentStep(step_type="observation", content=observation_text, tool_result=result)
            stats["output_tokens"] += estimate_tokens(observation_text)

            # ---- grade ----
            kept: list[tuple] = []
            for doc, score in candidates:
                verdict = _ask(self.llm, _GRADE_RELEVANCE_SYSTEM, f"Question: {current_query}\n\nPassage: {doc.text[:800]}")
                stats["llm_calls"] += 1
                stats["output_tokens"] += estimate_tokens(verdict)
                if _parse_verdict(verdict, "relevant", default=True):
                    kept.append((doc, score))
            if not kept:
                kept = candidates[: SETTINGS.top_k]
                yield AgentStep(step_type="thought", content="No passage graded relevant -- falling back to top raw results.")

            # ---- generate ----
            context = "\n\n".join(doc.text[:800] for doc, _score in kept)
            answer = _ask(self.llm, _GENERATE_SYSTEM, f"Question: {current_query}\n\nPassages:\n{context}")
            stats["llm_calls"] += 1
            stats["output_tokens"] += estimate_tokens(answer)
            stats["final_answer"] = answer
            yield AgentStep(step_type="thought", content=f"Draft answer (attempt {attempt + 1}): {answer}")

            # ---- verify: grounded, then useful ----
            grounded_verdict = _ask(self.llm, _GROUNDED_SYSTEM, f"Passages:\n{context}\n\nAnswer:\n{answer}")
            stats["llm_calls"] += 1
            stats["output_tokens"] += estimate_tokens(grounded_verdict)
            grounded = _parse_verdict(grounded_verdict, "grounded", default=True)

            useful = True
            if grounded:
                useful_verdict = _ask(self.llm, _USEFUL_SYSTEM, f"Question: {current_query}\n\nAnswer:\n{answer}")
                stats["llm_calls"] += 1
                stats["output_tokens"] += estimate_tokens(useful_verdict)
                useful = _parse_verdict(useful_verdict, "useful", default=True)

            if grounded and useful:
                yield AgentStep(step_type="final_answer", content=answer)
                return

            reason = "not grounded" if not grounded else "not useful"
            yield AgentStep(step_type="thought", content=f"Self-check failed ({reason}).")

            if attempt == self.max_retries:
                stats["final_answer"] = f"{answer}\n\n[unverified: self-check flagged '{reason}' after {self.max_retries} retries]"
                yield AgentStep(step_type="final_answer", content=stats["final_answer"])
                return

            # ---- rewrite query and loop back to retrieve ----
            current_query = _ask(self.llm, _REWRITE_SYSTEM, current_query)
            stats["llm_calls"] += 1
            stats["output_tokens"] += estimate_tokens(current_query)
            stats["retries_used"] += 1
            yield AgentStep(step_type="thought", content=f"Rewriting query -> {current_query}")

    def run(self, query: str, max_steps: int = 8) -> AgentRunResult:
        with timer() as t:
            stats: dict = {}
            steps = list(self._run_iter(query, stats))

        return AgentRunResult(
            query=query,
            final_answer=stats["final_answer"],
            steps=steps,
            latency_seconds=t["seconds"],
            tool_calls_made=stats["tool_calls"],
            llm_calls_made=stats["llm_calls"],
            estimated_input_tokens=estimate_tokens(query),
            estimated_output_tokens=stats["output_tokens"],
            self_correction_retries=stats["retries_used"],
        )

    def run_stream(self, query: str, max_steps: int = 8) -> Iterator[AgentStep]:
        stats: dict = {}
        yield from self._run_iter(query, stats)
