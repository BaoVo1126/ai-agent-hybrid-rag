"""
Self-correcting RAG agent.

A fourth strategy alongside ReAct / function-calling / plan-execute, built
to answer a question the other three don't ask: when the retrieved-grounded
answer might be wrong, does the *system* notice and fix it, or does the
mistake just ship as the final answer?

This is the same corrective-RAG idea as the companion project
`agentic-rag-local` (a standalone LangGraph pipeline) reimplemented against
this repo's own Tool / LLMClient / AgentStrategy interfaces, so it drops
into the existing factory + benchmark harness as just another named
strategy -- no change to how the API, CLI, or evaluation code select or run
an agent.

Loop, capped at `SETTINGS.self_correction_max_retries` retries:
  1. retrieve   -- reranked hybrid search (src/retrieval/reranker.py) for
                   the current query
  2. grade      -- LLM judges each candidate passage's relevance; keep only
                   the ones marked relevant (falls back to the raw top
                   results if none pass, rather than answering from nothing)
  3. generate   -- answer using only the kept passages
  4. verify     -- LLM checks (a) is the answer grounded in those passages
                   (not hallucinated), and (b) does it actually address the
                   question
  5. if either check fails and retries remain: rewrite the query and go
     back to step 1. Otherwise return the answer -- flagged as unverified
     if it still failed a check after the last allowed retry, so a failure
     is visible rather than silently shipped as if it were confident.

Streaming: the loop body lives in `_run_iter()`, a generator that `yield`s
each AgentStep the moment it's produced instead of appending to a list.
`run()` just materializes that generator into a list (unchanged external
behavior/contract); `run_stream()` (used by api/main.py's SSE endpoint)
yields directly, so the UI sees each retrieval / grading verdict / retry
appear in real time instead of all at once at the end. Per-call stats
(llm_calls, tokens, retries...) are threaded through a plain `stats` dict
passed into the generator rather than closed-over locals, since a generator
can't easily hand back extra values alongside each yield.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

from src.agents.base import estimate_tokens, timer
from src.config import SETTINGS
from src.core.interfaces import AgentRunResult, AgentStep, AgentStrategy, LLMClient, ToolCall, ToolResult
from src.tools.registry import ToolRegistry

_GRADE_RELEVANCE_SYSTEM = (
    "You judge whether a retrieved passage is relevant to a question. "
    "Reply with exactly one word on its own line: yes or no."
)

_GENERATE_SYSTEM = (
    "Answer the question using ONLY the provided passages. If the passages "
    "do not contain enough information, say so explicitly instead of "
    "guessing. Be concise."
)

_GROUNDED_SYSTEM = (
    "You judge whether an answer is fully supported by the given passages "
    "-- i.e. it does not state anything the passages don't back up. "
    "Reply with exactly one word on its own line: yes or no."
)

_USEFUL_SYSTEM = (
    "You judge whether an answer actually addresses the question asked "
    "(on-topic and responsive), regardless of whether it is factually "
    "correct. Reply with exactly one word on its own line: yes or no."
)

_REWRITE_SYSTEM = (
    "Rewrite the question to be clearer and more specific for a document "
    "search, while keeping the original intent. Reply with only the "
    "rewritten question, nothing else."
)

_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _ask(llm: LLMClient, system: str, user: str) -> str:
    """One-shot free-text call -- graders/rewriter don't need tool-calling,
    so `react_text` mode (plain text in, plain text out) is the cheapest fit
    against the existing LLMClient contract."""
    response = llm.complete([{"role": "user", "content": user}], tools=[], system=system, mode="react_text")
    return " ".join(b["text"] for b in response["content"] if b.get("type") == "text").strip()


def _yes_no(text: str, default: bool) -> bool:
    """Parses a loose yes/no out of free text rather than requiring strict
    JSON -- more robust across both MockLLMClient's canned phrasing and a
    real model that might add a stray word around the answer. `default`
    covers mock mode / unparseable output without treating that as a hard
    failure (see module docstring in core/llm_client.py: MockLLMClient
    exists for structural testing, not for real grading accuracy)."""
    match = _YES_NO_RE.search(text)
    if not match:
        return default
    return match.group(1).lower() == "yes"


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
                if _yes_no(verdict, default=True):
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
            grounded = _yes_no(grounded_verdict, default=True)

            useful = True
            if grounded:
                useful_verdict = _ask(self.llm, _USEFUL_SYSTEM, f"Question: {current_query}\n\nAnswer:\n{answer}")
                stats["llm_calls"] += 1
                stats["output_tokens"] += estimate_tokens(useful_verdict)
                useful = _yes_no(useful_verdict, default=True)

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
