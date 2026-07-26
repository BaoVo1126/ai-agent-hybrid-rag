"""
GuardedAgentStrategy -- wraps ANY AgentStrategy and short-circuits obviously
unusable input (empty, gibberish, keyboard-mash) before it ever reaches an
LLM call or a retrieval query, using core/query_guard.py::assess_query().
Wired in once, centrally, by agents/factory.py::get_agent(), so every
strategy (ReAct, function-calling, plan-execute, self-correcting RAG) gets
this protection automatically without any of the four duplicating the
check -- the same "wrap the interface once" shape as RerankedRetriever
wrapping a base retriever (retrieval/reranker.py).

On a rejected query: returns a normal AgentRunResult/AgentStep (not an
exception), with the guard's clarifying message as the final answer and
zero tool/LLM calls -- so callers (API, CLI, benchmark) need no
special-case handling, and the response shape callers already expect never
changes.
"""

from __future__ import annotations

from typing import Iterator

from src.agents.base import timer
from src.core.interfaces import AgentRunResult, AgentStep, AgentStrategy
from src.core.query_guard import assess_query


class GuardedAgentStrategy(AgentStrategy):
    def __init__(self, inner: AgentStrategy) -> None:
        self.inner = inner
        self.name = inner.name

    def _rejection_result(self, query: str, message: str) -> AgentRunResult:
        with timer() as t:
            pass
        return AgentRunResult(
            query=query,
            final_answer=message,
            steps=[AgentStep(step_type="final_answer", content=message)],
            latency_seconds=t["seconds"],
            tool_calls_made=0,
            llm_calls_made=0,
            estimated_input_tokens=0,
            estimated_output_tokens=0,
        )

    def run(self, query: str, max_steps: int = 8) -> AgentRunResult:
        assessment = assess_query(query)
        if not assessment.ok:
            return self._rejection_result(query, assessment.message)
        return self.inner.run(query, max_steps=max_steps)

    def run_stream(self, query: str, max_steps: int = 8) -> Iterator[AgentStep]:
        assessment = assess_query(query)
        if not assessment.ok:
            yield AgentStep(step_type="final_answer", content=assessment.message)
            return
        yield from self.inner.run_stream(query, max_steps=max_steps)
