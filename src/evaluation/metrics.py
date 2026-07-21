"""
Evaluation metrics.

Tool-usage / keyword correctness (score_example) stays the shallow check it
always was -- this project doesn't know in advance what document you'll
drop into data/, so it can't ship a ground-truth answer key for document
content. What's new: `score_groundedness` (evaluation/groundedness.py) adds
an LLM-as-judge signal on top, which IS meaningful regardless of which
document you use, and is specifically what differentiates
self_correcting_rag from the other three strategies in the benchmark table.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.interfaces import AgentRunResult
from src.evaluation.eval_dataset import EvalExample


@dataclass
class ExampleScore:
    query: str
    used_expected_tool: bool
    keyword_match: bool | None  # None when the example has no ground-truth keywords
    grounded: bool | None  # None when the strategy never called document_search
    self_correction_retries: int
    passed: bool


def score_example(example: EvalExample, result: AgentRunResult, grounded: bool | None) -> ExampleScore:
    used_tools = {
        step.tool_call.tool_name
        for step in result.steps
        if step.step_type == "tool_call" and step.tool_call is not None
    }
    used_expected_tool = example.expected_tool is None or example.expected_tool in used_tools

    keyword_match: bool | None = None
    if example.expected_keywords:
        answer_lower = result.final_answer.lower()
        keyword_match = any(kw.lower() in answer_lower for kw in example.expected_keywords)

    # A confidently-wrong answer (grounded=False) still fails even if the
    # right tool was called and a keyword happened to match -- this is the
    # whole point of adding groundedness rather than treating tool-usage as
    # a sufficient correctness signal on its own.
    passed = used_expected_tool and (keyword_match is not False) and (grounded is not False)

    return ExampleScore(
        query=example.query,
        used_expected_tool=used_expected_tool,
        keyword_match=keyword_match,
        grounded=grounded,
        self_correction_retries=result.self_correction_retries,
        passed=passed,
    )


@dataclass
class StrategyBenchmark:
    strategy_name: str
    n_examples: int
    avg_latency_seconds: float
    avg_tool_calls: float
    avg_llm_calls: float
    avg_estimated_tokens: float
    pass_rate: float
    avg_groundedness: float | None  # None when no example in the set called document_search
    avg_self_correction_retries: float


def aggregate(strategy_name: str, results: list[AgentRunResult], scores: list[ExampleScore]) -> StrategyBenchmark:
    n = len(results) or 1

    grounded_scores = [s.grounded for s in scores if s.grounded is not None]
    avg_groundedness = (sum(1 for g in grounded_scores if g) / len(grounded_scores)) if grounded_scores else None

    return StrategyBenchmark(
        strategy_name=strategy_name,
        n_examples=len(results),
        avg_latency_seconds=sum(r.latency_seconds for r in results) / n,
        avg_tool_calls=sum(r.tool_calls_made for r in results) / n,
        avg_llm_calls=sum(r.llm_calls_made for r in results) / n,
        avg_estimated_tokens=sum(r.estimated_input_tokens + r.estimated_output_tokens for r in results) / n,
        pass_rate=sum(1 for s in scores if s.passed) / n,
        avg_groundedness=avg_groundedness,
        avg_self_correction_retries=sum(s.self_correction_retries for s in scores) / n,
    )
