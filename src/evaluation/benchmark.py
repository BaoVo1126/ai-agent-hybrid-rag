"""
Benchmark harness: run every registered agent strategy over the same eval
set with the same tool registry and LLM client, then produce a comparison
table. Run `python scripts/run_benchmark.py` -- it writes
`results/benchmark.md` (and a chart if matplotlib is available).

Now includes `self_correcting_rag` alongside the original three strategies,
plus two metrics that specifically justify comparing it against them:
  - avg_groundedness: LLM-as-judge check of whether each strategy's answer
    was actually supported by what it retrieved (evaluation/groundedness.py)
  - avg_self_correction_retries: how many retrieve->generate->verify loops
    self_correcting_rag needed (always 0 for the other three, which have no
    retry mechanism)
"""

from __future__ import annotations

import os

from src.agents.factory import get_agent
from src.core.interfaces import AgentRunResult
from src.core.llm_client import get_llm_client
from src.evaluation.eval_dataset import DEFAULT_EVAL_SET, EvalExample
from src.evaluation.groundedness import score_groundedness
from src.evaluation.metrics import StrategyBenchmark, aggregate, score_example
from src.tools.registry import build_default_registry

STRATEGY_NAMES = ["react", "function_calling", "plan_execute", "self_correcting_rag"]


def run_benchmark(eval_set: list[EvalExample] | None = None) -> dict[str, StrategyBenchmark]:
    eval_set = eval_set or DEFAULT_EVAL_SET
    llm = get_llm_client()
    registry = build_default_registry()

    benchmarks: dict[str, StrategyBenchmark] = {}
    for strategy_name in STRATEGY_NAMES:
        agent = get_agent(strategy_name, llm, registry)
        results: list[AgentRunResult] = []
        scores = []
        for example in eval_set:
            result = agent.run(example.query)
            results.append(result)
            grounded = score_groundedness(result, llm)
            scores.append(score_example(example, result, grounded))
        benchmarks[strategy_name] = aggregate(strategy_name, results, scores)

    return benchmarks


def format_markdown(benchmarks: dict[str, StrategyBenchmark], mode_label: str) -> str:
    lines = [
        "# Agent Strategy Benchmark",
        "",
        f"Mode: **{mode_label}**  ",
        f"Examples per strategy: **{next(iter(benchmarks.values())).n_examples}**",
        "",
        "| Strategy | Pass rate | Groundedness | Avg retries | Avg latency (s) | Avg tool calls | Avg LLM calls | Avg tokens (est.) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b in benchmarks.values():
        grounded_str = f"{b.avg_groundedness:.0%}" if b.avg_groundedness is not None else "n/a"
        lines.append(
            f"| {b.strategy_name} | {b.pass_rate:.0%} | {grounded_str} | {b.avg_self_correction_retries:.1f} | "
            f"{b.avg_latency_seconds:.3f} | {b.avg_tool_calls:.1f} | {b.avg_llm_calls:.1f} | {b.avg_estimated_tokens:.0f} |"
        )

    lines += [
        "",
        "## How to read this",
        "- **Pass rate**: fraction of eval examples where the expected tool was used, "
        "the expected keyword (if any) appeared in the final answer, AND the answer "
        "wasn't flagged as ungrounded.",
        "- **Groundedness**: LLM-as-judge fraction of `document_search`-backed answers "
        "that were actually supported by the retrieved passages (n/a for examples that "
        "never called document_search, e.g. pure arithmetic). This is the metric "
        "`self_correcting_rag` is specifically designed to improve, at the cost of more "
        "LLM calls and higher latency per query -- that trade-off is the point of this "
        "table, not a bug in one strategy.",
        "- **Avg retries**: retrieve->generate->verify loops beyond the first attempt. "
        "Always 0 for react/function_calling/plan_execute, which have no self-correction step.",
        "- **Avg tool calls / LLM calls**: proxies for cost and latency overhead per query.",
        "- **Avg tokens (est.)**: `len(text)//4` heuristic in mock mode; the real `usage` "
        "field from Ollama's response is used automatically once LLM_BACKEND=ollama is set.",
        "",
        "Re-run with `python scripts/run_benchmark.py` after adding document-specific "
        "examples to `src/evaluation/eval_dataset.py` for a more meaningful accuracy signal.",
    ]
    return "\n".join(lines)


def try_plot(benchmarks: dict[str, StrategyBenchmark], out_path: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    names = list(benchmarks.keys())
    latencies = [benchmarks[n].avg_latency_seconds for n in names]
    tool_calls = [benchmarks[n].avg_tool_calls for n in names]
    pass_rates = [benchmarks[n].pass_rate * 100 for n in names]
    groundedness = [
        (benchmarks[n].avg_groundedness * 100 if benchmarks[n].avg_groundedness is not None else 0) for n in names
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].bar(names, latencies, color="#4C72B0")
    axes[0].set_title("Avg latency (s)")
    axes[1].bar(names, tool_calls, color="#DD8452")
    axes[1].set_title("Avg tool calls")
    axes[2].bar(names, pass_rates, color="#55A868")
    axes[2].set_title("Pass rate (%)")
    axes[3].bar(names, groundedness, color="#8172B2")
    axes[3].set_title("Groundedness (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    from src.config import SETTINGS

    mode_label = "real (Ollama)" if SETTINGS.is_real_mode else "mock (offline)"
    benchmarks = run_benchmark()
    md = format_markdown(benchmarks, mode_label)

    os.makedirs("results", exist_ok=True)
    md_path = os.path.join("results", "benchmark.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    chart_path = os.path.join("results", "benchmark_chart.png")
    plotted = try_plot(benchmarks, chart_path)

    print(md)
    print(f"\nWritten to {md_path}" + (f" and {chart_path}" if plotted else " (install matplotlib for a chart too)"))


if __name__ == "__main__":
    main()
