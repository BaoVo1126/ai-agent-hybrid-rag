#!/usr/bin/env python
"""Interactive CLI chat, e.g.:

    python scripts/run_agent_cli.py --strategy react
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.factory import get_agent  # noqa: E402
from src.config import SETTINGS  # noqa: E402
from src.core.llm_client import get_llm_client  # noqa: E402
from src.tools.registry import build_default_registry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["react", "function_calling", "plan_execute", "self_correcting_rag"],
        default="function_calling",
    )
    args = parser.parse_args()

    mode = "real (Ollama)" if SETTINGS.is_real_mode else "mock (offline)"
    print(f"Agent Lab CLI -- strategy={args.strategy}, mode={mode}")
    print("Type a question, or 'exit' to quit.\n")

    llm = get_llm_client()
    registry = build_default_registry()
    agent = get_agent(args.strategy, llm, registry)

    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("exit", "quit"):
            break

        result = agent.run(query)
        for step in result.steps:
            if step.step_type == "tool_call":
                print(f"  [action] {step.content}")
            elif step.step_type == "observation":
                print(f"  [observation] {step.content[:200]}")
        print(f"agent> {result.final_answer}")
        print(f"       ({result.tool_calls_made} tool calls, {result.llm_calls_made} LLM calls, {result.latency_seconds:.2f}s)\n")


if __name__ == "__main__":
    main()
