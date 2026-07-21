"""
Plan-and-execute agent.

Instead of interleaving reasoning and tool calls one step at a time (as
ReAct and function-calling do), this strategy front-loads a lightweight
decomposition of the query into subtasks, executes each subtask's tool call
independently, then makes a single synthesis call over all gathered
observations. Trade-off it's built to demonstrate in the benchmark: usually
fewer LLM round-trips for multi-part questions, at the cost of not being
able to adapt the plan based on what earlier steps found.

The planner here is a deliberately simple heuristic splitter (not an LLM
call) -- keeping planning cheap and dependency-free lets the benchmark
isolate the *execution* trade-off rather than also varying planning quality.
"""

from __future__ import annotations

import re
import uuid

from src.agents.base import estimate_tokens, timer
from src.core.interfaces import AgentRunResult, AgentStep, AgentStrategy, LLMClient, ToolCall, ToolResult
from src.tools.base import to_tool_schema
from src.tools.registry import ToolRegistry

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools. Use document_search "
    "to ground answers in the ingested document(s); use calculator for "
    "arithmetic; use summarize_text to condense long passages before "
    "answering."
)

_SPLIT_RE = re.compile(r"\s+and\s+|;\s*", re.IGNORECASE)


def _plan(query: str, max_subtasks: int = 3) -> list[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(query) if p.strip()]
    return parts[:max_subtasks] if len(parts) > 1 else [query]


class PlanExecuteAgent(AgentStrategy):
    name = "plan_execute"

    def __init__(self, llm: LLMClient, registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = registry

    def run(self, query: str, max_steps: int = 8) -> AgentRunResult:
        with timer() as t:
            steps: list[AgentStep] = []
            tool_schemas = [to_tool_schema(tool) for tool in self.registry.all()]
            llm_calls = 0
            tool_calls = 0
            input_tokens = estimate_tokens(SYSTEM_PROMPT + query)
            output_tokens = 0

            subtasks = _plan(query, max_subtasks=max(1, max_steps - 1))
            steps.append(AgentStep(step_type="thought", content=f"Plan: {subtasks}"))

            observations: list[str] = []
            for subtask in subtasks:
                messages = [{"role": "user", "content": subtask}]
                response = self.llm.complete(messages, tools=tool_schemas, system=SYSTEM_PROMPT, mode="tool_use")
                llm_calls += 1
                tool_use_blocks = [b for b in response["content"] if b.get("type") == "tool_use"]

                if not tool_use_blocks:
                    text = " ".join(b["text"] for b in response["content"] if b.get("type") == "text")
                    output_tokens += estimate_tokens(text)
                    continue

                block = tool_use_blocks[0]  # one tool call per subtask keeps this strategy O(subtasks), not O(steps)
                call = ToolCall(tool_name=block["name"], arguments=block["input"], call_id=block.get("id", str(uuid.uuid4())))
                steps.append(AgentStep(step_type="tool_call", content=f"[{subtask}] -> {call.arguments}", tool_call=call))
                tool_calls += 1

                try:
                    tool = self.registry.get(call.tool_name)
                    output = tool.run(**call.arguments)
                    is_error = False
                except Exception as exc:  # noqa: BLE001
                    output = f"Tool error: {exc}"
                    is_error = True

                result = ToolResult(call_id=call.call_id, tool_name=call.tool_name, output=output, is_error=is_error)
                steps.append(AgentStep(step_type="observation", content=output, tool_result=result))
                output_tokens += estimate_tokens(output)
                observations.append(output)

            # Single synthesis call over every observation gathered across all subtasks.
            synth_messages: list[dict] = [{"role": "user", "content": query}]
            for obs in observations:
                synth_messages.append({"role": "tool", "tool_use_id": "synthesis", "content": obs})
            final_response = self.llm.complete(synth_messages, tools=[], system=SYSTEM_PROMPT, mode="tool_use")
            llm_calls += 1
            final_answer = " ".join(b["text"] for b in final_response["content"] if b.get("type") == "text").strip()
            final_answer = final_answer or "(no answer produced)"
            output_tokens += estimate_tokens(final_answer)
            steps.append(AgentStep(step_type="final_answer", content=final_answer))

        return AgentRunResult(
            query=query,
            final_answer=final_answer,
            steps=steps,
            latency_seconds=t["seconds"],
            tool_calls_made=tool_calls,
            llm_calls_made=llm_calls,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
        )
