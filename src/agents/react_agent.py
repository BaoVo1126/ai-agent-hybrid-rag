"""
ReAct agent (Reason + Act), same loop shape used in rag-from-scratch's
ReAct agent with RAGTool/CalculatorTool, extended here to the general tool
registry. Kept as a first-class alternative strategy specifically so it can
be benchmarked against the native function_calling_agent -- the whole point
of this project is measuring that trade-off, not just picking one.
"""

from __future__ import annotations

import re
import uuid

from src.agents.base import estimate_tokens, timer
from src.core.interfaces import AgentRunResult, AgentStep, AgentStrategy, LLMClient, ToolCall, ToolResult
from src.tools.base import to_tool_schema, to_prompt_description
from src.tools.registry import ToolRegistry

ACTION_RE = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _build_system_prompt(registry: ToolRegistry) -> str:
    tool_lines = "\n".join(to_prompt_description(t) for t in registry.all())
    return (
        "You are an AI agent that reasons step by step using the ReAct format.\n"
        "Available tools:\n"
        f"{tool_lines}\n\n"
        "On each turn respond with EITHER:\n"
        "  Thought: <your reasoning>\n"
        "  Action: <tool_name>\n"
        "  Action Input: {\"arg_name\": \"value\"}\n"
        "OR, once you have enough information:\n"
        "  Thought: <your reasoning>\n"
        "  Final Answer: <your answer>\n"
        "Never call a tool that is not in the list above."
    )


class ReActAgent(AgentStrategy):
    name = "react"

    def __init__(self, llm: LLMClient, registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = _build_system_prompt(registry)

    def run(self, query: str, max_steps: int = 8) -> AgentRunResult:
        with timer() as t:
            steps: list[AgentStep] = []
            messages: list[dict] = [{"role": "user", "content": query}]
            # Tool schemas are still passed through so the mock client can
            # match on tool *names*; real Claude ignores `tools` in react_text mode
            # (see RealLLMClient.complete) and only reads the text system prompt.
            tool_schemas = [to_tool_schema(t) for t in self.registry.all()]

            llm_calls = 0
            tool_calls = 0
            input_tokens = estimate_tokens(self.system_prompt + query)
            output_tokens = 0
            final_answer = ""

            for _ in range(max_steps):
                response = self.llm.complete(messages, tools=tool_schemas, system=self.system_prompt, mode="react_text")
                llm_calls += 1
                text = response["content"][0]["text"] if response["content"] else ""
                output_tokens += estimate_tokens(text)
                messages.append({"role": "assistant", "content": text})

                final_match = FINAL_ANSWER_RE.search(text)
                if final_match:
                    final_answer = final_match.group(1).strip()
                    steps.append(AgentStep(step_type="final_answer", content=final_answer))
                    break

                action_match = ACTION_RE.search(text)
                input_match = ACTION_INPUT_RE.search(text)
                steps.append(AgentStep(step_type="thought", content=text))

                if not action_match:
                    final_answer = text.strip() or "(no action or final answer parsed)"
                    steps.append(AgentStep(step_type="final_answer", content=final_answer))
                    break

                tool_name = action_match.group(1)
                try:
                    import json

                    args = json.loads(input_match.group(1)) if input_match else {}
                except Exception:  # noqa: BLE001
                    args = {}

                call = ToolCall(tool_name=tool_name, arguments=args, call_id=str(uuid.uuid4()))
                steps.append(AgentStep(step_type="tool_call", content=str(args), tool_call=call))
                tool_calls += 1

                try:
                    tool = self.registry.get(tool_name)
                    output = tool.run(**args)
                    is_error = False
                except Exception as exc:  # noqa: BLE001
                    output = f"Tool error: {exc}"
                    is_error = True

                result = ToolResult(call_id=call.call_id, tool_name=tool_name, output=output, is_error=is_error)
                steps.append(AgentStep(step_type="observation", content=output, tool_result=result))
                output_tokens += estimate_tokens(output)
                messages.append({"role": "tool", "tool_use_id": call.call_id, "content": f"Observation: {output}"})
            else:
                final_answer = "(max steps reached without a final answer)"

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
