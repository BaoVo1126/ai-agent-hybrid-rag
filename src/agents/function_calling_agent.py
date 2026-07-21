"""
Function-calling agent.

Uses the model's *native* tool-use capability (tool_use content blocks --
Ollama's OpenAI-style function-calling and Anthropic's tool_use converge on
the same shape here) rather than prompting it to emit a specific text
format. This is generally the most reliable and lowest-overhead strategy for
models trained for tool use, which is exactly what the benchmark in
evaluation/ measures against the more manual react_agent.py.
"""

from __future__ import annotations

import uuid

from src.agents.base import estimate_tokens, timer
from src.core.interfaces import AgentRunResult, AgentStep, AgentStrategy, LLMClient, ToolCall, ToolResult
from src.tools.base import to_tool_schema
from src.tools.registry import ToolRegistry

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools. Use document_search "
    "to ground answers in the ingested document(s); use calculator for "
    "arithmetic; use summarize_text to condense long passages before "
    "answering. Only call a tool when it's actually needed, and give a "
    "concise final answer once you have enough information."
)


class FunctionCallingAgent(AgentStrategy):
    name = "function_calling"

    def __init__(self, llm: LLMClient, registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = registry

    def run(self, query: str, max_steps: int = 8) -> AgentRunResult:
        with timer() as t:
            steps: list[AgentStep] = []
            messages: list[dict] = [{"role": "user", "content": query}]
            tool_schemas = [to_tool_schema(tool) for tool in self.registry.all()]

            llm_calls = 0
            tool_calls = 0
            input_tokens = estimate_tokens(SYSTEM_PROMPT + query)
            output_tokens = 0
            final_answer = ""

            for _ in range(max_steps):
                response = self.llm.complete(messages, tools=tool_schemas, system=SYSTEM_PROMPT, mode="tool_use")
                llm_calls += 1
                messages.append({"role": "assistant", "content": response["content"]})

                text_parts = [b["text"] for b in response["content"] if b.get("type") == "text"]
                tool_use_blocks = [b for b in response["content"] if b.get("type") == "tool_use"]

                for text in text_parts:
                    output_tokens += estimate_tokens(text)
                    steps.append(AgentStep(step_type="thought", content=text))

                if not tool_use_blocks:
                    final_answer = " ".join(text_parts).strip() or "(no answer produced)"
                    steps.append(AgentStep(step_type="final_answer", content=final_answer))
                    break

                for block in tool_use_blocks:
                    call = ToolCall(tool_name=block["name"], arguments=block["input"], call_id=block.get("id", str(uuid.uuid4())))
                    steps.append(AgentStep(step_type="tool_call", content=str(call.arguments), tool_call=call))
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
                    messages.append({"role": "tool", "tool_use_id": call.call_id, "content": output})
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
