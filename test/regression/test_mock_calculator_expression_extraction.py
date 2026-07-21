from __future__ import annotations

from src.core.llm_client import MockLLMClient
from src.tools.base import to_tool_schema
from src.tools.calculator_tool import CalculatorTool


def test_mock_extracts_full_parenthesized_expression():
    """
    Real bug found while running the benchmark (results/benchmark.md showed
    an 80% pass rate, one point below expected): the original regex
    `\\d+\\s*[+\\-*/]\\s*\\d+` only captured the FIRST two-number pair in a
    query, so "What is (25 + 15) / 2?" was reduced to just "25 + 15" (= 40)
    instead of the full expression (= 20). Fixed by matching a maximal run of
    arithmetic-looking characters instead of a fixed two-operand pattern.
    """
    llm = MockLLMClient()
    tools = [to_tool_schema(CalculatorTool())]
    messages = [{"role": "user", "content": "What is (25 + 15) / 2?"}]

    response = llm.complete(messages, tools=tools, mode="tool_use")
    tool_use = [b for b in response["content"] if b["type"] == "tool_use"][0]
    expression = tool_use["input"]["expression"]

    calculator = CalculatorTool()
    result = calculator.run(expression=expression)
    assert result == "20.0", f"Expected the full expression to evaluate to 20.0, got expression={expression!r} -> {result}"


def test_mock_still_extracts_simple_expression():
    llm = MockLLMClient()
    tools = [to_tool_schema(CalculatorTool())]
    messages = [{"role": "user", "content": "What is 12 * 8?"}]

    response = llm.complete(messages, tools=tools, mode="tool_use")
    tool_use = [b for b in response["content"] if b["type"] == "tool_use"][0]
    calculator = CalculatorTool()
    assert calculator.run(**tool_use["input"]) == "96"
