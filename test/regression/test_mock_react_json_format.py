from __future__ import annotations

import json
import re

from src.core.llm_client import MockLLMClient
from src.tools.base import to_tool_schema
from src.tools.calculator_tool import CalculatorTool

ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})", re.IGNORECASE | re.DOTALL)


def test_mock_react_action_input_is_valid_json_not_python_repr():
    """
    Real bug found while building this project: the mock's react_text branch
    originally formatted Action Input with an f-string over a Python dict,
    e.g. `{tool_use['input']}` -> "{'expression': '6 * 7'}". That is valid
    Python repr but NOT valid JSON (single quotes). react_agent.py parses
    Action Input with `json.loads`, which raised, was silently swallowed by
    a bare except, and the agent then called the tool with an empty argument
    dict -- producing a confusing "Tool error: 'expression'" KeyError instead
    of a real answer. The fix formats Action Input with `json.dumps`.

    This test locks in the fix: the emitted Action Input must round-trip
    through `json.loads` cleanly.
    """
    llm = MockLLMClient()
    tools = [to_tool_schema(CalculatorTool())]
    messages = [{"role": "user", "content": "What is 6 * 7?"}]

    response = llm.complete(messages, tools=tools, system="sys", mode="react_text")
    text = response["content"][0]["text"]

    match = ACTION_INPUT_RE.search(text)
    assert match, f"Expected an 'Action Input: {{...}}' line, got: {text!r}"

    parsed = json.loads(match.group(1))  # must not raise
    assert parsed == {"expression": "6 * 7"}


def test_react_agent_end_to_end_no_longer_fails_on_arithmetic(tool_registry):
    """Full end-to-end regression check for the same bug via the public agent API."""
    from src.agents.factory import get_agent

    agent = get_agent("react", MockLLMClient(), tool_registry)
    result = agent.run("What is 6 * 7?")
    assert "42" in result.final_answer
    assert "Tool error" not in result.final_answer
