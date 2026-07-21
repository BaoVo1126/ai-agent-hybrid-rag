from __future__ import annotations

from src.core.llm_client import MockLLMClient
from src.tools.base import to_tool_schema
from src.tools.calculator_tool import CalculatorTool
from src.tools.rag_tool import DocumentSearchTool


def _tools(hybrid_retriever):
    return [to_tool_schema(CalculatorTool()), to_tool_schema(DocumentSearchTool(hybrid_retriever))]


def test_mock_picks_calculator_for_arithmetic(hybrid_retriever):
    llm = MockLLMClient()
    messages = [{"role": "user", "content": "what is 4 + 5"}]
    response = llm.complete(messages, tools=_tools(hybrid_retriever), mode="tool_use")
    tool_use = [b for b in response["content"] if b["type"] == "tool_use"]
    assert tool_use and tool_use[0]["name"] == "calculator"


def test_mock_picks_document_search_for_document_question(hybrid_retriever):
    llm = MockLLMClient()
    messages = [{"role": "user", "content": "explain what the document says about agents"}]
    response = llm.complete(messages, tools=_tools(hybrid_retriever), mode="tool_use")
    tool_use = [b for b in response["content"] if b["type"] == "tool_use"]
    assert tool_use and tool_use[0]["name"] == "document_search"


def test_mock_does_not_leak_earlier_turn_keywords_into_later_decision(hybrid_retriever):
    """
    Regression test for the exact bug class documented in chatbot-rag: the
    mock's tool-selection must only look at the CURRENT user turn. A math
    keyword in an earlier turn must not cause the calculator to fire again
    on an unrelated later question.
    """
    llm = MockLLMClient()
    messages = [
        {"role": "user", "content": "what is 4 + 5"},
        {"role": "assistant", "content": [{"type": "text", "text": "9"}]},
        {"role": "user", "content": "explain what the document says about evaluation"},
    ]
    response = llm.complete(messages, tools=_tools(hybrid_retriever), mode="tool_use")
    tool_use = [b for b in response["content"] if b["type"] == "tool_use"]
    assert tool_use and tool_use[0]["name"] == "document_search", (
        "Mock incorrectly re-triggered a tool based on an earlier turn's keywords."
    )


def test_mock_synthesizes_final_answer_after_tool_result(hybrid_retriever):
    llm = MockLLMClient()
    messages = [
        {"role": "user", "content": "what is 4 + 5"},
        {"role": "assistant", "content": [{"type": "text", "text": "using calculator"}]},
        {"role": "tool", "tool_use_id": "call_1", "content": "9"},
    ]
    response = llm.complete(messages, tools=_tools(hybrid_retriever), mode="tool_use")
    assert response["stop_reason"] == "end_turn"
    text = response["content"][0]["text"]
    assert "9" in text
