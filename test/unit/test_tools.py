from __future__ import annotations

import pytest

from src.tools.calculator_tool import CalculatorTool
from src.tools.rag_tool import DocumentSearchTool
from src.tools.summarize_tool import SummarizeTool


def test_calculator_basic_arithmetic():
    tool = CalculatorTool()
    assert tool.run(expression="2 + 3 * 4") == "14"


def test_calculator_handles_parentheses():
    tool = CalculatorTool()
    assert tool.run(expression="(2 + 3) * 4") == "20"


def test_calculator_rejects_unsafe_input():
    tool = CalculatorTool()
    result = tool.run(expression="__import__('os').system('echo hi')")
    assert "Error" in result


def test_document_search_returns_relevant_chunk(hybrid_retriever):
    tool = DocumentSearchTool(hybrid_retriever, top_k=2)
    result = tool.run(query="what is an AI agent")
    assert "agent" in result.lower()


def test_document_search_empty_index_returns_message():
    from src.retrieval.hybrid import HybridRetriever

    empty_retriever = HybridRetriever()
    empty_retriever.fit([])
    tool = DocumentSearchTool(empty_retriever)
    result = tool.run(query="anything")
    assert "no relevant" in result.lower()


def test_summarize_tool_shrinks_text():
    tool = SummarizeTool(max_sentences=1)
    long_text = (
        "Retrieval augmented generation is a technique. It combines retrieval and generation. "
        "The weather today is unrelated. Agents can use tools to act."
    )
    result = tool.run(text=long_text)
    assert len(result) < len(long_text)


@pytest.mark.parametrize("tool_cls", [CalculatorTool, SummarizeTool])
def test_tool_schema_has_required_fields(tool_cls):
    tool = tool_cls()
    schema = tool.input_schema()
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema
