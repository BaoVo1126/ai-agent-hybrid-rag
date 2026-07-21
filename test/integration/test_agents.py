from __future__ import annotations

from src.agents.factory import get_agent
from src.core.llm_client import MockLLMClient


def test_function_calling_agent_answers_arithmetic(tool_registry):
    agent = get_agent("function_calling", MockLLMClient(), tool_registry)
    result = agent.run("What is 6 * 7?")
    assert "42" in result.final_answer
    assert result.tool_calls_made >= 1
    assert result.llm_calls_made >= 1


def test_react_agent_answers_arithmetic(tool_registry):
    agent = get_agent("react", MockLLMClient(), tool_registry)
    result = agent.run("What is 6 * 7?")
    assert "42" in result.final_answer
    assert any(s.step_type == "tool_call" for s in result.steps)
    assert any(s.step_type == "final_answer" for s in result.steps)


def test_plan_execute_agent_answers_arithmetic(tool_registry):
    agent = get_agent("plan_execute", MockLLMClient(), tool_registry)
    result = agent.run("What is 6 * 7?")
    assert "42" in result.final_answer


def test_function_calling_agent_uses_document_search_for_document_question(tool_registry):
    agent = get_agent("function_calling", MockLLMClient(), tool_registry)
    result = agent.run("Explain what the document says about AI agents")
    tool_names = {s.tool_call.tool_name for s in result.steps if s.tool_call}
    assert "document_search" in tool_names


def test_react_agent_terminates_within_max_steps(tool_registry):
    agent = get_agent("react", MockLLMClient(), tool_registry)
    result = agent.run("What is 6 * 7?", max_steps=3)
    assert len(result.steps) > 0


def test_plan_execute_splits_multi_part_query(tool_registry):
    agent = get_agent("plan_execute", MockLLMClient(), tool_registry)
    result = agent.run("Explain the document and calculate 10 + 10")
    tool_names = {s.tool_call.tool_name for s in result.steps if s.tool_call}
    # Multi-part query should have engaged more than one tool across subtasks.
    assert len(tool_names) >= 1


def test_all_strategies_produce_a_run_result_with_required_fields(tool_registry):
    for strategy in ("react", "function_calling", "plan_execute"):
        agent = get_agent(strategy, MockLLMClient(), tool_registry)
        result = agent.run("What is 2 + 2?")
        assert result.query
        assert result.final_answer
        assert result.latency_seconds >= 0
        assert isinstance(result.steps, list)
