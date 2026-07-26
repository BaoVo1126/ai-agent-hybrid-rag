from __future__ import annotations

from src.agents.factory import get_agent
from src.agents.guarded_agent import GuardedAgentStrategy
from src.core.interfaces import AgentRunResult, AgentStep, AgentStrategy
from src.core.llm_client import MockLLMClient


class _StubAgent(AgentStrategy):
    name = "stub"

    def __init__(self) -> None:
        self.run_called = False

    def run(self, query: str, max_steps: int = 8) -> AgentRunResult:
        self.run_called = True
        return AgentRunResult(
            query=query,
            final_answer="real answer",
            steps=[AgentStep(step_type="final_answer", content="real answer")],
            latency_seconds=0.01,
            tool_calls_made=1,
            llm_calls_made=1,
            estimated_input_tokens=5,
            estimated_output_tokens=5,
        )


def test_rejects_empty_query_without_calling_inner_agent():
    inner = _StubAgent()
    guarded = GuardedAgentStrategy(inner)

    result = guarded.run("")

    assert not inner.run_called
    assert result.tool_calls_made == 0
    assert result.llm_calls_made == 0
    assert result.final_answer  # a clarifying message, not empty


def test_rejects_gibberish_query_without_calling_inner_agent():
    inner = _StubAgent()
    guarded = GuardedAgentStrategy(inner)

    result = guarded.run("asdkfjhaslkdjfh qwoprkjhsdflkj")

    assert not inner.run_called
    assert result.llm_calls_made == 0


def test_passes_through_valid_query_to_inner_agent():
    inner = _StubAgent()
    guarded = GuardedAgentStrategy(inner)

    result = guarded.run("What does the document say about AI agents?")

    assert inner.run_called
    assert result.final_answer == "real answer"


def test_run_stream_rejects_nonsensical_query_without_calling_inner_agent():
    inner = _StubAgent()
    guarded = GuardedAgentStrategy(inner)

    steps = list(guarded.run_stream("!!!!!?????#####"))

    assert not inner.run_called
    assert len(steps) == 1
    assert steps[0].step_type == "final_answer"


def test_get_agent_wraps_every_strategy_with_guard(tool_registry):
    for strategy in ("react", "function_calling", "plan_execute", "self_correcting_rag"):
        agent = get_agent(strategy, MockLLMClient(), tool_registry)
        assert isinstance(agent, GuardedAgentStrategy)
        assert agent.name == strategy


def test_end_to_end_nonsense_query_through_factory_costs_zero_llm_calls(tool_registry):
    agent = get_agent("function_calling", MockLLMClient(), tool_registry)
    result = agent.run("asdkfjhaslkdjfh qwoprkjhsdflkj")
    assert result.llm_calls_made == 0
    assert result.tool_calls_made == 0
