"""Factory so the API/CLI/benchmark can select a strategy by name string
(e.g. from a UI dropdown or a --strategy CLI flag) without importing every
agent class individually."""

from __future__ import annotations

from src.core.interfaces import AgentStrategy, LLMClient
from src.tools.registry import ToolRegistry

STRATEGIES = {
    "react": "src.agents.react_agent.ReActAgent",
    "function_calling": "src.agents.function_calling_agent.FunctionCallingAgent",
    "plan_execute": "src.agents.plan_execute_agent.PlanExecuteAgent",
    "self_correcting_rag": "src.agents.self_correcting_rag_agent.SelfCorrectingRAGAgent",
}


def get_agent(strategy: str, llm: LLMClient, registry: ToolRegistry) -> AgentStrategy:
    if strategy == "react":
        from src.agents.react_agent import ReActAgent

        return ReActAgent(llm, registry)
    if strategy == "function_calling":
        from src.agents.function_calling_agent import FunctionCallingAgent

        return FunctionCallingAgent(llm, registry)
    if strategy == "plan_execute":
        from src.agents.plan_execute_agent import PlanExecuteAgent

        return PlanExecuteAgent(llm, registry)
    if strategy == "self_correcting_rag":
        from src.agents.self_correcting_rag_agent import SelfCorrectingRAGAgent

        return SelfCorrectingRAGAgent(llm, registry)
    raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(STRATEGIES)}")
