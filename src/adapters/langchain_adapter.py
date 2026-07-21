"""
LangChain adapter.

This is the concrete answer to "make it usable with any framework or tool":
the Tool ABC in core/interfaces.py never imports LangChain, LlamaIndex, or
any other framework. This module is a thin, optional bridge that wraps our
Tool objects as LangChain `StructuredTool`s so they can be dropped straight
into a LangChain/LangGraph agent executor if that's the ecosystem you'd
rather use for a given deployment.

The import is guarded so the rest of the project (API, CLI, benchmark,
tests) never requires langchain to be installed -- this module is only
touched if you explicitly choose to use it.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Tool
from src.tools.registry import ToolRegistry


def to_langchain_tool(tool: Tool) -> Any:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is not installed. Install with `pip install langchain-core` "
            "to use this adapter -- it is entirely optional and unused elsewhere in the project."
        ) from exc

    def _run(**kwargs: Any) -> str:
        return tool.run(**kwargs)

    return StructuredTool.from_function(
        func=_run,
        name=tool.name,
        description=tool.description,
    )


def to_langchain_tools(registry: ToolRegistry) -> list[Any]:
    """Wrap every tool in the registry as a LangChain tool in one call."""
    return [to_langchain_tool(tool) for tool in registry.all()]
