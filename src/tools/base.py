"""
Tool base class.

Every tool subclasses `src.core.interfaces.Tool`. This module just adds a
small convenience: `to_tool_schema()`, which formats a Tool as a
name/description/input_schema dict -- the shape both Ollama's and
Anthropic's native tool-use APIs expect (OllamaLLMClient does a trivial
key-rename on top of this, see `_to_ollama_tool`). The ReAct agent formats
the same tool differently (as text in a prompt) -- one Tool implementation,
multiple presentations, which is the whole point of decoupling "what a tool
does" from "how a given agent strategy calls it".
"""

from __future__ import annotations

from src.core.interfaces import Tool


def to_tool_schema(tool: Tool) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema(),
    }


def to_prompt_description(tool: Tool) -> str:
    """A one-line textual description used inside the ReAct prompt."""
    schema = tool.input_schema()
    args = ", ".join(schema.get("properties", {}).keys())
    return f"- {tool.name}({args}): {tool.description}"
