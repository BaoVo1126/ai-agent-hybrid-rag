"""
Raw Ollama adapter.

This module doesn't add new logic -- `src/core/llm_client.py::OllamaLLMClient`
already talks to a local Ollama server directly over plain HTTP, no framework
in between. This file exists just to make that "framework-agnostic and
free-by-default" choice explicit and documented, and to give the same three
tools a second calling convention: a single raw `/api/chat` call with
`tools=[...]` attached, no agent loop at all -- useful if you want to inspect
exactly what the model decides to do with one turn before wrapping it in any
agent strategy.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from src.config import SETTINGS
from src.tools.base import to_tool_schema
from src.tools.registry import ToolRegistry


def single_turn_tool_call(query: str, registry: ToolRegistry, system: str = "") -> dict[str, Any]:
    """One raw call to a local Ollama server with tools attached -- no loop,
    no framework, no API key. Requires `ollama serve` running and a
    tool-capable model already pulled (e.g. `ollama pull llama3.1`)."""
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in (to_tool_schema(t) for t in registry.all())
    ]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": query})

    payload = {"model": SETTINGS.ollama_model, "messages": messages, "tools": tool_schemas, "stream": False}
    request = urllib.request.Request(
        f"{SETTINGS.ollama_host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=SETTINGS.request_timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))
