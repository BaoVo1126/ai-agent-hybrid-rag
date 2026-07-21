"""
Dual-mode LLM client.

Same philosophy as chatbot-rag: the whole system must run offline, with zero
setup, so the agent loop, tool-calling contract, and evaluation harness can
all be exercised and unit-tested without any external dependency. Set
LLM_BACKEND=ollama (and run a local Ollama server) and everything
transparently switches from the deterministic mock to a real, locally-hosted
model -- free, no API key, no account, nothing leaves your machine.

Bug learned the hard way in the previous project and deliberately avoided
here: the mock's tool-selection logic must only look at the *current* user
turn, never the full running transcript. Scanning the whole conversation
caused false keyword matches on earlier turns (e.g. a past message mentioning
"calculate" made the mock keep re-triggering the calculator tool on unrelated
later questions). Scope the pattern-matching input tightly.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
import uuid
from typing import Any

from src.config import SETTINGS
from src.core.interfaces import LLMClient


class MockLLMClient(LLMClient):
    """
    A small rule-based stand-in for a real model, used unless LLM_BACKEND=ollama
    is set.

    It does two things a real model would do inside the agent loop:
      1. Decide whether a tool is needed for the *current* user message.
      2. Produce a final natural-language answer once observations exist.

    It intentionally does NOT try to be a good chatbot — it exists purely so
    the surrounding agent/tool/evaluation machinery can be built, tested, and
    demoed without network access or a running Ollama server.
    """

    # Matches a maximal run of arithmetic-looking characters; validated (has a
    # digit AND an operator) before being treated as an expression. A narrower
    # two-number-only pattern was tried first and silently truncated
    # parenthesized/multi-operator expressions like "(25 + 15) / 2" down to
    # just "25 + 15" -- caught via the benchmark eval set, see
    # tests/regression/test_mock_calculator_expression_extraction.py.
    _MATH_CHARS_PATTERN = re.compile(r"[\d\.\s\+\-\*/x×\(\)]{3,}")

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        mode: str = "tool_use",
    ) -> dict[str, Any]:
        if mode == "react_text":
            return self._complete_react_text(messages, tools or [])
        return self._complete_tool_use(messages, tools or [])

    def _complete_tool_use(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        last_user_msg = self._last_user_text(messages)
        has_tool_result = any(m.get("role") == "tool" for m in messages)

        if not has_tool_result and tools:
            tool_use = self._maybe_pick_tool(last_user_msg, tools)
            if tool_use:
                return {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"I should look this up using {tool_use['name']}."},
                        {"type": "tool_use", "id": tool_use["id"], "name": tool_use["name"], "input": tool_use["input"]},
                    ],
                    "stop_reason": "tool_use",
                }

        # No (more) tools needed -> synthesize a final answer from whatever
        # tool observations are already in the transcript.
        observations = [m["content"] for m in messages if m.get("role") == "tool"]
        answer = self._synthesize_answer(last_user_msg, observations)
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
            "stop_reason": "end_turn",
        }

    def _complete_react_text(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Emits plain text in the Thought/Action/Action Input format that
        react_agent.py parses. Only the *most recent* user turn (the
        original query) and the *most recent* observation (if any) drive the
        decision -- scanning the whole transcript here is exactly the bug
        class this project's memory notes warn about.
        """
        last_user_msg = self._last_user_text(messages)
        last_observation = self._last_observation_text(messages)
        tool_names = {t["name"] for t in tools}

        if last_observation is None:
            tool_use = self._maybe_pick_tool(last_user_msg, tools)
            if tool_use:
                import json

                # NOTE: this must be valid JSON (json.dumps), not Python's dict
                # repr (str(...)), which uses single quotes and silently fails
                # react_agent.py's json.loads -> the agent would then call the
                # tool with an empty argument dict and raise a confusing
                # "Tool error: 'expression'" KeyError. Caught via test
                # tests/regression/test_mock_react_json_format.py.
                text = (
                    f"Thought: I need more information to answer '{last_user_msg}'.\n"
                    f"Action: {tool_use['name']}\n"
                    f"Action Input: {json.dumps(tool_use['input'])}"
                )
                return {"role": "assistant", "content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}

        answer = self._synthesize_answer(last_user_msg, [last_observation] if last_observation else [])
        text = f"Thought: I now have enough information.\nFinal Answer: {answer}"
        return {"role": "assistant", "content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}

    @staticmethod
    def _last_observation_text(messages: list[dict[str, Any]]) -> str | None:
        for m in reversed(messages):
            if m.get("role") == "tool":
                return m.get("content")
        return None

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                return content if isinstance(content, str) else str(content)
        return ""

    def _maybe_pick_tool(self, text: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Scoped ONLY to `text` (the current user turn) -- see module docstring."""
        lowered = text.lower()
        tool_names = {t["name"] for t in tools}

        if "calculator" in tool_names:
            expr = self._extract_math_expression(lowered)
            if expr:
                return {"id": "call_calc_1", "name": "calculator", "input": {"expression": expr}}

        if "document_search" in tool_names and any(
            kw in lowered for kw in ("document", "pdf", "paper", "according to", "what does", "explain", "summarize", "chunk", "section")
        ):
            return {"id": "call_search_1", "name": "document_search", "input": {"query": text}}

        return None

    def _extract_math_expression(self, lowered_text: str) -> str | None:
        best_match = None
        for match in self._MATH_CHARS_PATTERN.finditer(lowered_text):
            candidate = match.group(0).strip()
            has_digit = any(c.isdigit() for c in candidate)
            has_operator = any(c in "+-*/x×" for c in candidate)
            if has_digit and has_operator and (best_match is None or len(candidate) > len(best_match)):
                best_match = candidate
        if best_match is None:
            return None
        return best_match.replace("x", "*").replace("×", "*")

    @staticmethod
    def _synthesize_answer(query: str, observations: list[str]) -> str:
        if not observations:
            return (
                f"[mock-mode] I don't have retrieval evidence for '{query}'. "
                "Set LLM_BACKEND=ollama (with a local Ollama server running) and re-run for a real generated answer."
            )
        joined = " | ".join(o[:220] for o in observations)
        return f"[mock-mode] Based on the retrieved evidence: {joined}"


class OllamaLLMClient(LLMClient):
    """
    Thin wrapper around a local Ollama server (https://ollama.com).

    Free, no API key, nothing leaves your machine. Requires:
      1. Ollama installed and running (`ollama serve`, or the desktop app).
      2. A tool-capable model pulled locally, e.g. `ollama pull llama3.1`.

    Talks to Ollama's native `/api/chat` endpoint over plain HTTP using only
    the standard library (`urllib`) -- no extra pip package needed, on top of
    everything else already dependency-free in mock mode.
    """

    def __init__(self) -> None:
        self.host = SETTINGS.ollama_host.rstrip("/")
        self.model = SETTINGS.ollama_model

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        mode: str = "tool_use",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_ollama_messages(messages, system),
            "stream": False,
        }
        # In "react_text" mode, tools are described in the system prompt as
        # plain text and the model replies in free text -- so we deliberately
        # do NOT pass native tool schemas in that mode (mirrors the mock).
        if mode == "tool_use" and tools:
            payload["tools"] = [self._to_ollama_tool(t) for t in tools]

        data = self._post("/api/chat", payload)
        message = data.get("message", {}) or {}

        content: list[dict[str, Any]] = []
        text = message.get("content") or ""
        if text:
            content.append({"type": "text", "text": text})

        for tool_call in message.get("tool_calls") or []:
            func = tool_call.get("function", {}) or {}
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or str(uuid.uuid4()),
                    "name": func.get("name", ""),
                    "input": args or {},
                }
            )

        if not content:
            content = [{"type": "text", "text": ""}]
        stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"

        return {
            "role": "assistant",
            "content": content,
            "stop_reason": stop_reason,
            "usage": {
                # Ollama reports these as prompt/eval token counts per response.
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
        }

    @staticmethod
    def _to_ollama_tool(tool_schema: dict[str, Any]) -> dict[str, Any]:
        """Our internal tool schema (name/description/input_schema) maps
        directly onto Ollama's OpenAI-style function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema.get("description", ""),
                "parameters": tool_schema.get("input_schema", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _to_ollama_messages(messages: list[dict[str, Any]], system: str | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            role = m.get("role")
            if role == "tool":
                out.append({"role": "tool", "content": m.get("content", "")})
            elif role == "assistant":
                content = m.get("content")
                if isinstance(content, list):
                    text = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
                    out.append({"role": "assistant", "content": text})
                else:
                    out.append({"role": "assistant", "content": content or ""})
            else:
                out.append({"role": "user", "content": m.get("content", "")})
        return out

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.host}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=SETTINGS.request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running? "
                f"Install it from https://ollama.com, then `ollama pull {self.model}` "
                "before setting LLM_BACKEND=ollama."
            ) from exc


def get_llm_client() -> LLMClient:
    """Factory: real mode iff LLM_BACKEND=ollama is set (see Settings.is_real_mode)."""
    return OllamaLLMClient() if SETTINGS.is_real_mode else MockLLMClient()
