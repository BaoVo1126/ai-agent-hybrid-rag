"""
Core abstract interfaces.

Why this file exists
---------------------
Every previous project in this learning track (rag-from-scratch, chatbot-rag)
leaned on the same principle: define the *contract* once as an ABC, then let
concrete implementations be swapped without touching the pipeline that calls
them. This project pushes that one level further — not just swapping
embedders/rerankers, but swapping the **agent reasoning strategy** and the
**tool set** itself, which is what makes the agent "framework agnostic".

If you later want to plug in LangChain, LlamaIndex, or a hand-rolled
Ollama/Anthropic tool-use loop, you only need to implement these interfaces
— the rest of the system (API, evaluation harness, UI) never needs to know
which one is running underneath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Document:
    """A single retrievable unit of text with metadata."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A request from the agent's reasoning loop to invoke a tool."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResult:
    """The observation returned after executing a ToolCall."""

    call_id: str
    tool_name: str
    output: str
    is_error: bool = False


@dataclass
class AgentStep:
    """
    One iteration of the agent loop, kept for transparency and evaluation.

    Recording every step (not just the final answer) is what lets the
    benchmark harness compare strategies on *how* they got the answer
    (tool calls, reasoning turns) and not only *whether* they got it right.
    """

    step_type: str  # "thought" | "tool_call" | "observation" | "final_answer"
    content: str
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


@dataclass
class AgentRunResult:
    """Full trace + final answer + basic performance counters for one query."""

    query: str
    final_answer: str
    steps: list[AgentStep]
    latency_seconds: float
    tool_calls_made: int
    llm_calls_made: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    # Only meaningful for self_correcting_rag_agent.py: how many
    # retrieve->generate->verify loops it needed beyond the first attempt.
    # Defaults to 0 so every other existing strategy is unaffected.
    self_correction_retries: int = 0


class Tool(ABC):
    """
    Base class every tool must implement.

    A tool is intentionally described the same way regardless of which
    agent strategy calls it — name + description + JSON schema + run().
    This is also the shape both Ollama's and Anthropic's native tool-use
    APIs expect (with a trivial key-renaming adapter, see
    `OllamaLLMClient._to_ollama_tool`), so the same Tool subclass can be
    handed directly to whichever backend is active in
    `function_calling_agent.py`, or manually formatted into a text prompt
    for `react_agent.py`. One implementation, multiple calling conventions.
    """

    name: str
    description: str

    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON schema describing the tool's expected arguments."""

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a plain-text observation."""


class LLMClient(ABC):
    """
    Abstract chat-completion client.

    Two concrete implementations exist: a MockLLMClient (deterministic,
    offline, used unless LLM_BACKEND=ollama is set) and an OllamaLLMClient
    (calls a free, locally-hosted model via https://ollama.com). Every agent
    strategy is written against this interface, so switching between
    mock/real mode never requires touching agent code — the same
    "dual-mode" pattern used in chatbot-rag.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        mode: str = "tool_use",
    ) -> dict[str, Any]:
        """
        Return a raw response dict shaped like the Anthropic/OpenAI-style
        Messages API (both Ollama's native tool-calling format and the
        Anthropic API converge on the same "list of content blocks with a
        `type` of `text` or `tool_use`" shape, which is what this internal
        representation mirrors).

        `mode` distinguishes two calling conventions used by different agent
        strategies against the *same* client:
          - "tool_use": native tool-calling (function_calling_agent.py) --
            `tools` is a list of JSON schemas, response may contain a
            `tool_use` content block.
          - "react_text": classic ReAct prompting (react_agent.py) -- tools
            are described in plain text inside `system`, and the model is
            expected to reply with free text in a Thought/Action/Action
            Input/Final Answer format that the agent parses itself.
        """


class AgentStrategy(ABC):
    """
    Base class for an agent reasoning strategy (ReAct, native function-calling,
    plan-and-execute, self-correcting RAG, ...). Each strategy owns its own
    loop but shares the same tool registry and LLM client, which is what the
    benchmark harness exploits to run a fair head-to-head comparison.
    """

    name: str

    @abstractmethod
    def run(self, query: str, max_steps: int = 8) -> AgentRunResult:
        """Execute the agent loop for a single query and return a full trace."""

    def run_stream(self, query: str, max_steps: int = 8) -> "Iterator[AgentStep]":
        """
        Yields AgentStep objects as they happen, for live SSE streaming
        (api/main.py POST /api/chat/stream), then implicitly finishes once
        the underlying run() completes.

        Default implementation: just runs to completion and yields the
        already-built steps -- correct output, but the UI sees the whole
        trace appear at once rather than progressively. This is what
        react_agent.py / function_calling_agent.py / plan_execute_agent.py
        currently get; self_correcting_rag_agent.py overrides this method to
        yield each step *as it's produced* instead (see that file), since
        its loop already runs one step at a time and refactoring it into a
        generator was a small, self-contained change. Give the other three
        the same treatment when they need true live streaming too -- the
        interface already supports it, no caller needs to change.
        """
        result = self.run(query, max_steps=max_steps)
        yield from result.steps


class DocumentLoader(ABC):
    """Loads raw files (pdf, txt, md, ...) into Document objects."""

    @abstractmethod
    def load(self, path: str) -> list[Document]:
        ...
