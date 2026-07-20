# AI Agentic Assistant

**A framework-agnostic AI agent that answers questions from any document you give it — with a built-in benchmark comparing three different reasoning strategies.**

Drop a PDF, `.txt`, or `.md` file into `data/`, pick a strategy (ReAct / native function-calling / plan-and-execute), and ask questions through a CLI, a web UI, or the FastAPI backend directly. Runs fully offline in mock mode with zero setup, or against a real, free, locally-hosted model via [Ollama](https://ollama.com) by setting one environment variable — no API key, no cloud account, nothing to pay for.

![status](https://img.shields.io/badge/tests-31%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![mode](https://img.shields.io/badge/offline--first-yes-informational) ![cost](https://img.shields.io/badge/real--mode-free%20(Ollama)-success)

---

## Why this project exists

This is a learning project built on the way to becoming an AI Engineer, focused specifically on **agents**: the piece that sits on top of retrieval (see the companion project [`rag-from-scratch`](../rag-from-scratch)) and turns "search + generate" into "reason, decide, act, and check your own work."

The goal wasn't to pick *one* agent architecture and ship it — it was to build **one tool registry that three different reasoning strategies can share**, then measure the trade-off between them instead of just asserting it. That comparison is the centerpiece of this repo (`results/benchmark.md`), and it's meant to be re-run, re-argued with, and extended.

## What's actually in the box

- **Three interchangeable agent strategies** over the same tools:
  - **ReAct** — classic `Thought → Action → Action Input → Observation` text loop.
  - **Function calling** — the model's native tool-use (OpenAI/Ollama-style `tool_calls`).
  - **Plan-and-execute** — decompose the query into subtasks, execute each independently, synthesize once at the end.
- **A small hybrid retriever** (BM25 + TF-IDF fused with Reciprocal Rank Fusion) as the `document_search` tool, plus a calculator and an extractive summarizer — enough tools to make strategy comparison meaningful without the retrieval layer being the point.
- **A dual-mode LLM client**: a deterministic mock (offline, fully testable, zero install) and a real client that talks to a **free, local Ollama server** — behind the *same* interface, so every agent strategy is written once and runs in both modes unmodified.
- **A benchmark harness** that runs every strategy over the same eval set and reports pass rate, latency, tool calls, and estimated token cost side by side.
- **A web UI** that streams the reasoning trace (Thought / Action / Observation / Final Answer) live, with a strategy picker and drag-and-drop file upload.
- **Framework adapters** — the `Tool` interface never imports any specific agent framework; thin adapters show how to hand the same tools to LangChain or call Ollama's raw HTTP API directly with no framework at all.
- **31 passing tests** across unit, integration, and regression levels, including two real bugs found and fixed while building this (see [`docs/bugs-found.md`](docs/bugs-found.md)).

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────────────────┐
│  data/*.pdf  │───▶│  ingestion/      │───▶│  retrieval/                │
│  *.txt *.md  │    │  loaders,        │    │  BM25 + TF-IDF -> RRF      │
└─────────────┘    │  chunking, index │    │  fusion (HybridRetriever)  │
                    └──────────────────┘    └──────────────┬─────────────┘
                                                            │
                                                   ┌────────▼─────────┐
                                                   │  tools/           │
                                                   │  document_search  │
                                                   │  calculator        │
                                                   │  summarize_text     │
                                                   └────────┬──────────┘
                                                            │  (one registry, shared)
                     ┌──────────────────────────────────────┼──────────────────────────────────────┐
                     ▼                                      ▼                                      ▼
            ┌────────────────┐                    ┌──────────────────┐                  ┌────────────────────┐
            │  ReAct agent   │                    │ Function-calling │                  │  Plan & execute     │
            │  (text loop)   │                    │ agent (native)   │                  │  agent              │
            └────────┬───────┘                    └────────┬─────────┘                  └──────────┬──────────┘
                      └──────────────────────┬──────────────┴──────────────────┬────────────────────┘
                                              ▼                                 ▼
                                     ┌─────────────────┐              ┌───────────────────┐
                                     │  core/llm_client │              │  evaluation/       │
                                     │  Mock ↔ Ollama   │              │  benchmark.py      │
                                     │  (dual mode)     │              │  -> results/*.md    │
                                     └────────┬─────────┘              └────────────────────┘
                                              ▼
                              ┌───────────────────────────────┐
                              │  api/main.py (FastAPI)         │
                              │  + web/ (chat UI, trace view)  │
                              │  + scripts/run_agent_cli.py     │
                              └────────────────────────────────┘
```

Every box on the left of the tool registry (ingestion, retrieval) and every box on the right (the three agents, the API, the CLI) only depends on the `Tool` / `LLMClient` / `AgentStrategy` abstract interfaces in `src/core/interfaces.py` — not on each other's concrete implementations. That's what makes it possible to add a fourth agent strategy, or swap the retriever for a vector database, without touching anything else.

## Quickstart

```bash
git clone <this-repo>
cd ai-agent-lab
pip install -r requirements.txt

# 1. Add your document (or use the bundled sample to try it immediately)
cp "/path/to/AI Engineering.pdf" data/

# 2. Build the retrieval index (train-once, serve-many -- not rebuilt on every request)
python scripts/build_index.py

# 3a. Chat from the terminal
python scripts/run_agent_cli.py --strategy function_calling

# 3b. ...or launch the web UI
uvicorn src.api.main:app --reload
# then open http://localhost:8000

# 4. Run the strategy benchmark
python scripts/run_benchmark.py
```

All of the above runs **fully offline** — no install, no key, no account. To use a real model instead of the mock, for free, via [Ollama](https://ollama.com):

```bash
# 1. Install Ollama: https://ollama.com/download, then start it
ollama serve   # or just open the Ollama desktop app

# 2. Pull a tool-capable model (a few good options, pick based on your RAM)
ollama pull llama3.1        # 8B, good default
# ollama pull qwen2.5:7b    # also solid at tool-calling
# ollama pull mistral-nemo  # larger, stronger reasoning

# 3. Flip the switch -- no code changes needed anywhere
export LLM_BACKEND=ollama
export OLLAMA_MODEL=llama3.1   # must match what you pulled

python scripts/run_agent_cli.py --strategy react
```

Every downstream component (agents, API, benchmark) automatically switches from mock to real mode based solely on `LLM_BACKEND` — see `src/config.py::Settings.is_real_mode`. `OllamaLLMClient` talks to Ollama over plain HTTP using only the Python standard library, so there's no extra package to install either.

### Docker

```bash
# Mock mode (default) -- nothing extra needed
docker compose up --build

# Real mode against Ollama already running on your host machine
LLM_BACKEND=ollama docker compose up --build

# Real mode with Ollama running in its own container too (nothing installed on host)
docker compose --profile with-ollama up --build
docker exec -it ai-agent-lab-ollama-1 ollama pull llama3.1   # first time only
LLM_BACKEND=ollama OLLAMA_HOST=http://ollama:11434 docker compose --profile with-ollama up
```

Then open http://localhost:8000.

## Benchmark: comparing the three strategies

Run `python scripts/run_benchmark.py` any time — it re-runs all three strategies over `src/evaluation/eval_dataset.py` and rewrites `results/benchmark.md` (plus a chart if `matplotlib` is installed). Sample output in mock mode, on 5 example queries:

| Strategy | Pass rate | Avg latency (s) | Avg tool calls | Avg LLM calls | Avg tokens (est.) |
|---|---|---|---|---|---|
| react | 100% | 0.001 | 1.0 | 2.0 | 366 |
| function_calling | 100% | 0.000 | 1.0 | 2.0 | 191 |
| plan_execute | 100% | 0.000 | 1.2 | 2.2 | 192 |

**How to read it, and why the numbers look like that:**
- **Token cost**: ReAct is ~2x more expensive here because every turn re-sends the full tool list as text inside the system prompt, whereas function-calling sends tool schemas once via the API's native `tools` parameter. This gap gets larger, not smaller, as you add more tools.
- **LLM calls**: plan-execute makes one call per subtask plus one synthesis call, so it scales with query complexity rather than with reasoning depth — cheaper for simple single-part questions, but it can't adapt its plan mid-way the way ReAct and function-calling can.
- **Pass rate** is intentionally shallow (tool-usage + keyword containment, not an LLM judge) because this project doesn't know what document you'll drop into `data/` and so can't ship a real ground-truth answer key. Add a few `EvalExample`s with `expected_keywords` drawn from facts in *your* document (see `src/evaluation/eval_dataset.py`) for a meaningful accuracy signal on your own content.
- These numbers are from **mock mode**; the shape of the comparison (ReAct costs more tokens, plan-execute makes fewer adaptive decisions) holds in real mode too, but re-run `python scripts/run_benchmark.py` with `LLM_BACKEND=ollama` set for real latency and real token counts from Ollama's `prompt_eval_count` / `eval_count` fields.

## Project layout

```
src/
  core/            interfaces.py (Tool, LLMClient, AgentStrategy ABCs), llm_client.py (mock/Ollama), config.py
  ingestion/       loaders.py (pdf/txt/md), chunking.py, indexer.py (train-once, serve-many)
  retrieval/       bm25.py, tfidf.py, hybrid.py (RRF fusion)
  tools/           base.py, rag_tool.py, calculator_tool.py, summarize_tool.py, registry.py
  agents/          react_agent.py, function_calling_agent.py, plan_execute_agent.py, factory.py
  adapters/        langchain_adapter.py, raw_ollama_adapter.py  (framework-agnostic bridges)
  evaluation/      eval_dataset.py, metrics.py, benchmark.py
  api/             main.py (FastAPI), schemas.py
web/               index.html, style.css, app.js  (reasoning-trace console UI)
scripts/           build_index.py, run_agent_cli.py, run_benchmark.py
tests/             unit/, integration/, regression/  (31 tests)
data/              drop your .pdf/.txt/.md here
results/           benchmark.md + benchmark_chart.png land here
```

## Design decisions worth knowing about

- **Abstract interfaces first.** `Tool`, `LLMClient`, and `AgentStrategy` are defined once in `src/core/interfaces.py`. Every concrete piece (mock vs real LLM, three agent strategies, three tools) is swappable through these contracts — the same pattern used throughout the earlier `rag-from-scratch` and `chatbot-rag` projects, extended here to the reasoning layer itself.
- **Framework-agnostic by construction, not by claim.** `Tool` subclasses never import LangChain, LlamaIndex, or any specific model SDK. `src/adapters/` shows three different ways to actually invoke them: native Ollama tool-use (`OllamaLLMClient`), a raw single-turn HTTP call with no agent loop (`raw_ollama_adapter.py`), and a LangChain `StructuredTool` bridge (`langchain_adapter.py`, optional import).
- **Free and local by default.** `OllamaLLMClient` talks to a locally-running Ollama server over plain HTTP using nothing but the Python standard library — no API key to manage, no cloud bill, nothing leaves your machine, and it's a one-line env var to turn on or off.
- **Train-once, serve-many.** `scripts/build_index.py` builds and pickles the retrieval index separately from the API/CLI, which just load it. The API's `/api/upload` endpoint explicitly rebuilds on new file upload rather than re-indexing on every chat request.
- **The mock LLM only reads the current turn.** A real bug surfaced while building `chatbot-rag` (a mock scanning the *whole* conversation for keywords caused stale matches from earlier turns). `MockLLMClient._maybe_pick_tool` is deliberately scoped to only the latest user message — see the regression test in `tests/unit/test_mock_llm_client.py`.
- **Shallow-but-honest evaluation.** Rather than faking a ground-truth answer key for a document this project has never seen, `metrics.py` measures what's objectively checkable for any document (tool usage + keyword containment for tasks with a deterministic answer like arithmetic) and says so plainly in the benchmark output.

## Known bugs found and fixed while building this

Documented in detail, with the exact failure and fix, in [`docs/bugs-found.md`](docs/bugs-found.md) — both are also locked in as regression tests so they can't silently come back:

1. **Mock ReAct output wasn't valid JSON.** The mock formatted `Action Input` with Python's dict repr (single quotes) instead of `json.dumps`, so the agent's `json.loads` silently failed and the tool got called with no arguments.
2. **Mock calculator only extracted the first two-number pair.** `"What is (25 + 15) / 2?"` was reduced to `"25 + 15"`, dropping the division — caught by the benchmark showing an 80% instead of 100% pass rate on a deterministic arithmetic example.

## Extending this project

- **Add a document-specific eval set**: edit `src/evaluation/eval_dataset.py` with `EvalExample`s whose `expected_keywords` match facts you know are in your own file, for real accuracy numbers instead of tool-usage proxies.
- **Add a fourth strategy**: subclass `AgentStrategy` in `src/agents/`, register it in `src/agents/factory.py`, and it's automatically eligible for the benchmark and the API's `strategy` field.
- **Swap the retriever**: implement anything with a `.search(query, top_k)` method and pass it into `DocumentSearchTool` — the rest of the system doesn't know or care whether it's BM25 or a vector database.
- **True token-level streaming**: `/api/chat/stream` currently re-streams an already-completed trace step by step (see the note in `src/api/main.py`) — the natural next step is making the agent loop itself `async`/generator-based so the API can forward tokens live from Ollama's own streaming response (`"stream": true` in `/api/chat`).

## Testing

```bash
pytest tests/ -v
```

29 unit + integration tests plus 2 regression tests for the bugs above, all offline (mock mode) so CI never needs an API key.
