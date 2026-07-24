# 🤖 AI Agents with Hybrid RAG

**A production-ready, framework-agnostic AI agent that answers questions from any document you give it — benchmarking four different reasoning strategies side by side, with real Postgres/Redis/pgvector infrastructure behind it, and an optional LoRA fine-tuning step to push accuracy further.**

![status](https://img.shields.io/badge/tests-43%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![cost](https://img.shields.io/badge/real--mode-free%20(Ollama)-success) ![infra](https://img.shields.io/badge/storage-Postgres%20%2B%20pgvector%20%2B%20Redis-336791) ![streaming](https://img.shields.io/badge/API-SSE%20streaming-informational) ![finetuning](https://img.shields.io/badge/finetuning-LoRA%20(optional)-orange)

🔗 **Live demo:** https://ai-agent-hybrid-rag.onrender.com/

---

## 📌 What this is

Drop a PDF/TXT/MD file in, ask questions through a web UI, CLI, or API — the agent retrieves the relevant passages, reasons about the answer, and (in its most advanced mode) **checks its own answer for hallucination before returning it**, retrying automatically if it's wrong. Runs fully offline with zero setup in mock mode, or against a real, free, local model via [Ollama](https://ollama.com) with one environment variable — no API key, no cloud bill. When retrieval + prompting alone aren't accurate enough on your specific document, an optional **LoRA fine-tuning** step adapts a small local model's weights to it directly.

## ✨ What it does

- 🧠 **Four interchangeable reasoning strategies** over the same tools — ReAct, native function-calling, plan-and-execute, and a **self-correcting RAG agent** that grades its own retrieved evidence and re-tries when its answer isn't grounded
- 🔍 **Hybrid retrieval** — BM25 + dense vector search fused via Reciprocal Rank Fusion, with an optional cross-encoder reranker on top
- 💬 **Persistent chat sessions** — full conversation history saved to Postgres, Redis-cached for fast reads
- ⚡ **Real-time streaming** — `/api/chat/stream` is true Server-Sent Events; the self-correcting agent streams each retrieval/verify/retry step live as it happens, not after the fact
- 📊 **Built-in benchmark** — quantitatively compares all four strategies on pass rate, latency, groundedness, and cost — not just a demo, a measurement tool
- 🎯 **Optional LoRA fine-tuning** — when accuracy on your own document still isn't high enough, adapt a small local model (`Qwen2.5-0.5B-Instruct` by default) to it: a dataset generator turns your document's own chunks into training examples, `peft` trains a LoRA adapter on top, and the result plugs in as a third `LLM_BACKEND` (`lora`) right alongside `mock`/`ollama` — no other code changes, and it's directly comparable in the same benchmark
- 🐳 **One-command production infra** — `docker compose up` brings up Postgres (pgvector) + Redis alongside the app

## 🛠️ Tech stack

| Layer | Tools |
|---|---|
| Agent orchestration | Custom `AgentStrategy` interface (ReAct / function-calling / plan-execute / self-correcting RAG) |
| LLM | [Ollama](https://ollama.com) (`llama3.1`, local & free) — dual-mode with a deterministic offline mock for CI |
| Fine-tuning | LoRA (`peft`) on a small local causal LM (`Qwen2.5-0.5B-Instruct` default) — dataset auto-generated from your document's own chunks; optional `torch`/`transformers`/`peft`/`datasets`/`accelerate` |
| Retrieval | BM25 + pgvector dense embeddings (`sentence-transformers`), RRF fusion, cross-encoder reranking |
| Vector storage | **Postgres + pgvector** (swappable for a zero-setup pickle file in dev) |
| Chat history | **Postgres**, **Redis** read-cache |
| API | **FastAPI**, Server-Sent Events streaming |
| Frontend | Vanilla JS chat console with a live reasoning-trace view |
| Testing | Pytest — 43 tests (unit / integration / regression), including real bugs caught by testing against a genuinely fresh environment |

## 🏗️ Architecture

```
data/*.pdf,txt,md → ingestion (chunk) → retrieval (BM25 + pgvector, RRF fusion)
                          │                              │
                          ▼                               │
              finetuning/ (dataset_builder,       tools/ (document_search, calculator, summarize)
              lora_trainer -> LoRA adapter)                │
                          │               ┌───────────────┬───────────────┬──────┴──────────┐
                          │               ▼               ▼               ▼                 ▼
                          │            ReAct        Function-calling  Plan & execute   Self-correcting RAG
                          │               └───────────────┴───────────────┴──────┬──────────┘
                          │                                                       ▼
                          └────────────────────────────────────────▶ core/llm_client (Mock ↔ Ollama ↔ LoRA)
                                                                                   │
                          api/main.py (FastAPI, SSE streaming)  ──▶  Postgres (chat history + vectors)
                                                                                   │                          ▲
                                                                              web/ (chat UI)          Redis (read cache)
```

Every agent, tool, and storage backend is swappable through abstract interfaces (`src/core/interfaces.py`) — adding a strategy, or moving from a pickle file to a real database, never touches the other layers. That's not a design claim, it's demonstrated: the self-correcting agent, the entire Postgres/Redis/pgvector layer, and the LoRA fine-tuning backend were all added after the original three-strategy version, with zero changes to the agents or API that didn't need them — `LoRALLMClient` implements the exact same `LLMClient` interface as the mock and Ollama clients, so every agent strategy, the API, the CLI, and the benchmark work with it unmodified.

## 🚀 Quickstart

```bash
git clone <this-repo> && cd ai-agent-lab
pip install -r requirements.txt

cp "/path/to/your.pdf" data/
python scripts/build_index.py
uvicorn src.api.main:app --reload   # open http://localhost:8000
```

Runs fully offline out of the box. For a real model (free): install [Ollama](https://ollama.com), `ollama pull llama3.1`, set `LLM_BACKEND=ollama`. For production storage: `docker compose up -d postgres redis`, set `VECTOR_BACKEND=postgres` and `CHAT_HISTORY_BACKEND=postgres` — see [Production upgrade](#production-upgrade) below. For higher accuracy on your own document: see [LoRA fine-tuning](#lora-finetuning) below.

## <a name="production-upgrade"></a>🐳 Production upgrade: real storage, not just a demo

| | Dev default (zero setup) | Production |
|---|---|---|
| Vector storage | pickle file, BM25+TFIDF | **Postgres + pgvector**, BM25 + dense embeddings |
| Chat history | in-process dict, lost on restart | **Postgres**, Redis-cached reads |
| Streaming | same endpoint | real **SSE** (`text/event-stream`) |

```bash
docker compose up -d postgres redis
export VECTOR_BACKEND=postgres CHAT_HISTORY_BACKEND=postgres
export POSTGRES_DSN="postgresql+psycopg2://agentlab:agentlab@localhost:5432/agentlab"
export REDIS_URL="redis://localhost:6379/0"
python scripts/build_index.py
uvicorn src.api.main:app --reload
```

Redis is a pure cache, not a hard dependency — if it's unreachable, reads just fall back to Postgres directly instead of failing.

## <a name="lora-finetuning"></a>🎯 LoRA fine-tuning: adapting the model to your documents

Retrieval and prompting only go so far when a small/free/local model has never seen anything like your document before. `src/finetuning/` adds a third `LLM_BACKEND` option, `lora`, that adapts the model's weights to your document directly instead of only conditioning on it at inference time — see [`docs/lora-finetuning.md`](docs/lora-finetuning.md) for the full write-up.

```bash
# 1. Generate a training dataset from the chunks already in data/
LLM_BACKEND=ollama OLLAMA_MODEL=llama3.1 python scripts/build_finetune_dataset.py

# 2. Install the optional training extras (not installed by default)
pip install torch transformers peft datasets accelerate

# 3. Train the LoRA adapter (CPU-feasible against the small default base model)
python scripts/train_lora.py

# 4. Use it -- same as flipping LLM_BACKEND=ollama, no other code changes
export LLM_BACKEND=lora
python scripts/run_agent_cli.py --strategy self_correcting_rag
```

Because it's implemented against the same `LLMClient` interface as `MockLLMClient`/`OllamaLLMClient`, it drops straight into every existing strategy, the API, the CLI, and `scripts/run_benchmark.py` — run the benchmark once per backend to get a real before/after accuracy number instead of just asserting fine-tuning helped.

## 📊 Benchmark: four strategies, measured not asserted

`python scripts/run_benchmark.py` runs every strategy over the same eval set and reports pass rate, latency, LLM/tool call count, and **groundedness** (LLM-as-judge check of whether the answer is actually supported by what it retrieved) side by side. The self-correcting agent trades more LLM calls and latency for measurably higher groundedness — the whole point of the comparison is making that trade-off visible instead of just claiming one strategy is "better." Re-run it with `LLM_BACKEND=lora` after training an adapter to see the same comparison for the fine-tuned model.

## 🐛 Engineering rigor

Every backend swap in this project (reranker, pgvector, chat history, LoRA inference) was tested against a **genuinely fresh environment** — an empty database, a missing package — not just the happy path. That process caught 4 real bugs, each documented with the exact failure, fix, and a regression test that locks it in: see [`docs/bugs-found.md`](docs/bugs-found.md).

## 📁 Project layout

```
src/
  core/         interfaces.py (Tool/LLMClient/AgentStrategy ABCs), llm_client.py, config.py
  ingestion/    loaders, chunking, indexer (memory + postgres backends)
  retrieval/    bm25, fusion (RRF), embeddings, pgvector_store, hybrid + hybrid_pgvector, reranker
  finetuning/   dataset_builder.py (chunks -> QA pairs), lora_trainer.py, lora_llm_client.py
  tools/        document_search, calculator, summarize, registry
  agents/       react, function_calling, plan_execute, self_correcting_rag, factory
  db/           chat session models, Postgres repository
  cache/        Redis read-cache wrapper
  evaluation/   eval_dataset, metrics, groundedness, benchmark
  api/          main.py (FastAPI + SSE), schemas.py
web/            chat UI + live reasoning-trace console
scripts/        build_index, build_finetune_dataset, train_lora, run_agent_cli, run_benchmark
docs/           bugs-found.md, lora-finetuning.md
tests/          unit / integration / regression (43 tests)
data/finetune/  generated LoRA training dataset (gitignored)
models/         trained LoRA adapter output (gitignored)
docker-compose.yml   Postgres (pgvector) + Redis + the app
```
