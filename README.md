# 🤖 AI Agents with Hybrid RAG

**A production-ready, framework-agnostic AI agent that answers questions from any document you give it — benchmarking four different reasoning strategies side by side, with real Postgres/Redis/pgvector infrastructure behind it, and BM25 + FAISS hybrid retrieval by default for real semantic search.**

![status](https://img.shields.io/badge/tests-89%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![cost](https://img.shields.io/badge/real--mode-free%20(Ollama)-success) ![infra](https://img.shields.io/badge/storage-Postgres%20%2B%20pgvector%20%2B%20Redis-336791) ![streaming](https://img.shields.io/badge/API-SSE%20streaming-informational) ![retrieval](https://img.shields.io/badge/retrieval-BM25%20%2B%20FAISS-informational)

🔗 **Live demo:** https://ai-agent-hybrid-rag.onrender.com/

---

## 📌 What this is

Drop a PDF/TXT/MD file in, ask questions through a web UI, CLI, or API — the agent retrieves the relevant passages, reasons about the answer, and (in its most advanced mode) **checks its own answer for hallucination before returning it**, retrying automatically if it's wrong. Runs fully offline with zero setup in mock mode, or against a real, free, local model via [Ollama](https://ollama.com) with one environment variable — no API key, no cloud bill. Retrieval defaults to BM25 fused with a real dense FAISS index for accurate semantic search, and documents are split with a paragraph/sentence-aware `RecursiveCharacterTextSplitter` instead of a fixed word count.

## ✨ What it does

- 🧠 **Four interchangeable reasoning strategies** over the same tools — ReAct, native function-calling, plan-and-execute, and a **self-correcting RAG agent** that grades its own retrieved evidence and re-tries when its answer isn't grounded
- 🔍 **Hybrid retrieval** — BM25 fused via Reciprocal Rank Fusion with a dense vector index: **FAISS** by default (zero extra infrastructure), or **pgvector** in production — with an optional cross-encoder reranker on top
- 📄 **Sentence-aware chunking** — a `RecursiveCharacterTextSplitter` tries paragraph, then sentence, then word boundaries before falling back to a raw character cut, so chunks stay coherent instead of being sliced at a fixed word count
- 💬 **Persistent chat sessions** — full conversation history saved to Postgres, Redis-cached for fast reads
- ⚡ **Real-time streaming** — `/api/chat/stream` is true Server-Sent Events; the self-correcting agent streams each retrieval/verify/retry step live as it happens, not after the fact
- 📊 **Built-in benchmark** — quantitatively compares all four strategies on pass rate, latency, groundedness, and cost — not just a demo, a measurement tool
- 🐳 **One-command production infra** — `docker compose up` brings up Postgres (pgvector) + Redis alongside the app

## 🛠️ Tech stack

| Layer | Tools |
|---|---|
| Agent orchestration | Custom `AgentStrategy` interface (ReAct / function-calling / plan-execute / self-correcting RAG) |
| LLM | [Ollama](https://ollama.com) (`llama3.1`, local & free) — dual-mode with a deterministic offline mock for CI |
| Retrieval | BM25 + FAISS dense embeddings (`sentence-transformers`, default) or pgvector in production, RRF fusion, cross-encoder reranking |
| Chunking | `RecursiveCharacterTextSplitter` (paragraph/sentence-aware, `ingestion/chunking.py`) |
| Vector storage | **Postgres + pgvector** (swappable for a zero-setup pickle file in dev) |
| Chat history | **Postgres**, **Redis** read-cache |
| API | **FastAPI**, Server-Sent Events streaming |
| Frontend | Vanilla JS chat console with a live reasoning-trace view |
| Testing | Pytest — 89 tests (unit / integration / regression), including real bugs caught by testing against a genuinely fresh environment |

## 🏗️ Architecture

```
data/*.pdf,txt,md → ingestion (chunk) → retrieval (BM25 + FAISS default, RRF fusion)
                                                            │
                                                   tools/ (document_search, calculator, summarize)
                                                            │
                     ┌───────────────┬───────────────┬──────┴──────────┐
                     ▼               ▼               ▼                 ▼
                  ReAct        Function-calling  Plan & execute   Self-correcting RAG
                     └───────────────┴───────────────┴──────┬──────────┘
                                                             ▼
                                                 core/llm_client (Mock ↔ Ollama)
                                                             │
                          api/main.py (FastAPI, SSE streaming)  ──▶  Postgres (chat history + vectors)
                                                                                   │                          ▲
                                                                              web/ (chat UI)          Redis (read cache)
```

Every agent, tool, and storage backend is swappable through abstract interfaces (`src/core/interfaces.py`) — adding a strategy, or moving from a pickle file to a real database, never touches the other layers. That's not a design claim, it's demonstrated: the self-correcting agent, the entire Postgres/Redis/pgvector layer, and the FAISS retrieval backend were all added after the original three-strategy version, with zero changes to the agents or API that didn't need them — `HybridFaissRetriever` implements the exact same `.search(query, top_k)` interface as `HybridPGVectorRetriever`, so every agent strategy, the API, the CLI, and the benchmark work with it unmodified.

## 🚀 Quickstart

```bash
git clone <this-repo> && cd ai-agent-lab
pip install -r requirements.txt

cp "/path/to/your.pdf" data/
python scripts/build_index.py
uvicorn src.api.main:app --reload   # open http://localhost:8000
```

Runs fully offline out of the box (`faiss-cpu`/`sentence-transformers` are in `requirements.txt` by default now, for the FAISS retrieval backend). For a real model (free): install [Ollama](https://ollama.com), `ollama pull llama3.1`, set `LLM_BACKEND=ollama`. For production storage: `docker compose up -d postgres redis`, set `VECTOR_BACKEND=postgres` and `CHAT_HISTORY_BACKEND=postgres` — see [Production upgrade](#production-upgrade) below.

## <a name="production-upgrade"></a>🐳 Production upgrade: real storage, not just a demo

| | Dev default (zero setup) | Production |
|---|---|---|
| Vector storage | pickle file, BM25 + FAISS (dense embeddings) | **Postgres + pgvector**, BM25 + dense embeddings |
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

## 📊 Benchmark: four strategies, measured not asserted

`python scripts/run_benchmark.py` runs every strategy over the same eval set and reports pass rate, latency, LLM/tool call count, and **groundedness** (LLM-as-judge check of whether the answer is actually supported by what it retrieved) side by side. The self-correcting agent trades more LLM calls and latency for measurably higher groundedness — the whole point of the comparison is making that trade-off visible instead of just claiming one strategy is "better."

## 🐛 Engineering rigor

Every backend swap in this project (reranker, faiss, pgvector, chat history) was tested against a **genuinely fresh environment** — an empty database, a missing package — not just the happy path. That process caught 6 real bugs, each documented with the exact failure, fix, and a regression test that locks it in: see [`docs/bugs-found.md`](docs/bugs-found.md).

## 📁 Project layout

```
src/
  core/         interfaces.py (Tool/LLMClient/AgentStrategy ABCs), llm_client.py, config.py
  ingestion/    loaders, chunking (RecursiveCharacterTextSplitter), indexer (memory + faiss + postgres backends)
  retrieval/    bm25, fusion (RRF), embeddings, faiss_store + hybrid_faiss, pgvector_store + hybrid_pgvector, reranker
  tools/        document_search, calculator, summarize, registry
  agents/       react, function_calling, plan_execute, self_correcting_rag, factory
  db/           chat session models, Postgres repository
  cache/        Redis read-cache wrapper
  evaluation/   eval_dataset, metrics, groundedness, benchmark
  api/          main.py (FastAPI + SSE), schemas.py
web/            chat UI + live reasoning-trace console
scripts/        build_index, run_agent_cli, run_benchmark
docs/           bugs-found.md
tests/          unit / integration / regression (89 tests)
docker-compose.yml   Postgres (pgvector) + Redis + the app
```
