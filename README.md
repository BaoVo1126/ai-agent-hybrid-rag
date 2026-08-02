# 🤖 AI Agents with Hybrid RAG

**A production-ready, framework-agnostic AI agent that answers questions from any document you give it — benchmarking four different reasoning strategies side by side, with real Postgres/Redis/pgvector infrastructure behind it, BM25 + FAISS hybrid retrieval by default, and an n8n orchestration layer wrapping the whole thing for real-world automation.**

![status](https://img.shields.io/badge/tests-89%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![cost](https://img.shields.io/badge/real--mode-free%20(Ollama)-success) ![infra](https://img.shields.io/badge/storage-Postgres%20%2B%20pgvector%20%2B%20Redis-336791) ![streaming](https://img.shields.io/badge/API-SSE%20streaming-informational) ![retrieval](https://img.shields.io/badge/retrieval-BM25%20%2B%20FAISS-informational) ![orchestration](https://img.shields.io/badge/orchestration-n8n-EA4B71)

🔗 **Live demo:** https://ai-agent-hybrid-rag.onrender.com/

🧩 **Algorithmic foundation:** the self-correcting RAG strategy below started as a standalone, easier-to-read reference implementation — see [`local-agentic-rag`](https://github.com/BaoVo1126/local-agentic-rag) — before being reimplemented here against this repo's `AgentStrategy` interface, alongside three other reasoning strategies.

---

## 📌 What this is

Drop a PDF/TXT/MD file in, ask questions through a web UI, CLI, API, or an n8n workflow — the agent retrieves the relevant passages, reasons about the answer, and (in its most advanced mode) **checks its own answer for hallucination before returning it**, retrying automatically if it's wrong. Runs fully offline with zero setup in mock mode, or against a real, free, local model via [Ollama](https://ollama.com) with one environment variable. Retrieval defaults to BM25 fused with a real dense FAISS index for accurate semantic search, and documents are split with a paragraph/sentence-aware `RecursiveCharacterTextSplitter` instead of a fixed word count.

## ✨ What it does

- 🧠 **Four interchangeable reasoning strategies** over the same tools — ReAct, native function-calling, plan-and-execute, and a **self-correcting RAG agent** that grades its own retrieved evidence and re-tries when its answer isn't grounded.
- 🎯 **Structured prompt contracts, not loose text parsing** — the self-correcting RAG agent's graders/verifiers now speak a strict JSON schema (`{"grounded": true|false, "reason": "..."}`) instead of a bare "yes/no" that silently defaulted to a pass on anything unparseable — see [Prompt engineering](#prompt-engineering) below.
- 🔗 **n8n orchestration layer** — a workflow-level automation layer sitting outside the agent, wiring `/api/upload` + `/api/chat` into webhooks, file-ingestion pipelines, and human-review routing when the agent's self-check flags an answer — see [`n8n/`](n8n/).
- 🔍 **Hybrid retrieval** — BM25 fused via Reciprocal Rank Fusion with a dense vector index: **FAISS** by default (zero extra infrastructure), or **pgvector** in production — with an optional cross-encoder reranker on top.
- 📄 **Sentence-aware chunking** — a `RecursiveCharacterTextSplitter` tries paragraph, then sentence, then word boundaries before falling back to a raw character cut, so chunks stay coherent instead of being sliced at a fixed word count.
- 💬 **Persistent chat sessions** — full conversation history saved to Postgres, Redis-cached for fast reads.
- ⚡ **Real-time streaming** — `/api/chat/stream` is true Server-Sent Events; the self-correcting agent streams each retrieval/verify/retry step live as it happens, not after the fact.
- 📊 **Built-in benchmark** — quantitatively compares all four strategies on pass rate, latency, groundedness, and cost — not just a demo, a measurement tool.
- 🐳 **One-command production infra** — `docker compose up` brings up Postgres (pgvector) + Redis alongside the app; `--profile with-n8n` adds the orchestration layer.

## 🛠️ Tech stack

| Layer | Tools |
|---|---|
| Agent orchestration | Custom `AgentStrategy` interface (ReAct / function-calling / plan-execute / self-correcting RAG) |
| Prompt engineering | `src/agents/prompts.py` — role + output-contract system prompts, separated from loop control flow |
| LLM | [Ollama](https://ollama.com) (`llama3.1`, local & free) — dual-mode with a deterministic offline mock for CI |
| Retrieval | BM25 + FAISS dense embeddings (`sentence-transformers`, default) or pgvector in production, RRF fusion, cross-encoder reranking |
| Chunking | `RecursiveCharacterTextSplitter` (paragraph/sentence-aware, `ingestion/chunking.py`) |
| Vector storage | **Postgres + pgvector** (swappable for a zero-setup pickle file in dev) |
| Chat history | **Postgres**, **Redis** read-cache |
| API | **FastAPI**, Server-Sent Events streaming |
| Workflow orchestration | **n8n** — webhook → ingest → ask-agent → route-on-confidence (`n8n/document_qa_workflow.json`) |
| Frontend | Vanilla JS chat console with a live reasoning-trace view |
| Testing | Pytest — 89 tests (unit / integration / regression), including real bugs caught by testing against a genuinely fresh environment |

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph ORCH["🔗 Orchestration (outside the agent)"]
        direction LR
        N8N["n8n\nwebhook → ingest? → ask → route-on-confidence"]
    end

    subgraph INGEST["📄 Ingestion"]
        direction LR
        DATA[("data/*.pdf,txt,md")] --> CHUNK["chunking\nRecursiveCharacterTextSplitter"]
    end

    subgraph RETRIEVAL["🔍 Retrieval"]
        direction LR
        HYBRID["BM25 + FAISS / pgvector\nRRF fusion + reranker"]
    end

    subgraph AGENTS["🧠 Agent strategies (AgentStrategy interface)"]
        direction LR
        REACT["ReAct"]
        FCALL["Function-calling"]
        PLAN["Plan & execute"]
        SELFRAG["Self-correcting RAG"]
    end

    subgraph BRAIN["🎯 Reasoning core"]
        direction LR
        PROMPTS["agents/prompts.py\nrole + JSON output contracts"]
        LLM["core/llm_client\nMock ↔ Ollama"]
    end

    subgraph SERVE["⚡ Serving"]
        direction LR
        API["api/main.py\nFastAPI + SSE"]
        WEB["web/ chat UI"]
    end

    subgraph STORE["💾 Storage"]
        direction LR
        PG[("Postgres\nchat history + vectors")]
        REDIS[("Redis\nread cache")]
    end

    N8N -->|HTTP| API
    DATA --> CHUNK --> HYBRID
    HYBRID --> TOOLS["tools/ (document_search, calculator, summarize)"]
    TOOLS --> REACT & FCALL & PLAN & SELFRAG
    REACT & FCALL & PLAN & SELFRAG --> PROMPTS --> LLM --> API
    API --> PG
    API --> WEB
    PG -.-> REDIS

    classDef orch fill:#FDF2F8,stroke:#DB2777,color:#831843;
    classDef ingest fill:#FEFCE8,stroke:#CA8A04,color:#713F12;
    classDef retrieval fill:#ECFEFF,stroke:#0891B2,color:#164E63;
    classDef agent fill:#F3E8FF,stroke:#9333EA,color:#581C87;
    classDef brain fill:#FFF7ED,stroke:#EA580C,color:#7C2D12;
    classDef serve fill:#E0F2FE,stroke:#0284C7,color:#0C4A6E;
    classDef store fill:#DCFCE7,stroke:#16A34A,color:#14532D;
    class N8N orch;
    class DATA,CHUNK ingest;
    class HYBRID,TOOLS retrieval;
    class REACT,FCALL,PLAN,SELFRAG agent;
    class PROMPTS,LLM brain;
    class API,WEB serve;
    class PG,REDIS store;
```

Every agent, tool, and storage backend is swappable through abstract interfaces (`src/core/interfaces.py`) — adding a strategy, or moving from a pickle file to a real database, never touches the other layers. That's not a design claim, it's demonstrated: the self-correcting agent, the entire Postgres/Redis/pgvector layer, and the FAISS retrieval backend were all added after the original three-strategy version, with zero changes to the agents or API that didn't need them — `HybridFaissRetriever` implements the exact same `.search(query, top_k)` interface as `HybridPGVectorRetriever`, so every agent strategy, the API, the CLI, and the benchmark work with it unmodified. n8n follows the same principle from the outside: it's one more caller of `/api/chat`, not a rewrite of the agent loop.

## <a name="prompt-engineering"></a>🧠 Prompt engineering & system directives

The self-correcting RAG agent's accuracy depends entirely on its graders and verifiers judging correctly — and the original version asked for a bare "yes" or "no" in free text, parsed with a regex that **defaulted to a pass (`True`) whenever the model's response didn't match cleanly**. That's a silently optimistic failure mode: any hedge, caveat, or off-format response from the model waved a passage or answer through instead of being caught as a real failure.

Following the same separation used in [`AI_AGENT_FROM_ZERO`](https://github.com/breslee1707/AI_AGENT_FROM_ZERO) (prompts kept in their own module, tool/output shape stated explicitly rather than implied), `src/agents/prompts.py` now gives each grader/verifier:

- **An explicit ROLE line** — what kind of judge it is, before the task.
- **A strict output contract** — a one-line JSON object with a fixed schema (`{"grounded": true|false, "reason": "..."}`), not a word the model has to guess the exact phrasing of.
- **Fail-closed parsing** — `_parse_verdict()` in `self_correcting_rag_agent.py` only accepts a clean match against that schema (with a loose yes/no fallback for the mock LLM's canned test phrasing); anything else is now treated as the check **failing**, not passing.

This directly targets the "accuracy still very low" symptom: a low-accuracy self-correcting loop is often not a retrieval problem, it's the *grader itself* rubber-stamping bad passages/answers because its own output wasn't being parsed reliably.

```mermaid
sequenceDiagram
    autonumber
    participant Agent as SelfCorrectingRAGAgent
    participant Ret as Hybrid Retriever
    participant LLM as LLM (grader/generator/verifier)

    Agent->>Ret: search(query)
    Ret-->>Agent: candidate passages
    loop each passage
        Agent->>LLM: GRADE_RELEVANCE_SYSTEM + passage
        LLM-->>Agent: {"relevant": true|false, "reason": "..."}
    end
    Agent->>LLM: GENERATE_SYSTEM + kept passages
    LLM-->>Agent: draft answer
    Agent->>LLM: GROUNDED_SYSTEM + answer
    LLM-->>Agent: {"grounded": true|false, "reason": "..."}
    alt grounded
        Agent->>LLM: USEFUL_SYSTEM + answer
        LLM-->>Agent: {"useful": true|false, "reason": "..."}
    end
    alt grounded and useful
        Agent-->>Agent: return answer
    else malformed or failing verdict
        Note over Agent: _parse_verdict() fails CLOSED,<br/>not open, on anything that isn't<br/>valid JSON matching the schema
        Agent->>LLM: REWRITE_SYSTEM + query
        LLM-->>Agent: rewritten query
        Agent->>Ret: search(rewritten query)
    end
```

## 🔗 n8n orchestration layer

`n8n/document_qa_workflow.json` puts a workflow-level automation layer **outside** the agent: a webhook receives a question (and optionally a file to ingest), calls `/api/upload` if needed, calls `/api/chat`, and routes the result — including notifying a human reviewer when the agent's own self-check flags the answer as `unverified`. The agent's reasoning loop stays exactly where it is; n8n only ever talks to it over HTTP, the same way the CLI, web UI, and benchmark do. See [`n8n/README.md`](n8n/README.md) for the full design rationale and setup steps.

```bash
docker compose --profile with-n8n up -d
# n8n UI at http://localhost:5678 — import n8n/document_qa_workflow.json
```

## 🚀 Quickstart

```bash
git clone <this-repo> && cd ai-agent-hybrid-rag
pip install -r requirements.txt

cp "/path/to/your.pdf" data/
python scripts/build_index.py
uvicorn src.api.main:app --reload   # open http://localhost:8000
```

Runs fully offline out of the box (`faiss-cpu`/`sentence-transformers` are in `requirements.txt` by default now, for the FAISS retrieval backend). For a real model (free): install [Ollama](https://ollama.com), `ollama pull llama3.1`, set `LLM_BACKEND=ollama`. For production storage: `docker compose up -d postgres redis`, set `VECTOR_BACKEND=postgres` and `CHAT_HISTORY_BACKEND=postgres` — see [Production upgrade](#production-upgrade) below. For workflow orchestration: `docker compose --profile with-n8n up -d`.

## <a name="production-upgrade"></a>🐳 Production upgrade: real storage, not just a demo

| | Dev default (zero setup) | Production |
|---|---|---|
| Vector storage | pickle file, BM25 + FAISS (dense embeddings) | **Postgres + pgvector**, BM25 + dense embeddings |
| Chat history | in-process dict, lost on restart | **Postgres**, Redis-cached reads |
| Streaming | same endpoint | real **SSE** (`text/event-stream`) |
| Orchestration | direct API calls | optional **n8n** workflows (`--profile with-n8n`) |

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
  agents/       react, function_calling, plan_execute, self_correcting_rag, prompts.py, factory
  db/           chat session models, Postgres repository
  cache/        Redis read-cache wrapper
  evaluation/   eval_dataset, metrics, groundedness, benchmark
  api/          main.py (FastAPI + SSE), schemas.py
web/            chat UI + live reasoning-trace console
n8n/            document_qa_workflow.json + setup/design notes
scripts/        build_index, run_agent_cli, run_benchmark
docs/           bugs-found.md
tests/          unit / integration / regression (89 tests)
docker-compose.yml   Postgres (pgvector) + Redis + n8n (optional) + the app
```
