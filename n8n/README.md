# n8n orchestration layer

`document_qa_workflow.json` wraps this repo's existing FastAPI endpoints
(`/api/upload`, `/api/chat`) in an n8n workflow, rather than reimplementing
orchestration logic a second time inside n8n. This repo's agent already
owns the hard part -- the Think→Act→Observe / self-correction loop lives in
`src/agents/*` and stays there. n8n's job is the layer *above* the agent:
receiving a request from anywhere (webhook, form, chat platform, schedule),
deciding whether a file needs to be ingested first, calling the agent, and
routing the result (respond to the caller, and/or notify someone when the
agent's own self-check flagged the answer as unverified).

## Why n8n sits outside the agent, not inside it

The agent's own orchestration (which of the four `AgentStrategy` implementations
runs, whether it retries) is a **reasoning-level** loop: it needs low latency,
tight control over LLM calls, and the exact interfaces in
`src/core/interfaces.py`. n8n orchestration is a **workflow-level** loop: which
external systems this agent gets wired into, and what happens before/after a
single `/api/chat` call. Keeping that boundary means adding a new automation
(a Slack bot, a scheduled re-index, a form submission) never touches agent
code -- it's a new n8n workflow calling the same two HTTP endpoints.

## Workflow structure

```
Webhook (question [+ optional file_url])
        │
   Has file to ingest? ──yes──▶ Download file ──▶ POST /api/upload ──┐
        │no                                                          │
        └──────────────────────────────────────────────────────────▶│
                                                                       ▼
                                                          POST /api/chat (strategy,
                                                          default: self_correcting_rag)
                                                                       │
                                                     Self-check flagged "unverified"?
                                                        │yes                  │no
                                                        ▼                     ▼
                                              Notify human reviewer   Respond to caller
                                                        │
                                                        ▼
                                                Respond to caller
```

The branch on `"unverified"` reads the exact flag the self-correcting RAG
agent already emits when it exhausts its retries without passing its own
groundedness/usefulness checks (see
`src/agents/self_correcting_rag_agent.py`) -- n8n doesn't re-implement that
judgment, it just routes on it.

## Setup

1. Import `document_qa_workflow.json` into n8n (Workflows → Import from File).
2. Set the `AGENT_API_BASE_URL` environment variable in your n8n instance to
   wherever this repo's API is running, e.g. `http://localhost:8000` or the
   deployed URL.
3. Replace the "Notify human reviewer" no-op node with a real notification
   node (Slack, Telegram, Email) — it's left as a placeholder so the
   workflow imports cleanly without requiring extra credentials.
4. Activate the workflow. It exposes a webhook at `POST /webhook/agent-ask`
   accepting `{"question": "...", "file_url": "...", "strategy": "...",
   "session_id": "..."}` (`file_url` and `strategy` are optional).

This is one example workflow, not the only way to use the API — any n8n
workflow can call `/api/chat` directly for simpler cases (no ingestion
branch), or add nodes before it (RSS feed → summarize → ask) since the
agent is just an HTTP call from n8n's perspective.
