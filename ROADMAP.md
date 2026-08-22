# Roadmap

This project is being built incrementally, one working piece at a time. Each
phase should run end-to-end before the next one starts. Check a phase off
only once its code runs and its output has been looked at, not just written.

## Phase 0 — Repo setup (this commit)

- Repo skeleton, `.gitignore`, `.env.example`, `requirements.txt`.
- `src/knowledge_ops/` package that later phases will grow into.
- No agent logic yet.

## Phase 1 — Hello World agent (this commit)

- A single LangChain LLM call wrapped in a tiny script (`run_hello_agent.py`).
- Answers basic ad-hoc questions typed at a prompt.
- No tools, no memory, no documents — just: question in, model answer out,
  with the raw prompt and response printed so nothing is hidden.
- Goal: prove the environment, the API key, and the LangChain wiring all
  work before adding any complexity.

## Phase 2 — Load and chunk documents (this commit)

- Source documents live in `src/knowledge_ops/data/documents/` (currently 5
  sample company policy PDFs). Loader is dispatched by file extension
  (`src/knowledge_ops/ingestion/loaders.py`) so new types (`.docx`, more
  `.pdf`s, etc.) can be added without touching the ingestion script.
- Split into chunks with `RecursiveCharacterTextSplitter`.
- Folded into the same run as Phase 3 below rather than kept as a separate
  step — for a document set this small there's nothing meaningful to
  inspect in between "loaded" and "embedded," so `run_ingest.py` does both.

## Phase 3 — Embeddings + Chroma vector store (this commit)

- Embed the chunks from Phase 2 (Google's `gemini-embedding-001`) and
  persist them in a local Chroma database (`run_ingest.py`, rebuilds from
  scratch on every run).
- `run_search_demo.py` runs a similarity search and prints the chunks it
  retrieves for a query, with source file, page number, and similarity
  score — retrieval quality inspected directly, before any LLM is involved.

## Phase 4 — RAG query answering (superseded by Phase 5, see below)

First version: a single LangGraph orchestrator node that decomposed a
question, retrieved per subtask, and synthesized one cited answer. Proved
that "decompose -> retrieve -> synthesize" works end to end (see git
history if you want the simpler 3-node version). Phase 5 replaced that
single node with five distinct agents; the underlying decompose/retrieve/
synthesize logic didn't disappear, it just moved into separate,
independently-inspectable agents.

## Phase 5 — Multi-agent architecture (this commit)

Feature: **Multi-Agent Task Routing** — instead of one orchestrator doing
everything, five agents each own one clearly scoped job, wired together
by a LangGraph graph (`src/knowledge_ops/orchestration/graph.py`):

- **Planning agent** (`agents/planner.py`) — the only agent that decides
  *what* needs to happen. Breaks the question into an ordered plan of
  subtasks (or a one-step plan, for a simple question).
- **Retrieval agent** (`agents/retrieval.py`) — fetches chunks from Chroma
  per subtask. Doesn't interpret anything.
- **Reasoning agent** (`agents/reasoning.py`) — drafts one coherent, cited
  answer to the *original* question from the retrieved context. Also the
  agent that revises its own draft if validation rejects it.
- **Validation agent** (`agents/validation.py`) — checks the draft's
  claims are actually grounded in the retrieved context (not outside
  knowledge, not the validator's own opinion). If it finds unsupported
  claims, the graph routes back to the Reasoning agent once
  (`config.MAX_REVISIONS`) with that specific feedback before returning
  an answer regardless — bounded so a genuinely unanswerable question
  can't loop forever.
- **Memory agent** (`agents/memory.py`) — two jobs: keeps the running
  (question, answer) history for the current session so follow-up
  questions can reference earlier ones, and persists every run's full
  step-by-step trace to `logs/agent_trace.jsonl` for later inspection.

Because each agent returns its own step into an additive `trace` list
(`agents/trace.py`), and `run_query.py` prints each agent's work as it
happens, every interaction is inspectable both live (console) and after
the fact (the JSONL log) — this is most of what Phase 6 below would have
built from scratch; Phase 6 now mainly adds *aggregating and visualizing*
what's already being logged, not the logging itself.

Deliberately still a bounded, mostly-linear graph (one conditional retry
edge), not a fully autonomous multi-agent system with arbitrary looping —
that's the smallest version of "distinct agents with routing between
them" that satisfies the acceptance criteria. Natural next steps: fan
retrieval out in parallel per subtask, add a router that skips planning
entirely for obviously-simple questions, or let the Validation agent's
feedback loop apply to specific subtasks rather than the whole answer.

## Phase 6 — Transparency & observability

- Phase 5's `logs/agent_trace.jsonl` already gives a durable, structured
  record of every agent's inputs/outputs per run. What's still open:
  latency and token-usage metrics per agent, a way to browse/search past
  runs instead of grepping a JSONL file by hand, and likely LangSmith (or
  an equivalent open tracer) for a proper visual replay of a run's graph
  execution.

## Phase 7 — AI governance & guardrails

- Input/output filtering, source-grounding checks (does the answer actually
  cite retrieved content?), rate limiting, and an audit log of every query
  and answer.
- A short written policy: what the agent is and isn't allowed to answer,
  and what happens when it's unsure.

## Ground rules for every phase

- Keep it runnable. Each phase ends with something you can execute and see
  output from, not just code sitting unused.
- Keep dependencies scoped to the current phase — `requirements.txt` grows
  phase by phase, not all at once.
- Keep the LLM swappable. All model access goes through
  `src/knowledge_ops/config.py` so switching providers later touches one
  file, not every script.
