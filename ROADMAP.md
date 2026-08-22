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

## Phase 4+5 — RAG query answering + LangGraph orchestration (this commit)

Built together as one feature: **Complex Query Handling** — a business
user asks a question that may require reasoning across more than one
document, and gets back one coherent, cited answer instead of a raw
single-pass retrieval.

- `src/knowledge_ops/orchestration/graph.py` is a LangGraph `StateGraph`
  with three nodes, run in sequence:
  1. **decompose** — an LLM call (structured output, not string-parsing)
     breaks the question into 1-5 self-contained subtasks. A simple,
     single-topic question just comes back as one subtask, unchanged.
  2. **retrieve** — runs a Chroma similarity search (Phase 3) per subtask,
     so a question spanning multiple policies actually pulls chunks from
     each relevant document rather than just whatever's closest to the
     question as a whole.
  3. **synthesize** — one LLM call answers the *original* question using
     only the retrieved chunks, citing the source document and page for
     every claim, and says so explicitly if the context doesn't cover
     part of the question rather than guessing.
- `run_query.py` runs the graph and prints every step: the subtask
  breakdown, what was retrieved for each one, and the final cited answer
  — the orchestration equivalent of Phase 1's "print the prompt and the
  response," extended to a multi-step agent.
- Deliberately a straight-line graph, not a looping/self-correcting one —
  that's the smallest orchestrator that satisfies "decompose, retrieve,
  synthesize." Natural next steps once this is proven out: fan the
  retrieve step out in parallel per subtask, add a router that skips
  decomposition for obviously-simple questions, or add a critique node
  that checks the answer against its cited context before returning it.

## Phase 6 — Transparency & observability

- Structured logging of every step: what was retrieved, what prompt was
  sent, what the model returned, how long it took, token usage.
- Likely LangSmith (or an equivalent open tracer) so a full run can be
  replayed and inspected, not just the final answer.

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
