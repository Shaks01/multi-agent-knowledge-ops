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

## Phase 2 — Load and chunk documents

- Point the project at a folder of source documents (start with a handful of
  `.txt`/`.md` files).
- Use LangChain document loaders + a text splitter to turn them into chunks.
- No embeddings or storage yet — just prove documents can be read and split
  sensibly.

## Phase 3 — Embeddings + Chroma vector store

- Embed the chunks from Phase 2 and persist them in a local Chroma database.
- Build a small script to run a similarity search and print the chunks it
  retrieves for a sample query, so retrieval quality can be inspected
  directly.

## Phase 4 — RAG query answering

- Wire retrieval (Phase 3) into the LLM call (Phase 1): retrieve relevant
  chunks, stuff them into the prompt, ask the model to answer using only
  that context.
- Print the retrieved sources alongside the answer, not just the answer —
  visibility into *why* the agent said what it said.

## Phase 5 — LangGraph orchestration

- Replace the linear script with a LangGraph graph: nodes for
  retrieve → decide → answer (and room to add more agents/nodes later,
  e.g. a router, a summarizer, a critique step).
- Multiple specialized nodes/agents collaborating is what makes this
  "multi-agent" — Phases 1-4 are single-agent building blocks.

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
