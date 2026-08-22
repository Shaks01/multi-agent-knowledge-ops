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

- Source documents live in `src/knowledge_ops/data/documents/` (currently 6
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

## Phase 6 — Explainability & transparency (this commit)

Feature: **Explainability and Transparency** — turning the raw trace
Phase 5 already logs into something a person can actually use to
understand and evaluate a decision, not just data sitting in a file.

- `src/knowledge_ops/explainability/report.py` + `run_explain.py`: given a
  `run_id` (or nothing, for the most recent run), reads the matching
  record from `logs/agent_trace.jsonl` and renders a plain-text
  walkthrough -- every agent's step, the final answer, its sources,
  whether it was blocked or flagged, all in one place instead of scrolled
  past in a terminal or buried in raw JSON.
- **Citation consistency check** (`check_citation_consistency`): extracts
  every `(Source: ...)` citation from the final answer and checks it
  against what the Retrieval agent actually returned for that run. A
  citation naming a document that was never retrieved is flagged as a
  likely hallucinated source -- this is the concrete, checkable version of
  "the explanation aligns with the final response and source documents,"
  not just an assertion that it does.
- `run_query.py` now points to `run_explain.py` after every answer, so the
  live console trace and the after-the-fact report are both one command
  away.

What's still open: latency/token-usage metrics per agent, a way to
browse/search many past runs at once instead of one at a time, and likely
LangSmith (or an equivalent open tracer) for a visual replay of a run's
graph execution rather than a text report.

## Phase 7 — AI governance & guardrails (this commit)

Feature: **Governance and Guardrails** — minimizing hallucinations and
handling uncertainty responsibly, plus a first line of defense against
malicious or malformed input.

- **Input Guard agent** (`agents/input_guard.py`) — new first node in the
  graph (`START -> input_guard`). Rule-based, not an LLM call: rejects
  empty input, input over `config.MAX_QUESTION_LENGTH`, and common
  prompt-injection phrasing ("ignore previous instructions," "reveal your
  system prompt," etc.) before any other agent -- including the
  LLM-backed ones -- ever sees the question. Deliberately simple and
  deterministic over a fancier LLM-based moderation layer, so every
  rejection is fully explainable by pointing at the rule that fired.
- **Confidence threshold** (`config.CONFIDENCE_THRESHOLD`, default 0.7) —
  the Validation agent (Phase 5) now also rates its confidence 0.0-1.0,
  not just approved/rejected. An approved answer built on thin or
  borderline context, not just an outright rejected one, now triggers the
  same retry-then-warn path if its confidence falls below the threshold.
- **Warnings and disclaimers** (`agents/memory.py`) — carried over from
  the "Response Validation" work and extended: a low-confidence or
  rejected answer gets a visible warning naming the specific unsupported
  claim(s), and every non-blocked answer gets a standing disclaimer
  naming what it was generated from and that it isn't legal advice. Both
  the warning and the disclaimer are logged, not just displayed, so
  they're part of the auditable record too.

Deliberately not yet doing: semantic/LLM-based content moderation (only
pattern-based), automatic escalation to a human for low-confidence
answers, or hard-blocking (vs. warning on) ungrounded answers -- see
`config.py` and the agents above for where each of those would plug in.

## Phase 8 — Evaluation, observability & failure detection (this commit)

Feature: **Evaluation, Observability, and Failure Detection** — explicit,
structured evaluation of each run's own behavior, separate from whether
the answer itself reads as grounded.

- **Evaluation agent** (`agents/evaluation.py`) — new node in the graph,
  `validator -> evaluator -> memory`, running once the Validation <->
  Reasoning retry loop has settled (skipped for a question the Input
  Guard blocked -- nothing was generated to evaluate). Rule-based, like
  the Input Guard agent: no LLM call, so every flag traces to an exact
  rule.
- **Retrieval relevance signals** — every chunk's similarity score is now
  captured in both the Retrieval agent's own trace output
  (`scores_per_subtask`) and the Evaluation agent's summary, not just
  printed to the console and discarded.
- **Failure detection** — the Evaluation agent flags, per run:
  `insufficient_retrieval` (a subtask returned zero chunks --
  `config.MIN_CHUNKS_PER_SUBTASK`), `weak_retrieval_relevance` (opt-in via
  `config.RELEVANCE_DISTANCE_THRESHOLD`, unset by default -- see the
  comment in `config.py` for why no default distance value is safe to
  assume), `low_grounding_confidence` and `answer_rejected` (carried over
  from Validation's verdict), and `conflicting_agent_outputs` -- the one
  new *cross-agent* check: Validation approved the answer, but the
  citation-consistency check independently found a citation that was
  never retrieved. That specific, structurally-detectable disagreement is
  what "conflicting agent outputs" means here; broader semantic
  contradiction-checking between agents (e.g. two subtask answers that
  logically conflict) is not implemented.
- **Structured, inspectable logging** — the full evaluation record
  (grounding, per-subtask retrieval detail, the citation check, and the
  failures list) is persisted under `record["evaluation"]` in
  `logs/agent_trace.jsonl` for every run, alongside the existing trace.
  `run_explain.py` renders it as an "Evaluation" section, falling back to
  recomputing just the citation check for older log lines that predate
  this agent.
- **Alongside the final response** — `run_query.py` prints an "Evaluation
  summary" (failures flagged, grounding, any unmatched citations)
  immediately under every answer, not only in the after-the-fact report.

One real bug this surfaced during testing, fixed as part of this phase:
the Reasoning agent only counted a Validation-triggered retry as a
"revision" if the verdict listed specific unsupported claims. A
low-confidence-but-approved verdict with no listed claims never
incremented `revision_count`, so `config.MAX_REVISIONS` didn't actually
bound that retry path -- it could loop forever. Fixed in
`agents/reasoning.py` so any call made because a prior validation verdict
exists counts as a revision, regardless of whether it named a specific
claim.

What's still open: no automated evaluation *scoring* across many runs at
once (this is per-run only -- see "browse/search many past runs" under
Phase 6), and `conflicting_agent_outputs` covers exactly one specific
disagreement, not general contradiction detection across agents or
subtasks.

## Ground rules for every phase

- Keep it runnable. Each phase ends with something you can execute and see
  output from, not just code sitting unused.
- Keep dependencies scoped to the current phase — `requirements.txt` grows
  phase by phase, not all at once.
- Keep the LLM swappable. All model access goes through
  `src/knowledge_ops/config.py` so switching providers later touches one
  file, not every script.
