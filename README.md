# multi-agent-knowledge-ops

A knowledge-operations agent system, built from scratch and in small,
runnable steps. Full plan and current phase: see [`ROADMAP.md`](ROADMAP.md).

Tech stack: Python, LangChain, LangGraph, embeddings + RAG, Chroma
(vector store). Each piece is introduced only in the phase that needs it —
see `requirements.txt`.

## Where things stand right now

**Phase 1: Hello World agent.** A single script that sends a question to an
LLM and prints the answer, with the raw prompt and response shown so
nothing happens off-screen.

**Phase 2 + 3: Document ingestion + Chroma vector store.** Source documents
in `src/knowledge_ops/data/documents/` get loaded, split into chunks,
embedded, and stored in a local Chroma database. A separate script lets you
run similarity searches against that store directly, so retrieval quality
can be checked before any LLM is involved.

**Phase 5: Multi-agent architecture.** Five distinct agents, each with one
job — Planning, Retrieval, Reasoning, Validation, and Memory — wired
together by a LangGraph graph instead of one orchestrator doing
everything. Validation can send a draft answer back to the Reasoning
agent for one bounded revision, and every agent's step is logged both to
the console and to `logs/agent_trace.jsonl` for later inspection.
Follow-up questions in the same session can reference earlier ones.

No governance/guardrail enforcement yet (only Validation's read-only
grounding check) — see `ROADMAP.md` for Phase 7.

## Setup

1. **Create a virtual environment** (run this on your own machine, not
   inside any sandbox — a venv isn't portable across operating systems):

   ```
   python -m venv venv
   ```

   Activate it:
   - Windows (PowerShell): `venv\Scripts\Activate.ps1`
   - Windows (cmd): `venv\Scripts\activate.bat`
   - macOS/Linux: `source venv/bin/activate`

2. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **Get a free API key** from Google AI Studio (no credit card required):
   https://aistudio.google.com/apikey

4. **Configure your key**

   ```
   copy .env.example .env      (Windows)
   cp .env.example .env        (macOS/Linux)
   ```

   Then open `.env` and paste your key in place of `your-key-here`.

## Run the Hello World agent

```
python run_hello_agent.py
```

Type a question and press Enter. Type `exit` or `quit` to stop. Every turn
prints the exact prompt sent to the model and the exact response it
returned — this is deliberate: transparency into what the agent is doing
is a goal of this project from day one, not something bolted on later.

## Ingest documents into the vector store

Drop source documents (currently `.pdf`, `.txt`, `.md` are supported) into
`src/knowledge_ops/data/documents/` — 5 sample company policy PDFs are
already there — then run:

```
python run_ingest.py
```

This loads every supported file, splits it into ~1000-character chunks,
embeds each chunk (one Gemini embedding API call per chunk), and writes
the result to a local Chroma database at `chroma_db/` (git-ignored —
regenerate it any time by re-running this script; it always rebuilds from
scratch rather than appending).

Then check what actually got retrieved for a question:

```
python run_search_demo.py
python run_search_demo.py "how many vacation days do employees get?"
```

This prints the top matching chunks with their source file, page number,
and similarity score — no LLM involved yet, just "did retrieval find the
right text." That's deliberate: Phase 4 (RAG) wires an LLM answer on top
of this, but if retrieval itself is off, a fluent-sounding wrong answer
would hide that. Check this step works well before moving on.

## Ask a complex, multi-document question

```
python run_query.py
python run_query.py "If an employee is accused of violating the non-disclosure policy during a dispute, does that go through arbitration, and could their attendance record also affect any disciplinary action?"
```

That example question genuinely spans three of the five sample documents
(Non-Disclosure Policy, Arbitration Policy, Attendance Policy) — good for
checking that planning and multi-document retrieval are actually doing
something, not just answering from whichever single document is closest
to the question as a whole.

Every run prints each agent's work as it happens: the Planning agent's
plan, what the Retrieval agent pulled from Chroma per subtask (source +
page + score), the Reasoning agent's draft, the Validation agent's verdict
(and a revised draft if it rejected the first one), and finally the
Memory agent confirming where the full trace was written. If the answer
looks wrong, that trace tells you which agent is responsible rather than
just having to guess.

Run it interactively (no arguments) and ask a follow-up question in the
same session — the Memory agent keeps a short history so something like
"what about attendance?" after an NDA question can be resolved using that
context.

Inspect past runs any time with:

```
type logs\agent_trace.jsonl    (Windows)
cat logs/agent_trace.jsonl     (macOS/Linux)
```

Each line is one complete JSON record of a run: the question, every
agent's inputs/outputs, the validation verdict, the final answer, and its
sources.

## Project layout

```
.
├── ROADMAP.md                    phased build plan
├── requirements.txt              deps, grouped by phase
├── .env.example                  copy to .env and fill in your key
├── run_hello_agent.py            entry point for Phase 1
├── run_ingest.py                 entry point for Phase 2+3 (load -> chunk -> embed -> store)
├── run_search_demo.py            Phase 3 verification: inspect retrieval directly
├── run_query.py                  entry point for Phase 5: multi-agent complex Q&A
└── src/knowledge_ops/
    ├── config.py                 loads .env, builds LLM/embeddings clients, shared settings
    ├── agents/
    │   ├── hello_agent.py        the Phase 1 agent logic
    │   ├── trace.py              shared step-logging helper used by every agent below
    │   ├── planner.py            Planning agent: question -> ordered subtasks
    │   ├── retrieval.py          Retrieval agent: subtasks -> Chroma chunks
    │   ├── reasoning.py          Reasoning agent: chunks -> cited draft answer
    │   ├── validation.py         Validation agent: draft -> grounding verdict
    │   └── memory.py             Memory agent: session history + trace log persistence
    ├── ingestion/
    │   ├── loaders.py            file-extension -> LangChain loader dispatch
    │   └── ingest.py             load -> split -> embed -> persist to Chroma
    ├── orchestration/
    │   └── graph.py              wires the 5 agents into a LangGraph graph
    └── data/documents/           source documents (sample company policy PDFs)
logs/
└── agent_trace.jsonl             one JSON record per run (git-ignored, local history)
```

Every later phase (observability, governance) builds on this same
`src/knowledge_ops/` package — see `ROADMAP.md` for what's next.

## Why Gemini for the LLM

This project defaults to Google's Gemini API (`langchain-google-genai`)
because it currently has a genuine free tier — usable without a credit
card, which makes it the easiest way to get this running today. The LLM
call is isolated in `src/knowledge_ops/config.py`, so swapping in
Anthropic's Claude API, OpenAI, or a local model via Ollama later is a
change in one file, not a rewrite.
