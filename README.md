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

**Phase 4 + 5: Complex Query Handling.** A LangGraph orchestrator breaks a
question into subtasks, retrieves relevant chunks per subtask, and
synthesizes one coherent, cited answer — so a question spanning more than
one policy document gets a real answer instead of whatever's closest to
the question as a whole.

No self-correction, memory across turns, or governance/guardrail layer
yet — see `ROADMAP.md` for Phases 6-7.

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
checking that decomposition and multi-document retrieval are actually
doing something, not just answering from whichever single document is
closest to the question as a whole.

Every run prints: the subtasks the orchestrator broke the question into,
what was retrieved from Chroma for each one (source + page + score, same
as the search demo), and the final answer with a "Sources:" list. If the
answer looks wrong, that trace tells you whether decomposition, retrieval,
or the final synthesis step is where it went wrong — rather than just
having to guess.

## Project layout

```
.
├── ROADMAP.md                    phased build plan
├── requirements.txt              deps, grouped by phase
├── .env.example                  copy to .env and fill in your key
├── run_hello_agent.py            entry point for Phase 1
├── run_ingest.py                 entry point for Phase 2+3 (load -> chunk -> embed -> store)
├── run_search_demo.py            Phase 3 verification: inspect retrieval directly
├── run_query.py                  entry point for Phase 4+5: complex, multi-document Q&A
└── src/knowledge_ops/
    ├── config.py                 loads .env, builds LLM/embeddings clients, shared settings
    ├── agents/
    │   └── hello_agent.py        the Phase 1 agent logic
    ├── ingestion/
    │   ├── loaders.py            file-extension -> LangChain loader dispatch
    │   └── ingest.py             load -> split -> embed -> persist to Chroma
    ├── orchestration/
    │   └── graph.py              LangGraph orchestrator: decompose -> retrieve -> synthesize
    └── data/documents/           source documents (sample company policy PDFs)
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
