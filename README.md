# multi-agent-knowledge-ops

A knowledge-operations agent system, built from scratch and in small,
runnable steps. Full plan and current phase: see [`ROADMAP.md`](ROADMAP.md).

Tech stack: Python, LangChain, LangGraph, embeddings + RAG, Chroma
(vector store). Each piece is introduced only in the phase that needs it —
see `requirements.txt`.

## Where things stand right now

**Phase 1: Hello World agent.** A single script that sends a question to an
LLM and prints the answer, with the raw prompt and response shown so
nothing happens off-screen. No documents, no retrieval, no orchestration
yet — those come in later phases.

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

## Project layout

```
.
├── ROADMAP.md                    phased build plan
├── requirements.txt              deps, grouped by phase
├── .env.example                  copy to .env and fill in your key
├── run_hello_agent.py            entry point for Phase 1
└── src/knowledge_ops/
    ├── config.py                 loads .env, builds the LLM client
    └── agents/
        └── hello_agent.py        the Phase 1 agent logic
```

Every later phase (document ingestion, embeddings, RAG, LangGraph
orchestration, observability, governance) builds on this same
`src/knowledge_ops/` package — see `ROADMAP.md` for what's next.

## Why Gemini for the LLM

This project defaults to Google's Gemini API (`langchain-google-genai`)
because it currently has a genuine free tier — usable without a credit
card, which makes it the easiest way to get this running today. The LLM
call is isolated in `src/knowledge_ops/config.py`, so swapping in
Anthropic's Claude API, OpenAI, or a local model via Ollama later is a
change in one file, not a rewrite.
