"""
Memory agent.

Role: the only agent responsible for *remembering things across time*.
Two distinct kinds of memory, both handled here so no other agent needs
to know about either:

1. Short-term conversational memory -- the running list of (question,
   answer) pairs for the current interactive session, so a follow-up
   question like "what about attendance?" can be resolved by the
   Planning/Reasoning agents. Lives only in memory for the process's
   lifetime (see ROADMAP.md for persisting this across restarts later).

2. Long-term audit trail -- every agent in the pipeline appends to the
   run's `trace` as it works; this agent is what actually commits that
   trace to disk (logs/agent_trace.jsonl, one JSON object per run) so a
   past interaction can be inspected later without having had the
   terminal output open at the time. This is what satisfies "agent
   interactions are traceable and logged for inspection."
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from knowledge_ops.agents.trace import log_step
from knowledge_ops.config import LOGS_DIR

AGENT_NAME = "memory"

TRACE_LOG_PATH = LOGS_DIR / "agent_trace.jsonl"


def run(
    question: str,
    answer: str,
    sources: List[str],
    trace: List[dict],
    conversation_history: List[dict],
) -> dict:
    run_id = uuid.uuid4().hex[:12]
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources,
        "trace": trace,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    updated_history = conversation_history + [{"question": question, "answer": answer}]

    print(f"--- {AGENT_NAME} agent: logged run {run_id} to {TRACE_LOG_PATH} ---\n")

    return {
        "answer": answer,
        "conversation_history": updated_history,
        "trace": log_step(
            AGENT_NAME,
            "persist_trace",
            output={"run_id": run_id, "log_path": str(TRACE_LOG_PATH)},
        ),
    }
