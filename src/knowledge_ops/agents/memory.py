"""
Memory agent.

Role: the only agent responsible for *remembering things across time*, and
the last stop before an answer is handed back -- which makes it the right
place to attach warnings/disclaimers and record what happened. Four jobs,
all handled here so no other agent needs to know about any of them:

1. Short-term conversational memory -- the running list of (question,
   answer) pairs for the current interactive session, so a follow-up
   question like "what about attendance?" can be resolved by the
   Planning/Reasoning agents. Lives only in memory for the process's
   lifetime (see ROADMAP.md for persisting this across restarts later).

2. Long-term audit trail -- every agent in the pipeline appends to the
   run's `trace` as it works; this agent is what actually commits that
   trace to disk (logs/agent_trace.jsonl, one JSON object per run) so a
   past interaction can be inspected later without having had the
   terminal output open at the time (see run_explain.py). This is what
   satisfies "agent interactions are traceable and logged for inspection."

3. Grounding/confidence warning -- by the time this agent runs, the
   Validator -> Reasoning retry loop (graph.py) is over, one way or
   another. If the final verdict is still unapproved OR its confidence is
   below config.CONFIDENCE_THRESHOLD, this agent prepends a plain-text
   warning -- naming the specific unsupported claim(s) -- rather than
   handing back an answer that looks no different from a fully grounded
   one. Nothing is redacted; see ROADMAP.md Phase 7 for stricter
   enforcement (e.g. refusing outright) as a possible next step.

4. Standing disclaimer -- every non-blocked answer gets a short, constant
   footer naming what it was generated from and that it isn't a
   substitute for checking the real policy. This is separate from #3:
   it's not conditional on anything going wrong, it's baseline source
   attribution for *any* answer this system produces.

A question rejected by the Input Guard agent (graph.py routes straight
here in that case) skips #3 and #4 entirely -- a "your question wasn't
processed" message doesn't need a grounding disclaimer, since nothing was
generated from the documents.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from knowledge_ops.agents.trace import log_step
from knowledge_ops.config import CONFIDENCE_THRESHOLD, LOGS_DIR

AGENT_NAME = "memory"

TRACE_LOG_PATH = LOGS_DIR / "agent_trace.jsonl"

UNVERIFIED_WARNING_HEADER = (
    "NOTE: This answer could not be fully verified against the source "
    "documents. Treat it with caution and check the underlying policies "
    "directly before relying on it."
)

STANDING_DISCLAIMER = (
    "This answer was generated from the company policy documents in this "
    "system's knowledge base and is provided for informational purposes "
    "only -- it is not legal advice. Confirm with HR/Legal before relying "
    "on it for an actual decision."
)


def _needs_grounding_warning(validation: Optional[dict]) -> bool:
    if not validation:
        return False
    if not validation.get("approved", True):
        return True
    confidence = validation.get("confidence")
    return confidence is not None and confidence < CONFIDENCE_THRESHOLD


def _apply_grounding_warning(answer: str, validation: Optional[dict]) -> str:
    if not _needs_grounding_warning(validation):
        return answer

    lines = [UNVERIFIED_WARNING_HEADER]
    issues = (validation or {}).get("issues") or []
    if issues:
        lines.append("Unsupported claim(s) flagged by the validation agent:")
        lines.extend(f"  - {issue}" for issue in issues)
    confidence = (validation or {}).get("confidence")
    if confidence is not None:
        lines.append(f"(validation confidence: {confidence:.2f})")

    return "\n".join(lines) + "\n\n" + answer


def run(
    question: str,
    answer: str,
    sources: List[str],
    trace: List[dict],
    conversation_history: List[dict],
    validation: Optional[dict] = None,
    guard: Optional[dict] = None,
) -> dict:
    blocked = bool(guard) and not guard.get("allowed", True)

    if blocked:
        final_answer = answer  # the Input Guard agent already set this to its reason
        was_flagged = False
    else:
        final_answer = _apply_grounding_warning(answer, validation)
        was_flagged = final_answer != answer
        final_answer = final_answer + "\n\n" + STANDING_DISCLAIMER

    run_id = uuid.uuid4().hex[:12]
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": final_answer,
        "blocked_by_input_guard": blocked,
        "guard_category": (guard or {}).get("category"),
        "flagged_low_confidence": was_flagged,
        "sources": sources,
        "trace": trace,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    updated_history = conversation_history + [
        {"question": question, "answer": final_answer}
    ]

    if blocked:
        print(f"--- {AGENT_NAME} agent: question was blocked, not answered ---")
    elif was_flagged:
        print(f"--- {AGENT_NAME} agent: flagged answer as low-confidence ---")
    print(f"--- {AGENT_NAME} agent: logged run {run_id} to {TRACE_LOG_PATH} ---\n")

    return {
        "answer": final_answer,
        "conversation_history": updated_history,
        "trace": log_step(
            AGENT_NAME,
            "persist_trace",
            output={
                "run_id": run_id,
                "log_path": str(TRACE_LOG_PATH),
                "blocked_by_input_guard": blocked,
                "flagged_low_confidence": was_flagged,
            },
        ),
    }
