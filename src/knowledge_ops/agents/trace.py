"""
Shared step-logging helper used by every agent so each one records what it
did into the run's trace, in the same shape, without re-implementing
timestamping/formatting per agent.

The orchestration graph's `trace` state field is additive (see
`orchestration/graph.py`'s `Annotated[List[dict], operator.add]`), so each
agent just returns the single-element list `log_step(...)` produces and
LangGraph appends it to whatever earlier agents already logged, rather
than overwriting it.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List


def log_step(agent: str, action: str, **details: Any) -> List[Dict]:
    entry = {
        "agent": agent,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    return [entry]
