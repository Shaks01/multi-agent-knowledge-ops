"""
Phase 6: turn a logged run into a human-readable explanation.

The Memory agent already writes every run's full step-by-step trace to
logs/agent_trace.jsonl (see agents/memory.py). This module is what makes
that raw JSONL actually *useful* to a person: it renders one run as a
plain-text walkthrough of what each agent did, and runs a citation
consistency check confirming every source cited in the final answer is
something the Retrieval agent actually found for that run -- not
something invented at the reasoning step.

This is the concrete implementation of "the explanation aligns with the
final response and source documents": it's not just an assertion, it's a
check you can run and see pass or fail.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from knowledge_ops.config import LOGS_DIR

TRACE_LOG_PATH = LOGS_DIR / "agent_trace.jsonl"

_CITATION_PATTERN = re.compile(r"\(Source:\s*([^)]+)\)")


def load_run(run_id: Optional[str] = None) -> dict:
    """Load one run record. `run_id=None` returns the most recent run."""
    if not TRACE_LOG_PATH.exists():
        raise FileNotFoundError(
            f"No trace log found at {TRACE_LOG_PATH}. Ask a question with "
            "run_query.py first."
        )

    records = []
    with open(TRACE_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"{TRACE_LOG_PATH} exists but has no runs logged yet.")

    if run_id is None:
        return records[-1]

    for record in records:
        if record.get("run_id") == run_id:
            return record

    raise ValueError(
        f"No run found with run_id={run_id!r}. Run `python run_explain.py` "
        "with no argument to see the most recent run's id, or check "
        f"{TRACE_LOG_PATH} directly."
    )


def _extract_citations(answer: str) -> List[str]:
    return [m.strip() for m in _CITATION_PATTERN.findall(answer)]


def _filename(label: str) -> str:
    """"Attendance Policy.pdf, p.2" -> "attendance policy.pdf" """
    return label.split(",")[0].strip().lower()


def _retrieved_filenames(trace: List[dict]) -> Set[str]:
    filenames = set()
    for step in trace:
        if step.get("agent") == "retrieval":
            for label in step.get("output", {}).get("sources", []):
                filenames.add(_filename(label))
    return filenames


def check_citation_consistency(record: dict) -> Dict:
    """Compare sources cited inline in the answer against what Retrieval
    actually returned for this run. Matches on filename only (not exact
    page string), since an LLM's citation formatting can vary slightly
    even when it's citing a real, retrieved document.
    """
    cited = _extract_citations(record.get("answer", ""))
    retrieved = _retrieved_filenames(record.get("trace", []))

    unmatched = [c for c in cited if _filename(c) not in retrieved]

    return {
        "cited": cited,
        "retrieved_filenames": sorted(retrieved),
        "unmatched": unmatched,
        "consistent": not unmatched,
    }


def build_explanation(record: dict) -> str:
    lines = []
    lines.append(f"Run {record.get('run_id')}  ({record.get('timestamp')})")
    lines.append(f"Question: {record.get('question')}")
    lines.append("")

    lines.append("--- Agent-by-agent trace ---")
    for step in record.get("trace", []):
        lines.append(f"[{step.get('agent')}] {step.get('action')}")
        output = step.get("output")
        if isinstance(output, dict):
            for key, value in output.items():
                lines.append(f"    {key}: {value}")
        elif output is not None:
            lines.append(f"    output: {output}")
    lines.append("")

    lines.append("--- Final answer ---")
    lines.append(record.get("answer", ""))
    lines.append("")
    lines.append(f"Sources listed: {record.get('sources', [])}")
    lines.append(f"Blocked by input guard: {record.get('blocked_by_input_guard', False)}")
    lines.append(f"Flagged low-confidence: {record.get('flagged_low_confidence', False)}")
    lines.append("")

    lines.append("--- Citation consistency check ---")
    check = check_citation_consistency(record)
    if not check["cited"]:
        lines.append("No inline (Source: ...) citations found in the answer.")
    elif check["consistent"]:
        lines.append(
            "OK: every cited source matches a document the retrieval agent "
            "actually retrieved for this run."
        )
    else:
        lines.append(
            "WARNING: the following cited source(s) do not match anything "
            "retrieved for this run -- possible hallucinated citation:"
        )
        for c in check["unmatched"]:
            lines.append(f"  - {c}")

    return "\n".join(lines)
