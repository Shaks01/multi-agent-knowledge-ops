"""
Phase 6+8: turn a logged run into a human-readable explanation.

The Memory agent already writes every run's full step-by-step trace to
logs/agent_trace.jsonl (see agents/memory.py). This module is what makes
that raw JSONL actually *useful* to a person: it renders one run as a
plain-text walkthrough of what each agent did, plus (Phase 8) the
Evaluation agent's structured verdict on the run itself -- grounding,
retrieval relevance, citation consistency, and any failures flagged.

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


def source_key(label: str) -> str:
    """"Attendance Policy.pdf, p.2" -> "attendance policy.pdf"

    Shared normalization used to compare a citation string against a
    retrieved-source label -- matches on filename only (not exact page),
    since an LLM's citation formatting can vary slightly even when it's
    citing a real, retrieved document. Exported (not "_"-prefixed) because
    the Evaluation agent (agents/evaluation.py) needs the exact same
    normalization to run this same check live, during a query, not just
    after the fact here.
    """
    return label.split(",")[0].strip().lower()


def _retrieved_filenames(trace: List[dict]) -> Set[str]:
    filenames = set()
    for step in trace:
        if step.get("agent") == "retrieval":
            for label in step.get("output", {}).get("sources", []):
                filenames.add(source_key(label))
    return filenames


def citation_consistency(answer: str, retrieved_filenames: Set[str]) -> Dict:
    """Compare sources cited inline in `answer` against the set of
    filenames actually retrieved. This is the low-level check both the
    Evaluation agent (live, during a query -- agents/evaluation.py) and
    `check_citation_consistency` below (after the fact, from a logged
    record) run -- kept as one function so both places can never drift
    apart on what "consistent" means.
    """
    cited = _extract_citations(answer)
    unmatched = [c for c in cited if source_key(c) not in retrieved_filenames]

    return {
        "cited": cited,
        "retrieved_filenames": sorted(retrieved_filenames),
        "unmatched": unmatched,
        "consistent": not unmatched,
    }


def check_citation_consistency(record: dict) -> Dict:
    """Recompute the citation check from a logged record's raw trace.

    Runs recorded from Phase 8 onward already carry this same check
    pre-computed in `record["evaluation"]["citation_check"]` (see
    `build_explanation`, which prefers that when present). This function
    stays for older log lines written before the Evaluation agent existed,
    and as a standalone way to re-check any record without touching the
    graph.
    """
    retrieved = _retrieved_filenames(record.get("trace", []))
    return citation_consistency(record.get("answer", ""), retrieved)


def _format_citation_check(check: Dict) -> List[str]:
    lines = []
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
    return lines


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

    lines.append("--- Evaluation ---")
    evaluation = record.get("evaluation")
    if record.get("blocked_by_input_guard"):
        lines.append(
            "N/A -- the question was blocked by the input guard before "
            "anything was generated, so there's nothing to evaluate."
        )
    elif evaluation is None:
        # A run logged before the Evaluation agent existed (Phase 8) --
        # fall back to recomputing just the citation check from the raw
        # trace, rather than showing nothing.
        lines.append(
            "(older run, predates the Evaluation agent -- recomputing "
            "citation check only)"
        )
        lines.extend(_format_citation_check(check_citation_consistency(record)))
    else:
        failures = evaluation.get("failures", [])
        lines.append(
            f"Failures flagged: {', '.join(failures) if failures else 'none'}"
        )
        grounding = evaluation.get("grounding", {})
        lines.append(
            f"Grounding: approved={grounding.get('approved')} "
            f"confidence={grounding.get('confidence')}"
        )
        if grounding.get("issues"):
            for issue in grounding["issues"]:
                lines.append(f"  unsupported claim: {issue}")

        retrieval_eval = evaluation.get("retrieval", {})
        for s in retrieval_eval.get("per_subtask", []):
            best = s.get("best_score")
            best_str = f"{best:.4f}" if isinstance(best, (int, float)) else "n/a"
            lines.append(
                f"Retrieval: {s.get('chunk_count')} chunk(s), best score "
                f"{best_str} -- {s.get('subtask')!r}"
            )
        if retrieval_eval.get("insufficient_subtasks"):
            lines.append(
                f"  insufficient retrieval for: "
                f"{retrieval_eval['insufficient_subtasks']}"
            )
        if retrieval_eval.get("weak_relevance_subtasks"):
            lines.append(
                f"  weak relevance for: {retrieval_eval['weak_relevance_subtasks']}"
            )

        lines.extend(_format_citation_check(evaluation.get("citation_check", {})))

    return "\n".join(lines)
