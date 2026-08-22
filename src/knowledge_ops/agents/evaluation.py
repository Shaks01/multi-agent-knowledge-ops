"""
Evaluation agent.

Role: after the Validation <-> Reasoning retry loop (graph.py) has settled
on a final draft and verdict, this agent runs a second, independent pass
over the SAME agent outputs -- retrieval, the draft answer, and the
grounding verdict -- and computes a structured set of observability
signals and failure flags, rather than just trusting whatever the other
agents already said about themselves.

This is deliberately a separate agent from Validation, not more code
bolted onto it: Validation judges the ANSWER against the CONTEXT (is
every claim supported). Evaluation judges the SYSTEM's own behavior this
run -- did retrieval actually find enough relevant material, does the
final answer's citations line up with what was retrieved, does the
confidence score agree with the citation check, etc. Keeping these
separate means a disagreement between two checks is itself a signal worth
surfacing (see "conflicting_agent_outputs" below), not something one
agent could silently paper over by being the only one asked.

Rule-based/computational, like the Input Guard agent -- no LLM call, so
every flag it raises can be traced to an exact rule, and it costs nothing
extra to run on every single query.

Skipped entirely for a question the Input Guard agent blocked (see
graph.py) -- nothing was retrieved or generated, so there's nothing here
to evaluate.
"""

from typing import List, Optional

from knowledge_ops.agents.retrieval import SubtaskResult, collect_sources
from knowledge_ops.agents.trace import log_step
from knowledge_ops.config import (
    CONFIDENCE_THRESHOLD,
    MIN_CHUNKS_PER_SUBTASK,
    RELEVANCE_DISTANCE_THRESHOLD,
)
from knowledge_ops.explainability.report import citation_consistency, source_key

AGENT_NAME = "evaluation"


def _retrieval_signals(subtask_results: List[SubtaskResult]) -> dict:
    per_subtask = []
    insufficient = []
    weak_relevance = []

    for result in subtask_results:
        chunks = result["chunks"]
        scores = [chunk["score"] for chunk in chunks]
        # Chroma's default distance function is smaller-is-more-similar,
        # so the "best" match is the minimum score, not the maximum.
        best_score = min(scores) if scores else None

        per_subtask.append(
            {
                "subtask": result["subtask"],
                "chunk_count": len(chunks),
                "best_score": best_score,
                "scores": scores,
            }
        )

        if len(chunks) < MIN_CHUNKS_PER_SUBTASK:
            insufficient.append(result["subtask"])
        elif RELEVANCE_DISTANCE_THRESHOLD is not None and all(
            score > RELEVANCE_DISTANCE_THRESHOLD for score in scores
        ):
            weak_relevance.append(result["subtask"])

    return {
        "per_subtask": per_subtask,
        "insufficient_subtasks": insufficient,
        "weak_relevance_subtasks": weak_relevance,
    }


def run(
    question: str,
    draft_answer: str,
    subtask_results: List[SubtaskResult],
    validation: Optional[dict],
    revision_count: int = 0,
) -> dict:
    validation = validation or {}

    retrieval_eval = _retrieval_signals(subtask_results)

    retrieved_filenames = {
        source_key(label) for label in collect_sources(subtask_results)
    }
    citation_check = citation_consistency(draft_answer, retrieved_filenames)

    failures = []
    if retrieval_eval["insufficient_subtasks"]:
        failures.append("insufficient_retrieval")
    if retrieval_eval["weak_relevance_subtasks"]:
        failures.append("weak_retrieval_relevance")

    approved = validation.get("approved", True)
    if not approved:
        failures.append("answer_rejected")

    confidence = validation.get("confidence")
    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        failures.append("low_grounding_confidence")

    if approved and not citation_check["consistent"]:
        # Validation found nothing wrong with the draft, but the citation
        # check independently found a cited source that was never
        # retrieved -- these two checks disagree with each other, which
        # is worth flagging even though neither one is individually
        # "wrong" on its own terms.
        failures.append("conflicting_agent_outputs")

    evaluation = {
        "grounding": {
            "approved": validation.get("approved"),
            "confidence": confidence,
            "issues": validation.get("issues", []),
        },
        "retrieval": retrieval_eval,
        "citation_check": citation_check,
        "revision_count": revision_count,
        "failures": failures,
    }

    print(f"--- {AGENT_NAME} agent: evaluation summary ---")
    print(f"  failures flagged: {', '.join(failures) if failures else 'none'}")
    for s in retrieval_eval["per_subtask"]:
        best = s["best_score"]
        best_str = f"{best:.4f}" if best is not None else "n/a"
        print(f"    retrieval: {s['chunk_count']} chunk(s), best score {best_str} -- {s['subtask']!r}")
    if not citation_check["consistent"]:
        print(f"  citation check: unmatched source(s): {citation_check['unmatched']}")
    print()

    return {
        "evaluation": evaluation,
        "trace": log_step(AGENT_NAME, "evaluate_run", output=evaluation),
    }
