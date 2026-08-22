"""
Validation agent.

Role: check the Reasoning agent's draft answer against the same context it
was given -- not against outside knowledge, and not against what the
validator "thinks" is true. Its only question is: is every claim in this
draft actually supported by the retrieved excerpts?

This is the governance-relevant agent: it's what would eventually enforce
"the agent doesn't say things its documents don't back up" (see
ROADMAP.md Phase 7). Right now it only flags issues and triggers one
bounded revision -- it doesn't block or redact anything itself.
"""

from typing import List

from pydantic import BaseModel, Field

from knowledge_ops.agents.retrieval import SubtaskResult, format_context_blocks
from knowledge_ops.agents.trace import log_step
from knowledge_ops.config import get_llm

AGENT_NAME = "validation"


class ValidationResult(BaseModel):
    approved: bool = Field(
        description="True only if every claim in the draft is supported by the context."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How confident you are in this verdict and in the answer's "
            "grounding overall, from 0.0 (not confident at all -- e.g. the "
            "context is thin, ambiguous, or only tangentially related) to "
            "1.0 (fully confident every claim is clearly and directly "
            "supported)."
        ),
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Specific claims in the draft that the context does NOT support. Empty if approved.",
    )
    notes: str = Field(
        default="", description="One short sentence explaining the verdict."
    )


SYSTEM_PROMPT = """You are the validation agent for a company knowledge-base \
system. You are given a draft answer and the context excerpts it was \
supposed to be based on. Check ONLY whether every factual claim in the \
draft is actually supported by that context, and rate how confident you \
are in that judgment.

Rules:
- Do not judge writing style, tone, or completeness -- only factual \
grounding.
- If the draft says the context doesn't cover something, that's honest, \
not an unsupported claim -- approve it.
- List each unsupported claim as its own short item, quoting or closely \
paraphrasing the claim.
- Confidence should reflect the context's quality, not just whether you \
approved the answer: e.g. an approved answer built on thin or borderline \
context deserves a lower confidence than one built on clear, direct \
context, even though both get approved=true."""


def run(
    question: str, draft_answer: str, subtask_results: List[SubtaskResult]
) -> dict:
    context = format_context_blocks(subtask_results)
    user_message = (
        f"Original question: {question}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Context excerpts:\n\n{context}"
    )

    llm = get_llm().with_structured_output(ValidationResult)
    result: ValidationResult = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    verdict = {
        "approved": result.approved,
        "confidence": result.confidence,
        "issues": result.unsupported_claims,
        "notes": result.notes,
    }

    print(f"--- {AGENT_NAME} agent: verdict ---")
    print(f"  approved: {verdict['approved']}  (confidence: {verdict['confidence']:.2f})")
    if verdict["issues"]:
        for issue in verdict["issues"]:
            print(f"    - unsupported: {issue}")
    if verdict["notes"]:
        print(f"  notes: {verdict['notes']}")
    print()

    return {
        "validation": verdict,
        "trace": log_step(
            AGENT_NAME,
            "validate_answer",
            input={"draft_answer": draft_answer},
            output=verdict,
        ),
    }
