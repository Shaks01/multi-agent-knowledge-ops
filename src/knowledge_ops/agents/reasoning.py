"""
Reasoning agent.

Role: turn retrieved context into a coherent, cited draft answer to the
*original* question (not a list of separate subtask answers). It only
ever reasons over what the Retrieval agent found -- no outside knowledge,
no guessing.

If the Validation agent rejects a previous draft, this agent gets called
again with the validation's specific issues appended, and is expected to
fix exactly those issues rather than starting over from scratch.
"""

from typing import List, Optional

from knowledge_ops.agents.retrieval import (
    SubtaskResult,
    collect_sources,
    format_context_blocks,
)
from knowledge_ops.agents.trace import log_step
from knowledge_ops.config import get_llm

AGENT_NAME = "reasoning"

SYSTEM_PROMPT = """You are the reasoning agent for a company knowledge-base \
system. Answer the user's question using ONLY the provided context \
excerpts -- do not use outside knowledge and do not guess.

Rules:
- Write one coherent answer to the ORIGINAL question, not a list of \
separate answers to each subtask.
- Cite the source for every claim inline, like (Source: Attendance \
Policy.pdf, p.2).
- If the context doesn't fully cover some part of the question, say so \
explicitly instead of filling the gap with a guess.
- Be concise. Do not repeat the question back before answering."""


def _format_history(conversation_history: List[dict]) -> str:
    if not conversation_history:
        return ""
    lines = ["Recent conversation (for context only):"]
    for turn in conversation_history[-3:]:
        lines.append(f"  Q: {turn['question']}")
        lines.append(f"  A: {turn['answer']}")
    return "\n".join(lines)


def run(
    question: str,
    subtask_results: List[SubtaskResult],
    conversation_history: Optional[List[dict]] = None,
    validation_feedback: Optional[dict] = None,
    revision_count: int = 0,
) -> dict:
    context = format_context_blocks(subtask_results)
    history_block = _format_history(conversation_history or [])

    parts = []
    if history_block:
        parts.append(history_block)
    parts.append(f"Original question: {question}")
    if validation_feedback and validation_feedback.get("issues"):
        issues = "\n".join(f"- {issue}" for issue in validation_feedback["issues"])
        parts.append(
            "Your previous draft answer had issues flagged by the "
            f"validation agent -- fix these specifically:\n{issues}"
        )
    parts.append(f"Context excerpts:\n\n{context}")
    user_message = "\n\n".join(parts)

    llm = get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    is_revision = bool(validation_feedback and validation_feedback.get("issues"))
    label = "revised draft" if is_revision else "draft"
    print(f"--- {AGENT_NAME} agent: {label} answer ---")
    print(response.content)
    print()

    return {
        "draft_answer": response.content,
        "sources": collect_sources(subtask_results),
        "revision_count": revision_count + 1 if is_revision else revision_count,
        "trace": log_step(
            AGENT_NAME,
            "revise_answer" if is_revision else "draft_answer",
            input={"question": question, "used_feedback": is_revision},
            output={"draft_answer": response.content},
        ),
    }
