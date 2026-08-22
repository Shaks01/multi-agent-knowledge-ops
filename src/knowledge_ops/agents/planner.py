"""
Planning agent.

Role: the only agent that decides *what needs to happen* to answer the
question. It doesn't retrieve, reason, or validate anything itself -- it
just breaks the question into an ordered plan of self-contained subtasks
for the Retrieval and Reasoning agents to work through.

A simple, single-topic question comes back as a one-step plan, unchanged --
this agent's job is to recognize when a question is already simple, not to
manufacture complexity.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from knowledge_ops.agents.trace import log_step
from knowledge_ops.config import MAX_SUBTASKS, get_llm

AGENT_NAME = "planner"


class Plan(BaseModel):
    """The ordered subtasks needed to fully answer the question."""

    subtasks: List[str] = Field(
        description=(
            "1 to 5 self-contained sub-questions, in a sensible order. Each "
            "should be answerable on its own by searching a document "
            "library, and together they should cover everything needed to "
            "answer the original question."
        )
    )


SYSTEM_PROMPT = """You are the planning agent for a company knowledge-base \
system. Your only job is to decide what steps are needed to answer the \
user's question -- you do not answer it yourself.

Rules:
- If the question is already simple and about one topic, return it as the \
only subtask, unchanged.
- Only split into multiple subtasks when the question genuinely spans more \
than one topic/policy (e.g. it asks about two different procedures, or how \
two different policies interact).
- Each subtask must make sense on its own, without seeing the others.
- Do not invent sub-questions the user didn't ask about.
- If recent conversation history is provided, use it to resolve references \
like "that policy" or "it," but plan only for the current question."""


def _format_history(conversation_history: List[dict]) -> str:
    if not conversation_history:
        return ""
    lines = ["Recent conversation (for context only):"]
    for turn in conversation_history[-3:]:
        lines.append(f"  Q: {turn['question']}")
        lines.append(f"  A: {turn['answer']}")
    return "\n".join(lines)


def run(question: str, conversation_history: Optional[List[dict]] = None) -> dict:
    conversation_history = conversation_history or []
    history_block = _format_history(conversation_history)
    user_content = question if not history_block else f"{history_block}\n\nCurrent question: {question}"

    llm = get_llm().with_structured_output(Plan)
    result: Plan = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )

    subtasks = [s.strip() for s in result.subtasks if s.strip()][:MAX_SUBTASKS]
    if not subtasks:
        subtasks = [question]

    print(f"--- {AGENT_NAME} agent: plan ---")
    for i, subtask in enumerate(subtasks, start=1):
        print(f"  {i}. {subtask}")
    print()

    return {
        "plan": subtasks,
        "trace": log_step(
            AGENT_NAME,
            "create_plan",
            input=question,
            output={"subtasks": subtasks},
        ),
    }
