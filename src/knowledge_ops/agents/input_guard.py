"""
Input Guard agent (guardrail).

Role: the first checkpoint every question passes through, before any
other agent -- including the LLM-backed ones -- ever sees it.

Deliberately rule-based, not an LLM call: cheap, deterministic, and its
reasoning is trivial to audit (you can point at the exact rule that
fired, rather than guess at a model's judgment about its own input).
That's a real tradeoff -- it will miss subtler misuse a model-based
moderation layer might catch -- but for a first governance pass it means
every rejection is fully explainable, which matters more here.

Checks, in order:
  1. Not empty.
  2. Not absurdly long (config.MAX_QUESTION_LENGTH).
  3. Not an obvious prompt-injection / instruction-override attempt.

Anything else is left to the Planning/Reasoning/Validation agents to
handle honestly (e.g. "the documents don't cover this") -- this agent's
job is only to catch inputs that shouldn't reach the LLM-backed agents at
all, not to judge whether a question is answerable.
"""

import re

from knowledge_ops.agents.trace import log_step
from knowledge_ops.config import MAX_QUESTION_LENGTH

AGENT_NAME = "input_guard"

# Short, high-precision patterns for the most common instruction-override
# phrasing. Deliberately not exhaustive -- a rule-based list like this is
# a floor, not a ceiling; see ROADMAP.md for adding an LLM-based check on
# top of this later.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all|any|the)?\s*(previous|prior|above)\s*instructions",
        r"disregard (all|any|the)?\s*(previous|prior|above)\s*instructions",
        r"reveal (your|the) system prompt",
        r"you are now (in )?(developer|debug|god)\s*mode",
        r"act as (if you (had|have) )?no (restrictions|rules|guidelines)",
        r"pretend (you have|to have) no (restrictions|guidelines|rules)",
        r"forget (that )?you('re| are) (an?|the) (ai|assistant|agent)",
    ]
]


def _verdict(allowed: bool, category: str, reason: str = "") -> dict:
    return {"allowed": allowed, "category": category, "reason": reason}


def check(question: str) -> dict:
    """Pure rule evaluation, no side effects -- kept separate from `run()`
    so it's independently unit-testable without a trace/print involved."""
    stripped = question.strip()

    if not stripped:
        return _verdict(False, "empty", "The question was empty.")

    if len(stripped) > MAX_QUESTION_LENGTH:
        return _verdict(
            False,
            "too_long",
            f"The question is too long ({len(stripped)} characters, limit "
            f"{MAX_QUESTION_LENGTH}). Please ask something more specific.",
        )

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            return _verdict(
                False,
                "injection_attempt",
                "This looks like an attempt to override the system's "
                "instructions rather than a question about company "
                "documents, so it wasn't processed.",
            )

    return _verdict(True, "ok")


def run(question: str) -> dict:
    verdict = check(question)

    print(f"--- {AGENT_NAME} agent: {'allowed' if verdict['allowed'] else 'BLOCKED'} ---")
    if not verdict["allowed"]:
        print(f"  category: {verdict['category']}")
        print(f"  reason: {verdict['reason']}")
    print()

    result = {
        "guard": verdict,
        "trace": log_step(AGENT_NAME, "check_input", input=question, output=verdict),
    }
    if not verdict["allowed"]:
        # Set these directly so the graph can route straight to the Memory
        # agent without every downstream node needing a special case for
        # "nothing ran" -- from Memory's point of view this looks like any
        # other draft answer + empty source list.
        result["draft_answer"] = verdict["reason"]
        result["sources"] = []

    return result
