"""
Phase 5+6+7: the multi-agent orchestrator, now with a guardrail entry
point.

Wires six distinct agents (src/knowledge_ops/agents/) into a LangGraph
graph, each with one clearly scoped job:

  input_guard -> rule-based safety/sanity check, before anything else runs
  planner     -> decides the sequence of subtasks needed (routing/planning)
  retrieval   -> fetches context per subtask from Chroma
  reasoning   -> drafts a coherent, cited answer from that context
  validation  -> checks the draft is grounded in that context, with a
                 confidence score
  memory      -> persists the run's full trace, updates conversation
                 history, and applies warnings/disclaimers to the answer

Graph shape:

    START -> input_guard --(blocked)-------------------+
                 |                                      |
              (allowed)                                 |
                 v                                       v
              planner -> retrieval -> reasoning -> validator --+
                                          ^                    |
                                          | (rejected or       | (approved &
                                          |  low confidence,   |  confident, or
                                          |  retry)            |  retry cap hit)
                                          +--------------------+
                                                               v
                                                             memory -> END

(Node is named "validator", not "validation" -- LangGraph doesn't allow a
node name that collides with a state field name, and `validation` is
already the state field holding the verdict dict.)

The validator -> reasoning edge is a bounded retry (see
config.MAX_REVISIONS): if the validator finds unsupported claims OR its
confidence is below config.CONFIDENCE_THRESHOLD, the Reasoning agent gets
one more attempt with that specific feedback before the answer is
returned regardless, so a genuinely unanswerable question can't loop
forever. The Memory agent applies a visible warning in that case (or if
the retry cap was hit before the Validation agent was satisfied) -- see
agents/memory.py.

A question the Input Guard agent blocks skips planner/retrieval/
reasoning/validator entirely and goes straight to Memory, which logs the
rejection and returns its reason as the answer, unchanged.
"""

import operator
from typing import Annotated, List, Optional, TypedDict

from knowledge_ops.agents import (
    input_guard,
    memory,
    planner,
    reasoning,
    retrieval,
    validation,
)
from knowledge_ops.agents.retrieval import SubtaskResult
from knowledge_ops.config import CONFIDENCE_THRESHOLD, MAX_REVISIONS


class AgentState(TypedDict):
    question: str
    conversation_history: List[dict]
    guard: dict
    plan: List[str]
    subtask_results: List[SubtaskResult]
    draft_answer: str
    validation: dict
    revision_count: int
    answer: str
    sources: List[str]
    # Additive: every agent appends its own step(s) rather than overwriting
    # what earlier agents already logged (see agents/trace.py).
    trace: Annotated[List[dict], operator.add]


# --- Node wrappers -----------------------------------------------------
# Thin adapters between LangGraph's "one dict argument" node signature and
# each agent's own plain-argument function signature (kept plain so each
# agent module is usable/testable on its own, without a graph involved).


def _input_guard_node(state: AgentState) -> dict:
    return input_guard.run(state["question"])


def _planner_node(state: AgentState) -> dict:
    return planner.run(state["question"], state.get("conversation_history"))


def _retrieval_node(state: AgentState) -> dict:
    return retrieval.run(state["plan"])


def _reasoning_node(state: AgentState) -> dict:
    return reasoning.run(
        question=state["question"],
        subtask_results=state["subtask_results"],
        conversation_history=state.get("conversation_history"),
        validation_feedback=state.get("validation"),
        revision_count=state.get("revision_count", 0),
    )


def _validation_node(state: AgentState) -> dict:
    return validation.run(
        question=state["question"],
        draft_answer=state["draft_answer"],
        subtask_results=state["subtask_results"],
    )


def _memory_node(state: AgentState) -> dict:
    return memory.run(
        question=state["question"],
        answer=state["draft_answer"],
        sources=state["sources"],
        trace=state["trace"],
        conversation_history=state.get("conversation_history", []),
        validation=state.get("validation"),
        guard=state.get("guard"),
    )


def _route_after_guard(state: AgentState) -> str:
    guard = state.get("guard", {})
    return "planner" if guard.get("allowed", True) else "memory"


def _route_after_validation(state: AgentState) -> str:
    verdict = state.get("validation", {})
    rejected = not verdict.get("approved", True)
    low_confidence = verdict.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
    if (rejected or low_confidence) and state.get("revision_count", 0) < MAX_REVISIONS:
        return "reasoning"
    return "memory"


# --- Graph assembly ------------------------------------------------------


def build_graph():
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(AgentState)
    builder.add_node("input_guard", _input_guard_node)
    builder.add_node("planner", _planner_node)
    builder.add_node("retrieval", _retrieval_node)
    builder.add_node("reasoning", _reasoning_node)
    # Named "validator", not "validation" -- that name is taken by the
    # `validation` state field (the verdict dict), and LangGraph raises
    # "'validation' is already being used as a state key" if a node reuses it.
    builder.add_node("validator", _validation_node)
    builder.add_node("memory", _memory_node)

    builder.add_edge(START, "input_guard")
    builder.add_conditional_edges(
        "input_guard", _route_after_guard, {"planner": "planner", "memory": "memory"}
    )
    builder.add_edge("planner", "retrieval")
    builder.add_edge("retrieval", "reasoning")
    builder.add_edge("reasoning", "validator")
    builder.add_conditional_edges(
        "validator", _route_after_validation, {"reasoning": "reasoning", "memory": "memory"}
    )
    builder.add_edge("memory", END)

    return builder.compile()


def answer_question(
    question: str, conversation_history: Optional[List[dict]] = None
) -> AgentState:
    """Run the full multi-agent graph on a single question.

    `conversation_history` is a list of {"question", "answer"} dicts from
    earlier turns in the same session (see run_query.py) -- pass the
    updated `conversation_history` this returns into the next call to
    keep short-term memory going across a conversation.
    """
    graph = build_graph()
    return graph.invoke(
        {
            "question": question,
            "conversation_history": conversation_history or [],
            "guard": {},
            "plan": [],
            "subtask_results": [],
            "draft_answer": "",
            "validation": {},
            "revision_count": 0,
            "answer": "",
            "sources": [],
            "trace": [],
        }
    )
