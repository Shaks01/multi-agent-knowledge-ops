"""
Entry point for Phase 5: ask a complex, possibly multi-document question
and get back an answer produced by five distinct agents -- planning,
retrieval, reasoning, validation, and memory (see
src/knowledge_ops/agents/ and orchestration/graph.py).

Every step prints what it did: the plan, what was retrieved per subtask,
the draft answer, the validation verdict (and a revised draft if it was
rejected), and finally where the full trace was logged for later
inspection. Multi-turn memory works within one run of this script --
follow-up questions can reference earlier ones in the same session.

Usage:
    python run_query.py
    python run_query.py "If an employee is accused of violating the \
non-disclosure policy during a dispute, does that go through arbitration, \
and could their attendance record also affect any disciplinary action?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_ops.orchestration.graph import answer_question


def _print_result(result: dict) -> None:
    print("=== Final answer ===")
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(f"  - {source}")
    if result.get("revision_count", 0) > 0:
        print(
            f"\n(Reasoning agent revised this answer "
            f"{result['revision_count']} time(s) after validation feedback.)"
        )
    print()


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = answer_question(question)
        _print_result(result)
        return

    print("Multi-agent query demo. Type a question, or 'exit'/'quit' to stop.\n")
    print(
        "Try one that spans more than one document, e.g.: "
        "\"If an employee is accused of violating the non-disclosure policy "
        "during a dispute, does that go through arbitration, and could "
        "their attendance record also affect any disciplinary action?\"\n"
    )

    conversation_history = []
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = answer_question(question, conversation_history=conversation_history)
        _print_result(result)
        conversation_history = result["conversation_history"]


if __name__ == "__main__":
    main()
