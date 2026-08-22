"""
Entry point for Phase 4+5: ask a complex, possibly multi-document question
and get back an orchestrated, cited answer.

The orchestrator (src/knowledge_ops/orchestration/graph.py) breaks the
question into subtasks, retrieves relevant chunks per subtask from the
Chroma store built by run_ingest.py, and synthesizes one coherent answer.
Every step prints what it did -- the subtask breakdown, what was retrieved
for each one, and the sources behind the final answer -- so nothing about
how the answer was produced is hidden.

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
    print()


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = answer_question(question)
        _print_result(result)
        return

    print("Complex query demo. Type a question, or 'exit'/'quit' to stop.\n")
    print(
        "Try one that spans more than one document, e.g.: "
        "\"If an employee is accused of violating the non-disclosure policy "
        "during a dispute, does that go through arbitration, and could "
        "their attendance record also affect any disciplinary action?\"\n"
    )
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

        result = answer_question(question)
        _print_result(result)


if __name__ == "__main__":
    main()
