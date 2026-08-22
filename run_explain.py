"""
Entry point for Phase 6 explainability: turns a logged run into a
human-readable explanation of how the agents got to that answer.

Usage:
    python run_explain.py            # most recent run
    python run_explain.py <run_id>   # a specific run (see the run_id
                                      # printed by run_query.py, or any
                                      # line in logs/agent_trace.jsonl)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_ops.explainability.report import build_explanation, load_run


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    record = load_run(run_id)
    print(build_explanation(record))


if __name__ == "__main__":
    main()
