"""
Entry point for the Phase 1 Hello World agent.

Usage:
    python run_hello_agent.py
"""

import sys
from pathlib import Path

# Allow running this script directly from the repo root without installing
# the package: add src/ to the import path.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_ops.agents.hello_agent import main

if __name__ == "__main__":
    main()
