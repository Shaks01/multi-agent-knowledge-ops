"""
Test package for multi-agent-knowledge-ops.

Adds src/ to sys.path the same way every run_*.py entry point does (see
e.g. run_query.py), so `import knowledge_ops...` resolves inside test
modules without needing the package installed (`pip install -e .`) or a
PYTHONPATH set by hand. This runs once, the first time anything under
`tests/` is imported -- i.e. as soon as `python -m unittest discover`
(run from the repo root) finds this package.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
