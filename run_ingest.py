"""
Entry point for the Phase 2+3 ingestion pipeline: loads every document in
src/knowledge_ops/data/documents/, splits it into chunks, embeds them, and
stores the result in a local Chroma vector database.

Usage:
    python run_ingest.py

Re-run this any time you add, remove, or change files in the documents
folder -- it always rebuilds the vector store from scratch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_ops.ingestion.ingest import run

if __name__ == "__main__":
    run()
