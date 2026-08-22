"""
Phase 3 verification: run similarity search against the persisted Chroma
store and print exactly what comes back -- the chunk text, its source
file/page, and its similarity score. This is deliberately not "the agent
answering a question" (that's Phase 4, RAG); it's a way to inspect
retrieval quality on its own, before an LLM's phrasing can hide whether
the right chunks were even found.

Usage:
    python run_search_demo.py
    python run_search_demo.py "how many vacation days do employees get?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_ops.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    get_embeddings,
)

TOP_K = 4


def _load_vector_store():
    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"No vector store found at {CHROMA_PERSIST_DIR}. "
            "Run `python run_ingest.py` first."
        )

    from langchain_chroma import Chroma

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


def search(query: str, vector_store=None, k: int = TOP_K) -> None:
    vector_store = vector_store or _load_vector_store()
    results = vector_store.similarity_search_with_score(query, k=k)

    print(f"\nQuery: {query!r}")
    print(f"Top {len(results)} chunk(s) (lower score = more similar):\n")
    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = f"{source}" + (f", page {page + 1}" if page is not None else "")
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"[{rank}] score={score:.4f}  ({location})")
        print(f"    {preview}\n")


def main() -> None:
    vector_store = _load_vector_store()

    if len(sys.argv) > 1:
        search(" ".join(sys.argv[1:]), vector_store=vector_store)
        return

    print("Retrieval demo. Type a question, or 'exit'/'quit' to stop.\n")
    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        search(query, vector_store=vector_store)


if __name__ == "__main__":
    main()
