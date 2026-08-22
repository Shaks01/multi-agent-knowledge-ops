"""
Retrieval agent.

Role: given the Planning agent's subtasks, fetch relevant chunks from the
Chroma vector store (Phase 3) for each one. It does not interpret or
answer anything -- that's the Reasoning agent's job. Keeping retrieval
and reasoning as separate agents means a bad answer can be diagnosed as
"retrieval found the wrong text" vs. "reasoning misused the right text"
just by looking at the trace.
"""

from typing import List, Optional, TypedDict

from knowledge_ops.agents.trace import log_step
from knowledge_ops.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    TOP_K_PER_SUBTASK,
    get_embeddings,
)

AGENT_NAME = "retrieval"


class RetrievedChunk(TypedDict):
    text: str
    source: str
    page: Optional[int]
    score: float


class SubtaskResult(TypedDict):
    subtask: str
    chunks: List[RetrievedChunk]


def format_context_blocks(subtask_results: List[SubtaskResult]) -> str:
    """Render retrieved chunks as labeled blocks for an LLM prompt.

    Shared by the Reasoning and Validation agents so both see the exact
    same context, formatted the exact same way.
    """
    blocks = []
    for result in subtask_results:
        for chunk in result["chunks"]:
            blocks.append(f"[{_label(chunk)}]\n{chunk['text']}")
    return "\n\n".join(blocks)


def collect_sources(subtask_results: List[SubtaskResult]) -> List[str]:
    """Deduplicated, first-seen-order list of "file, page" labels."""
    seen = set()
    sources = []
    for result in subtask_results:
        for chunk in result["chunks"]:
            label = _label(chunk)
            if label not in seen:
                seen.add(label)
                sources.append(label)
    return sources


def _label(chunk: RetrievedChunk) -> str:
    page = chunk.get("page")
    return f"{chunk['source']}" + (f", p.{page + 1}" if page is not None else "")


def run(plan: List[str]) -> dict:
    from langchain_chroma import Chroma

    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"No vector store found at {CHROMA_PERSIST_DIR}. "
            "Run `python run_ingest.py` first."
        )

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    print(f"--- {AGENT_NAME} agent: retrieving per subtask ---")
    subtask_results: List[SubtaskResult] = []
    for subtask in plan:
        hits = vector_store.similarity_search_with_score(subtask, k=TOP_K_PER_SUBTASK)
        chunks: List[RetrievedChunk] = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "score": score,
            }
            for doc, score in hits
        ]
        subtask_results.append({"subtask": subtask, "chunks": chunks})

        print(f"  subtask: {subtask!r}")
        for chunk in chunks:
            print(f"    - {_label(chunk)} (score={chunk['score']:.4f})")
    print()

    return {
        "subtask_results": subtask_results,
        "trace": log_step(
            AGENT_NAME,
            "retrieve_per_subtask",
            input={"plan": plan},
            output={
                "sources": collect_sources(subtask_results),
                "chunks_per_subtask": [len(r["chunks"]) for r in subtask_results],
            },
        ),
    }
