"""
Phase 4+5: the orchestrator agent for complex, multi-document queries.

Implements the "Complex Query Handling" story: a question that may need
reasoning across more than one document is decomposed into subtasks, each
subtask is answered by retrieving from the Chroma store built in Phase 3,
and a final node synthesizes everything into one coherent, cited answer.

This is intentionally a straight-line graph (decompose -> retrieve ->
synthesize), not a looping/self-correcting agent -- that's the smallest
version of "orchestrator" that satisfies the acceptance criteria. Natural
next steps once this is proven out: fan the retrieve step out in parallel
per subtask (LangGraph's Send API), add a router node that skips
decomposition entirely for obviously-simple questions, or add a critique
node that checks the answer against the retrieved context before
returning it (ties into Phase 6/7's transparency and governance goals).
"""

from typing import List, TypedDict

from pydantic import BaseModel, Field

from knowledge_ops.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    MAX_SUBTASKS,
    TOP_K_PER_SUBTASK,
    get_embeddings,
    get_llm,
)


# --- State ----------------------------------------------------------------


class RetrievedChunk(TypedDict):
    text: str
    source: str
    page: int | None
    score: float


class SubtaskResult(TypedDict):
    subtask: str
    chunks: List[RetrievedChunk]


class QueryState(TypedDict):
    question: str
    subtasks: List[str]
    subtask_results: List[SubtaskResult]
    answer: str
    sources: List[str]


# --- Structured output for the decomposition step --------------------------


class SubtaskList(BaseModel):
    """The set of self-contained sub-questions needed to fully answer the
    original question. If the original question is already simple and
    single-topic, this should just contain that one question unchanged."""

    subtasks: List[str] = Field(
        description=(
            "1 to 5 self-contained sub-questions. Each should be answerable "
            "on its own by searching a document library, and together they "
            "should cover everything needed to answer the original question."
        )
    )


DECOMPOSE_SYSTEM_PROMPT = """You are the orchestrator for a company knowledge \
base agent. Break the user's question into a small number of self-contained \
sub-questions that, together, cover everything needed to answer it fully.

Rules:
- If the question is already simple and about one topic, return it as the \
only subtask, unchanged.
- Only split when the question genuinely spans more than one topic/policy \
(e.g. it asks about two different procedures, or how two different \
policies interact).
- Each subtask must make sense on its own, without seeing the others.
- Do not invent sub-questions the user didn't ask about."""


SYNTHESIZE_SYSTEM_PROMPT = """You are a company knowledge-base assistant. \
Answer the user's question using ONLY the provided context excerpts -- do \
not use outside knowledge and do not guess.

Rules:
- Write one coherent answer to the ORIGINAL question, not a list of \
separate answers to each subtask.
- Cite the source for every claim inline, like (Source: Attendance \
Policy.pdf, p.2).
- If the context doesn't fully cover some part of the question, say so \
explicitly instead of filling the gap with a guess.
- Be concise. Do not repeat the question back before answering."""


# --- Nodes ------------------------------------------------------------------


def decompose(state: QueryState) -> dict:
    llm = get_llm().with_structured_output(SubtaskList)
    result: SubtaskList = llm.invoke(
        [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": state["question"]},
        ]
    )

    subtasks = [s.strip() for s in result.subtasks if s.strip()][:MAX_SUBTASKS]
    if not subtasks:
        subtasks = [state["question"]]

    print("--- orchestrator: decomposed into subtasks ---")
    for i, subtask in enumerate(subtasks, start=1):
        print(f"  {i}. {subtask}")
    print()

    return {"subtasks": subtasks}


def retrieve(state: QueryState) -> dict:
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

    print("--- orchestrator: retrieving per subtask ---")
    subtask_results: List[SubtaskResult] = []
    for subtask in state["subtasks"]:
        hits = vector_store.similarity_search_with_score(
            subtask, k=TOP_K_PER_SUBTASK
        )
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
            page_label = f", p.{chunk['page'] + 1}" if chunk["page"] is not None else ""
            print(f"    - {chunk['source']}{page_label} (score={chunk['score']:.4f})")
    print()

    return {"subtask_results": subtask_results}


def synthesize(state: QueryState) -> dict:
    context_blocks = []
    all_sources = []
    for result in state["subtask_results"]:
        for chunk in result["chunks"]:
            page_label = f", p.{chunk['page'] + 1}" if chunk["page"] is not None else ""
            label = f"{chunk['source']}{page_label}"
            all_sources.append(label)
            context_blocks.append(f"[{label}]\n{chunk['text']}")

    context = "\n\n".join(context_blocks)
    user_message = (
        f"Original question: {state['question']}\n\n"
        f"Context excerpts:\n\n{context}"
    )

    llm = get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    # De-duplicate sources while preserving first-seen order, for a clean
    # "Sources:" footer.
    seen = set()
    unique_sources = []
    for label in all_sources:
        if label not in seen:
            seen.add(label)
            unique_sources.append(label)

    return {"answer": response.content, "sources": unique_sources}


# --- Graph assembly ----------------------------------------------------


def build_graph():
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(QueryState)
    builder.add_node("decompose", decompose)
    builder.add_node("retrieve", retrieve)
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "retrieve")
    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


def answer_question(question: str) -> QueryState:
    """Run the full orchestrator graph on a single question."""
    graph = build_graph()
    return graph.invoke(
        {
            "question": question,
            "subtasks": [],
            "subtask_results": [],
            "answer": "",
            "sources": [],
        }
    )
