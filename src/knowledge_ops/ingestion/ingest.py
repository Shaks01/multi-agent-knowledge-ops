"""
Phase 2 + 3: load documents, split into chunks, embed, and persist to Chroma.

Every run rebuilds the vector store from scratch (existing persisted data
in CHROMA_PERSIST_DIR is deleted first). That keeps this simple and
deterministic for a project this size -- no partial re-ingest, no
duplicate-chunk bookkeeping to worry about. Once the document set is
large enough that a full rebuild is too slow, that's the point to add
incremental/upsert logic; not before.
"""

import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge_ops.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    get_embeddings,
)
from knowledge_ops.ingestion.loaders import load_documents


def run() -> None:
    print(f"Loading documents from {DOCUMENTS_DIR} ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} page(s)/section(s) total.\n")

    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Produced {len(chunks)} chunk(s).\n")

    if CHROMA_PERSIST_DIR.exists():
        print(f"Removing existing vector store at {CHROMA_PERSIST_DIR} ...")
        shutil.rmtree(CHROMA_PERSIST_DIR)

    print("Embedding chunks and writing to Chroma (this calls the embedding API "
          f"once per chunk -- {len(chunks)} call(s)) ...")
    embeddings = get_embeddings()

    # Imported here, same reasoning as the lazy imports in config.py: don't
    # require the package until this function actually runs.
    from langchain_chroma import Chroma

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    count = vector_store._collection.count()
    print(
        f"\nDone. {count} chunk(s) stored in Chroma collection "
        f"'{CHROMA_COLLECTION_NAME}' at {CHROMA_PERSIST_DIR}."
    )


if __name__ == "__main__":
    run()
