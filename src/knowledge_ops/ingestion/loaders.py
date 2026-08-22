"""
Phase 2: turn files in DOCUMENTS_DIR into LangChain Documents.

Loader choice is dispatched by file extension so new document types can be
added without touching the ingestion script itself -- add an entry to
`LOADERS_BY_EXTENSION` (and the matching package to requirements.txt).
"""

from pathlib import Path
from typing import List

from knowledge_ops.config import DOCUMENTS_DIR


def _load_pdf(path: Path) -> List:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path)).load()


def _load_text(path: Path) -> List:
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding="utf-8").load()


# Extension (lowercase, with leading dot) -> loader function.
# Add entries here as new document types show up, e.g.:
#   ".docx": _load_docx   (needs `docx2txt` -- see requirements.txt)
LOADERS_BY_EXTENSION = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_text,
}


def discover_files(documents_dir: Path = DOCUMENTS_DIR) -> List[Path]:
    """Return every file in `documents_dir` with a supported extension."""
    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {documents_dir}\n"
            "Create it and drop your source documents in, then re-run."
        )

    return sorted(
        path
        for path in documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() in LOADERS_BY_EXTENSION
    )


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> List:
    """Load every supported file in `documents_dir` into LangChain Documents.

    Each returned Document's metadata always includes `source` set to the
    plain filename (not the full path) -- that's what gets surfaced later
    as a citation when the agent answers a question, so it should stay
    short and human-readable.

    Files with an unsupported extension are skipped, with a warning
    printed for each one, rather than failing the whole run.
    """
    files = discover_files(documents_dir)
    if not files:
        raise FileNotFoundError(
            f"No supported documents found in {documents_dir} "
            f"(looked for: {', '.join(sorted(LOADERS_BY_EXTENSION))})"
        )

    all_skipped = [
        path
        for path in documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() not in LOADERS_BY_EXTENSION
    ]
    for path in all_skipped:
        print(f"  skipping {path.name} (no loader registered for {path.suffix!r})")

    documents = []
    for path in files:
        loader = LOADERS_BY_EXTENSION[path.suffix.lower()]
        loaded = loader(path)
        for doc in loaded:
            doc.metadata["source"] = path.name
        documents.extend(loaded)
        print(f"  loaded {path.name}: {len(loaded)} page(s)/section(s)")

    return documents
