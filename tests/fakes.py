"""
Small fake objects shared across the test suite.

None of these depend on the real langchain_google_genai / langchain_chroma
/ langgraph packages -- they only reproduce the tiny slice of interface
each agent actually calls (`.invoke(messages)`, `.with_structured_output(
cls).invoke(messages)`, `.similarity_search_with_score(query, k=...)`).
That's what makes the tests in this suite unit tests rather than
integration tests against a real LLM or vector store: every network- or
API-key-dependent call is replaced with one of these before the agent
code under test ever runs.
"""


class FakeLLMResponse:
    """Stands in for a LangChain chat model's response object -- the
    agents only ever read `.content` off of it."""

    def __init__(self, content):
        self.content = content


class FakeChatModel:
    """Stands in for `config.get_llm()`'s return value.

    Supports both call shapes the agents use:
      - `llm.invoke(messages)` (reasoning.py, hello_agent.py)
      - `llm.with_structured_output(SomeSchema).invoke(messages)`
        (planner.py, validation.py)

    Every call is recorded in `.calls` (each entry is the exact `messages`
    list passed in) so a test can assert on what a "revision" prompt
    contained, for example, without needing a real model to have produced
    it.
    """

    def __init__(self, content=None, structured_result=None):
        self._content = content
        self._structured_result = structured_result
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeLLMResponse(self._content)

    def with_structured_output(self, schema_cls):
        return _FakeStructuredModel(self, schema_cls)


class _FakeStructuredModel:
    def __init__(self, parent, schema_cls):
        self._parent = parent
        self._schema_cls = schema_cls

    def invoke(self, messages):
        self._parent.calls.append(messages)
        return self._parent._structured_result


def make_chunk(text, source, page=0, score=0.1):
    return {"text": text, "source": source, "page": page, "score": score}


def make_subtask_result(subtask, chunks):
    return {"subtask": subtask, "chunks": chunks}


class FakeChroma:
    """Stands in for `langchain_chroma.Chroma`.

    `hits` is a list of (doc, score) tuples returned verbatim (up to `k`)
    for every call to `similarity_search_with_score`, regardless of the
    query -- good enough for testing the Retrieval agent's own plumbing
    (does it build the right chunk dicts, does it log the right trace),
    which is all this agent is responsible for. Actual retrieval quality
    is checked with real documents via `run_search_demo.py`, not here.
    """

    def __init__(self, hits=None, collection_name=None, embedding_function=None, persist_directory=None):
        self.hits = hits if hits is not None else []

    def similarity_search_with_score(self, query, k=4):
        return self.hits[:k]


class FakeDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
