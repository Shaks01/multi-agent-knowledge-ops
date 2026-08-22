"""
hello_agent.py imports langchain_core.messages at module level, so this
whole module needs that package importable. langchain_core is a
transitive dependency of `langchain` (already in requirements.txt), so on
a normal `pip install -r requirements.txt` setup it's present; this only
skips in an environment that hasn't installed the project's own
dependencies yet.
"""

import unittest

try:
    from knowledge_ops.agents import hello_agent
    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover -- environment-dependent
    hello_agent = None
    _IMPORT_ERROR = exc


class FakeHelloLLM:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return type("FakeResponse", (), {"content": self._content})()


@unittest.skipUnless(hello_agent is not None, f"langchain_core not importable: {_IMPORT_ERROR}")
class TestAsk(unittest.TestCase):
    def test_returns_the_llm_response_content(self):
        fake_llm = FakeHelloLLM("42, per the knowledge base.")
        answer = hello_agent.ask("What is the answer?", llm=fake_llm)
        self.assertEqual(answer, "42, per the knowledge base.")

    def test_sends_system_prompt_and_question_to_the_model(self):
        fake_llm = FakeHelloLLM("some answer")
        hello_agent.ask("a specific question", llm=fake_llm)

        [messages] = fake_llm.calls
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, hello_agent.SYSTEM_PROMPT)
        self.assertEqual(messages[1].content, "a specific question")


if __name__ == "__main__":
    unittest.main()
