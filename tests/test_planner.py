import unittest
from unittest import mock

from knowledge_ops.agents import planner
from knowledge_ops.agents.planner import Plan
from knowledge_ops.config import MAX_SUBTASKS
from tests.fakes import FakeChatModel


class TestRun(unittest.TestCase):
    def test_simple_question_becomes_a_one_step_plan(self):
        fake_llm = FakeChatModel(structured_result=Plan(subtasks=["What is the arbitration process?"]))
        with mock.patch.object(planner, "get_llm", lambda: fake_llm):
            result = planner.run("What is the arbitration process?")

        self.assertEqual(result["plan"], ["What is the arbitration process?"])
        self.assertEqual(result["trace"][0]["agent"], "planner")

    def test_blank_subtasks_are_stripped(self):
        fake_llm = FakeChatModel(structured_result=Plan(subtasks=["real subtask", "  ", ""]))
        with mock.patch.object(planner, "get_llm", lambda: fake_llm):
            result = planner.run("some question")
        self.assertEqual(result["plan"], ["real subtask"])

    def test_falls_back_to_original_question_if_everything_is_blank(self):
        fake_llm = FakeChatModel(structured_result=Plan(subtasks=["", "   "]))
        with mock.patch.object(planner, "get_llm", lambda: fake_llm):
            result = planner.run("the original question")
        self.assertEqual(result["plan"], ["the original question"])

    def test_subtasks_are_capped_at_max_subtasks(self):
        too_many = [f"subtask {i}" for i in range(MAX_SUBTASKS + 5)]
        fake_llm = FakeChatModel(structured_result=Plan(subtasks=too_many))
        with mock.patch.object(planner, "get_llm", lambda: fake_llm):
            result = planner.run("a sprawling question")
        self.assertEqual(len(result["plan"]), MAX_SUBTASKS)

    def test_conversation_history_is_included_in_the_prompt(self):
        fake_llm = FakeChatModel(structured_result=Plan(subtasks=["follow-up subtask"]))
        history = [{"question": "What does the NDA cover?", "answer": "Confidential info, per the NDA."}]

        with mock.patch.object(planner, "get_llm", lambda: fake_llm):
            planner.run("what about arbitration?", conversation_history=history)

        [messages] = fake_llm.calls
        user_content = messages[-1]["content"]
        self.assertIn("What does the NDA cover?", user_content)
        self.assertIn("what about arbitration?", user_content)

    def test_no_history_means_plain_question_in_prompt(self):
        fake_llm = FakeChatModel(structured_result=Plan(subtasks=["a subtask"]))
        with mock.patch.object(planner, "get_llm", lambda: fake_llm):
            planner.run("a plain question", conversation_history=None)

        [messages] = fake_llm.calls
        self.assertEqual(messages[-1]["content"], "a plain question")


if __name__ == "__main__":
    unittest.main()
