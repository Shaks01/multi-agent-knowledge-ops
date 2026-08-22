import unittest
from unittest import mock

from knowledge_ops.agents import reasoning
from tests.fakes import FakeChatModel, make_chunk, make_subtask_result


SUBTASK_RESULTS = [
    make_subtask_result(
        "Does the NDA cover contractors?",
        [make_chunk("Confidential information includes...", "Non-Disclosure Policy.pdf", page=0)],
    )
]


class TestRun(unittest.TestCase):
    def test_first_pass_is_not_a_revision(self):
        fake_llm = FakeChatModel(content="A cited draft answer.")
        with mock.patch.object(reasoning, "get_llm", lambda: fake_llm):
            result = reasoning.run(
                question="Does the NDA cover contractors?",
                subtask_results=SUBTASK_RESULTS,
                revision_count=0,
            )

        self.assertEqual(result["draft_answer"], "A cited draft answer.")
        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(result["trace"][0]["action"], "draft_answer")

    def test_revision_with_specific_issues_increments_revision_count(self):
        fake_llm = FakeChatModel(content="A revised, more careful answer.")
        feedback = {"approved": False, "confidence": 0.2, "issues": ["the retention period claim is unsupported"]}

        with mock.patch.object(reasoning, "get_llm", lambda: fake_llm):
            result = reasoning.run(
                question="Does the NDA cover contractors?",
                subtask_results=SUBTASK_RESULTS,
                validation_feedback=feedback,
                revision_count=0,
            )

        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["trace"][0]["action"], "revise_answer")
        # The specific issue should have been passed into the prompt sent
        # to the model, not silently dropped.
        [messages] = fake_llm.calls
        user_message = messages[-1]["content"]
        self.assertIn("the retention period claim is unsupported", user_message)

    def test_approved_but_low_confidence_with_no_issues_still_counts_as_a_revision(self):
        # Regression test: earlier, is_revision was computed from
        # `bool(validation_feedback and validation_feedback.get("issues"))`,
        # so an approved-but-low-confidence verdict with an EMPTY issues
        # list (a real case -- see agents/validation.py, confidence can be
        # low even when approved=True) never counted as a revision. That
        # meant revision_count never incremented, so config.MAX_REVISIONS
        # never actually capped this retry path in graph.py -- the
        # validator -> reasoning loop could run forever. Fixed by basing
        # is_revision on "was there a prior validation verdict at all"
        # rather than "did it list specific issues."
        fake_llm = FakeChatModel(content="Same answer, tightened up.")
        feedback = {"approved": True, "confidence": 0.4, "issues": []}

        with mock.patch.object(reasoning, "get_llm", lambda: fake_llm):
            result = reasoning.run(
                question="Does the NDA cover contractors?",
                subtask_results=SUBTASK_RESULTS,
                validation_feedback=feedback,
                revision_count=0,
            )

        self.assertEqual(
            result["revision_count"], 1,
            "an approved-but-low-confidence retry must still increment "
            "revision_count, or config.MAX_REVISIONS can't bound it",
        )
        self.assertEqual(result["trace"][0]["action"], "revise_answer")
        # With no specific issues to list, the agent should still get an
        # actionable prompt (not silently just "try again").
        [messages] = fake_llm.calls
        user_message = messages[-1]["content"]
        self.assertIn("0.40", user_message)

    def test_sources_come_from_subtask_results_not_the_llm(self):
        fake_llm = FakeChatModel(content="An answer that doesn't even mention sources itself.")
        with mock.patch.object(reasoning, "get_llm", lambda: fake_llm):
            result = reasoning.run(
                question="Does the NDA cover contractors?",
                subtask_results=SUBTASK_RESULTS,
            )
        self.assertEqual(result["sources"], ["Non-Disclosure Policy.pdf, p.1"])


if __name__ == "__main__":
    unittest.main()
