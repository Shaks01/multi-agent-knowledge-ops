import unittest
from unittest import mock

import pydantic

from knowledge_ops.agents import validation
from knowledge_ops.agents.validation import ValidationResult
from tests.fakes import FakeChatModel, make_chunk, make_subtask_result


SUBTASK_RESULTS = [
    make_subtask_result(
        "Does the NDA cover contractors?",
        [make_chunk("Confidential information includes...", "Non-Disclosure Policy.pdf", page=0)],
    )
]


class TestValidationResultSchema(unittest.TestCase):
    def test_confidence_must_be_between_0_and_1(self):
        ValidationResult(approved=True, confidence=0.0)  # boundary, should not raise
        ValidationResult(approved=True, confidence=1.0)  # boundary, should not raise
        with self.assertRaises(pydantic.ValidationError):
            ValidationResult(approved=True, confidence=1.5)
        with self.assertRaises(pydantic.ValidationError):
            ValidationResult(approved=True, confidence=-0.1)

    def test_unsupported_claims_and_notes_default_empty(self):
        result = ValidationResult(approved=True, confidence=0.9)
        self.assertEqual(result.unsupported_claims, [])
        self.assertEqual(result.notes, "")


class TestRun(unittest.TestCase):
    def test_approved_high_confidence_verdict_shape(self):
        fake_result = ValidationResult(approved=True, confidence=0.95, unsupported_claims=[], notes="fine")
        fake_llm = FakeChatModel(structured_result=fake_result)

        with mock.patch.object(validation, "get_llm", lambda: fake_llm):
            result = validation.run(
                question="Does the NDA cover contractors?",
                draft_answer="Yes, per the NDA.",
                subtask_results=SUBTASK_RESULTS,
            )

        verdict = result["validation"]
        self.assertTrue(verdict["approved"])
        self.assertEqual(verdict["confidence"], 0.95)
        self.assertEqual(verdict["issues"], [])
        self.assertEqual(result["trace"][0]["agent"], "validation")

    def test_rejected_verdict_carries_unsupported_claims_as_issues(self):
        fake_result = ValidationResult(
            approved=False,
            confidence=0.2,
            unsupported_claims=["the retention period claim is not in any document"],
            notes="never satisfied",
        )
        fake_llm = FakeChatModel(structured_result=fake_result)

        with mock.patch.object(validation, "get_llm", lambda: fake_llm):
            result = validation.run(
                question="Does the NDA cover contractors?",
                draft_answer="Records are retained for 10 years.",
                subtask_results=SUBTASK_RESULTS,
            )

        verdict = result["validation"]
        self.assertFalse(verdict["approved"])
        self.assertEqual(verdict["issues"], ["the retention period claim is not in any document"])

    def test_draft_answer_reaches_the_prompt(self):
        fake_result = ValidationResult(approved=True, confidence=0.9)
        fake_llm = FakeChatModel(structured_result=fake_result)

        with mock.patch.object(validation, "get_llm", lambda: fake_llm):
            validation.run(
                question="Does the NDA cover contractors?",
                draft_answer="A very specific draft answer to check for.",
                subtask_results=SUBTASK_RESULTS,
            )

        [messages] = fake_llm.calls
        self.assertIn("A very specific draft answer to check for.", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
