import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from knowledge_ops.agents import memory


class TestGroundingWarningLogic(unittest.TestCase):
    """Pure logic: _needs_grounding_warning / _apply_grounding_warning."""

    def test_no_validation_means_no_warning(self):
        self.assertFalse(memory._needs_grounding_warning(None))
        self.assertEqual(memory._apply_grounding_warning("answer", None), "answer")

    def test_rejected_verdict_needs_warning(self):
        self.assertTrue(memory._needs_grounding_warning({"approved": False, "confidence": 0.9}))

    def test_approved_high_confidence_does_not_need_warning(self):
        self.assertFalse(memory._needs_grounding_warning({"approved": True, "confidence": 0.95}))

    def test_approved_low_confidence_still_needs_warning(self):
        # This is the Phase 7 behavior: approved=True alone isn't enough
        # to skip the warning if confidence is below the threshold.
        self.assertTrue(memory._needs_grounding_warning({"approved": True, "confidence": 0.3}))

    def test_warning_header_and_specific_issue_are_both_present(self):
        validation = {"approved": False, "confidence": 0.2, "issues": ["the retention claim is unsupported"]}
        warned = memory._apply_grounding_warning("Original answer text.", validation)
        self.assertIn(memory.UNVERIFIED_WARNING_HEADER, warned)
        self.assertIn("the retention claim is unsupported", warned)
        self.assertIn("Original answer text.", warned)

    def test_no_warning_leaves_answer_unchanged(self):
        validation = {"approved": True, "confidence": 0.99}
        self.assertEqual(memory._apply_grounding_warning("Original answer text.", validation), "Original answer text.")


class TestRun(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        logs_dir = Path(self._tmpdir.name) / "logs"
        self._patches = [
            mock.patch.object(memory, "LOGS_DIR", logs_dir),
            mock.patch.object(memory, "TRACE_LOG_PATH", logs_dir / "agent_trace.jsonl"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmpdir.cleanup()

    def _read_records(self):
        with open(memory.TRACE_LOG_PATH, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_approved_answer_gets_disclaimer_but_no_warning(self):
        result = memory.run(
            question="q", answer="A clean, grounded answer.", sources=["A.pdf"], trace=[],
            conversation_history=[], validation={"approved": True, "confidence": 0.95},
        )
        self.assertNotIn(memory.UNVERIFIED_WARNING_HEADER, result["answer"])
        self.assertIn(memory.STANDING_DISCLAIMER, result["answer"])

    def test_low_confidence_answer_gets_both_warning_and_disclaimer(self):
        result = memory.run(
            question="q", answer="A shaky answer.", sources=["A.pdf"], trace=[],
            conversation_history=[], validation={"approved": True, "confidence": 0.3},
        )
        self.assertIn(memory.UNVERIFIED_WARNING_HEADER, result["answer"])
        self.assertIn(memory.STANDING_DISCLAIMER, result["answer"])

    def test_blocked_question_gets_neither_warning_nor_disclaimer(self):
        guard = {"allowed": False, "category": "injection_attempt", "reason": "blocked reason text"}
        result = memory.run(
            question="q", answer="blocked reason text", sources=[], trace=[],
            conversation_history=[], validation=None, guard=guard,
        )
        self.assertEqual(result["answer"], "blocked reason text")
        self.assertNotIn(memory.STANDING_DISCLAIMER, result["answer"])

        [record] = self._read_records()
        self.assertTrue(record["blocked_by_input_guard"])
        self.assertEqual(record["guard_category"], "injection_attempt")
        self.assertIsNone(record["evaluation"])

    def test_persisted_record_includes_evaluation_when_provided(self):
        evaluation = {"failures": ["low_grounding_confidence"], "grounding": {"approved": True, "confidence": 0.3}}
        memory.run(
            question="q", answer="answer", sources=["A.pdf"], trace=[],
            conversation_history=[], validation={"approved": True, "confidence": 0.3},
            evaluation=evaluation,
        )
        [record] = self._read_records()
        self.assertEqual(record["evaluation"], evaluation)

    def test_conversation_history_is_appended_not_replaced(self):
        history = [{"question": "earlier q", "answer": "earlier a"}]
        result = memory.run(
            question="new q", answer="new a", sources=[], trace=[],
            conversation_history=history, validation={"approved": True, "confidence": 0.9},
        )
        self.assertEqual(len(result["conversation_history"]), 2)
        self.assertEqual(result["conversation_history"][0], history[0])
        self.assertEqual(result["conversation_history"][1]["question"], "new q")

    def test_each_run_appends_exactly_one_jsonl_line(self):
        for _ in range(3):
            memory.run(
                question="q", answer="a", sources=[], trace=[],
                conversation_history=[], validation={"approved": True, "confidence": 0.9},
            )
        self.assertEqual(len(self._read_records()), 3)

    def test_run_ids_are_unique_across_calls(self):
        r1 = memory.run(question="q", answer="a", sources=[], trace=[], conversation_history=[])
        r2 = memory.run(question="q", answer="a", sources=[], trace=[], conversation_history=[])
        records = self._read_records()
        self.assertNotEqual(records[0]["run_id"], records[1]["run_id"])


if __name__ == "__main__":
    unittest.main()
