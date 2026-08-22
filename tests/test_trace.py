import unittest
from datetime import datetime

from knowledge_ops.agents.trace import log_step


class TestLogStep(unittest.TestCase):
    def test_returns_single_element_list(self):
        result = log_step("planner", "create_plan")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_entry_has_agent_and_action(self):
        [entry] = log_step("retrieval", "retrieve_per_subtask")
        self.assertEqual(entry["agent"], "retrieval")
        self.assertEqual(entry["action"], "retrieve_per_subtask")

    def test_timestamp_is_iso_and_parseable(self):
        [entry] = log_step("memory", "persist_trace")
        # Should not raise -- confirms it's a real ISO-8601 timestamp, not
        # just some string.
        parsed = datetime.fromisoformat(entry["timestamp"])
        self.assertIsNotNone(parsed.tzinfo, "timestamp should be timezone-aware (UTC)")

    def test_extra_details_are_merged_in(self):
        [entry] = log_step("validation", "validate_answer", input={"x": 1}, output={"y": 2})
        self.assertEqual(entry["input"], {"x": 1})
        self.assertEqual(entry["output"], {"y": 2})

    def test_passing_agent_or_action_again_as_a_detail_kwarg_raises(self):
        # agent/action are ordinary named parameters on log_step, not
        # just dict keys -- so a caller can't accidentally clobber them
        # via **details; Python itself rejects the duplicate argument.
        with self.assertRaises(TypeError):
            log_step("planner", "create_plan", agent="duplicate")


if __name__ == "__main__":
    unittest.main()
