"""
Tests for the graph's pure routing functions.

_route_after_guard and _route_after_validation only ever read from a
plain dict (the AgentState "instance" -- TypedDict is just a dict at
runtime) and return a string node name. Neither one touches LangGraph
itself, which is only imported lazily inside build_graph() -- so these
are true unit tests of the routing *decisions*, independent of whether
the langgraph package is even installed.
"""

import unittest
from unittest import mock

from knowledge_ops.orchestration import graph


class TestRouteAfterGuard(unittest.TestCase):
    def test_allowed_goes_to_planner(self):
        self.assertEqual(graph._route_after_guard({"guard": {"allowed": True}}), "planner")

    def test_blocked_goes_to_memory(self):
        self.assertEqual(graph._route_after_guard({"guard": {"allowed": False}}), "memory")

    def test_missing_guard_key_defaults_to_allowed(self):
        self.assertEqual(graph._route_after_guard({}), "planner")


class TestRouteAfterValidation(unittest.TestCase):
    def test_rejected_with_retries_remaining_goes_to_reasoning(self):
        state = {"validation": {"approved": False, "confidence": 0.2}, "revision_count": 0}
        self.assertEqual(graph._route_after_validation(state), "reasoning")

    def test_rejected_with_retry_cap_hit_goes_to_evaluator(self):
        with mock.patch.object(graph, "MAX_REVISIONS", 1):
            state = {"validation": {"approved": False, "confidence": 0.2}, "revision_count": 1}
            self.assertEqual(graph._route_after_validation(state), "evaluator")

    def test_approved_high_confidence_goes_straight_to_evaluator(self):
        state = {"validation": {"approved": True, "confidence": 0.95}, "revision_count": 0}
        self.assertEqual(graph._route_after_validation(state), "evaluator")

    def test_approved_low_confidence_with_retries_remaining_goes_to_reasoning(self):
        # This is the Phase 7 behavior: approval alone isn't enough to
        # skip the retry loop if confidence is below threshold.
        state = {"validation": {"approved": True, "confidence": 0.1}, "revision_count": 0}
        self.assertEqual(graph._route_after_validation(state), "reasoning")

    def test_approved_low_confidence_with_retry_cap_hit_goes_to_evaluator(self):
        with mock.patch.object(graph, "MAX_REVISIONS", 1):
            state = {"validation": {"approved": True, "confidence": 0.1}, "revision_count": 1}
            self.assertEqual(graph._route_after_validation(state), "evaluator")

    def test_missing_validation_key_defaults_to_approved(self):
        state = {"revision_count": 0}
        self.assertEqual(graph._route_after_validation(state), "evaluator")


if __name__ == "__main__":
    unittest.main()
