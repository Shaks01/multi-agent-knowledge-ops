import unittest
from unittest import mock

from knowledge_ops.agents import evaluation
from tests.fakes import make_chunk, make_subtask_result

CLEAN_ANSWER = "Per the arbitration policy (Source: Arbitration Policy.pdf, p.1), disputes go to arbitration."
FABRICATED_CITATION_ANSWER = (
    "Per (Source: Arbitration Policy.pdf, p.1) and "
    "(Source: Made Up Nonexistent Policy.pdf, p.9), disputes go to arbitration."
)

RETRIEVED_ONE_CHUNK = [
    make_subtask_result(
        "Does arbitration apply?",
        [make_chunk("Disputes are resolved via arbitration.", "Arbitration Policy.pdf", page=0, score=0.12)],
    )
]


class TestRetrievalSignals(unittest.TestCase):
    def test_reports_chunk_count_and_best_score(self):
        signals = evaluation._retrieval_signals(RETRIEVED_ONE_CHUNK)
        [per_subtask] = signals["per_subtask"]
        self.assertEqual(per_subtask["chunk_count"], 1)
        self.assertEqual(per_subtask["best_score"], 0.12)
        self.assertEqual(signals["insufficient_subtasks"], [])

    def test_best_score_is_the_minimum_distance_not_maximum(self):
        results = [
            make_subtask_result(
                "s1",
                [
                    make_chunk("a", "A.pdf", score=0.5),
                    make_chunk("b", "A.pdf", score=0.1),
                    make_chunk("c", "A.pdf", score=0.3),
                ],
            )
        ]
        signals = evaluation._retrieval_signals(results)
        self.assertEqual(signals["per_subtask"][0]["best_score"], 0.1)

    def test_zero_chunks_flags_insufficient_retrieval(self):
        results = [make_subtask_result("unanswerable subtask", [])]
        signals = evaluation._retrieval_signals(results)
        self.assertEqual(signals["insufficient_subtasks"], ["unanswerable subtask"])
        self.assertIsNone(signals["per_subtask"][0]["best_score"])

    def test_relevance_threshold_unset_by_default_means_no_weak_relevance_flag(self):
        # config.RELEVANCE_DISTANCE_THRESHOLD is None unless explicitly
        # configured -- confirm the "opt-in" behavior described in
        # config.py actually holds.
        self.assertIsNone(evaluation.RELEVANCE_DISTANCE_THRESHOLD)
        signals = evaluation._retrieval_signals(RETRIEVED_ONE_CHUNK)
        self.assertEqual(signals["weak_relevance_subtasks"], [])

    def test_relevance_threshold_flags_weak_relevance_when_set(self):
        with mock.patch.object(evaluation, "RELEVANCE_DISTANCE_THRESHOLD", 0.1):
            signals = evaluation._retrieval_signals(RETRIEVED_ONE_CHUNK)
        self.assertEqual(signals["weak_relevance_subtasks"], ["Does arbitration apply?"])

    def test_min_chunks_per_subtask_configurable(self):
        with mock.patch.object(evaluation, "MIN_CHUNKS_PER_SUBTASK", 2):
            signals = evaluation._retrieval_signals(RETRIEVED_ONE_CHUNK)
        # Only 1 chunk was retrieved but 2 are now required.
        self.assertEqual(signals["insufficient_subtasks"], ["Does arbitration apply?"])


class TestRun(unittest.TestCase):
    def test_no_failures_when_everything_checks_out(self):
        validation = {"approved": True, "confidence": 0.95, "issues": []}
        result = evaluation.run(
            question="q", draft_answer=CLEAN_ANSWER,
            subtask_results=RETRIEVED_ONE_CHUNK, validation=validation,
        )
        self.assertEqual(result["evaluation"]["failures"], [])
        self.assertEqual(result["trace"][0]["agent"], "evaluation")

    def test_rejected_answer_flags_answer_rejected_and_low_confidence(self):
        validation = {"approved": False, "confidence": 0.2, "issues": ["some claim"]}
        result = evaluation.run(
            question="q", draft_answer=CLEAN_ANSWER,
            subtask_results=RETRIEVED_ONE_CHUNK, validation=validation,
        )
        failures = result["evaluation"]["failures"]
        self.assertIn("answer_rejected", failures)
        self.assertIn("low_grounding_confidence", failures)
        # Approved is False, so the conflicting-outputs check (which only
        # applies to an *approved* answer) must not fire here even though
        # the citation check would also fail for this input.
        self.assertNotIn("conflicting_agent_outputs", failures)

    def test_approved_but_fabricated_citation_flags_conflicting_agent_outputs(self):
        # This is the headline case: Validation is satisfied (high
        # confidence, approved), but the answer cites a document that was
        # never retrieved. Neither check is "wrong" in isolation -- the
        # disagreement between them is the signal.
        validation = {"approved": True, "confidence": 0.95, "issues": []}
        result = evaluation.run(
            question="q", draft_answer=FABRICATED_CITATION_ANSWER,
            subtask_results=RETRIEVED_ONE_CHUNK, validation=validation,
        )
        evaluation_record = result["evaluation"]
        self.assertIn("conflicting_agent_outputs", evaluation_record["failures"])
        self.assertEqual(
            evaluation_record["citation_check"]["unmatched"],
            ["Made Up Nonexistent Policy.pdf, p.9"],
        )
        self.assertFalse(evaluation_record["citation_check"]["consistent"])

    def test_insufficient_retrieval_flag(self):
        empty_results = [make_subtask_result("unanswerable", [])]
        validation = {"approved": True, "confidence": 0.9, "issues": []}
        result = evaluation.run(
            question="q", draft_answer="No context to cite.",
            subtask_results=empty_results, validation=validation,
        )
        self.assertIn("insufficient_retrieval", result["evaluation"]["failures"])

    def test_missing_validation_defaults_gracefully(self):
        # validation=None can happen structurally even though graph.py
        # always supplies a dict in practice -- shouldn't raise.
        result = evaluation.run(
            question="q", draft_answer=CLEAN_ANSWER,
            subtask_results=RETRIEVED_ONE_CHUNK, validation=None,
        )
        self.assertIsInstance(result["evaluation"]["failures"], list)


if __name__ == "__main__":
    unittest.main()
