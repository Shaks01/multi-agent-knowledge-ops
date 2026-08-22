import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from knowledge_ops.agents import retrieval
from tests.fakes import FakeChroma, FakeDoc, make_chunk, make_subtask_result


class TestLabel(unittest.TestCase):
    def test_label_with_page(self):
        chunk = make_chunk("text", "Attendance Policy.pdf", page=1)
        self.assertEqual(retrieval._label(chunk), "Attendance Policy.pdf, p.2")

    def test_label_without_page(self):
        chunk = make_chunk("text", "Attendance Policy.pdf", page=None)
        self.assertEqual(retrieval._label(chunk), "Attendance Policy.pdf")

    def test_page_zero_displays_as_page_one(self):
        # page is 0-indexed internally; the label should be human ("p.1"),
        # not "p.0".
        chunk = make_chunk("text", "Attendance Policy.pdf", page=0)
        self.assertEqual(retrieval._label(chunk), "Attendance Policy.pdf, p.1")


class TestFormatContextBlocks(unittest.TestCase):
    def test_joins_chunks_from_multiple_subtasks(self):
        results = [
            make_subtask_result("subtask 1", [make_chunk("first chunk", "A.pdf", page=0)]),
            make_subtask_result("subtask 2", [make_chunk("second chunk", "B.pdf", page=2)]),
        ]
        blocks = retrieval.format_context_blocks(results)
        self.assertIn("[A.pdf, p.1]\nfirst chunk", blocks)
        self.assertIn("[B.pdf, p.3]\nsecond chunk", blocks)

    def test_empty_subtask_results_gives_empty_string(self):
        self.assertEqual(retrieval.format_context_blocks([]), "")


class TestCollectSources(unittest.TestCase):
    def test_deduplicates_preserving_first_seen_order(self):
        results = [
            make_subtask_result(
                "s1",
                [
                    make_chunk("t1", "A.pdf", page=0),
                    make_chunk("t2", "B.pdf", page=0),
                ],
            ),
            make_subtask_result(
                "s2",
                [
                    make_chunk("t3", "A.pdf", page=0),  # duplicate of A.pdf, p.1
                    make_chunk("t4", "C.pdf", page=0),
                ],
            ),
        ]
        self.assertEqual(
            retrieval.collect_sources(results),
            ["A.pdf, p.1", "B.pdf, p.1", "C.pdf, p.1"],
        )

    def test_no_chunks_gives_empty_list(self):
        self.assertEqual(retrieval.collect_sources([make_subtask_result("s1", [])]), [])


class TestRun(unittest.TestCase):
    """retrieval.run() itself, with Chroma and the embeddings client both
    faked out -- no real vector store or API key involved."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.persist_dir = Path(self._tmpdir.name) / "chroma_db"
        self.persist_dir.mkdir()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_raises_if_persist_dir_missing(self):
        # retrieval.run() does `from langchain_chroma import Chroma` before
        # its CHROMA_PERSIST_DIR.exists() check, so that import needs to
        # succeed even on this path -- a fake module is enough since the
        # Chroma class itself is never instantiated before the check runs.
        missing_dir = Path(self._tmpdir.name) / "does_not_exist"
        fake_chroma_module = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"langchain_chroma": fake_chroma_module}), \
                mock.patch.object(retrieval, "CHROMA_PERSIST_DIR", missing_dir):
            with self.assertRaises(FileNotFoundError):
                retrieval.run(["some subtask"])

    def test_run_builds_subtask_results_and_trace_from_fake_chroma(self):
        hits = [
            (FakeDoc("relevant excerpt", {"source": "Arbitration Policy.pdf", "page": 0}), 0.05),
        ]
        fake_chroma_module = mock.MagicMock()
        fake_chroma_module.Chroma = lambda **kwargs: FakeChroma(hits=hits)

        with mock.patch.dict(sys.modules, {"langchain_chroma": fake_chroma_module}), \
                mock.patch.object(retrieval, "CHROMA_PERSIST_DIR", self.persist_dir), \
                mock.patch.object(retrieval, "get_embeddings", lambda: object()):
            result = retrieval.run(["Does the NDA cover contractors?"])

        [subtask_result] = result["subtask_results"]
        self.assertEqual(subtask_result["subtask"], "Does the NDA cover contractors?")
        [chunk] = subtask_result["chunks"]
        self.assertEqual(chunk["source"], "Arbitration Policy.pdf")
        self.assertEqual(chunk["score"], 0.05)

        [trace_entry] = result["trace"]
        self.assertEqual(trace_entry["agent"], "retrieval")
        self.assertEqual(trace_entry["output"]["sources"], ["Arbitration Policy.pdf, p.1"])
        self.assertEqual(trace_entry["output"]["chunks_per_subtask"], [1])
        self.assertEqual(trace_entry["output"]["scores_per_subtask"], [[0.05]])

    def test_run_handles_zero_hits_per_subtask(self):
        fake_chroma_module = mock.MagicMock()
        fake_chroma_module.Chroma = lambda **kwargs: FakeChroma(hits=[])

        with mock.patch.dict(sys.modules, {"langchain_chroma": fake_chroma_module}), \
                mock.patch.object(retrieval, "CHROMA_PERSIST_DIR", self.persist_dir), \
                mock.patch.object(retrieval, "get_embeddings", lambda: object()):
            result = retrieval.run(["an unanswerable subtask"])

        [subtask_result] = result["subtask_results"]
        self.assertEqual(subtask_result["chunks"], [])
        self.assertEqual(result["trace"][0]["output"]["chunks_per_subtask"], [0])


if __name__ == "__main__":
    unittest.main()
