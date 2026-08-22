"""
Tests for ingestion/loaders.py's file-discovery and dispatch logic.

Deliberately does NOT exercise the real _load_pdf/_load_text functions --
those need langchain_community actually installed to import PyPDFLoader/
TextLoader. discover_files() and load_documents()'s dispatch behavior are
pure filesystem + dict-lookup logic and don't need any loader library at
all, so that's what's tested here; load_documents() is exercised with a
fake entry swapped into LOADERS_BY_EXTENSION instead of a real loader.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from knowledge_ops.ingestion import loaders


class FakeDoc:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}


class TestDiscoverFiles(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.docs_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_raises_if_directory_does_not_exist(self):
        missing = self.docs_dir / "does_not_exist"
        with self.assertRaises(FileNotFoundError):
            loaders.discover_files(missing)

    def test_only_supported_extensions_are_returned(self):
        (self.docs_dir / "a.pdf").write_text("x")
        (self.docs_dir / "b.txt").write_text("x")
        (self.docs_dir / "c.md").write_text("x")
        (self.docs_dir / "d.docx").write_text("x")  # unsupported, no loader registered
        (self.docs_dir / "notes").mkdir()  # a directory, not a file -- must be skipped

        found = {p.name for p in loaders.discover_files(self.docs_dir)}
        self.assertEqual(found, {"a.pdf", "b.txt", "c.md"})

    def test_empty_directory_returns_empty_list(self):
        self.assertEqual(loaders.discover_files(self.docs_dir), [])

    def test_results_are_sorted(self):
        (self.docs_dir / "z.txt").write_text("x")
        (self.docs_dir / "a.txt").write_text("x")
        names = [p.name for p in loaders.discover_files(self.docs_dir)]
        self.assertEqual(names, ["a.txt", "z.txt"])


class TestLoadDocuments(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.docs_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_raises_if_no_supported_files_found(self):
        (self.docs_dir / "unsupported.docx").write_text("x")
        with self.assertRaises(FileNotFoundError):
            loaders.load_documents(self.docs_dir)

    def test_dispatches_to_the_registered_loader_and_stamps_source_metadata(self):
        (self.docs_dir / "policy.txt").write_text("policy content")
        fake_loader = mock.Mock(return_value=[FakeDoc("policy content")])

        with mock.patch.object(loaders, "LOADERS_BY_EXTENSION", {".txt": fake_loader}):
            documents = loaders.load_documents(self.docs_dir)

        fake_loader.assert_called_once()
        [doc] = documents
        self.assertEqual(doc.metadata["source"], "policy.txt")

    def test_unsupported_files_are_skipped_not_fatal(self):
        (self.docs_dir / "policy.txt").write_text("policy content")
        (self.docs_dir / "unsupported.docx").write_text("x")
        fake_loader = mock.Mock(return_value=[FakeDoc("policy content")])

        with mock.patch.object(loaders, "LOADERS_BY_EXTENSION", {".txt": fake_loader}):
            documents = loaders.load_documents(self.docs_dir)

        self.assertEqual(len(documents), 1)


if __name__ == "__main__":
    unittest.main()
