"""Streamlit interactions reuse the real backend with deterministic embeddings."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from knowledge_base.retrieval import RetrievedChunk
from knowledge_base.storage import ChromaStore
from knowledge_base.uploads import index_upload
from tests.test_ingestion import write_pdf
from tests.test_retrieval import DeterministicEmbeddings

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.embeddings = DeterministicEmbeddings()
        self.stores = []

        def create_store():
            store = ChromaStore(self.root / "vectors", embeddings=self.embeddings)
            self.stores.append(store)
            return store

        self.factory = patch("knowledge_base.storage.ChromaStore", side_effect=create_store).start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self.close_stores)
        st.cache_resource.clear()
        self.app = AppTest.from_file(str(APP), default_timeout=30).run()

    def close_stores(self):
        st.cache_resource.clear()
        for store in self.stores:
            store.close()

    def upload_txt(self):
        self.app.file_uploader[0].set_value([
            ("cats.txt", b"Cats are mammals", "text/plain"),
            ("fish.txt", b"Fish live underwater", "text/plain"),
        ]).run()
        self.app.button(key="index").click().run()

    def search(self, text="Tell me about cats", top_k=4):
        self.app.text_input(key="query").set_value(text)
        self.app.number_input(key="top_k").set_value(top_k)
        self.app.button[-1].click().run()

    def test_initial_page_is_lazy_and_requires_files(self):
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.button(key="index").disabled)
        self.assertTrue(self.app.file_uploader[0].accept_multiple_files)
        self.assertEqual(self.app.number_input(key="top_k").value, 4)
        self.factory.assert_not_called()

    def test_multiple_uploads_search_and_duplicate_handling(self):
        self.upload_txt()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.success), 2)
        self.assertEqual(self.stores[0]._collection.count(), 2)
        self.search(top_k=1)
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.expander), 1)
        self.assertIn("Source: cats.txt", [item.value for item in self.app.text])
        self.assertIn("Cats are mammals", [item.value for item in self.app.text])
        self.assertTrue(any("Similarity score: 1.000" in item.value for item in self.app.caption))
        self.app.run()
        self.assertEqual(len(self.app.expander), 1)
        self.app.button(key="index").click().run()
        self.assertEqual(len(self.app.info), 2)
        self.assertEqual(self.stores[0]._collection.count(), 2)
        self.assertEqual(len(self.embeddings.document_calls), 2)
        self.assertEqual(len(self.app.expander), 0)
        self.factory.assert_called_once_with()

    def test_pdf_upload_displays_original_page(self):
        pdf = self.root / "sample.pdf"
        write_pdf(pdf, ["", "Cats are mammals"])
        self.app.file_uploader[0].set_value(("sample.pdf", pdf.read_bytes(), "application/pdf")).run()
        self.app.button(key="index").click().run()
        self.search()
        self.assertFalse(self.app.exception)
        self.assertIn("Source: sample.pdf", [item.value for item in self.app.text])
        self.assertIn("PDF page: 2", [item.value for item in self.app.caption])

    def test_bad_upload_does_not_block_valid_file_and_temp_files_are_removed(self):
        self.app.file_uploader[0].set_value([
            ("broken.txt", b"\xff", "text/plain"),
            ("empty.txt", b"", "text/plain"),
            ("cats.txt", b"Cats are mammals", "text/plain"),
        ]).run()
        self.app.button(key="index").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.error), 1)
        self.assertIn("broken.txt", self.app.error[0].value)
        self.assertEqual(len(self.app.info), 1)
        self.assertEqual(len(self.app.success), 1)
        self.assertEqual(self.stores[0]._collection.count(), 1)

    def test_blank_query_does_not_open_store(self):
        self.search(" \n ")
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.warning[0].value, "Enter a search query.")
        self.factory.assert_not_called()

    def test_search_empty_collection(self):
        self.search()
        self.assertFalse(self.app.exception)
        self.assertIn("No matching chunks", self.app.info[0].value)
        self.assertEqual(self.embeddings.query_calls, [])

    def test_database_open_failure(self):
        self.factory.side_effect = RuntimeError("Database unavailable")
        self.search()
        self.assertFalse(self.app.exception)
        self.assertIn("Database unavailable", self.app.error[0].value)
        self.assertNotIn("search_results", self.app.session_state)

    def test_retrieval_error_clears_previous_results(self):
        self.upload_txt()
        self.search()
        with patch("knowledge_base.retrieval.retrieve", side_effect=RuntimeError("Model unavailable")):
            self.search()
        self.assertFalse(self.app.exception)
        self.assertIn("Model unavailable", self.app.error[0].value)
        self.assertEqual(len(self.app.expander), 0)

    def test_missing_scores_and_pages_are_not_displayed(self):
        result = RetrievedChunk("id", "Plain content", {"source": "notes.txt"}, None, None)
        with patch("knowledge_base.retrieval.retrieve", return_value=[result]):
            self.search()
        self.assertFalse(self.app.exception)
        self.assertIn("Plain content", [item.value for item in self.app.text])
        self.assertFalse(any("PDF page:" in item.value or "Similarity score:" in item.value
                             for item in self.app.caption))


class UploadTests(unittest.TestCase):
    def test_temporary_file_preserves_basename_and_is_cleaned_up(self):
        observed = []

        def index(path, store):
            observed.append(path)
            self.assertEqual(path.name, "notes.txt")
            self.assertEqual(path.read_bytes(), b"document")
            return ["chunk-id"]

        with patch("knowledge_base.uploads.index_document", side_effect=index):
            ids = index_upload("..\\folder/notes.txt", b"document", object())
        self.assertEqual(ids, ["chunk-id"])
        self.assertFalse(observed[0].exists())

    def test_temporary_file_is_removed_after_indexing_failure(self):
        observed = []

        def fail(path, store):
            observed.append(path)
            raise RuntimeError("Indexing failed")

        with patch("knowledge_base.uploads.index_document", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "Indexing failed"):
                index_upload("notes.txt", b"document", object())
        self.assertFalse(observed[0].exists())

    def test_unsupported_upload_does_not_reach_ingestion(self):
        with patch("knowledge_base.uploads.index_document") as index:
            with self.assertRaisesRegex(ValueError, "PDF or TXT"):
                index_upload("notes.csv", b"document", object())
            index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
