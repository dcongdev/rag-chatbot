"""Streamlit interactions reuse the real backend with deterministic embeddings."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from knowledge_base.retrieval import RetrievedChunk
from knowledge_base.generation import (
    INSUFFICIENT_CONTEXT, GenerationError, MissingAPIKeyError, QuestionAnswer,
    SourceReference,
)
from knowledge_base.storage import ChromaStore
from langchain_core.documents import Document
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
        self.app.button(key="search_submit").click().run()

    def test_initial_page_lists_storage_without_loading_embeddings(self):
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.button(key="index").disabled)
        self.assertTrue(self.app.file_uploader[0].accept_multiple_files)
        self.assertEqual(self.app.number_input(key="top_k").value, 4)
        self.factory.assert_called_once_with()
        self.assertEqual(self.embeddings.document_calls, [])
        self.assertEqual(self.embeddings.query_calls, [])

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
        self.assertEqual(len(self.app.expander), 1)
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
        self.factory.assert_called_once_with()

    def test_search_empty_collection(self):
        self.search()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("No matching chunks" in item.value for item in self.app.info))
        self.assertEqual(self.embeddings.query_calls, [])

    def test_database_open_failure(self):
        st.cache_resource.clear()
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
        self.assertEqual(len(self.app.expander), 1)

    def test_missing_scores_and_pages_are_not_displayed(self):
        result = RetrievedChunk("id", "Plain content", {"source": "notes.txt"}, None, None)
        with patch("knowledge_base.retrieval.retrieve", return_value=[result]):
            self.search()
        self.assertFalse(self.app.exception)
        self.assertIn("Plain content", [item.value for item in self.app.text])
        self.assertFalse(any("PDF page:" in item.value or "Similarity score:" in item.value
                             for item in self.app.caption))

    def ask(self, question="What do the documents say?"):
        self.app.chat_input(key="question").set_value(question).run()

    def test_question_answer_and_sources_persist_on_rerun(self):
        answer = QuestionAnswer("Cats are mammals. [1]", [
            SourceReference(1, "chunk", "animals.pdf", 2),
        ], True)
        with patch("knowledge_base.generation.answer_question", return_value=answer) as generate:
            self.app.number_input(key="answer_top_k").set_value(2)
            self.ask()
            self.assertFalse(self.app.exception)
            self.assertEqual(generate.call_args.kwargs["top_k"], 2)
            self.assertIn("Cats are mammals. [1]", [item.value for item in self.app.markdown])
            self.assertIn("[1] animals.pdf — PDF page 2", [item.value for item in self.app.text])
            self.app.run()
            generate.assert_called_once()
            self.assertIn("Cats are mammals. [1]", [item.value for item in self.app.markdown])
        self.upload_txt()
        self.assertEqual(self.app.session_state["chat_messages"], [])

    def test_blank_question_does_not_open_store_or_generate(self):
        with patch("knowledge_base.generation.answer_question") as generate:
            self.ask("  ")
            generate.assert_not_called()
        self.factory.assert_called_once_with()
        self.assertEqual(self.app.warning[0].value, "Enter a question.")

    def test_question_missing_credentials_and_timeout(self):
        for error in [MissingAPIKeyError("Set GEMINI_API_KEY"), GenerationError("Gemini timed out")]:
            with self.subTest(error=str(error)):
                with patch("knowledge_base.generation.answer_question", side_effect=error):
                    self.ask()
                self.assertFalse(self.app.exception)
                self.assertIn(str(error), self.app.error[-1].value)
                self.assertEqual(self.app.session_state["chat_messages"][-1]["kind"], "error")

    def test_insufficient_context_displays_information(self):
        with patch("knowledge_base.generation.answer_question", return_value=
                   QuestionAnswer(INSUFFICIENT_CONTEXT, [], False)):
            self.ask()
        self.assertFalse(self.app.exception)
        self.assertIn(INSUFFICIENT_CONTEXT, [item.value for item in self.app.info])

    def test_failed_question_preserves_history_and_hides_internal_error(self):
        with patch("knowledge_base.generation.answer_question", return_value=
                   QuestionAnswer("Previous answer", [], True)):
            self.ask()
        with patch("knowledge_base.generation.answer_question", side_effect=RuntimeError("secret-key")):
            self.ask("Another question")
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.session_state["chat_messages"]), 4)
        self.assertIn("Previous answer", [item.value for item in self.app.markdown])
        self.assertNotIn("secret-key", self.app.error[0].value)

    def test_management_list_refreshes_after_upload(self):
        self.upload_txt()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.dataframe[0].value.to_dict("records"), [
            {"Filename": "cats.txt", "Chunks": 1}, {"Filename": "fish.txt", "Chunks": 1},
        ])
        self.assertTrue(self.app.button(key="replace").disabled)

    def test_add_existing_filename_requires_explicit_replacement(self):
        self.upload_txt()
        self.app.file_uploader(key="uploads").set_value(
            ("cats.txt", b"Kittens drink milk", "text/plain")
        ).run()
        self.app.button(key="index").click().run()
        self.assertTrue(any("Use Replace document" in item.value for item in self.app.info))
        records = self.stores[0]._collection.get(where={"source": "cats.txt"})
        self.assertEqual(records["documents"], ["Cats are mammals"])

    def test_partial_upload_failure_does_not_merge_later_same_name(self):
        self.app.file_uploader(key="uploads").set_value([
            ("cats.txt", b"Cats are mammals", "text/plain"),
            ("cats.txt", b"Kittens drink milk", "text/plain"),
            ("fish.txt", b"Fish live underwater", "text/plain"),
        ]).run()

        def fail_after_write(filename, content, store):
            ids = index_upload(filename, content, store)
            if content == b"Cats are mammals":
                raise RuntimeError("Indexing acknowledgement failed after saving a chunk")
            return ids

        with patch("knowledge_base.uploads.index_upload", side_effect=fail_after_write):
            self.app.button(key="index").click().run()
        self.assertFalse(self.app.exception)
        records = self.stores[0]._collection.get(where={"source": "cats.txt"})
        self.assertEqual(records["documents"], ["Cats are mammals"])
        self.assertEqual(self.stores[0]._collection.count(), 2)
        self.assertTrue(any("Use Replace document" in item.value for item in self.app.info))

    def test_delete_requires_confirmation_and_removes_retrieval_results(self):
        self.upload_txt()
        self.search()
        self.app.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Old answer", "kind": "answer"},
        ]
        self.app.button(key="request_delete").click().run()
        self.assertEqual(self.stores[0]._collection.count(), 2)
        self.assertIn("cats.txt", self.app.warning[0].value)
        self.app.button(key="confirm_delete").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.dataframe[0].value.to_dict("records"), [
            {"Filename": "fish.txt", "Chunks": 1},
        ])
        self.assertTrue(any("Deleted cats.txt" in item.value for item in self.app.success))
        self.assertNotIn("search_results", self.app.session_state)
        self.assertEqual(self.app.session_state["chat_messages"], [])
        self.search(top_k=20)
        self.assertNotIn("Source: cats.txt", [item.value for item in self.app.text])
        self.assertIn("Source: fish.txt", [item.value for item in self.app.text])
        self.app.button(key="request_delete").click().run()
        self.app.button(key="confirm_delete").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.dataframe), 0)
        self.assertTrue(any("No indexed documents" in item.value for item in self.app.info))

    def test_delete_cancel_and_selection_change_reset_confirmation(self):
        self.upload_txt()
        self.app.button(key="request_delete").click().run()
        self.app.button(key="cancel_delete").click().run()
        self.assertNotIn("pending_delete", self.app.session_state)
        self.assertEqual(self.stores[0]._collection.count(), 2)
        self.app.button(key="request_delete").click().run()
        self.app.selectbox(key="managed_document").set_value("fish.txt").run()
        self.assertNotIn("pending_delete", self.app.session_state)
        self.assertEqual(self.stores[0]._collection.count(), 2)

    def test_replace_pdf_removes_stale_chunks_and_updates_search(self):
        self.upload_txt()
        self.stores[0].index_chunks([
            Document(page_content="Dogs enjoy walks", metadata={"source": "cats.txt"}),
        ])
        self.app.run()
        old_ids = self.stores[0]._collection.get(where={"source": "cats.txt"})["ids"]
        self.search()
        pdf = self.root / "replacement.pdf"
        write_pdf(pdf, ["", "Kittens drink milk"])
        self.app.file_uploader(key="replacement:cats.txt").set_value(
            ("replacement.pdf", pdf.read_bytes(), "application/pdf")
        ).run()
        self.app.button(key="replace").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Replaced cats.txt: 1 chunks" in item.value for item in self.app.success))
        self.assertEqual(self.app.dataframe[0].value.to_dict("records")[0],
                         {"Filename": "cats.txt", "Chunks": 1})
        records = self.stores[0]._collection.get(where={"source": "cats.txt"})
        self.assertTrue(set(old_ids).isdisjoint(records["ids"]))
        self.assertEqual(records["metadatas"], [{"source": "cats.txt", "page": 2}])
        self.assertNotIn("search_results", self.app.session_state)
        self.search(top_k=20)
        texts = [item.value for item in self.app.text]
        self.assertIn("Kittens drink milk", texts)
        self.assertNotIn("Cats are mammals", texts)
        self.assertNotIn("Dogs enjoy walks", texts)
        self.assertIn("PDF page: 2", [item.value for item in self.app.caption])

    def test_failed_replacement_preserves_document_and_shows_error(self):
        self.upload_txt()
        before = self.stores[0]._collection.get()
        for content in [b"\xff", b""]:
            with self.subTest(content=content):
                self.app.file_uploader(key="replacement:cats.txt").set_value(
                    ("invalid.txt", content, "text/plain")
                ).run()
                self.app.button(key="replace").click().run()
                self.assertFalse(self.app.exception)
                self.assertTrue(any("Could not replace cats.txt" in item.value for item in self.app.error))
                self.assertEqual(self.stores[0]._collection.get(), before)
                self.assertEqual(len(self.app.dataframe[0].value), 2)

    def test_delete_failure_keeps_document_and_shows_error(self):
        self.upload_txt()
        self.app.button(key="request_delete").click().run()
        with patch.object(self.stores[0], "delete_document", side_effect=RuntimeError("database unavailable")):
            self.app.button(key="confirm_delete").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Could not delete cats.txt" in item.value for item in self.app.error))
        self.assertEqual(self.stores[0]._collection.count(), 2)

    def test_listing_failure_does_not_remove_search_or_question_forms(self):
        with patch.object(self.stores[0], "list_documents", side_effect=RuntimeError("database unavailable")):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Could not list indexed documents" in item.value for item in self.app.error))
        self.assertEqual(self.app.text_input(key="query").label, "Search query")
        self.assertEqual(len(self.app.chat_input), 1)

    def test_tabs_and_optional_retrieval_layout(self):
        self.assertEqual([tab.label for tab in self.app.tabs], ["AI Chat", "Knowledge Base"])
        self.assertEqual(len(self.app.tabs[0].chat_input), 1)
        self.assertEqual(len(self.app.tabs[0].file_uploader), 0)
        self.assertEqual(len(self.app.tabs[1].file_uploader), 1)
        self.assertEqual(self.app.expander[0].label, "Semantic search (optional)")
        self.assertFalse(self.app.expander[0].proto.expanded)

    def test_chat_history_has_roles_sources_and_independent_questions(self):
        first = QuestionAnswer("First answer [1]", [SourceReference(1, "a", "first.pdf", 2)], True)
        second = QuestionAnswer("Second answer [1]", [SourceReference(1, "b", "second.txt", None)], True)
        with patch("knowledge_base.generation.answer_question", side_effect=[first, second]) as generate:
            self.ask("First question")
            self.ask("Second question")
            self.assertEqual([message.name for message in self.app.chat_message],
                             ["user", "assistant", "user", "assistant"])
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(generate.call_args_list[0].args, ("First question", self.stores[0]))
            self.assertEqual(generate.call_args_list[1].args, ("Second question", self.stores[0]))
            self.assertEqual(generate.call_args_list[1].kwargs, {"top_k": 4})
            self.assertIn("[1] first.pdf — PDF page 2", [item.value for item in self.app.chat_message[1].text])
            self.assertIn("[1] second.txt", [item.value for item in self.app.chat_message[3].text])
            self.app.run()
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(len(self.app.chat_message), 4)

    def test_clear_chat_preserves_documents_and_does_not_call_model(self):
        self.upload_txt()
        with patch("knowledge_base.generation.answer_question", return_value=
                   QuestionAnswer("Answer", [], True)) as generate:
            self.ask()
            self.app.button(key="clear_chat").click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.session_state["chat_messages"], [])
            self.assertEqual(len(self.app.chat_message), 0)
            generate.assert_called_once()
        self.assertEqual(self.stores[0]._collection.count(), 2)

    def test_new_session_has_no_chat_history(self):
        with patch("knowledge_base.generation.answer_question", return_value=
                   QuestionAnswer("Answer", [], True)):
            self.ask()
        another = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(another.exception)
        self.assertEqual(another.session_state["chat_messages"], [])
        self.assertEqual(len(another.chat_message), 0)

    def test_error_history_survives_rerun_without_retrying_model(self):
        with patch("knowledge_base.generation.answer_question", side_effect=GenerationError("Gemini timed out")) as generate:
            self.ask()
            self.app.run()
            generate.assert_called_once()
        self.assertEqual([message.name for message in self.app.chat_message], ["user", "assistant"])
        self.assertIn("Gemini timed out", self.app.chat_message[1].error[0].value)


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
