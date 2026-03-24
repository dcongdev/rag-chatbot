"""Document operations against real persistent Chroma with deterministic vectors."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from knowledge_base.ingestion import IngestionError, index_document, replace_document
from knowledge_base.storage import (
    ChromaStore, DocumentNotFoundError, IndexedDocument, ReplacementError, chunk_id,
)
from tests.test_ingestion import write_pdf
from tests.test_storage import FakeEmbeddings


class DocumentManagementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.embeddings = FakeEmbeddings()
        self.store = ChromaStore(self.root / "vectors", embeddings=self.embeddings)
        self.addCleanup(self.store.close)
        self.old = [Document(page_content=f"original {page}",
                             metadata={"source": "notes.pdf", "page": page})
                    for page in [1, 2, 3]]
        self.other = Document(page_content="unrelated text", metadata={"source": "notes.pdf.backup"})
        self.store.index_chunks([*self.old, self.other])

    def replacement(self):
        return [self.old[0], Document(page_content="updated page", metadata={"source": "notes.pdf", "page": 2})]

    def records(self):
        data = self.store._collection.get(include=["documents", "metadatas", "embeddings"])
        return {identifier: (text, metadata, vector.tolist())
                for identifier, text, metadata, vector in zip(
                    data["ids"], data["documents"], data["metadatas"], data["embeddings"])}

    def assert_original(self, expected):
        self.assertEqual(self.records(), expected)

    def test_listing_and_empty_collection(self):
        self.assertEqual(self.store.list_documents(), [
            IndexedDocument("notes.pdf", 3), IndexedDocument("notes.pdf.backup", 1)])
        self.store.index_chunks(self.old)
        self.assertEqual(self.store.list_documents()[0].chunk_count, 3)
        with ChromaStore(self.root / "empty") as empty:
            self.assertEqual(empty.list_documents(), [])

    def test_listing_spans_multiple_metadata_pages(self):
        documents = [Document(page_content=f"item {index}", metadata={"source": "large.txt"})
                     for index in range(1001)]
        self.store.index_chunks(documents)
        listing = {item.filename: item.chunk_count for item in self.store.list_documents()}
        self.assertEqual(listing, {"large.txt": 1001, "notes.pdf": 3, "notes.pdf.backup": 1})

    def test_delete_exact_filename_leaves_other_sources(self):
        self.assertEqual(self.store.delete_document("notes.pdf"), 3)
        self.assertEqual(self.store.list_documents(), [IndexedDocument("notes.pdf.backup", 1)])
        self.assertEqual(set(self.records()), {chunk_id(self.other)})
        with self.assertRaises(DocumentNotFoundError):
            self.store.delete_document("notes.pdf")

    def test_filenames_are_literal_and_case_sensitive(self):
        before = self.records()
        for filename in ["notes", "NOTES.PDF", "*", "notes.pdf "]:
            with self.subTest(filename=filename):
                with self.assertRaises(DocumentNotFoundError):
                    self.store.delete_document(filename)
        self.assert_original(before)

    def test_replacement_removes_stale_chunks_and_keeps_other_documents(self):
        documents = self.replacement()
        ids = self.store.replace_chunks("notes.pdf", documents, batch_size=1)
        self.assertEqual(ids, [chunk_id(document) for document in documents])
        self.assertEqual(set(self.records()), set(ids) | {chunk_id(self.other)})
        self.assertEqual(self.store.list_documents()[0], IndexedDocument("notes.pdf", 2))
        self.assertEqual(self.records()[ids[1]][1], {"source": "notes.pdf", "page": 2})

    def test_identical_replacement_does_not_reembed_or_duplicate(self):
        self.embeddings.calls.clear()
        self.assertEqual(self.store.replace_chunks("notes.pdf", self.old + self.old),
                         [chunk_id(document) for document in self.old])
        self.assertEqual(self.embeddings.calls, [])
        self.assertEqual(len(self.records()), 4)

    def test_missing_replacement_target_does_not_index(self):
        before = self.records()
        with self.assertRaises(DocumentNotFoundError):
            self.store.replace_chunks("missing.pdf", self.replacement())
        self.assert_original(before)

    def test_invalid_or_empty_replacement_does_not_modify_store(self):
        before = self.records()
        for documents in [[], [Document(page_content=" \n", metadata={"source": "notes.pdf"})],
                          [self.other], [Document(page_content="text", metadata={"source": "notes.pdf", "page": 0})]]:
            with self.subTest(documents=documents):
                with self.assertRaises(ValueError):
                    self.store.replace_chunks("notes.pdf", documents)
        self.assert_original(before)

    def test_embedding_failure_preserves_original(self):
        before = self.records()
        with patch.object(self.embeddings, "embed_documents", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(ReplacementError, "preserved"):
                self.store.replace_chunks("notes.pdf", self.replacement())
        self.assert_original(before)

    def test_partial_indexing_failure_removes_staged_chunks(self):
        before = self.records()
        upsert = self.store._collection.upsert
        count = 0

        def fail_second(**kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("write failed")
            return upsert(**kwargs)

        documents = [Document(page_content=f"new {page}", metadata={"source": "notes.pdf", "page": page})
                     for page in [1, 2]]
        with patch.object(self.store._collection, "upsert", side_effect=fail_second):
            with self.assertRaisesRegex(ReplacementError, "preserved"):
                self.store.replace_chunks("notes.pdf", documents, batch_size=1)
        self.assert_original(before)

    def test_failed_upsert_acknowledgement_removes_written_replacement(self):
        before = self.records()
        upsert = self.store._collection.upsert

        def fail_after_write(**kwargs):
            upsert(**kwargs)
            raise RuntimeError("write acknowledgement failed")

        with patch.object(self.store._collection, "upsert", side_effect=fail_after_write):
            with self.assertRaisesRegex(ReplacementError, "preserved"):
                self.store.replace_chunks("notes.pdf", self.replacement())
        self.assert_original(before)

    def test_delete_failure_after_mutation_restores_old_records_and_vectors(self):
        before = self.records()
        delete = self.store._collection.delete
        count = 0

        def fail_after_delete(**kwargs):
            nonlocal count
            count += 1
            result = delete(**kwargs)
            if count == 1:
                raise RuntimeError("delete acknowledgement failed")
            return result

        with patch.object(self.store._collection, "delete", side_effect=fail_after_delete):
            with self.assertRaisesRegex(ReplacementError, "preserved"):
                self.store.replace_chunks("notes.pdf", self.replacement(), batch_size=1)
        self.assert_original(before)

    def test_rollback_failure_is_explicit(self):
        with patch.object(self.store._collection, "delete", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(ReplacementError, "rollback was incomplete"):
                self.store.replace_chunks("notes.pdf", self.replacement())

    def test_reprocessing_new_pdf_preserves_target_identity_and_new_pages(self):
        replacement = self.root / "new-name.pdf"
        write_pdf(replacement, ["", "Updated evidence"])
        ids = replace_document("notes.pdf", replacement, self.store)
        self.assertEqual(len(ids), 1)
        record = self.records()[ids[0]]
        self.assertEqual(record[0].strip(), "Updated evidence")
        self.assertEqual(record[1], {"source": "notes.pdf", "page": 2})

    def test_txt_replacement_pipeline_and_repeated_replacement(self):
        original = self.root / "guide.txt"
        original.write_text("Old guide content", encoding="utf-8")
        old_ids = index_document(original, self.store)
        replacement = self.root / "updated.txt"
        replacement.write_text("New guide content is longer", encoding="utf-8")
        ids = replace_document("guide.txt", replacement, self.store, chunk_size=10, chunk_overlap=2)
        self.assertTrue(set(old_ids).isdisjoint(self.records()))
        self.assertTrue(all(self.records()[identifier][1] == {"source": "guide.txt"} for identifier in ids))
        self.embeddings.calls.clear()
        self.assertEqual(replace_document("guide.txt", replacement, self.store,
                                          chunk_size=10, chunk_overlap=2), ids)
        self.assertEqual(self.embeddings.calls, [])

    def test_parsing_and_empty_file_failures_leave_original_unchanged(self):
        before = self.records()
        path = self.root / "replacement.txt"
        path.write_bytes(b"\xff")
        with self.assertRaises(IngestionError):
            replace_document("notes.pdf", path, self.store)
        path.write_bytes(b"")
        with self.assertRaises(ValueError):
            replace_document("notes.pdf", path, self.store)
        self.assert_original(before)

    def test_operations_persist_after_reopen(self):
        ids = self.store.replace_chunks("notes.pdf", self.replacement())
        self.store.delete_document("notes.pdf.backup")
        self.store.close()
        with ChromaStore(self.root / "vectors") as reopened:
            self.assertEqual(reopened.list_documents(), [IndexedDocument("notes.pdf", 2)])
            self.assertEqual(set(reopened._collection.get()["ids"]), set(ids))
            self.assertEqual(reopened.delete_document("notes.pdf"), 2)
        with ChromaStore(self.root / "vectors") as reopened:
            self.assertEqual(reopened.list_documents(), [])

    def test_invalid_names_parameters_and_closed_store(self):
        for filename in ["", " ", None, "../notes.pdf", "folder\\notes.pdf"]:
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    self.store.delete_document(filename)
                with self.assertRaises(ValueError):
                    self.store.replace_chunks(filename, self.old)
        for size in [0, True, 1.5]:
            with self.assertRaises(ValueError):
                self.store.replace_chunks("notes.pdf", self.old, batch_size=size)
        self.store.close()
        for operation in [self.store.list_documents, lambda: self.store.delete_document("notes.pdf"),
                          lambda: self.store.replace_chunks("notes.pdf", self.old)]:
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation()


if __name__ == "__main__":
    unittest.main()
