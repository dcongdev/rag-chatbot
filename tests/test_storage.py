"""Real persistent Chroma storage with model computation replaced by test doubles."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from knowledge_base.embeddings import MODEL_NAME
from knowledge_base.ingestion import index_document, ingest_document
from knowledge_base.storage import ChromaStore, chunk_id
from tests.test_ingestion import write_pdf


class FakeEmbeddings(Embeddings):
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        raise AssertionError("Query embedding is outside this milestone.")


class StorageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.embeddings = FakeEmbeddings()
        self.store = ChromaStore(self.root / "vectors", embeddings=self.embeddings)
        self.addCleanup(self.store.close)

    def test_persists_text_embeddings_and_metadata_across_processes(self):
        document = Document(page_content="Persist this evidence",
                            metadata={"source": "sample.pdf", "page": 2})
        ids = self.store.index_chunks([document])
        self.store.close()
        code = """
import json, sys
from knowledge_base.storage import ChromaStore
with ChromaStore(sys.argv[1]) as store:
    data = store._collection.get(include=['documents', 'metadatas', 'embeddings'])
    data['embeddings'] = data['embeddings'].tolist()
    print(json.dumps(data))
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.root / "vectors")],
            capture_output=True, text=True, check=True, timeout=60,
        )
        data = json.loads(result.stdout)
        self.assertEqual(data["ids"], ids)
        self.assertEqual(data["documents"], [document.page_content])
        self.assertEqual(data["metadatas"], [document.metadata])
        self.assertEqual(data["embeddings"], [[21.0, 1.0, 0.5]])

    def test_duplicates_in_batch_and_on_reopen_do_not_reembed(self):
        document = Document(page_content="Repeated", metadata={"source": "a.txt"})
        ids = self.store.index_chunks([document, document])
        self.assertEqual(ids, [chunk_id(document)])
        self.assertEqual(self.embeddings.calls, [["Repeated"]])
        self.assertEqual(self.store.index_chunks([document]), [])
        self.store.close()
        with ChromaStore(self.root / "vectors") as reopened:
            with patch("knowledge_base.storage.create_embeddings") as factory:
                self.assertEqual(reopened.index_chunks([document, document]), [])
                factory.assert_not_called()
            self.assertEqual(reopened._collection.count(), 1)

    def test_same_text_keeps_distinct_sources_and_pages(self):
        documents = [
            Document(page_content="Shared text", metadata=metadata)
            for metadata in [{"source": "a.pdf", "page": 1},
                             {"source": "a.pdf", "page": 2},
                             {"source": "b.pdf", "page": 1}]
        ]
        ids = self.store.index_chunks(documents)
        self.assertEqual(len(set(ids)), 3)
        stored = self.store._collection.get(ids=ids)
        by_id = dict(zip(stored["ids"], stored["metadatas"]))
        for document in documents:
            self.assertEqual(by_id[chunk_id(document)], document.metadata)

    def test_ids_are_deterministic_and_metadata_order_independent(self):
        first = Document(page_content="text", metadata={"source": "x.pdf", "page": 1})
        reordered = Document(page_content="text", metadata={"page": 1, "source": "x.pdf"})
        changed = Document(page_content="text!", metadata=first.metadata)
        self.assertEqual(chunk_id(first), chunk_id(reordered))
        self.assertEqual(len(chunk_id(first)), 64)
        self.assertNotEqual(chunk_id(first), chunk_id(changed))

    def test_batches_only_embed_new_chunks(self):
        documents = [Document(page_content=f"chunk {i}", metadata={"source": "a.txt"})
                     for i in range(5)]
        self.store.index_chunks(documents[:1])
        self.embeddings.calls.clear()
        ids = self.store.index_chunks(documents, batch_size=2)
        self.assertEqual(len(ids), 4)
        self.assertEqual(self.embeddings.calls,
                         [["chunk 1"], ["chunk 2", "chunk 3"], ["chunk 4"]])
        self.assertEqual(self.store._collection.count(), 5)

    def test_empty_chunks_do_not_load_model(self):
        with ChromaStore(self.root / "empty") as store:
            with patch("knowledge_base.storage.create_embeddings") as factory:
                self.assertEqual(store.index_chunks([]), [])
                self.assertEqual(store.index_chunks([
                    Document(page_content=" \n", metadata={})]), [])
                factory.assert_not_called()
            self.assertEqual(store._collection.count(), 0)

    def test_embedding_failure_can_be_retried_without_duplicates(self):
        documents = [Document(page_content=f"chunk {i}", metadata={"source": "a.txt"})
                     for i in range(2)]
        with patch.object(self.embeddings, "embed_documents",
                          side_effect=[[[1.0, 1.0, 0.5]], RuntimeError("model failed")]):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                self.store.index_chunks(documents, batch_size=1)
        self.assertEqual(self.store._collection.count(), 1)
        self.assertEqual(self.store.index_chunks(documents), [chunk_id(documents[1])])
        self.assertEqual(self.store._collection.count(), 2)

    def test_wrong_vector_count_is_not_stored(self):
        document = Document(page_content="text", metadata={"source": "a.txt"})
        with patch.object(self.embeddings, "embed_documents", return_value=[[1., 2.], [3., 4.]]):
            with self.assertRaises(ValueError):
                self.store.index_chunks([document])
        self.assertEqual(self.store._collection.count(), 0)

    def test_invalid_metadata_rejected_before_embedding(self):
        for metadata in [{}, {"source": ""}, {"source": "a.pdf", "page": 0},
                         {"source": "a.pdf", "page": True},
                         {"source": "a.txt", "nested": {}},
                         {"source": "a.txt", "value": float("nan")}]:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    self.store.index_chunks([Document(page_content="text", metadata=metadata)])
        self.assertEqual(self.embeddings.calls, [])

    def test_pipeline_indexes_txt_and_pdf(self):
        txt = self.root / "notes.txt"
        txt.write_text("abcdefghijklmnop", encoding="utf-8")
        pdf = self.root / "notes.pdf"
        write_pdf(pdf, ["First page", "", "Third page"])
        for path in [txt, pdf]:
            expected = ingest_document(path, chunk_size=8, chunk_overlap=2)
            ids = index_document(path, self.store, chunk_size=8, chunk_overlap=2)
            self.assertEqual(ids, [chunk_id(document) for document in expected])
            stored = self.store._collection.get(ids=ids)
            records = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))
            for document in expected:
                self.assertEqual(records[chunk_id(document)],
                                 (document.page_content, document.metadata))
            self.assertEqual(index_document(path, self.store, chunk_size=8, chunk_overlap=2), [])

    def test_configurable_directory(self):
        with patch.dict(os.environ, {"CHROMA_PERSIST_DIR": str(self.root / "configured")}):
            with ChromaStore(embeddings=self.embeddings) as configured:
                self.assertEqual(configured.persist_directory, (self.root / "configured").resolve())
                configured.index_chunks([Document(page_content="text", metadata={"source": "a.txt"})])
            with ChromaStore(self.root / "explicit", embeddings=self.embeddings) as explicit:
                self.assertEqual(explicit.persist_directory, (self.root / "explicit").resolve())
                self.assertEqual(explicit._collection.count(), 0)

    def test_invalid_batch_size_and_closed_store(self):
        for size in [0, -1, True, 1.5]:
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    self.store.index_chunks([], batch_size=size)
        self.store.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.store.index_chunks([])

    def test_collection_model_guard(self):
        self.store._collection.modify(metadata={"embedding_model": "different-model"})
        with self.assertRaisesRegex(ValueError, "embedding model"):
            ChromaStore(self.root / "vectors", embeddings=self.embeddings)

    def test_real_langchain_adapter_with_mocked_sentence_transformer(self):
        module = ModuleType("sentence_transformers")
        model = Mock()
        model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        module.SentenceTransformer = Mock(return_value=model)
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            with ChromaStore(self.root / "adapter") as store:
                document = Document(page_content="LangChain embedding", metadata={"source": "a.txt"})
                store.index_chunks([document])
                store.index_chunks([document])
                module.SentenceTransformer.assert_called_once_with(
                    MODEL_NAME, cache_folder=None, device="cpu")
                model.encode.assert_called_once_with(
                    [document.page_content], show_progress_bar=False, normalize_embeddings=True)
                vectors = store._collection.get(include=["embeddings"])["embeddings"]
                np.testing.assert_allclose(vectors, [[0.1, 0.2, 0.3]])


if __name__ == "__main__":
    unittest.main()
