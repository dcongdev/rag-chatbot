"""Deterministic query vectors exercise real persistent Chroma retrieval."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from knowledge_base.retrieval import retrieve
from knowledge_base.storage import ChromaStore, chunk_id


class DeterministicEmbeddings(Embeddings):
    vectors = {
        "Cats are mammals": [1.0, 0.0],
        "Kittens drink milk": [0.8, 0.6],
        "Dogs enjoy walks": [0.0, 1.0],
        "Birds have feathers": [-0.6, 0.8],
        "Fish live underwater": [-1.0, 0.0],
    }

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [self.vectors[text] for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [1.0, 0.0] if text == "Tell me about cats" else [-1.0, 0.0]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "vectors"
        self.embeddings = DeterministicEmbeddings()
        self.store = ChromaStore(self.path, embeddings=self.embeddings)
        self.addCleanup(self.store.close)
        self.documents = [
            Document(page_content=text, metadata={"source": "animals.pdf", "page": page})
            for page, text in enumerate(self.embeddings.vectors, start=1)
        ]
        self.store.index_chunks(self.documents)

    def test_ranking_content_metadata_and_scores(self):
        results = retrieve("Tell me about cats", self.store, top_k=3)
        self.assertEqual([r.content for r in results],
                         [d.page_content for d in self.documents[:3]])
        for result, document, distance in zip(results, self.documents, [0.0, 0.4, 2.0]):
            self.assertEqual(result.id, chunk_id(document))
            self.assertEqual(result.metadata, document.metadata)
            self.assertAlmostEqual(result.distance, distance, places=5)
            self.assertAlmostEqual(result.similarity_score, 1 / (1 + distance), places=5)
        self.assertEqual(self.embeddings.query_calls, ["Tell me about cats"])
        self.assertEqual(len(self.embeddings.document_calls), 1)

    def test_default_and_configurable_top_k(self):
        self.assertEqual(len(retrieve("Tell me about cats", self.store)), 4)
        self.assertEqual(len(retrieve("Tell me about cats", self.store, top_k=1)), 1)
        self.assertEqual(len(retrieve("Tell me about cats", self.store, top_k=100)), 5)

    def test_query_vector_changes_nearest_chunk(self):
        result = retrieve("Where do fish live?", self.store, top_k=1)[0]
        self.assertEqual(result.content, "Fish live underwater")
        self.assertAlmostEqual(result.distance, 0)

    def test_blank_queries_do_not_embed_or_query_database(self):
        with patch.object(self.store._collection, "query") as query:
            for text in ["", " ", "\n\t"]:
                self.assertEqual(retrieve(text, self.store), [])
            query.assert_not_called()
        self.assertEqual(self.embeddings.query_calls, [])

    def test_query_is_trimmed(self):
        results = retrieve(" \nTell me about cats\t ", self.store, top_k=1)
        self.assertEqual(results[0].content, "Cats are mammals")
        self.assertEqual(self.embeddings.query_calls, ["Tell me about cats"])

    def test_invalid_parameters(self):
        for top_k in [0, -1, True, False, 1.5, "3", None]:
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    retrieve("cats", self.store, top_k=top_k)
        for query in [None, 123, [], b"cats"]:
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "query"):
                    retrieve(query, self.store)
        self.assertEqual(self.embeddings.query_calls, [])

    def test_empty_collection_does_not_load_model(self):
        with ChromaStore(self.path.parent / "empty") as empty:
            with patch("knowledge_base.storage.create_embeddings") as factory:
                self.assertEqual(retrieve("cats", empty), [])
                factory.assert_not_called()

    def test_reopened_collection_uses_shared_embedding_factory_once(self):
        self.store.close()
        embeddings = DeterministicEmbeddings()
        with ChromaStore(self.path) as reopened:
            with patch("knowledge_base.storage.create_embeddings", return_value=embeddings) as factory:
                results = retrieve("Tell me about cats", reopened, top_k=1)
                retrieve("Where do fish live?", reopened, top_k=1)
                factory.assert_called_once_with()
            self.assertEqual(results[0].metadata, {"source": "animals.pdf", "page": 1})
            self.assertEqual(embeddings.document_calls, [])
            self.assertEqual(len(embeddings.query_calls), 2)

    def test_txt_metadata_and_result_mutation_do_not_affect_store(self):
        document = Document(page_content="Cats are mammals", metadata={"source": "notes.txt"})
        with ChromaStore(self.path.parent / "txt", embeddings=self.embeddings) as store:
            store.index_chunks([document])
            result = retrieve("Tell me about cats", store)[0]
            self.assertEqual(result.metadata, {"source": "notes.txt"})
            result.metadata["source"] = "changed"
            self.assertEqual(retrieve("Tell me about cats", store)[0].metadata,
                             {"source": "notes.txt"})

    def test_optional_distances(self):
        with patch.object(self.store, "query_chunks", return_value={
            "ids": [["a"]], "documents": [["Some content"]],
            "metadatas": [[{"source": "notes.txt"}]], "distances": None,
        }):
            result = retrieve("cats", self.store)[0]
        self.assertIsNone(result.distance)
        self.assertIsNone(result.similarity_score)

    def test_closed_store(self):
        self.store.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            retrieve("cats", self.store)

    def test_embedding_failure_propagates_without_changing_collection(self):
        with patch.object(self.embeddings, "embed_query", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                retrieve("cats", self.store)
        self.assertEqual(self.store._collection.count(), 5)


if __name__ == "__main__":
    unittest.main()
