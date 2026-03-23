"""Gemini RAG contracts with mocked retrieval and official SDK responses."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from google.genai import errors, types

from knowledge_base.generation import (
    DEFAULT_MODEL, INSUFFICIENT_CONTEXT, GenerationError, MissingAPIKeyError,
    answer_question,
)
from knowledge_base.retrieval import RetrievedChunk


def sdk_response(payload):
    return types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(parts=[
            types.Part.from_text(text=json.dumps(payload)),
        ])),
    ])


class GenerationTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        loader = patch("knowledge_base.generation.load_dotenv")
        self.loader = loader.start()
        self.addCleanup(loader.stop)
        retrieval = patch("knowledge_base.generation.retrieve", return_value=[
            RetrievedChunk("chunk-1", "Paris is the capital of France.",
                           {"source": "cities.pdf", "page": 3}, 0.1, 0.9),
            RetrievedChunk("chunk-2", "Cats are mammals.", {"source": "animals.txt"}, 0.5, 0.6),
        ])
        self.retrieve = retrieval.start()
        self.addCleanup(retrieval.stop)
        factory = patch("knowledge_base.generation.genai.Client")
        self.factory = factory.start()
        self.addCleanup(factory.stop)
        self.client = self.factory.return_value.__enter__.return_value
        self.generate = self.client.models.generate_content
        self.generate.return_value = sdk_response({
            "answer": "Paris is the capital of France. [1]",
            "source_ids": [1], "has_answer": True,
        })
        self.store = object()

    def test_grounded_prompt_and_reference_mapping(self):
        result = answer_question(" What is the capital of France? ", self.store, top_k=2)
        self.retrieve.assert_called_once_with("What is the capital of France?", self.store, top_k=2)
        self.assertTrue(result.has_answer)
        self.assertEqual(result.answer, "Paris is the capital of France. [1]")
        self.assertEqual(result.sources[0].filename, "cities.pdf")
        self.assertEqual(result.sources[0].page, 3)
        self.assertEqual(result.sources[0].chunk_id, "chunk-1")
        request = self.generate.call_args.kwargs
        self.assertEqual(request["model"], DEFAULT_MODEL)
        prompt = json.loads(request["contents"])
        self.assertEqual(prompt["excerpts"][0]["text"], "Paris is the capital of France.")
        self.assertEqual(prompt["excerpts"][0]["page"], 3)
        self.assertNotIn("test-secret", request["contents"])
        self.assertIn("untrusted data", request["config"].system_instruction)
        self.assertEqual(request["config"].response_mime_type, "application/json")
        options = self.factory.call_args.kwargs["http_options"]
        self.assertEqual(options.timeout, 30000)
        self.assertEqual(options.retry_options.attempts, 1)
        self.assertFalse(self.factory.call_args.kwargs["vertexai"])
        self.factory.return_value.__exit__.assert_called_once()

    def test_empty_knowledge_base_never_contacts_gemini(self):
        self.retrieve.return_value = []
        os.environ.pop("GEMINI_API_KEY")
        result = answer_question("Question?", self.store)
        self.assertEqual(result.answer, INSUFFICIENT_CONTEXT)
        self.assertFalse(result.has_answer)
        self.assertEqual(result.sources, [])
        self.factory.assert_not_called()

    def test_missing_or_placeholder_api_key(self):
        for value in ["", "   ", "your_gemini_api_key_here"]:
            with self.subTest(value=value):
                os.environ["GEMINI_API_KEY"] = value
                with self.assertRaises(MissingAPIKeyError):
                    answer_question("Question?", self.store)
        self.factory.assert_not_called()

    def test_insufficient_context_normalizes_model_abstention(self):
        self.generate.return_value = sdk_response({
            "answer": "I don't know.", "source_ids": [], "has_answer": False,
        })
        result = answer_question("Unknown fact?", self.store)
        self.assertEqual(result.answer, INSUFFICIENT_CONTEXT)
        self.assertEqual(result.sources, [])

    def test_unknown_and_missing_source_ids_are_rejected(self):
        for source_ids in [[3], [0], [-1], [], [True], ["1"]]:
            with self.subTest(source_ids=source_ids):
                self.generate.return_value = sdk_response({
                    "answer": "A claim.", "source_ids": source_ids, "has_answer": True,
                })
                with self.assertRaises(GenerationError):
                    answer_question("Question?", self.store)

    def test_duplicate_references_and_txt_page(self):
        self.generate.return_value = sdk_response({
            "answer": "Cats are mammals. [2]", "source_ids": [2, 2], "has_answer": True,
        })
        result = answer_question("Question?", self.store)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].filename, "animals.txt")
        self.assertIsNone(result.sources[0].page)

    def test_json_text_response_fallback(self):
        self.generate.return_value = SimpleNamespace(parsed=None, text=json.dumps({
            "answer": "Paris. [1]", "source_ids": [1], "has_answer": True,
        }))
        self.assertEqual(answer_question("Question?", self.store).answer, "Paris. [1]")

    def test_sdk_parsed_schema_response(self):
        answer_question("Question?", self.store)
        schema = self.generate.call_args.kwargs["config"].response_schema
        self.generate.return_value = types.GenerateContentResponse(parsed=schema(
            answer="Paris. [1]", source_ids=[1], has_answer=True,
        ))
        self.assertEqual(answer_question("Question?", self.store).answer, "Paris. [1]")

    def test_real_sdk_with_in_memory_http_transport(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"candidates": [{
                "content": {"role": "model", "parts": [{"text": json.dumps({
                    "answer": "Paris. [1]", "source_ids": [1], "has_answer": True,
                })}]},
                "finishReason": "STOP",
            }]})

        # Retain the real SDK constructor while replacing only its HTTP transport.
        from google.genai.client import Client

        def create_client(**kwargs):
            kwargs["http_options"].client_args = {"transport": httpx.MockTransport(respond)}
            return Client(**kwargs)

        self.factory.side_effect = create_client
        result = answer_question("What is the capital?", self.store)
        self.assertEqual(result.answer, "Paris. [1]")
        self.assertEqual(result.sources[0].page, 3)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].headers["x-goog-api-key"], "test-secret")
        request_body = json.loads(requests[0].content)
        self.assertIn("responseSchema", request_body["generationConfig"])

    def test_unknown_inline_reference_is_rejected(self):
        self.generate.return_value = sdk_response({
            "answer": "Paris. [99]", "source_ids": [1], "has_answer": True,
        })
        with self.assertRaisesRegex(GenerationError, "valid document references"):
            answer_question("Question?", self.store)

    def test_empty_blocked_or_malformed_response(self):
        for text in [None, "", "not JSON", "{}"]:
            with self.subTest(text=text):
                self.generate.return_value = SimpleNamespace(parsed=None, text=text)
                with self.assertRaisesRegex(GenerationError, "invalid answer"):
                    answer_question("Question?", self.store)

    def test_timeout_errors_are_sanitized(self):
        for error in [httpx.ReadTimeout("test-secret"), TimeoutError("test-secret")]:
            with self.subTest(error=type(error)):
                self.generate.side_effect = error
                with self.assertRaisesRegex(GenerationError, "timed out") as raised:
                    answer_question("Question?", self.store)
                self.assertNotIn("test-secret", str(raised.exception))

    def test_api_errors_are_sanitized(self):
        for code, expected in [(403, "credentials"), (429, "quota"), (500, "request"), (504, "timed out")]:
            with self.subTest(code=code):
                self.generate.side_effect = errors.APIError(code, {"message": "test-secret"})
                with self.assertRaisesRegex(GenerationError, expected) as raised:
                    answer_question("Question?", self.store)
                self.assertNotIn("test-secret", str(raised.exception))

    def test_network_errors_are_sanitized(self):
        self.generate.side_effect = httpx.ConnectError("test-secret")
        with self.assertRaisesRegex(GenerationError, "connection") as raised:
            answer_question("Question?", self.store)
        self.assertNotIn("test-secret", str(raised.exception))

    def test_retrieval_failure_does_not_contact_api(self):
        self.retrieve.side_effect = RuntimeError("Local store unavailable")
        with self.assertRaisesRegex(GenerationError, "retrieve"):
            answer_question("Question?", self.store)
        self.factory.assert_not_called()

    def test_invalid_parameters(self):
        for question in ["", " \n", None, 123]:
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    answer_question(question, self.store)
        for arguments in [{"top_k": 0}, {"top_k": True}, {"top_k": 1.5},
                          {"timeout_seconds": 0}, {"timeout_seconds": float("nan")},
                          {"timeout_seconds": True}, {"model": ""}]:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    answer_question("Question?", self.store, **arguments)
        self.retrieve.assert_not_called()
        self.factory.assert_not_called()

    def test_dotenv_loads_key_without_overriding_environment(self):
        from dotenv import load_dotenv

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("GEMINI_API_KEY=dotenv-secret\nGEMINI_MODEL=configured-model\n", encoding="utf-8")
            with patch("knowledge_base.generation.load_dotenv", wraps=load_dotenv):
                answer_question("Question?", self.store, dotenv_path=path)
                self.assertEqual(self.factory.call_args.kwargs["api_key"], "test-secret")
                os.environ.pop("GEMINI_API_KEY")
                answer_question("Question?", self.store, dotenv_path=path)
                self.assertEqual(self.factory.call_args.kwargs["api_key"], "dotenv-secret")
                self.assertEqual(self.generate.call_args.kwargs["model"], "configured-model")

    def test_model_and_timeout_overrides(self):
        os.environ["GEMINI_MODEL"] = "environment-model"
        answer_question("Question?", self.store, timeout_seconds=5, model="explicit-model")
        self.assertEqual(self.generate.call_args.kwargs["model"], "explicit-model")
        self.assertEqual(self.factory.call_args.kwargs["http_options"].timeout, 5000)


if __name__ == "__main__":
    unittest.main()
