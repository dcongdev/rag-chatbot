"""Small real documents exercise ingestion without network or external services."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from knowledge_base.ingestion import (
    IngestionError,
    UnsupportedFormatError,
    ingest_document,
    load_document,
    split_documents,
)


def write_pdf(path: Path, pages: list[str], *, password: str | None = None) -> None:
    """Generate tiny PDFs with actual text streams, including optional blank pages."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if password is not None:
        writer.encrypt(password)
    with path.open("wb") as handle:
        writer.write(handle)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_utf8_txt_and_bom(self):
        path = self.root / "notes.TXT"
        path.write_text("Xin chào\nKnowledge base", encoding="utf-8-sig")
        documents = load_document(str(path))
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "Xin chào\nKnowledge base")
        self.assertEqual(documents[0].metadata, {"source": "notes.TXT"})

    def test_pdf_page_numbers_survive_skipped_blank_pages(self):
        path = self.root / "sample.PDF"
        write_pdf(path, ["First page content", "", "Third page content"])
        documents = load_document(path)
        self.assertEqual([d.page_content.strip() for d in documents],
                         ["First page content", "Third page content"])
        self.assertEqual([d.metadata for d in documents],
                         [{"source": "sample.PDF", "page": 1},
                          {"source": "sample.PDF", "page": 3}])
        chunks = ingest_document(path, chunk_size=12, chunk_overlap=3)
        self.assertGreater(len(chunks), 2)
        self.assertEqual({c.metadata["page"] for c in chunks}, {1, 3})
        self.assertTrue(all(c.metadata["source"] == path.name for c in chunks))
        self.assertTrue(all(len(c.page_content) <= 12 for c in chunks))

    def test_empty_and_whitespace_files(self):
        for name, content in [("empty.txt", b""), ("empty.pdf", b""),
                              ("spaces.txt", b" \n\t")]:
            with self.subTest(name=name):
                path = self.root / name
                path.write_bytes(content)
                self.assertEqual(ingest_document(path), [])

    def test_blank_pdf(self):
        path = self.root / "blank.pdf"
        write_pdf(path, [""])
        self.assertEqual(ingest_document(path), [])

    def test_unsupported_format(self):
        path = self.root / "notes.csv"
        path.write_text("hello", encoding="utf-8")
        with self.assertRaisesRegex(UnsupportedFormatError, "PDF or TXT"):
            load_document(path)

    def test_missing_file(self):
        with self.assertRaises(IngestionError) as raised:
            load_document(self.root / "missing.txt")
        self.assertIsInstance(raised.exception.__cause__, FileNotFoundError)

    def test_invalid_utf8(self):
        path = self.root / "invalid.txt"
        path.write_bytes(b"\xff\xfe\x80")
        with self.assertRaises(IngestionError) as raised:
            load_document(path)
        self.assertIsInstance(raised.exception.__cause__, UnicodeDecodeError)

    def test_corrupt_pdf(self):
        path = self.root / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\nnot a valid PDF")
        with self.assertRaisesRegex(IngestionError, "broken.pdf") as raised:
            load_document(path)
        self.assertIsNotNone(raised.exception.__cause__)

    def test_encrypted_pdf(self):
        path = self.root / "secret.pdf"
        write_pdf(path, ["Secret"], password="password")
        with self.assertRaisesRegex(IngestionError, "password"):
            load_document(path)

    def test_read_error(self):
        path = self.root / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            with self.assertRaises(IngestionError) as raised:
                load_document(path)
        self.assertIsInstance(raised.exception.__cause__, PermissionError)

    def test_chunks_have_overlap_and_preserve_independent_metadata(self):
        document = Document(page_content="abcdefghijklmnopqrstuvwxyz",
                            metadata={"source": "letters.txt", "custom": {"id": 1}})
        chunks = split_documents(iter([document]), chunk_size=10, chunk_overlap=3)
        self.assertEqual([c.page_content for c in chunks],
                         ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"])
        self.assertTrue(all(c.metadata == document.metadata for c in chunks))
        chunks[0].metadata["custom"]["id"] = 2
        self.assertEqual(document.metadata["custom"]["id"], 1)
        self.assertEqual(chunks[1].metadata["custom"]["id"], 1)

    def test_empty_documents(self):
        self.assertEqual(split_documents([]), [])
        self.assertEqual(split_documents([Document(page_content=" \n\t")]), [])

    def test_invalid_chunk_parameters(self):
        for size, overlap in [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11),
                              (1.5, 0), (10, 1.5), (True, 0), (10, False)]:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    split_documents([], chunk_size=size, chunk_overlap=overlap)

    def test_txt_end_to_end(self):
        path = self.root / "letters.txt"
        path.write_text("abcdefghijklmnop", encoding="utf-8")
        chunks = ingest_document(path, chunk_size=8, chunk_overlap=2)
        self.assertEqual([c.page_content for c in chunks],
                         ["abcdefgh", "ghijklmn", "mnop"])
        self.assertTrue(all(c.metadata == {"source": "letters.txt"} for c in chunks))


if __name__ == "__main__":
    unittest.main()
