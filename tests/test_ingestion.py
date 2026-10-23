"""
Unit tests for document ingestion, PDF/Markdown/HTML parsing, section-aware chunking,
and validation of malformed, empty, oversized, or unsupported files.
"""

from __future__ import annotations

import io
import unittest

from pypdf import PdfWriter

from groundcheck.ingestion import IngestionError, ingest_document


class TestDocumentIngestion(unittest.TestCase):
    def test_ingest_markdown_with_sections(self) -> None:
        md_content = (
            "# Distributed Storage Engine\n\n"
            "## 1. Write-Ahead Log Layout\n"
            "Every mutation is appended to a 64 MB segment file protected by CRC32C checksums "
            "before returning acknowledgment to the client.\n\n"
            "## 2. Snapshot Compaction\n"
            "Background copy-on-write compaction runs whenever the active log exceeds 50000 entries "
            "or 128 MB of uncompacted data.\n"
        ).encode("utf-8")

        summary, chunks = ingest_document(
            filename="storage_spec.md",
            raw_bytes=md_content,
            chunk_size=300,
            chunk_overlap=50,
        )
        self.assertEqual(summary.file_type, "markdown")
        self.assertEqual(summary.title, "Distributed Storage Engine")
        self.assertGreaterEqual(len(chunks), 2)
        sections = [c.section_title for c in chunks]
        self.assertTrue(any("Write-Ahead Log" in s for s in sections))
        self.assertTrue(any("Snapshot Compaction" in s for s in sections))

    def test_ingest_html_document(self) -> None:
        html_bytes = (
            "<html><head><title>Ignored</title><script>var secret=1;</script></head>"
            "<body><h1>Telemetry Gateway Guide</h1>"
            "<h2>1. Frame Validation</h2>"
            "<p>All incoming Modbus frames are validated against a 16-bit cyclic redundancy check.</p>"
            "</body></html>"
        ).encode("utf-8")

        summary, chunks = ingest_document("gateway.html", html_bytes)
        self.assertEqual(summary.file_type, "html")
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("Modbus", chunks[0].text)
        self.assertNotIn("secret", chunks[0].text)

    def test_reject_unsupported_extension(self) -> None:
        with self.assertRaises(IngestionError) as ctx:
            ingest_document("archive.exe", b"MZ\x90\x00binary")
        self.assertIn("Unsupported file extension", str(ctx.exception))

    def test_reject_empty_file(self) -> None:
        with self.assertRaises(IngestionError) as ctx:
            ingest_document("empty.md", b"")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_reject_binary_null_bytes_in_text_file(self) -> None:
        with self.assertRaises(IngestionError) as ctx:
            ingest_document("corrupt.txt", b"Valid header\x00\x00\x00binary payload")
        self.assertIn("binary null bytes", str(ctx.exception).lower())

    def test_reject_oversized_file(self) -> None:
        two_mb = b"A" * (2 * 1024 * 1024)
        with self.assertRaises(IngestionError) as ctx:
            ingest_document("huge.txt", two_mb, max_upload_mb=1)
        self.assertIn("exceeds the 1 MB limit", str(ctx.exception))

    def test_reject_malformed_pdf(self) -> None:
        with self.assertRaises(IngestionError) as ctx:
            ingest_document("broken.pdf", b"%PDF-1.4\nnot a valid pdf trailer")
        self.assertIn("Failed to parse PDF", str(ctx.exception))

    def test_reject_blank_page_pdf(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        with self.assertRaises(IngestionError) as ctx:
            ingest_document("blank_scan.pdf", buf.getvalue())
        self.assertIn("no extractable text", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
