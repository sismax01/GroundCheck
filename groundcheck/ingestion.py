"""
Document ingestion, validation, metadata extraction, and section-aware chunking.
Supports PDF (.pdf via pypdf), Markdown (.md), Plain Text (.txt), and HTML (.html/.htm).
"""

from __future__ import annotations

import hashlib
import io
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Tuple

from pypdf import PdfReader

from groundcheck.database import utc_now_iso
from groundcheck.models import ChunkModel, DocumentSummary


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}


class IngestionError(ValueError):
    """Raised when a document is malformed, empty, unsupported, or exceeds limits."""


class _SimpleHTMLTextExtractor(HTMLParser):
    """Lightweight HTML-to-text parser that preserves heading boundaries."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: List[str] = []
        self._in_ignored_tag = False
        self._heading_level: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript"):
            self._in_ignored_tag = True
        elif t in ("h1", "h2", "h3", "h4"):
            self._heading_level = int(t[1])
            self._pieces.append("\n\n" + ("#" * self._heading_level) + " ")
        elif t in ("p", "div", "section", "article", "li", "tr", "br"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript"):
            self._in_ignored_tag = False
        elif t in ("h1", "h2", "h3", "h4"):
            self._heading_level = 0
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._in_ignored_tag:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw = "".join(self._pieces)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def estimate_tokens(text: str) -> int:
    """Estimate BPE token count (~1.33 tokens per whitespace word or char/4)."""
    words = len(text.split())
    return max(1, int(max(words * 1.3, len(text) / 4.2)))


def slugify_doc_id(filename: str, content_hash: str) -> str:
    """Create a clean, deterministic document identifier."""
    stem = Path(filename).stem.lower()
    clean_stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")[:36]
    if not clean_stem:
        clean_stem = "doc"
    return f"doc-{clean_stem}-{content_hash[:8]}"


def _extract_title_from_text(filename: str, pages: List[Tuple[Optional[int], str]]) -> str:
    """Infer a human-readable document title from headings or filename."""
    for _, page_text in pages:
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
                if len(heading) >= 4:
                    return heading[:120]
            if len(stripped) >= 6 and len(stripped) <= 110:
                return stripped
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def parse_document_bytes(
    filename: str,
    raw_bytes: bytes,
    max_upload_mb: int = 25,
) -> Tuple[str, List[Tuple[Optional[int], str]]]:
    """
    Validate and extract (page_number, text) pairs from raw file bytes.
    Raises IngestionError on invalid, empty, or corrupted inputs.
    """
    if not filename or not filename.strip():
        raise IngestionError("Filename must not be empty.")

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestionError(f"Unsupported file extension '{ext}'. Supported types: {allowed}")

    if not raw_bytes or len(raw_bytes) == 0:
        raise IngestionError(f"Uploaded file '{filename}' is empty (0 bytes).")

    max_bytes = max_upload_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise IngestionError(
            f"File '{filename}' ({len(raw_bytes) / (1024*1024):.1f} MB) exceeds the {max_upload_mb} MB limit."
        )

    pages: List[Tuple[Optional[int], str]] = []

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            if reader.is_encrypted:
                raise IngestionError(f"PDF '{filename}' is password-protected and cannot be indexed.")
            for idx, page in enumerate(reader.pages, start=1):
                extracted = (page.extract_text() or "").strip()
                if extracted:
                    pages.append((idx, extracted))
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Failed to parse PDF file '{filename}': {exc}") from exc

        if not pages:
            raise IngestionError(
                f"PDF '{filename}' contains no extractable text (it may be an image-only scan or empty)."
            )
        return "pdf", pages

    if b"\x00" in raw_bytes[:2048]:
        raise IngestionError(f"File '{filename}' appears to contain binary null bytes rather than valid text.")

    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded = raw_bytes.decode("latin-1")
        except Exception as exc:
            raise IngestionError(f"Could not decode text encoding for '{filename}': {exc}") from exc

    if ext in (".html", ".htm"):
        extractor = _SimpleHTMLTextExtractor()
        try:
            extractor.feed(decoded)
            text_content = extractor.get_text()
        except Exception as exc:
            raise IngestionError(f"Malformed HTML in '{filename}': {exc}") from exc
        file_type = "html"
    elif ext in (".md", ".markdown"):
        text_content = decoded.strip()
        file_type = "markdown"
    else:
        text_content = decoded.strip()
        file_type = "text"

    if not text_content or len(re.sub(r"\s+", "", text_content)) < 10:
        raise IngestionError(f"File '{filename}' contains insufficient readable text content.")

    raw_pages = [p.strip() for p in text_content.split("\f") if p.strip()]
    for idx, p_text in enumerate(raw_pages, start=1):
        pages.append((idx, p_text))

    return file_type, pages


_HEADING_PATTERN = re.compile(
    r"^(?:#{1,4}\s+(.+)|(?:SECTION|CHAPTER|PART)\s+\d+[:.\-\s]+(.+)|(\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9\s\-:,/()]{3,75}))$"
)


def segment_pages_into_chunks(
    doc_id: str,
    filename: str,
    doc_title: str,
    pages: List[Tuple[Optional[int], str]],
    chunk_size: int = 850,
    chunk_overlap: int = 120,
) -> List[ChunkModel]:
    """
    Split document pages into section-aware, overlapping passages.
    Strips raw Markdown `#` heading lines into `section_title` metadata so chunks contain
    clean prose and bullet points without orphaned heading fragments.
    """
    chunks: List[ChunkModel] = []
    current_section = "Overview & Introduction"
    chunk_idx = 0

    for page_num, page_text in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        buffer_lines: List[str] = []
        buffer_len = 0
        buffer_section = current_section

        def flush_buffer(buf: List[str], sec: str) -> None:
            nonlocal chunk_idx
            combined = "\n\n".join(buf).strip()
            if not combined or len(combined) < 15:
                return
            cid = f"{doc_id}::ch-{chunk_idx:03d}"
            chunks.append(
                ChunkModel(
                    chunk_id=cid,
                    doc_id=doc_id,
                    filename=filename,
                    doc_title=doc_title,
                    chunk_index=chunk_idx,
                    page_number=page_num,
                    section_title=sec,
                    text=combined,
                    char_count=len(combined),
                    token_estimate=estimate_tokens(combined),
                )
            )
            chunk_idx += 1

        for para in paragraphs:
            raw_lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
            if not raw_lines:
                continue

            body_lines: List[str] = []
            for ln in raw_lines:
                h_match = _HEADING_PATTERN.match(ln)
                if h_match:
                    detected_heading = (
                        h_match.group(1) or h_match.group(2) or h_match.group(3) or ""
                    ).strip()
                    if detected_heading:
                        # Skip top-level document H1 title if it matches doc_title
                        if ln.startswith("# ") and detected_heading.lower() == doc_title.lower():
                            continue
                        if buffer_lines:
                            flush_buffer(buffer_lines, buffer_section)
                            buffer_lines = []
                            buffer_len = 0
                        current_section = detected_heading[:100]
                        buffer_section = current_section
                        continue
                body_lines.append(ln)

            if not body_lines:
                continue

            para_clean = "\n".join(body_lines)

            if len(para_clean) > chunk_size:
                # Split on sentence punctuation only when not preceded by a standalone section digit
                sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\-])", para_clean)
                for sent in sentences:
                    if not sent.strip():
                        continue
                    if buffer_len + len(sent) + 2 > chunk_size and buffer_lines:
                        flush_buffer(buffer_lines, buffer_section)
                        overlap_buf: List[str] = []
                        overlap_chars = 0
                        for prev in reversed(buffer_lines):
                            if overlap_chars + len(prev) <= chunk_overlap:
                                overlap_buf.insert(0, prev)
                                overlap_chars += len(prev)
                            else:
                                break
                        buffer_lines = overlap_buf
                        buffer_len = overlap_chars
                        buffer_section = current_section
                    buffer_lines.append(sent.strip())
                    buffer_len += len(sent) + 1
            else:
                if buffer_len + len(para_clean) + 2 > chunk_size and buffer_lines:
                    flush_buffer(buffer_lines, buffer_section)
                    overlap_buf = []
                    overlap_chars = 0
                    for prev in reversed(buffer_lines):
                        if overlap_chars + len(prev) <= chunk_overlap:
                            overlap_buf.insert(0, prev)
                            overlap_chars += len(prev)
                        else:
                            break
                    buffer_lines = overlap_buf
                    buffer_len = overlap_chars
                    buffer_section = current_section

                buffer_lines.append(para_clean)
                buffer_len += len(para_clean) + 2

        if buffer_lines:
            flush_buffer(buffer_lines, buffer_section)

    return chunks


def ingest_document(
    filename: str,
    raw_bytes: bytes,
    chunk_size: int = 850,
    chunk_overlap: int = 120,
    max_upload_mb: int = 25,
    is_sample: bool = False,
    custom_doc_id: Optional[str] = None,
) -> Tuple[DocumentSummary, List[ChunkModel]]:
    """
    End-to-end document ingestion pipeline.
    """
    file_type, pages = parse_document_bytes(
        filename=filename,
        raw_bytes=raw_bytes,
        max_upload_mb=max_upload_mb,
    )
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    doc_id = custom_doc_id or slugify_doc_id(filename, content_hash)
    title = _extract_title_from_text(filename, pages)

    chunks = segment_pages_into_chunks(
        doc_id=doc_id,
        filename=filename,
        doc_title=title,
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise IngestionError(f"No valid passages could be extracted from '{filename}'.")

    total_tokens = sum(c.token_estimate for c in chunks)
    summary = DocumentSummary(
        doc_id=doc_id,
        filename=filename,
        title=title,
        file_type=file_type,
        file_size_bytes=len(raw_bytes),
        page_count=len(pages),
        chunk_count=len(chunks),
        total_tokens_est=total_tokens,
        content_hash=content_hash,
        is_sample=is_sample,
        created_at=utc_now_iso(),
    )
    return summary, chunks
