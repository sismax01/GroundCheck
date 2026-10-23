"""
SQLite persistence layer for GroundCheck.
Stores ingested documents, segmented chunks, query audit logs, and evaluation benchmark runs.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from groundcheck.models import (
    model_to_json,
    ChunkModel,
    DocumentDetail,
    DocumentSummary,
    EvaluationRunSummary,
    QueryResponse,
)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Database:
    """Thread-safe SQLite storage manager for GroundCheck."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    title TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    total_tokens_est INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    is_sample INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    doc_title TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    section_title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    token_estimate INTEGER NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

                CREATE TABLE IF NOT EXISTS query_history (
                    query_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    retrieval_mode TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    groundedness_score REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    benchmark_id TEXT NOT NULL,
                    retrieval_mode TEXT NOT NULL,
                    generation_mode TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    alpha REAL NOT NULL,
                    mean_precision_at_k REAL NOT NULL,
                    mean_recall_at_k REAL NOT NULL,
                    mean_mrr REAL NOT NULL,
                    mean_ndcg_at_k REAL NOT NULL,
                    mean_groundedness REAL NOT NULL,
                    abstention_accuracy REAL NOT NULL,
                    overall_pass_rate REAL NOT NULL,
                    latency_p50_ms REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    # -------------------------------------------------------------------------
    # Document & Chunk Persistence
    # -------------------------------------------------------------------------

    def save_document(self, doc: DocumentSummary, chunks: List[ChunkModel]) -> DocumentSummary:
        """Insert or replace a document and its segmented chunks."""
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc.doc_id,))
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, filename, title, file_type, file_size_bytes,
                    page_count, chunk_count, total_tokens_est, content_hash,
                    is_sample, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    doc.filename,
                    doc.title,
                    doc.file_type,
                    doc.file_size_bytes,
                    doc.page_count,
                    doc.chunk_count,
                    doc.total_tokens_est,
                    doc.content_hash,
                    1 if doc.is_sample else 0,
                    doc.created_at,
                ),
            )
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, filename, doc_title, chunk_index,
                    page_number, section_title, text, char_count, token_estimate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.filename,
                        c.doc_title,
                        c.chunk_index,
                        c.page_number,
                        c.section_title,
                        c.text,
                        c.char_count,
                        c.token_estimate,
                    )
                    for c in chunks
                ],
            )
        return doc

    def list_documents(self) -> List[DocumentSummary]:
        """Return all stored document summaries ordered by creation timestamp."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY is_sample DESC, created_at DESC, filename ASC"
            ).fetchall()
        return [
            DocumentSummary(
                doc_id=r["doc_id"],
                filename=r["filename"],
                title=r["title"],
                file_type=r["file_type"],
                file_size_bytes=r["file_size_bytes"],
                page_count=r["page_count"],
                chunk_count=r["chunk_count"],
                total_tokens_est=r["total_tokens_est"],
                content_hash=r["content_hash"],
                is_sample=bool(r["is_sample"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_document(self, doc_id: str) -> Optional[DocumentDetail]:
        """Fetch a single document and all its ordered chunks."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if not row:
                return None
            chunk_rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC", (doc_id,)
            ).fetchall()

        chunks = [
            ChunkModel(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                filename=c["filename"],
                doc_title=c["doc_title"],
                chunk_index=c["chunk_index"],
                page_number=c["page_number"],
                section_title=c["section_title"],
                text=c["text"],
                char_count=c["char_count"],
                token_estimate=c["token_estimate"],
            )
            for c in chunk_rows
        ]
        return DocumentDetail(
            doc_id=row["doc_id"],
            filename=row["filename"],
            title=row["title"],
            file_type=row["file_type"],
            file_size_bytes=row["file_size_bytes"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            total_tokens_est=row["total_tokens_est"],
            content_hash=row["content_hash"],
            is_sample=bool(row["is_sample"]),
            created_at=row["created_at"],
            chunks=chunks,
        )

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks by ID."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cur.rowcount > 0

    def get_all_chunks(self, doc_ids: Optional[List[str]] = None) -> List[ChunkModel]:
        """Return all chunks in the corpus, optionally filtered by document IDs."""
        with self._connect() as conn:
            if doc_ids:
                placeholders = ",".join("?" for _ in doc_ids)
                query = f"SELECT * FROM chunks WHERE doc_id IN ({placeholders}) ORDER BY doc_id, chunk_index ASC"
                rows = conn.execute(query, tuple(doc_ids)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM chunks ORDER BY doc_id, chunk_index ASC").fetchall()

        return [
            ChunkModel(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                filename=r["filename"],
                doc_title=r["doc_title"],
                chunk_index=r["chunk_index"],
                page_number=r["page_number"],
                section_title=r["section_title"],
                text=r["text"],
                char_count=r["char_count"],
                token_estimate=r["token_estimate"],
            )
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # Query History Persistence
    # -------------------------------------------------------------------------

    def save_query_response(self, response: QueryResponse) -> None:
        """Store a completed query and its full verification audit payload."""
        payload_str = model_to_json(response)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO query_history (
                    query_id, question, retrieval_mode, verdict,
                    groundedness_score, latency_ms, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.query_id,
                    response.question,
                    response.retrieval_mode,
                    response.verification.verdict,
                    response.verification.overall_groundedness_score,
                    response.telemetry.total_latency_ms,
                    payload_str,
                    response.created_at,
                ),
            )

    def list_queries(self, limit: int = 30) -> List[Dict[str, Any]]:
        """List recent queries with summary metadata."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT query_id, question, retrieval_mode, verdict,
                       groundedness_score, latency_ms, created_at
                FROM query_history
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_query_response(self, query_id: str) -> Optional[QueryResponse]:
        """Reconstruct a full QueryResponse from the history table."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM query_history WHERE query_id = ?", (query_id,)
            ).fetchone()
        if not row:
            return None
        data = json.loads(row["payload_json"])
        return QueryResponse(**data)

    # -------------------------------------------------------------------------
    # Evaluation Benchmark Persistence
    # -------------------------------------------------------------------------

    def save_evaluation_run(self, summary: EvaluationRunSummary) -> None:
        """Persist a benchmark evaluation run and its per-question breakdown."""
        payload_str = model_to_json(summary)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs (
                    run_id, experiment_name, benchmark_id, retrieval_mode,
                    generation_mode, top_k, alpha, mean_precision_at_k,
                    mean_recall_at_k, mean_mrr, mean_ndcg_at_k, mean_groundedness,
                    abstention_accuracy, overall_pass_rate, latency_p50_ms,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.run_id,
                    summary.experiment_name,
                    summary.benchmark_id,
                    summary.retrieval_mode,
                    summary.generation_mode,
                    summary.top_k,
                    summary.alpha,
                    summary.mean_precision_at_k,
                    summary.mean_recall_at_k,
                    summary.mean_mrr,
                    summary.mean_ndcg_at_k,
                    summary.mean_groundedness,
                    summary.abstention_accuracy,
                    summary.overall_pass_rate,
                    summary.latency_p50_ms,
                    payload_str,
                    summary.created_at,
                ),
            )

    def list_evaluation_runs(self, limit: int = 25) -> List[EvaluationRunSummary]:
        """Return stored evaluation runs ordered newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM evaluation_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results: List[EvaluationRunSummary] = []
        for r in rows:
            data = json.loads(r["payload_json"])
            results.append(EvaluationRunSummary(**data))
        return results

    def get_evaluation_run(self, run_id: str) -> Optional[EvaluationRunSummary]:
        """Fetch a specific evaluation run by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM evaluation_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if not row:
            return None
        return EvaluationRunSummary(**json.loads(row["payload_json"]))

    def delete_evaluation_run(self, run_id: str) -> bool:
        """Delete an evaluation run by ID."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM evaluation_runs WHERE run_id = ?", (run_id,))
            return cur.rowcount > 0
