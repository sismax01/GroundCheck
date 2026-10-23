"""
High-level orchestration service for GroundCheck.
Connects document ingestion, SQLite persistence, hybrid retrieval, answer synthesis,
claim-level groundedness verification, and reproducible benchmarking.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from groundcheck.config import Settings
from groundcheck.database import Database, utc_now_iso
from groundcheck.evaluator import (
    METRIC_DOCUMENTATION,
    execute_evaluation_run,
    load_benchmark_dataset,
)
from groundcheck.generator import generate_llm_answer, synthesize_extractive_answer
from groundcheck.ingestion import IngestionError, estimate_tokens, ingest_document
from groundcheck.models import (
    model_to_dict,
    CostAndLatencyMetrics,
    DocumentDetail,
    DocumentSummary,
    EvaluationRunRequest,
    EvaluationRunSummary,
    MetricExplanation,
    QueryRequest,
    QueryResponse,
)
from groundcheck.retrieval import HybridRetriever
from groundcheck.verifier import verify_answer


class GroundCheckService:
    """Core application service managing state, indexing, Q&A, and evaluation."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings.from_env()
        self.db = Database(self.settings.db_path)
        if self.settings.auto_seed_sample_docs:
            self.ensure_sample_corpus_seeded()

    def ensure_sample_corpus_seeded(self) -> List[DocumentSummary]:
        """Seed the reference technical documents if the database has no documents yet."""
        existing = self.db.list_documents()
        if existing:
            return existing
        return self.seed_sample_corpus()

    def seed_sample_corpus(self) -> List[DocumentSummary]:
        """Ingest all markdown/PDF/text files in `sample_docs/` with stable IDs."""
        seeded: List[DocumentSummary] = []
        sample_dir = self.settings.sample_docs_dir
        if not sample_dir.exists():
            return seeded

        for file_path in sorted(sample_dir.iterdir()):
            if file_path.is_file() and not file_path.name.startswith("."):
                raw_bytes = file_path.read_bytes()
                stem_slug = file_path.stem.lower().replace("_", "-")
                stable_id = f"doc-{stem_slug}"
                doc, chunks = ingest_document(
                    filename=file_path.name,
                    raw_bytes=raw_bytes,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                    max_upload_mb=self.settings.max_upload_mb,
                    is_sample=True,
                    custom_doc_id=stable_id,
                )
                self.db.save_document(doc, chunks)
                seeded.append(doc)
        return seeded

    def upload_document(self, filename: str, raw_bytes: bytes) -> DocumentSummary:
        """Validate, chunk, and persist a user-uploaded document."""
        doc, chunks = ingest_document(
            filename=filename,
            raw_bytes=raw_bytes,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            max_upload_mb=self.settings.max_upload_mb,
            is_sample=False,
        )
        return self.db.save_document(doc, chunks)

    def list_documents(self) -> List[DocumentSummary]:
        return self.db.list_documents()

    def get_document(self, doc_id: str) -> Optional[DocumentDetail]:
        return self.db.get_document(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        return self.db.delete_document(doc_id)

    def answer_question(self, request: QueryRequest, save_history: bool = True) -> QueryResponse:
        """
        Execute the full GroundCheck RAG & Verification pipeline:
        1. Load active chunks and build/run HybridRetriever.
        2. Synthesize grounded answer (local extractive or external LLM).
        3. Audit every claim, citation, and cross-passage contradiction.
        4. Record latency and token/cost telemetry.
        """
        question_clean = request.question.strip()
        if len(question_clean) < 3:
            raise ValueError("Question must be at least 3 characters long.")

        mode = request.retrieval_mode.lower().strip()
        if mode not in ("hybrid", "bm25", "semantic"):
            raise ValueError(f"Invalid retrieval_mode '{mode}'. Choose from: hybrid, bm25, semantic.")

        gen_mode = (request.generation_mode or self.settings.generation_mode).lower().strip()
        if gen_mode not in ("extractive", "llm"):
            raise ValueError(f"Invalid generation_mode '{gen_mode}'. Choose from: extractive, llm.")

        t0 = time.perf_counter()
        chunks = self.db.get_all_chunks(doc_ids=request.doc_ids)
        retriever = HybridRetriever(chunks)
        passages = retriever.search(
            query=question_clean,
            mode=mode,
            top_k=request.top_k,
            alpha=request.alpha,
        )
        t_retrieval = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        if gen_mode == "llm":
            answer_md, is_abstention, exec_label = generate_llm_answer(
                question=question_clean,
                passages=passages,
                settings=self.settings,
                min_evidence_score=request.min_evidence_score,
            )
            is_live_llm = True
        else:
            answer_md, is_abstention = synthesize_extractive_answer(
                question=question_clean,
                passages=passages,
                min_evidence_score=request.min_evidence_score,
            )
            exec_label = "local_extractive (Zero-Cost Deterministic Synthesis)"
            is_live_llm = False
        t_gen = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        verification = verify_answer(
            question=question_clean,
            answer_markdown=answer_md,
            passages=passages,
            min_evidence_score=request.min_evidence_score,
            explicit_abstention=is_abstention,
        )
        t_ver = (time.perf_counter() - t2) * 1000.0
        total_lat = t_retrieval + t_gen + t_ver

        # Estimate token usage and reference commercial API cost
        context_text = "\n".join(p.text for p in passages)
        prompt_tokens = estimate_tokens(question_clean) + estimate_tokens(context_text) + 120
        completion_tokens = estimate_tokens(answer_md)
        total_tokens = prompt_tokens + completion_tokens

        ref_cost = (
            (prompt_tokens * self.settings.cost_input_per_1m)
            + (completion_tokens * self.settings.cost_output_per_1m)
        ) / 1_000_000.0
        actual_cost = ref_cost if is_live_llm else 0.0

        telemetry = CostAndLatencyMetrics(
            retrieval_latency_ms=round(t_retrieval, 2),
            generation_latency_ms=round(t_gen, 2),
            verification_latency_ms=round(t_ver, 2),
            total_latency_ms=round(total_lat, 2),
            prompt_tokens_est=prompt_tokens,
            completion_tokens_est=completion_tokens,
            total_tokens_est=total_tokens,
            actual_api_cost_usd=round(actual_cost, 6),
            reference_llm_cost_usd=round(ref_cost, 6),
            execution_mode=exec_label,
        )

        resp = QueryResponse(
            query_id=f"qry-{uuid.uuid4().hex[:8]}",
            question=question_clean,
            answer_markdown=answer_md,
            retrieval_mode=mode,
            top_k=request.top_k,
            alpha=request.alpha,
            retrieved_passages=passages,
            verification=verification,
            telemetry=telemetry,
            created_at=utc_now_iso(),
        )

        if save_history:
            self.db.save_query_response(resp)

        return resp

    def run_benchmark(self, request: EvaluationRunRequest) -> EvaluationRunSummary:
        """Run the reproducible benchmark suite and store the results."""
        # Ensure reference sample documents are present before running the benchmark
        self.ensure_sample_corpus_seeded()
        return execute_evaluation_run(
            request=request,
            query_executor=self.answer_question,
            db=self.db,
            settings=self.settings,
        )

    def get_benchmark_metadata(self) -> Dict[str, Any]:
        """Return the benchmark dataset definition and metric mathematical explanations."""
        dataset = load_benchmark_dataset(self.settings.benchmarks_dir, "groundcheck_eval_v1")
        return {
            "benchmark": dataset,
            "metrics_guide": [model_to_dict(m) for m in METRIC_DOCUMENTATION],
        }
