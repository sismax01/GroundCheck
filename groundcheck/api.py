"""
FastAPI REST API routes and static frontend serving for GroundCheck.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from groundcheck import __version__
from groundcheck.config import Settings
from groundcheck.ingestion import IngestionError
from groundcheck.models import (
    DocumentDetail,
    DocumentSummary,
    EvaluationRunRequest,
    EvaluationRunSummary,
    QueryRequest,
    QueryResponse,
)
from groundcheck.service import GroundCheckService


STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the GroundCheck FastAPI application."""
    cfg = settings or Settings.from_env()
    service = GroundCheckService(settings=cfg)

    app = FastAPI(
        title="GroundCheck API",
        description=(
            "AI-Powered Document Research, Claim-Level Groundedness Verification, "
            "and Reproducible RAG Evaluation Platform."
        ),
        version=__version__,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.service = service
    app.state.settings = cfg

    def get_service() -> GroundCheckService:
        return app.state.service

    # -------------------------------------------------------------------------
    # System & Configuration Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/health")
    def health_check(svc: GroundCheckService = Depends(get_service)) -> Dict[str, Any]:
        docs = svc.list_documents()
        runs = svc.db.list_evaluation_runs(limit=5)
        return {
            "status": "ok",
            "version": __version__,
            "document_count": len(docs),
            "total_chunks": sum(d.chunk_count for d in docs),
            "evaluation_runs_count": len(runs),
            "generation_mode": svc.settings.generation_mode,
            "llm_configured": bool(svc.settings.llm_api_key),
            "llm_model": svc.settings.llm_model if svc.settings.llm_api_key else None,
            "default_retrieval_mode": svc.settings.default_retrieval_mode,
            "default_top_k": svc.settings.default_top_k,
            "default_alpha": svc.settings.default_alpha,
            "min_evidence_score": svc.settings.min_evidence_score,
        }

    # -------------------------------------------------------------------------
    # Document Management Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/documents", response_model=List[DocumentSummary])
    def list_documents(svc: GroundCheckService = Depends(get_service)) -> List[DocumentSummary]:
        return svc.list_documents()

    @app.get("/api/documents/{doc_id}", response_model=DocumentDetail)
    def get_document(doc_id: str, svc: GroundCheckService = Depends(get_service)) -> DocumentDetail:
        doc = svc.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return doc

    @app.post("/api/documents/upload", response_model=DocumentSummary, status_code=201)
    async def upload_document(
        file: UploadFile = File(...),
        svc: GroundCheckService = Depends(get_service),
    ) -> DocumentSummary:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided.")
        raw_bytes = await file.read()
        try:
            return svc.upload_document(filename=file.filename, raw_bytes=raw_bytes)
        except IngestionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Unexpected ingestion error: {exc}") from exc

    @app.post("/api/documents/seed", response_model=List[DocumentSummary])
    def reseed_sample_documents(svc: GroundCheckService = Depends(get_service)) -> List[DocumentSummary]:
        """Restore or re-index the built-in technical reference corpus."""
        svc.seed_sample_corpus()
        return svc.list_documents()

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str, svc: GroundCheckService = Depends(get_service)) -> Dict[str, Any]:
        deleted = svc.delete_document(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return {"deleted": True, "doc_id": doc_id}

    # -------------------------------------------------------------------------
    # Evidence-Grounded Question Answering & History Endpoints
    # -------------------------------------------------------------------------

    @app.post("/api/query", response_model=QueryResponse)
    def query_documents(
        payload: QueryRequest,
        svc: GroundCheckService = Depends(get_service),
    ) -> QueryResponse:
        try:
            return svc.answer_question(payload, save_history=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/queries")
    def list_query_history(
        limit: int = 30,
        svc: GroundCheckService = Depends(get_service),
    ) -> List[Dict[str, Any]]:
        return svc.db.list_queries(limit=max(1, min(100, limit)))

    @app.get("/api/queries/{query_id}", response_model=QueryResponse)
    def get_query_detail(
        query_id: str,
        svc: GroundCheckService = Depends(get_service),
    ) -> QueryResponse:
        resp = svc.db.get_query_response(query_id)
        if not resp:
            raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")
        return resp

    # -------------------------------------------------------------------------
    # AI Evaluation & Reproducible Benchmarking Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/evaluations/benchmark-info")
    def get_benchmark_info(svc: GroundCheckService = Depends(get_service)) -> Dict[str, Any]:
        return svc.get_benchmark_metadata()

    @app.get("/api/evaluations/runs", response_model=List[EvaluationRunSummary])
    def list_evaluation_runs(
        limit: int = 25,
        svc: GroundCheckService = Depends(get_service),
    ) -> List[EvaluationRunSummary]:
        return svc.db.list_evaluation_runs(limit=max(1, min(100, limit)))

    @app.get("/api/evaluations/runs/{run_id}", response_model=EvaluationRunSummary)
    def get_evaluation_run(
        run_id: str,
        svc: GroundCheckService = Depends(get_service),
    ) -> EvaluationRunSummary:
        run = svc.db.get_evaluation_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found.")
        return run

    @app.post("/api/evaluations/run", response_model=EvaluationRunSummary)
    def trigger_evaluation_run(
        payload: EvaluationRunRequest,
        svc: GroundCheckService = Depends(get_service),
    ) -> EvaluationRunSummary:
        try:
            return svc.run_benchmark(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/evaluations/compare-suite", response_model=List[EvaluationRunSummary])
    def run_comparison_suite(
        top_k: int = 5,
        svc: GroundCheckService = Depends(get_service),
    ) -> List[EvaluationRunSummary]:
        """
        Execute a 3-way reproducible ablation experiment comparing:
        1. Pure Lexical (BM25)
        2. Pure Semantic (Subword/N-gram TF-IDF Cosine)
        3. Hybrid RRF (α = 0.55)
        """
        k = max(1, min(15, top_k))
        configs = [
            EvaluationRunRequest(
                experiment_name=f"Lexical BM25 Baseline (k={k})",
                retrieval_mode="bm25",
                top_k=k,
                alpha=1.0,
            ),
            EvaluationRunRequest(
                experiment_name=f"Semantic N-Gram Vector (k={k})",
                retrieval_mode="semantic",
                top_k=k,
                alpha=0.0,
            ),
            EvaluationRunRequest(
                experiment_name=f"Hybrid BM25 + Semantic + RRF (k={k}, α=0.55)",
                retrieval_mode="hybrid",
                top_k=k,
                alpha=0.55,
            ),
        ]
        results = [svc.run_benchmark(c) for c in configs]
        return results

    @app.delete("/api/evaluations/runs/{run_id}")
    def delete_evaluation_run(
        run_id: str,
        svc: GroundCheckService = Depends(get_service),
    ) -> Dict[str, Any]:
        deleted = svc.db.delete_evaluation_run(run_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found.")
        return {"deleted": True, "run_id": run_id}

    # -------------------------------------------------------------------------
    # Static Frontend Serving
    # -------------------------------------------------------------------------

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_index() -> Any:
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h1>GroundCheck API is running.</h1>")

    return app
