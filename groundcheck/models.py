"""
Pydantic data models and API schemas for GroundCheck.
Compatible with both Pydantic v1 and v2.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkModel(BaseModel):
    """Represents a segmented, searchable passage within a technical document."""

    chunk_id: str = Field(..., description="Deterministic or unique identifier for the chunk")
    doc_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original source filename")
    doc_title: str = Field(..., description="Human-readable title of the document")
    chunk_index: int = Field(..., description="Sequential index of the chunk in the document")
    page_number: Optional[int] = Field(None, description="1-based page number if applicable (e.g. PDF)")
    section_title: str = Field("General", description="Detected section header or context")
    text: str = Field(..., description="Verbatim passage text")
    char_count: int = Field(..., description="Character length of the passage")
    token_estimate: int = Field(..., description="Estimated subword token count")


class DocumentSummary(BaseModel):
    """Metadata summary for an ingested document."""

    doc_id: str
    filename: str
    title: str
    file_type: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    total_tokens_est: int
    content_hash: str
    is_sample: bool = False
    created_at: str


class DocumentDetail(DocumentSummary):
    """Detailed view of a document including all its segmented passages."""

    chunks: List[ChunkModel] = Field(default_factory=list)


class RetrievedPassage(BaseModel):
    """A passage returned by the GroundCheck retrieval engine with transparent scoring breakdown."""

    rank: int = Field(..., description="1-based rank in retrieval results")
    chunk_id: str
    doc_id: str
    filename: str
    doc_title: str
    page_number: Optional[int] = None
    section_title: str = "General"
    text: str
    bm25_score: float = Field(..., description="Normalized lexical BM25 score [0.0, 1.0]")
    semantic_score: float = Field(..., description="Normalized TF-IDF n-gram cosine similarity [0.0, 1.0]")
    rrf_score: float = Field(..., description="Reciprocal Rank Fusion score")
    combined_score: float = Field(..., description="Final calibrated relevance score [0.0, 1.0]")
    matched_terms: List[str] = Field(default_factory=list, description="Exact query terms found in passage")
    relevance_tier: str = Field(..., description="High | Moderate | Low | Marginal")


class ClaimVerification(BaseModel):
    """Verification breakdown for a single sentence/claim in the generated answer."""

    claim_index: int
    claim_text: str
    cited_chunk_ids: List[str] = Field(default_factory=list)
    status: str = Field(
        ...,
        description="supported | partially_supported | unsupported | conflicted | abstention_notice",
    )
    support_score: float = Field(..., description="Measured lexical + semantic entailment score [0.0, 1.0]")
    best_supporting_chunk_id: Optional[str] = None
    best_supporting_quote: Optional[str] = None
    explanation: str = Field(..., description="Human-readable reason for the verification status")


class ConflictPair(BaseModel):
    """Details a detected numeric, polarity, or specification conflict between two retrieved passages."""

    chunk_id_a: str
    source_a: str
    excerpt_a: str
    chunk_id_b: str
    source_b: str
    excerpt_b: str
    conflict_description: str


class VerificationReport(BaseModel):
    """Overall groundedness and epistemic reliability report for an answer."""

    verdict: str = Field(
        ...,
        description="grounded | partially_grounded | conflicted_evidence | insufficient_evidence",
    )
    overall_groundedness_score: float = Field(..., description="Aggregate claim support score [0.0, 1.0]")
    citation_precision: float = Field(
        ..., description="Fraction of cited passages that genuinely support the claim they are attached to"
    )
    supported_claim_count: int
    total_substantive_claims: int
    has_conflict: bool = False
    conflicts: List[ConflictPair] = Field(default_factory=list)
    claims: List[ClaimVerification] = Field(default_factory=list)
    epistemic_note: str = Field(..., description="Clear explanation of confidence, evidence coverage, or gaps")


class CostAndLatencyMetrics(BaseModel):
    """Telemetry for query execution, distinguishing real measurements from reference estimates."""

    retrieval_latency_ms: float
    generation_latency_ms: float
    verification_latency_ms: float
    total_latency_ms: float
    prompt_tokens_est: int
    completion_tokens_est: int
    total_tokens_est: int
    actual_api_cost_usd: float = Field(0.0, description="Actual billed API cost (0.0 in local extractive mode)")
    reference_llm_cost_usd: float = Field(
        ..., description="Estimated cost if executed via reference commercial LLM API"
    )
    execution_mode: str = Field(
        ..., description="local_extractive (Zero-Cost Deterministic) | external_llm (Live API)"
    )


class QueryRequest(BaseModel):
    """Request payload for asking a question against uploaded documents."""

    question: str = Field(..., min_length=3, max_length=1500, description="Technical question to investigate")
    retrieval_mode: str = Field("hybrid", description="hybrid | bm25 | semantic")
    top_k: int = Field(5, ge=1, le=15, description="Number of passages to retrieve")
    alpha: float = Field(0.55, ge=0.0, le=1.0, description="Lexical (1.0) vs Semantic (0.0) weight")
    min_evidence_score: float = Field(0.18, ge=0.01, le=0.95, description="Minimum relevance threshold")
    doc_ids: Optional[List[str]] = Field(None, description="Optional subset of document IDs to filter by")
    generation_mode: Optional[str] = Field(None, description="Override generation mode: extractive | llm")


class QueryResponse(BaseModel):
    """Full inspectable response returned after answering a question."""

    query_id: str
    question: str
    answer_markdown: str
    retrieval_mode: str
    top_k: int
    alpha: float
    retrieved_passages: List[RetrievedPassage]
    verification: VerificationReport
    telemetry: CostAndLatencyMetrics
    created_at: str


class EvalCaseResult(BaseModel):
    """Detailed evaluation metrics for a single benchmark question."""

    case_id: str
    category: str = Field(..., description="factual | multi_hop | unanswerable | conflicting")
    question: str
    expected_behavior: str = Field(..., description="answer | abstain | report_conflict")
    actual_verdict: str
    passed_behavior_check: bool
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    term_coverage: float
    citation_precision: float
    groundedness_score: float
    latency_ms: float
    reference_cost_usd: float
    retrieved_doc_ids: List[str]
    expected_doc_ids: List[str]
    answer_preview: str


class EvaluationRunRequest(BaseModel):
    """Parameters for triggering a reproducible benchmark run."""

    experiment_name: Optional[str] = Field(None, description="Custom label for the experiment")
    retrieval_mode: str = Field("hybrid", description="hybrid | bm25 | semantic")
    top_k: int = Field(5, ge=1, le=15)
    alpha: float = Field(0.55, ge=0.0, le=1.0)
    min_evidence_score: float = Field(0.18, ge=0.01, le=0.95)
    generation_mode: str = Field("extractive", description="extractive | llm")


class MetricExplanation(BaseModel):
    """Documentation explaining what a metric measures and its mathematical formula."""

    metric_key: str
    name: str
    formula: str
    interpretation: str
    measurement_type: str = Field(..., description="Measured | Estimated")


class EvaluationRunSummary(BaseModel):
    """Aggregate benchmark results across all test cases in an experiment run."""

    run_id: str
    experiment_name: str
    benchmark_id: str
    retrieval_mode: str
    generation_mode: str
    top_k: int
    alpha: float
    min_evidence_score: float
    total_cases: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    hit_rate_at_k: float
    mean_term_coverage: float
    mean_citation_precision: float
    mean_groundedness: float
    abstention_accuracy: float
    conflict_detection_rate: float
    overall_pass_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_duration_ms: float
    total_actual_cost_usd: float
    total_reference_cost_usd: float
    created_at: str
    case_results: List[EvalCaseResult] = Field(default_factory=list)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model to a dict across both Pydantic v1 and v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def model_to_json(model: BaseModel, indent: Optional[int] = None) -> str:
    """Serialize a Pydantic model to a JSON string across both Pydantic v1 and v2."""
    import json
    return json.dumps(model_to_dict(model), indent=indent)
