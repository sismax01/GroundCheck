"""
Reproducible AI Evaluation & Benchmarking Engine for GroundCheck.
Computes standard Information Retrieval (IR) metrics (Precision@K, Recall@K, MRR, NDCG@K, HitRate@K),
Answer Quality & Faithfulness metrics (Term Coverage, Citation Precision, Claim Groundedness),
Epistemic Reliability metrics (Abstention Accuracy, Conflict Detection Rate),
and Operational Telemetry (Latency p50/p95, Token & Cost Estimation).
"""

from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from groundcheck.config import Settings
from groundcheck.database import Database, utc_now_iso
from groundcheck.ingestion import estimate_tokens
from groundcheck.models import (
    EvalCaseResult,
    EvaluationRunRequest,
    EvaluationRunSummary,
    MetricExplanation,
    QueryRequest,
    QueryResponse,
)
from groundcheck.retrieval import tokenize_lexical


METRIC_DOCUMENTATION: List[MetricExplanation] = [
    MetricExplanation(
        metric_key="mean_precision_at_k",
        name="Precision@K (Doc-Level)",
        formula="P@K = |{d in TopK : d in ExpectedDocs}| / K_unique_docs",
        interpretation="Fraction of distinct documents appearing in the top-K retrieved passages that belong to the ground-truth expected source set for answerable queries.",
        measurement_type="Measured",
    ),
    MetricExplanation(
        metric_key="mean_recall_at_k",
        name="Relevant-Evidence Recall@K",
        formula="Recall@K = |{d in TopK : d in ExpectedDocs}| / |ExpectedDocs|",
        interpretation="Proportion of required ground-truth source documents successfully retrieved within the top-K passages.",
        measurement_type="Measured",
    ),
    MetricExplanation(
        metric_key="mean_mrr",
        name="Mean Reciprocal Rank (MRR)",
        formula="MRR = (1/|Q|) * sum(1 / rank_first_relevant)",
        interpretation="Measures how high up the first relevant ground-truth passage appears in the ranked list (1.0 means rank #1).",
        measurement_type="Measured",
    ),
    MetricExplanation(
        metric_key="mean_ndcg_at_k",
        name="Normalized Discounted Cumulative Gain (NDCG@K)",
        formula="DCG@K / IDCG@K where DCG@K = sum(rel_i / log2(i + 1))",
        interpretation="Position-aware ranking quality metric that rewards placing relevant passages at the very top of the retrieval list.",
        measurement_type="Measured",
    ),
    MetricExplanation(
        metric_key="mean_citation_precision",
        name="Citation Support Precision",
        formula="ValidCitations / TotalCitations",
        interpretation="Fraction of inline [chunk_id] citations in the answer whose underlying passage genuinely entails the claim (support >= 0.38).",
        measurement_type="Measured",
    ),
    MetricExplanation(
        metric_key="mean_groundedness",
        name="Claim Groundedness Score",
        formula="(1/|C|) * sum(SupportScore(claim_i, CitedPassages))",
        interpretation="Mean lexical, bigram, and numeric-fidelity entailment score across all substantive claims in generated answers.",
        measurement_type="Measured",
    ),
    MetricExplanation(
        metric_key="abstention_accuracy",
        name="Epistemic Abstention & Conflict Accuracy",
        formula="CorrectEpistemicDecisions / TotalBenchmarkCases",
        interpretation="Measures whether the system answers answerable questions, flags conflicts on contradictory specs, and abstains on unanswerable queries.",
        measurement_type="Measured",
    ),
    MetricExplanation(
        metric_key="latency_p50_ms",
        name="End-to-End Latency (p50 / p95)",
        formula="Median and 95th percentile wall-clock execution time in milliseconds",
        interpretation="Real wall-clock time measured via time.perf_counter() covering retrieval, synthesis, and claim verification.",
        measurement_type="Measured",
    ),
    MetricExplanation(
        metric_key="total_reference_cost_usd",
        name="Reference LLM API Cost Projection",
        formula="(PromptTokens * $0.15 + CompletionTokens * $0.60) / 1,000,000",
        interpretation="Estimated cost if the exact prompt/completion token volume were billed at reference commercial API rates. Local extractive runs incur $0.00 actual cost.",
        measurement_type="Estimated",
    ),
]


def _doc_matches_expected(retrieved_doc_id: str, expected_prefixes: List[str]) -> bool:
    """Match a deterministic doc_id (which has a hash suffix) against benchmark expected prefixes."""
    for exp in expected_prefixes:
        if retrieved_doc_id == exp or retrieved_doc_id.startswith(exp):
            return True
    return False


def compute_ir_metrics_for_case(
    retrieved_doc_ids: List[str],
    expected_doc_ids: List[str],
) -> Dict[str, float]:
    """
    Calculate Precision@K, Recall@K, MRR, NDCG@K, and HitRate for a single query.
    """
    if not expected_doc_ids:
        # For unanswerable questions with no expected docs, IR metrics are not applicable (1.0 if not needed)
        return {
            "precision_at_k": 1.0,
            "recall_at_k": 1.0,
            "mrr": 1.0,
            "ndcg_at_k": 1.0,
            "hit_rate": 1.0,
        }

    if not retrieved_doc_ids:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "hit_rate": 0.0,
        }

    # Relevance vector at each rank position
    rel_at_rank: List[int] = [
        1 if _doc_matches_expected(did, expected_doc_ids) else 0 for did in retrieved_doc_ids
    ]

    # Distinct expected docs matched
    matched_expected = set()
    for exp in expected_doc_ids:
        if any(did == exp or did.startswith(exp) for did in retrieved_doc_ids):
            matched_expected.add(exp)

    recall = len(matched_expected) / float(len(expected_doc_ids))
    hit_rate = 1.0 if len(matched_expected) > 0 else 0.0

    # Passage-level / unique-doc Precision@K
    relevant_passages = sum(rel_at_rank)
    precision = relevant_passages / float(len(retrieved_doc_ids))

    # Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for idx, r in enumerate(rel_at_rank, start=1):
        if r == 1:
            mrr = 1.0 / float(idx)
            break

    # NDCG@K (binary relevance with deduplicated document gain or passage gain)
    dcg = 0.0
    for idx, r in enumerate(rel_at_rank, start=1):
        if r > 0:
            dcg += float(r) / math.log2(idx + 1.0)

    ideal_rels = sorted(rel_at_rank, reverse=True)
    # Ensure IDCG reflects at least the number of expected documents
    ideal_count = max(sum(ideal_rels), min(len(retrieved_doc_ids), len(expected_doc_ids)))
    idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_count + 1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "precision_at_k": round(float(precision), 4),
        "recall_at_k": round(float(recall), 4),
        "mrr": round(float(mrr), 4),
        "ndcg_at_k": round(float(min(1.0, ndcg)), 4),
        "hit_rate": round(float(hit_rate), 4),
    }


def load_benchmark_dataset(benchmarks_dir: Path, benchmark_id: str = "groundcheck_eval_v1") -> Dict[str, Any]:
    """Load the JSON benchmark dataset from disk."""
    path = benchmarks_dir / f"{benchmark_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def execute_evaluation_run(
    request: EvaluationRunRequest,
    query_executor: Callable[[QueryRequest, bool], QueryResponse],
    db: Database,
    settings: Settings,
) -> EvaluationRunSummary:
    """
    Execute every test case in `groundcheck_eval_v1.json` through the live GroundCheck pipeline,
    compute real metrics, persist the experiment run to SQLite, and return the summary.
    """
    t_start = time.perf_counter()
    benchmark = load_benchmark_dataset(settings.benchmarks_dir, "groundcheck_eval_v1")
    cases = benchmark.get("cases", [])

    case_results: List[EvalCaseResult] = []
    answerable_ir_cases: List[EvalCaseResult] = []
    substantive_answer_cases: List[EvalCaseResult] = []
    abstain_cases: List[EvalCaseResult] = []
    conflict_cases: List[EvalCaseResult] = []

    latencies: List[float] = []
    total_actual_cost = 0.0
    total_ref_cost = 0.0

    for case in cases:
        q_req = QueryRequest(
            question=case["question"],
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            alpha=request.alpha,
            min_evidence_score=request.min_evidence_score,
            generation_mode=request.generation_mode,
        )
        # Execute live query (without polluting user's manual query history list)
        q_resp = query_executor(q_req, False)

        retrieved_doc_ids = [p.doc_id for p in q_resp.retrieved_passages]
        expected_doc_ids = case.get("expected_doc_ids", [])
        expected_terms = [t.lower() for t in case.get("expected_terms", [])]
        expected_behavior = case.get("expected_behavior", "answer")
        actual_verdict = q_resp.verification.verdict

        ir_metrics = compute_ir_metrics_for_case(retrieved_doc_ids, expected_doc_ids)

        # Expected term coverage in answer
        if expected_terms:
            ans_lower = q_resp.answer_markdown.lower()
            ans_tokens = set(tokenize_lexical(ans_lower, remove_stopwords=False))
            matched_t = sum(
                1 for t in expected_terms if (t in ans_lower or normalize_term_match(t, ans_tokens))
            )
            term_cov = matched_t / float(len(expected_terms))
        else:
            term_cov = 1.0

        # Evaluate whether the epistemic behavior matches the ground-truth expectation
        if expected_behavior == "abstain":
            passed_behavior = actual_verdict == "insufficient_evidence"
        elif expected_behavior == "report_conflict":
            passed_behavior = actual_verdict == "conflicted_evidence" or q_resp.verification.has_conflict
        else:
            passed_behavior = (
                actual_verdict in ("grounded", "partially_grounded")
                and ir_metrics["recall_at_k"] >= 0.5
                and term_cov >= 0.4
            )

        preview = q_resp.answer_markdown.replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:217] + "..."

        c_res = EvalCaseResult(
            case_id=case["case_id"],
            category=case["category"],
            question=case["question"],
            expected_behavior=expected_behavior,
            actual_verdict=actual_verdict,
            passed_behavior_check=passed_behavior,
            precision_at_k=ir_metrics["precision_at_k"],
            recall_at_k=ir_metrics["recall_at_k"],
            mrr=ir_metrics["mrr"],
            ndcg_at_k=ir_metrics["ndcg_at_k"],
            hit_rate=ir_metrics["hit_rate"],
            term_coverage=round(float(term_cov), 4),
            citation_precision=q_resp.verification.citation_precision,
            groundedness_score=q_resp.verification.overall_groundedness_score,
            latency_ms=q_resp.telemetry.total_latency_ms,
            reference_cost_usd=q_resp.telemetry.reference_llm_cost_usd,
            retrieved_doc_ids=retrieved_doc_ids,
            expected_doc_ids=expected_doc_ids,
            answer_preview=preview,
        )
        case_results.append(c_res)
        latencies.append(q_resp.telemetry.total_latency_ms)
        total_actual_cost += q_resp.telemetry.actual_api_cost_usd
        total_ref_cost += q_resp.telemetry.reference_llm_cost_usd

        if expected_doc_ids:
            answerable_ir_cases.append(c_res)
        if expected_behavior in ("answer", "report_conflict"):
            substantive_answer_cases.append(c_res)
        if expected_behavior == "abstain":
            abstain_cases.append(c_res)
        if expected_behavior == "report_conflict":
            conflict_cases.append(c_res)

    def _mean(vals: List[float]) -> float:
        return round(float(sum(vals) / len(vals)), 4) if vals else 0.0

    ir_pool = answerable_ir_cases if answerable_ir_cases else case_results
    ans_pool = substantive_answer_cases if substantive_answer_cases else case_results

    abstention_acc = (
        sum(1 for c in abstain_cases if c.passed_behavior_check) / float(len(abstain_cases))
        if abstain_cases
        else 1.0
    )
    conflict_rate = (
        sum(1 for c in conflict_cases if c.passed_behavior_check) / float(len(conflict_cases))
        if conflict_cases
        else 1.0
    )
    overall_pass = (
        sum(1 for c in case_results if c.passed_behavior_check) / float(len(case_results))
        if case_results
        else 0.0
    )

    lat_arr = np.array(latencies, dtype=np.float64) if latencies else np.array([0.0])
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    total_dur_ms = (time.perf_counter() - t_start) * 1000.0

    run_id = f"eval-{uuid.uuid4().hex[:8]}"
    exp_name = (
        request.experiment_name.strip()
        if request.experiment_name and request.experiment_name.strip()
        else f"{request.retrieval_mode.upper()} (k={request.top_k}, α={request.alpha:.2f})"
    )

    summary = EvaluationRunSummary(
        run_id=run_id,
        experiment_name=exp_name,
        benchmark_id=benchmark.get("benchmark_id", "groundcheck_eval_v1"),
        retrieval_mode=request.retrieval_mode,
        generation_mode=request.generation_mode,
        top_k=request.top_k,
        alpha=request.alpha,
        min_evidence_score=request.min_evidence_score,
        total_cases=len(case_results),
        mean_precision_at_k=_mean([c.precision_at_k for c in ir_pool]),
        mean_recall_at_k=_mean([c.recall_at_k for c in ir_pool]),
        mean_mrr=_mean([c.mrr for c in ir_pool]),
        mean_ndcg_at_k=_mean([c.ndcg_at_k for c in ir_pool]),
        hit_rate_at_k=_mean([c.hit_rate for c in ir_pool]),
        mean_term_coverage=_mean([c.term_coverage for c in ans_pool]),
        mean_citation_precision=_mean([c.citation_precision for c in ans_pool]),
        mean_groundedness=_mean([c.groundedness_score for c in ans_pool]),
        abstention_accuracy=round(float(abstention_acc), 4),
        conflict_detection_rate=round(float(conflict_rate), 4),
        overall_pass_rate=round(float(overall_pass), 4),
        latency_p50_ms=round(p50, 2),
        latency_p95_ms=round(p95, 2),
        total_duration_ms=round(total_dur_ms, 2),
        total_actual_cost_usd=round(total_actual_cost, 6),
        total_reference_cost_usd=round(total_ref_cost, 6),
        created_at=utc_now_iso(),
        case_results=case_results,
    )

    db.save_evaluation_run(summary)
    return summary


def normalize_term_match(term: str, token_set: set) -> bool:
    """Helper to check if normalized token form is present in token_set."""
    from groundcheck.retrieval import normalize_token
    return normalize_token(term) in token_set
