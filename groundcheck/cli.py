"""
Command-Line Interface (CLI) for GroundCheck.
Supports starting the web server, seeding sample documents, running single queries,
and executing reproducible evaluation benchmarks from the terminal or CI pipelines.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import uvicorn

from groundcheck.config import Settings
from groundcheck.models import EvaluationRunRequest, QueryRequest, model_to_json
from groundcheck.service import GroundCheckService


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = Settings.from_env()
    host = args.host or cfg.host
    port = args.port or cfg.port
    print(f"[GroundCheck] Starting server on http://{host}:{port} ...")
    uvicorn.run("groundcheck.main:app", host=host, port=port, reload=args.reload)
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    svc = GroundCheckService()
    docs = svc.seed_sample_corpus()
    print(f"[GroundCheck] Seeded {len(docs)} sample technical documents:")
    for d in docs:
        print(f"  - {d.doc_id}: {d.title} ({d.chunk_count} chunks, ~{d.total_tokens_est} tokens)")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    svc = GroundCheckService()
    req = QueryRequest(
        question=args.question,
        retrieval_mode=args.mode,
        top_k=args.top_k,
        alpha=args.alpha,
        min_evidence_score=args.min_score,
    )
    resp = svc.answer_question(req)
    print("=" * 78)
    print(f"QUESTION : {resp.question}")
    print(f"VERDICT  : {resp.verification.verdict.upper()} "
          f"(Groundedness: {resp.verification.overall_groundedness_score*100:.1f}%, "
          f"Citation Precision: {resp.verification.citation_precision*100:.1f}%)")
    print(f"LATENCY  : {resp.telemetry.total_latency_ms:.2f} ms | MODE: {resp.telemetry.execution_mode}")
    print("-" * 78)
    print("ANSWER:")
    print(resp.answer_markdown)
    print("-" * 78)
    print("TOP RETRIEVED EVIDENCE PASSAGES:")
    for p in resp.retrieved_passages:
        print(
            f"  #{p.rank} [{p.chunk_id}] ({p.filename} - {p.section_title}) "
            f"| Combined={p.combined_score:.3f} (BM25={p.bm25_score:.3f}, Sem={p.semantic_score:.3f})"
        )
    print("=" * 78)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    svc = GroundCheckService()
    req = EvaluationRunRequest(
        experiment_name=args.name,
        retrieval_mode=args.mode,
        top_k=args.top_k,
        alpha=args.alpha,
        min_evidence_score=args.min_score,
    )
    summary = svc.run_benchmark(req)
    if args.json:
        print(model_to_json(summary, indent=2))
        return 0

    print("=" * 84)
    print(f"GROUNDCHECK BENCHMARK EVALUATION REPORT — {summary.experiment_name}")
    print("=" * 84)
    print(f"Run ID                 : {summary.run_id}")
    print(f"Benchmark Dataset      : {summary.benchmark_id} ({summary.total_cases} cases)")
    print(f"Configuration          : mode={summary.retrieval_mode}, top_k={summary.top_k}, alpha={summary.alpha}")
    print(f"Execution Mode         : {summary.generation_mode} (Actual API Cost: ${summary.total_actual_cost_usd:.4f}, "
          f"Ref LLM Est: ${summary.total_reference_cost_usd:.5f})")
    print("-" * 84)
    print("AGGREGATE METRICS (Calculated from Live Execution):")
    print(f"  Overall Behavior Pass Rate : {summary.overall_pass_rate * 100:6.1f}%")
    print(f"  Relevant-Evidence Recall@K : {summary.mean_recall_at_k * 100:6.1f}%")
    print(f"  Precision@K (Passage-Level): {summary.mean_precision_at_k * 100:6.1f}%")
    print(f"  Mean Reciprocal Rank (MRR) : {summary.mean_mrr:6.3f}")
    print(f"  NDCG@K                     : {summary.mean_ndcg_at_k:6.3f}")
    print(f"  Citation Support Precision : {summary.mean_citation_precision * 100:6.1f}%")
    print(f"  Mean Claim Groundedness    : {summary.mean_groundedness * 100:6.1f}%")
    print(f"  Epistemic Abstention Acc.  : {summary.abstention_accuracy * 100:6.1f}%")
    print(f"  Conflict Detection Rate    : {summary.conflict_detection_rate * 100:6.1f}%")
    print(f"  Latency p50 / p95          : {summary.latency_p50_ms:.2f} ms / {summary.latency_p95_ms:.2f} ms")
    print("-" * 84)
    print(f"{'ID':<6} {'Category':<13} {'Expected':<16} {'Actual Verdict':<22} {'Recall@K':<9} {'Ground.':<8} {'Status'}")
    print("-" * 84)
    for c in summary.case_results:
        status_str = "PASS" if c.passed_behavior_check else "FAIL"
        print(
            f"{c.case_id:<6} {c.category:<13} {c.expected_behavior:<16} "
            f"{c.actual_verdict:<22} {c.recall_at_k:>7.2f}   {c.groundedness_score:>6.2f}   {status_str}"
        )
    print("=" * 84)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundcheck",
        description="GroundCheck — Evidence-Grounded Document Research & RAG Evaluation CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Launch the FastAPI web application")
    p_serve.add_argument("--host", type=str, default=None, help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=None, help="Bind port (default: 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # seed
    sub.add_parser("seed", help="Seed or reset the reference technical document corpus")

    # query
    p_query = sub.add_parser("query", help="Ask a technical question from the command line")
    p_query.add_argument("question", type=str, help="Question text")
    p_query.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "bm25", "semantic"])
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--alpha", type=float, default=0.55)
    p_query.add_argument("--min-score", type=float, default=0.18)

    # eval
    p_eval = sub.add_parser("eval", help="Run the reproducible evaluation benchmark suite")
    p_eval.add_argument("--name", type=str, default=None, help="Experiment label")
    p_eval.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "bm25", "semantic"])
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--alpha", type=float, default=0.55)
    p_eval.add_argument("--min-score", type=float, default=0.18)
    p_eval.add_argument("--json", action="store_true", help="Output raw JSON summary")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "seed":
        return cmd_seed(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "eval":
        return cmd_eval(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
