"""
Unit tests for HybridRetriever (BM25 lexical, subword/n-gram TF-IDF semantic cosine,
Reciprocal Rank Fusion), answer synthesis, claim-level verification, numeric fidelity,
contradiction detection, and epistemic abstention.
"""

from __future__ import annotations

import unittest

from groundcheck.generator import synthesize_extractive_answer
from groundcheck.models import ChunkModel
from groundcheck.retrieval import HybridRetriever
from groundcheck.verifier import (
    compute_claim_passage_support,
    detect_passage_conflicts,
    verify_answer,
)


def _make_sample_chunks() -> list[ChunkModel]:
    return [
        ChunkModel(
            chunk_id="doc-ctrl::ch-000",
            doc_id="doc-ctrl",
            filename="controller_v1.md",
            doc_title="Controller Spec v1",
            chunk_index=0,
            page_number=1,
            section_title="Kalman Filter Setup",
            text=(
                "The Extended Kalman Filter (EKF) executes state estimation at a fixed sampling "
                "period of 50 ms (20 Hz). The CUSUM fault detection threshold J_th is set to 14.5."
            ),
            char_count=176,
            token_estimate=42,
        ),
        ChunkModel(
            chunk_id="doc-ctrl::ch-001",
            doc_id="doc-ctrl",
            filename="controller_v1.md",
            doc_title="Controller Spec v1",
            chunk_index=1,
            page_number=2,
            section_title="Safety Interlock",
            text=(
                "According to the legacy plant safety manual, the emergency pump shutdown timeout "
                "is configured at 45 ms after an over-current fault signal."
            ),
            char_count=142,
            token_estimate=34,
        ),
        ChunkModel(
            chunk_id="doc-gateway::ch-000",
            doc_id="doc-gateway",
            filename="gateway_v2.md",
            doc_title="Gateway Errata v2",
            chunk_index=0,
            page_number=1,
            section_title="Safety Interlock Errata",
            text=(
                "For PLC telemetry bridges, the emergency pump shutdown timeout is configured at "
                "120 ms to accommodate Modbus-TCP frame turnaround."
            ),
            char_count=135,
            token_estimate=32,
        ),
        ChunkModel(
            chunk_id="doc-webhook::ch-000",
            doc_id="doc-webhook",
            filename="webhook_spec.md",
            doc_title="Webhook Delivery Spec",
            chunk_index=0,
            page_number=1,
            section_title="HMAC Verification",
            text=(
                "Outbound webhook payloads are signed using HMAC-SHA256 and verified via "
                "hmac.compare_digest with a 300 second replay tolerance window."
            ),
            char_count=138,
            token_estimate=33,
        ),
    ]


class TestRetrievalAndVerification(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = _make_sample_chunks()
        self.retriever = HybridRetriever(self.chunks)

    def test_hybrid_bm25_and_semantic_modes_rank_relevant_first(self) -> None:
        q = "What is the EKF sampling period and CUSUM threshold J_th?"
        for mode in ("hybrid", "bm25", "semantic"):
            results = self.retriever.search(q, mode=mode, top_k=3)
            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0].chunk_id, "doc-ctrl::ch-000")
            self.assertGreater(results[0].combined_score, 0.35)

    def test_unanswerable_query_gets_low_relevance_and_abstains(self) -> None:
        q = "How many quantum qubits are inside the superconducting dilution refrigerator?"
        results = self.retriever.search(q, mode="hybrid", top_k=3)
        ans, abstained = synthesize_extractive_answer(q, results, min_evidence_score=0.18)
        self.assertTrue(abstained)
        self.assertIn("Insufficient evidence", ans)

        report = verify_answer(q, ans, results, min_evidence_score=0.18, explicit_abstention=abstained)
        self.assertEqual(report.verdict, "insufficient_evidence")

    def test_conflict_detection_surfaces_both_competing_passages(self) -> None:
        q = "What is the configured emergency pump shutdown timeout?"
        results = self.retriever.search(q, mode="hybrid", top_k=4)
        conflicts = detect_passage_conflicts(q, results, min_relevance=0.15)
        self.assertEqual(len(conflicts), 1)
        self.assertIn("45", conflicts[0].conflict_description)
        self.assertIn("120", conflicts[0].conflict_description)

        ans, abstained = synthesize_extractive_answer(q, results, min_evidence_score=0.15)
        self.assertFalse(abstained)
        self.assertIn("doc-ctrl::ch-001", ans)
        self.assertIn("doc-gateway::ch-000", ans)

        report = verify_answer(q, ans, results, min_evidence_score=0.15, explicit_abstention=False)
        self.assertEqual(report.verdict, "conflicted_evidence")
        self.assertTrue(report.has_conflict)

    def test_numeric_fidelity_penalizes_fabricated_numbers(self) -> None:
        passage = "The EKF runs at a fixed sampling period of 50 ms (20 Hz)."
        genuine_claim = "The EKF runs at a sampling period of 50 ms (20 Hz). [doc-ctrl::ch-000]"
        fabricated_claim = "The EKF runs at a sampling period of 999 ms (85 Hz). [doc-ctrl::ch-000]"

        score_true, _ = compute_claim_passage_support(genuine_claim, passage)
        score_fake, _ = compute_claim_passage_support(fabricated_claim, passage)
        self.assertGreater(score_true, 0.80)
        self.assertLess(score_fake, score_true * 0.60)


if __name__ == "__main__":
    unittest.main()
