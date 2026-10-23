"""
End-to-end integration tests for the GroundCheck FastAPI backend, SQLite persistence,
document upload/deletion, grounded Q&A, error handling on unavailable external LLM,
and reproducible benchmark evaluation runs.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from groundcheck.api import create_app
from groundcheck.config import Settings


class TestAPIAndEvaluationIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_groundcheck.db"
        self.settings = Settings.from_env(db_override=self.db_path)
        self.app = create_app(settings=self.settings)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_health_and_seeded_sample_corpus(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["document_count"], 4)
        self.assertGreater(data["total_chunks"], 10)

        docs_resp = self.client.get("/api/documents")
        self.assertEqual(docs_resp.status_code, 200)
        docs = docs_resp.json()
        self.assertEqual(len(docs), 4)

        # Fetch single document detail with chunks
        doc_id = docs[0]["doc_id"]
        detail_resp = self.client.get(f"/api/documents/{doc_id}")
        self.assertEqual(detail_resp.status_code, 200)
        self.assertGreater(len(detail_resp.json()["chunks"]), 0)

    def test_upload_and_delete_custom_document(self) -> None:
        custom_md = (
            "# Custom Autonomous Drone Navigation Spec\n\n"
            "## 1. Visual Odometry Update Rate\n"
            "The stereo visual-inertial odometry pipeline fuses IMU measurements at 400 Hz "
            "with keyframe camera poses at 30 Hz.\n"
        ).encode("utf-8")

        up_resp = self.client.post(
            "/api/documents/upload",
            files={"file": ("drone_vio_spec.md", custom_md, "text/markdown")},
        )
        self.assertEqual(up_resp.status_code, 201)
        uploaded = up_resp.json()
        doc_id = uploaded["doc_id"]

        # Query the newly uploaded document
        q_resp = self.client.post(
            "/api/query",
            json={
                "question": "At what frequency does the stereo visual-inertial odometry pipeline fuse IMU measurements?",
                "retrieval_mode": "hybrid",
                "top_k": 3,
            },
        )
        self.assertEqual(q_resp.status_code, 200)
        q_data = q_resp.json()
        self.assertEqual(q_data["verification"]["verdict"], "grounded")
        self.assertIn("400", q_data["answer_markdown"])

        # Delete the custom document
        del_resp = self.client.delete(f"/api/documents/{doc_id}")
        self.assertEqual(del_resp.status_code, 200)
        self.assertTrue(del_resp.json()["deleted"])

    def test_upload_invalid_file_returns_400(self) -> None:
        bad_resp = self.client.post(
            "/api/documents/upload",
            files={"file": ("malware.bin", b"\x00\x01\x02", "application/octet-stream")},
        )
        self.assertEqual(bad_resp.status_code, 400)
        self.assertIn("Unsupported file extension", bad_resp.json()["detail"])

    def test_external_llm_mode_without_api_key_returns_clear_503(self) -> None:
        resp = self.client.post(
            "/api/query",
            json={
                "question": "What are the calibrated valve ratios in QuadTank-FDI?",
                "retrieval_mode": "hybrid",
                "generation_mode": "llm",
            },
        )
        self.assertEqual(resp.status_code, 503)
        self.assertIn("GROUNDCHECK_LLM_API_KEY is not configured", resp.json()["detail"])

    def test_reproducible_benchmark_evaluation_and_comparison_suite(self) -> None:
        info_resp = self.client.get("/api/evaluations/benchmark-info")
        self.assertEqual(info_resp.status_code, 200)
        self.assertEqual(len(info_resp.json()["benchmark"]["cases"]), 10)

        run_resp = self.client.post(
            "/api/evaluations/run",
            json={
                "experiment_name": "CI Integration Test Run",
                "retrieval_mode": "hybrid",
                "top_k": 5,
                "alpha": 0.55,
            },
        )
        self.assertEqual(run_resp.status_code, 200)
        summary = run_resp.json()
        self.assertEqual(summary["total_cases"], 10)
        self.assertGreaterEqual(summary["mean_recall_at_k"], 0.90)
        self.assertGreaterEqual(summary["abstention_accuracy"], 1.0)
        self.assertGreaterEqual(summary["conflict_detection_rate"], 1.0)
        self.assertGreaterEqual(summary["overall_pass_rate"], 0.90)


if __name__ == "__main__":
    unittest.main()
