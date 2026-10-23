"""
Configuration management for GroundCheck.
Loads settings from environment variables or .env file with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # Storage paths
    db_path: Path = PROJECT_ROOT / "data" / "groundcheck.db"
    sample_docs_dir: Path = PROJECT_ROOT / "sample_docs"
    benchmarks_dir: Path = PROJECT_ROOT / "benchmarks"
    auto_seed_sample_docs: bool = True
    max_upload_mb: int = 25

    # Chunking parameters
    chunk_size: int = 850
    chunk_overlap: int = 110

    # Retrieval defaults
    default_retrieval_mode: str = "hybrid"  # hybrid | bm25 | semantic
    default_top_k: int = 5
    default_alpha: float = 0.55
    min_evidence_score: float = 0.18

    # Generation & LLM settings
    generation_mode: str = "extractive"  # extractive | llm
    llm_api_key: Optional[str] = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 20

    # Transparent Cost Reference Rates (USD per 1,000,000 tokens)
    cost_input_per_1m: float = 0.15
    cost_output_per_1m: float = 0.60

    @classmethod
    def from_env(cls, db_override: Optional[Path] = None) -> "Settings":
        """Instantiate settings from environment variables."""
        raw_db = os.getenv("GROUNDCHECK_DB_PATH", "data/groundcheck.db")
        db_path = db_override if db_override is not None else (
            Path(raw_db) if Path(raw_db).is_absolute() else PROJECT_ROOT / raw_db
        )

        auto_seed_raw = os.getenv("GROUNDCHECK_AUTO_SEED_SAMPLE_DOCS", "true").strip().lower()
        auto_seed = auto_seed_raw in ("1", "true", "yes", "on")

        api_key = os.getenv("GROUNDCHECK_LLM_API_KEY", "").strip()
        if api_key in ("", "your_api_key_here", "none", "null"):
            api_key = None

        gen_mode = os.getenv("GROUNDCHECK_GENERATION_MODE", "extractive").strip().lower()
        if gen_mode not in ("extractive", "llm"):
            gen_mode = "extractive"

        ret_mode = os.getenv("GROUNDCHECK_DEFAULT_RETRIEVAL_MODE", "hybrid").strip().lower()
        if ret_mode not in ("hybrid", "bm25", "semantic"):
            ret_mode = "hybrid"

        return cls(
            host=os.getenv("GROUNDCHECK_HOST", "127.0.0.1"),
            port=int(os.getenv("GROUNDCHECK_PORT", "8000")),
            log_level=os.getenv("GROUNDCHECK_LOG_LEVEL", "INFO"),
            db_path=db_path,
            sample_docs_dir=PROJECT_ROOT / "sample_docs",
            benchmarks_dir=PROJECT_ROOT / "benchmarks",
            auto_seed_sample_docs=auto_seed,
            max_upload_mb=max(1, int(os.getenv("GROUNDCHECK_MAX_UPLOAD_MB", "25"))),
            chunk_size=max(150, int(os.getenv("GROUNDCHECK_CHUNK_SIZE", "850"))),
            chunk_overlap=max(0, int(os.getenv("GROUNDCHECK_CHUNK_OVERLAP", "110"))),
            default_retrieval_mode=ret_mode,
            default_top_k=max(1, min(20, int(os.getenv("GROUNDCHECK_DEFAULT_TOP_K", "5")))),
            default_alpha=max(0.0, min(1.0, float(os.getenv("GROUNDCHECK_DEFAULT_ALPHA", "0.55")))),
            min_evidence_score=max(0.01, min(0.95, float(os.getenv("GROUNDCHECK_MIN_EVIDENCE_SCORE", "0.18")))),
            generation_mode=gen_mode,
            llm_api_key=api_key,
            llm_base_url=os.getenv("GROUNDCHECK_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            llm_model=os.getenv("GROUNDCHECK_LLM_MODEL", "gpt-4o-mini"),
            llm_timeout_seconds=max(3, int(os.getenv("GROUNDCHECK_LLM_TIMEOUT_SECONDS", "20"))),
            cost_input_per_1m=max(0.0, float(os.getenv("GROUNDCHECK_COST_INPUT_PER_1M", "0.15"))),
            cost_output_per_1m=max(0.0, float(os.getenv("GROUNDCHECK_COST_OUTPUT_PER_1M", "0.60"))),
        )
