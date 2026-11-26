# GroundCheck

**Evidence-Grounded Document Research, Claim-Level Verification & Reproducible RAG Evaluation Platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.92%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![GroundCheck CI](https://github.com/sismax01/GroundCheck/actions/workflows/ci.yml/badge.svg)](https://github.com/sismax01/GroundCheck/actions/workflows/ci.yml)
 
GroundCheck is an open-source AI engineering platform that bridges **technical document research**, **evidence-grounded question answering**, and **systematic, repeatable RAG evaluation**.

Unlike generic PDF chat wrappers that treat retrieved snippets as unverified context and silently hallucinate when evidence is missing, GroundCheck makes every answer inspectable at the sentence level:
- **Evidence Traceability**: Every claim links directly to its supporting passage (`[doc_id::ch-XXX]`) with source filename, section hierarchy, page number, and lexical/semantic relevance breakdowns.
- **Claim-Level Groundedness & Numeric Fidelity Auditing**: Answers are decomposed into atomic claims and verified for content-word entailment, contiguous phrase overlap, and exact numerical/unit fidelity against cited passages.
- **Epistemic Reliability (Abstention & Conflict Detection)**: When retrieved passages fall below the calibrated evidence threshold ($\tau_{\min} = 0.18$) or lack query-term coverage, GroundCheck abstains honestly rather than fabricating an answer. When two documents state conflicting quantitative specifications for the same attribute (e.g., `45 ms` vs. `120 ms`), GroundCheck flags a `conflicted_evidence` verdict and presents both cited excerpts side by side.
- **Reproducible Evaluation & Ablation Benchmarking**: Includes a 10-case technical benchmark (`benchmarks/groundcheck_eval_v1.json`) and an evaluation engine computing **Precision@K**, **Relevant-Evidence Recall@K**, **Mean Reciprocal Rank (MRR)**, **NDCG@K**, **Citation Support Precision**, **Claim Groundedness**, **Abstention Accuracy**, **Conflict Detection Rate**, **Latency (p50/p95)**, and **Token/Cost Projections**.

---

## Core Capabilities

### 1. Document Ingestion & Section-Aware Chunking (`groundcheck/ingestion.py`)
- Supports **PDF** (`.pdf` via `pypdf`), **Markdown** (`.md`), **Plain Text** (`.txt`), and **HTML** (`.html`, `.htm`).
- Extracts document titles, page numbers, and section headers automatically while stripping script/style noise.
- Segments documents into overlapping, section-coherent passages (`chunk_size=850`, `chunk_overlap=120`) with deterministic chunk IDs (`{doc_id}::ch-{index:03d}`) and SHA-256 content hashes.
- Validates all uploads against empty files, binary null-byte injection, password-protected/corrupt PDFs, image-only blank scans, and configurable size limits.

### 2. Calibrated Hybrid Retrieval (`groundcheck/retrieval.py`)
- **Lexical Engine (Okapi BM25)**: Matches exact technical identifiers (`HMAC-SHA256`, `HTTP 409`, `CRC32C`, `CUSUM`, `GLRT`, `EKF`) with Robertson-Sparck Jones IDF smoothing ($k_1 = 1.5, b = 0.75$).
- **Subword & Phrase Semantic Vector Engine**: Projects passages into a log-weighted TF-IDF vector space over word unigrams, word bigrams, and character $4$-grams to capture morphological roots and phrase paraphrases via cosine similarity.
- **Reciprocal Rank Fusion (RRF) & Absolute Score Calibration**: Combines lexical and semantic rankings via configurable convex blending ($\alpha \in [0, 1]$) and RRF ($k_{\text{rrf}} = 20$). Avoids per-query relative min-max scaling so out-of-domain queries receive genuinely low scores and trigger abstention reliably.

### 3. Pluggable Answer Synthesis (`groundcheck/generator.py`)
- **Deterministic Local Extractive Mode (`extractive`, Default)**: Zero-cost local synthesis using greedy submodular / maximal marginal query-term coverage across retrieved passages. Requires **no external API keys** and incurs **$0.00** cost.
- **External OpenAI-Compatible LLM Mode (`llm`)**: Connects to any OpenAI-compatible `/v1/chat/completions` API (`GROUNDCHECK_LLM_API_KEY`, `GROUNDCHECK_LLM_BASE_URL`, `GROUNDCHECK_LLM_MODEL`) with strict citation and abstention constraints.

### 4. Claim-Level Verification & Conflict Detection (`groundcheck/verifier.py`)
- Decomposes generated answers into individual claims and evaluates each claim against its cited `[doc_id::ch-XXX]` passages.
- Penalizes claims that introduce unverified numerical literals by $65\%$.
- Detects cross-document quantitative conflicts when two relevant passages specify different values for the same technical attribute and unit.

### 5. Reproducible AI Evaluation Workbench (`groundcheck/evaluator.py`)
- Runs repeatable experiments over `benchmarks/groundcheck_eval_v1.json` covering **factual**, **multi-hop**, **conflicting**, and **unanswerable** test cases.
- Clearly separates **Measured Values** (IR metrics, groundedness, abstention accuracy, wall-clock p50/p95 latency) from **Estimated Values** (reference commercial LLM token cost projections).

---

## Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend API** | Python 3.10+, FastAPI, Uvicorn, Pydantic | REST API, schema validation, OpenAPI docs (`/docs`), static serving |
| **Storage & Audit Log** | SQLite3 (`groundcheck/database.py`) | Zero-config transactional persistence for documents, chunks, query history, and evaluation runs |
| **Document Parsing** | `pypdf`, `html.parser` | Page-aware PDF extraction, Markdown/HTML/Text parsing, input sanitization |
| **Retrieval & Math** | `numpy` | Vectorized BM25, subword/n-gram TF-IDF cosine similarity, RRF, latency percentiles |
| **External LLM Client** | `httpx` | Optional OpenAI-compatible chat completion integration with timeout handling |
| **Frontend UI** | Vanilla ES6 JavaScript, HTML5, CSS3 | Responsive 3-tab engineering workbench (Dark/Light mode, zero frontend build step required) |
| **Testing & CI** | `unittest` / `pytest`, GitHub Actions | 17 automated unit and integration tests across Python 3.10, 3.11, and 3.12 |

---

## Repository Structure

```text
GroundCheck/
├── .github/workflows/ci.yml        # GitHub Actions CI pipeline (tests + live benchmark check)
├── benchmarks/
│   └── groundcheck_eval_v1.json    # Documented 10-case evaluation dataset & ground-truth criteria
├── docs/
│   └── ARCHITECTURE.md             # Deep-dive engineering architecture & mathematical formulas
├── groundcheck/
│   ├── __init__.py                 # Package metadata
│   ├── api.py                      # FastAPI application factory & REST endpoints
│   ├── cli.py                      # CLI for serving, seeding, querying, and benchmarking
│   ├── config.py                   # Environment configuration loader
│   ├── database.py                 # SQLite persistence for docs, chunks, queries, and eval runs
│   ├── evaluator.py                # IR, groundedness, abstention & latency benchmark calculator
│   ├── generator.py                # Local extractive synthesizer & OpenAI-compatible LLM client
│   ├── ingestion.py                # PDF/MD/TXT/HTML parser, validator & section-aware chunker
│   ├── main.py                     # ASGI server entrypoint
│   ├── models.py                   # Pydantic data schemas
│   ├── retrieval.py                # Hybrid BM25 + Subword/N-Gram TF-IDF + RRF retrieval engine
│   ├── service.py                  # Core orchestration service
│   ├── verifier.py                 # Claim-level entailment auditor & conflict detector
│   └── static/                     # Interactive Single-Page Application (index.html, styles.css, app.js)
├── sample_docs/                    # Built-in technical engineering corpus (seeded automatically)
│   ├── distributed_consensus_storage.md
│   ├── hybrid_rag_evaluation_guide.md
│   ├── quadtank_fdi_control_spec.md
│   └── webhook_reliability_spec.md
├── tests/                          # Automated unit and FastAPI integration test suite
│   ├── test_api_and_evaluation.py
│   ├── test_ingestion.py
│   └── test_retrieval_and_verifier.py
├── .env.example                    # Example environment configuration (placeholders only)
├── .gitignore                      # Clean git ignore rules
├── Dockerfile                      # Reproducible container image definition
├── LICENSE                         # MIT License
├── Makefile                        # Developer convenience targets
├── pyproject.toml                  # Standard Python package metadata & build config
├── requirements.txt                # Runtime dependencies
├── README.md                       # Project documentation
└── THIRD_PARTY_NOTICES.md          # Open-source dependency attributions
```

---

## Quickstart & Setup Instructions

### Prerequisites
- **Python 3.10 or newer**
- No external database server or paid API key is required. GroundCheck uses embedded SQLite and defaults to its deterministic **Local Extractive Mode** (`$0.00` cost) out of the box.

### 1. Clone & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env` if you want to customize ports, chunk sizes, or enable an external OpenAI-compatible LLM:
```bash
cp .env.example .env
```

### 3. Start the GroundCheck Web Application
```bash
python3 -m groundcheck.cli serve --host 127.0.0.1 --port 8000
```
Open **http://127.0.0.1:8000** in your browser:
- **Research & Evidence Q&A Tab**: Ask technical questions, click diagnostic presets (including cross-document conflict and abstention tests), click inline `[doc_id::ch-XXX]` citations to highlight the exact supporting passage, and inspect claim-by-claim verification scores.
- **Document Corpus Tab**: Upload your own `.pdf`, `.md`, `.txt`, or `.html` files and inspect how they are segmented into searchable chunks.
- **Evaluation & Benchmarks Tab**: Execute single benchmark configurations or run a 3-way ablation study comparing **Lexical BM25**, **Semantic N-Gram Vector**, and **Hybrid RRF** side by side.
- **Interactive API Docs**: Visit **http://127.0.0.1:8000/docs** for the Swagger OpenAPI interface.

---

## Command-Line Interface (CLI) Usage

GroundCheck includes a full CLI (`python3 -m groundcheck.cli`) for terminal workflows and CI pipelines:

### Run the Reproducible Evaluation Benchmark
```bash
python3 -m groundcheck.cli eval --mode hybrid --top-k 5 --alpha 0.55
```
Example real output computed by the evaluation engine:
```text
====================================================================================
GROUNDCHECK BENCHMARK EVALUATION REPORT — HYBRID (k=5, α=0.55)
====================================================================================
Run ID                 : eval-cde4fc56
Benchmark Dataset      : groundcheck_eval_v1 (10 cases)
Configuration          : mode=hybrid, top_k=5, alpha=0.55
Execution Mode         : extractive (Actual API Cost: $0.0000, Ref LLM Est: $0.00226)
------------------------------------------------------------------------------------
AGGREGATE METRICS (Calculated from Live Execution):
  Overall Behavior Pass Rate :  100.0%
  Relevant-Evidence Recall@K :  100.0%
  Precision@K (Passage-Level):   87.5%
  Mean Reciprocal Rank (MRR) :  1.000
  NDCG@K                     :  1.000
  Citation Support Precision :  100.0%
  Mean Claim Groundedness    :   95.7%
  Epistemic Abstention Acc.  :  100.0%
  Conflict Detection Rate    :  100.0%
  Latency p50 / p95          : 47.14 ms / 58.13 ms
------------------------------------------------------------------------------------
ID     Category      Expected         Actual Verdict         Recall@K  Ground.  Status
------------------------------------------------------------------------------------
GC-01  factual       answer           grounded                  1.00     0.84   PASS
GC-02  factual       answer           grounded                  1.00     0.88   PASS
GC-03  factual       answer           grounded                  1.00     0.97   PASS
GC-04  factual       answer           grounded                  1.00     0.98   PASS
GC-05  factual       answer           grounded                  1.00     1.00   PASS
GC-06  factual       answer           grounded                  1.00     1.00   PASS
GC-07  multi_hop     answer           grounded                  1.00     0.98   PASS
GC-08  conflicting   report_conflict  conflicted_evidence       1.00     1.00   PASS
GC-09  unanswerable  abstain          insufficient_evidence     1.00     1.00   PASS
GC-10  unanswerable  abstain          insufficient_evidence     1.00     1.00   PASS
====================================================================================
```

### Ask a Question from the Terminal
```bash
python3 -m groundcheck.cli query "What is the configured emergency pump shutdown timeout for the QuadTank-FDI controller?"
```

### Seed or Reset the Built-in Technical Corpus
```bash
python3 -m groundcheck.cli seed
```

---

## Running Automated Tests

Run the full unit and integration test suite using Python's built-in `unittest` runner (or `pytest` if installed):

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

The test suite verifies:
1. **Document Ingestion & Validation (`tests/test_ingestion.py`)**: Markdown section tracking, HTML script stripping, rejection of unsupported file extensions, empty files, binary null-byte files, oversized payloads, corrupt PDFs, and blank image-only PDFs.
2. **Hybrid Retrieval & Verification (`tests/test_retrieval_and_verifier.py`)**: Ranking accuracy across `hybrid`, `bm25`, and `semantic` modes, calibrated low scores and abstention on out-of-domain queries, cross-document quantitative conflict detection (`45 ms` vs. `120 ms`), and the $65\%$ numeric fidelity penalty on fabricated numbers.
3. **API & Evaluation Integration (`tests/test_api_and_evaluation.py`)**: End-to-end FastAPI endpoints (`/api/health`, `/api/documents`, `/api/documents/upload`, `/api/query`, `/api/evaluations/run`), clear `HTTP 503` reporting when external LLM mode is requested without an API key, and live 10-case benchmark verification.

---

## Known Limitations & Honest Engineering Disclosure

1. **Subword/N-Gram TF-IDF vs. Dense Neural Embeddings**: To ensure GroundCheck runs deterministically offline on any machine without multi-gigabyte GPU model downloads or paid embedding APIs, the semantic retriever uses log-weighted TF-IDF vectors over word unigrams, word bigrams, and character 4-grams. While effective for technical terminology and morphological variations, it cannot capture zero-overlap abstract conceptual synonyms as deeply as a neural bi-encoder (e.g., `bge-large` or `text-embedding-3`).
2. **Rule-Based & Lexical-Entailment Verification**: The claim verifier checks content-word recall, contiguous phrase overlap, and exact numerical/unit fidelity rather than running a heavy neural Natural Language Inference (NLI) cross-encoder. Complex logical negations or subtle pragmatic contradictions that do not involve quantitative specifications or explicit negation markers may not be caught automatically.
3. **Scanned / OCR-Required PDFs**: PDF extraction uses `pypdf` on embedded text streams. Scanned image-only PDFs without an OCR text layer are detected and rejected with an explicit error message rather than silently indexing blank pages.

---

## License

Released under the [MIT License](LICENSE).
