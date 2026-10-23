# GroundCheck — Architecture & Mathematical Specification

## 1. System Architecture Overview

GroundCheck is structured as a modular, layered Python system with a zero-build single-page web interface:

```text
+-------------------------------------------------------------------------+
|                    Web Workbench UI (HTML5 / CSS3 / ES6)                |
|  [Research & Q&A]     [Document Corpus & Chunks]   [Eval & Benchmarks]  |
+------------------------------------+------------------------------------+
                                     | REST JSON API
+------------------------------------v------------------------------------+
|                     FastAPI Layer (groundcheck/api.py)                  |
|         Input Validation, Error Handling, Multipart Upload, CORS        |
+------------------------------------+------------------------------------+
                                     |
+------------------------------------v------------------------------------+
|               Orchestration Service (groundcheck/service.py)            |
+-------+--------------------+--------------------+---------------+-------+
        |                    |                    |               |
        v                    v                    v               v
+---------------+  +------------------+  +----------------+  +------------+
| Ingestion &   |  | HybridRetriever  |  | Synthesizer &  |  | Benchmark  |
| Section Chunk |  | BM25 + Subword   |  | Claim Verifier |  | Evaluator  |
| (ingestion.py)|  | TF-IDF + RRF     |  | (generator.py, |  | (evaluator |
|               |  | (retrieval.py)   |  |  verifier.py)  |  |  .py)      |
+-------+-------+  +--------+---------+  +--------+-------+  +-----+------+
        |                   |                     |                |
        +-------------------+----------+----------+----------------+
                                       |
                        +--------------v--------------+
                        |  SQLite3 Persistence Layer  |
                        |  (groundcheck/database.py)  |
                        +-----------------------------+
```

---

## 2. Hybrid Retrieval & Absolute Score Calibration

### 2.1 Okapi BM25 Lexical Scoring
Given a tokenized query $Q = \{q_1, \dots, q_m\}$ and a document chunk $d$ with length $|d|$ and corpus average chunk length $\text{avgdl}$, GroundCheck computes:
$$\text{BM25}(Q, d) = \sum_{t \in Q} \ln\left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right) \cdot \frac{\text{tf}(t, d) \cdot (k_1 + 1)}{\text{tf}(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$
where $k_1 = 1.5$ and $b = 0.75$.

### 2.2 Subword & N-Gram TF-IDF Semantic Similarity
To capture morphological variants and phrase structure without external neural weights, each passage is mapped to a sparse feature vector $\mathbf{v}(d)$ comprising:
- Normalized word unigrams (weight $1.5$)
- Contiguous word bigrams $w_i\_w_{i+1}$ (weight $1.25$)
- Padded character $4$-grams `<tok>` (weight $0.35$)

Log-normalized TF-IDF cosine similarity is computed between $\mathbf{v}(Q)$ and $\mathbf{v}(d)$.

### 2.3 Reciprocal Rank Fusion (RRF) & Calibration
Rather than normalizing by the maximum score of the current query (which inflates irrelevant results on out-of-domain queries), GroundCheck normalizes BM25 against an ideal saturation ceiling $\max(3.5, 1.65 \cdot |Q_{\text{unique}}|)$ and blends lexical, semantic, query-term coverage, and RRF ($k_{\text{rrf}} = 20$):
$$S_{\text{hybrid}}(Q, d) = 0.65 \left(\alpha \hat{S}_{\text{BM25}} + (1 - \alpha) \hat{S}_{\text{sem}}\right) + 0.25\,\text{Coverage}(Q, d) + 0.10\,\text{RRF}(Q, d)$$

---

## 3. Claim Verification & Epistemic Calibration

Each sentence $c_i$ in an answer is verified against its cited passage $d_j$:
1. **Token Entailment Recall**: $R_{\text{tok}}(c_i, d_j) = \frac{|\text{tokens}(c_i) \cap \text{tokens}(d_j)|}{|\text{tokens}(c_i)|}$
2. **Best Sentence Overlap**: $S_{\text{sent}}(c_i, d_j) = \max_{s \in d_j} \left(0.70\,R_{\text{tok}}(c_i, s) + 0.30\,R_{\text{bigram}}(c_i, s)\right)$
3. **Numeric Fidelity Multiplier**: $\eta_{\text{num}} = 1.0$ if all numbers in $c_i$ appear in $d_j$, else $0.35 + 0.65 \frac{|\text{nums}(c_i) \cap \text{nums}(d_j)|}{|\text{nums}(c_i)|}$
4. **Support Score**: $\text{Support}(c_i, d_j) = \left(0.55\,R_{\text{tok}} + 0.45\,S_{\text{sent}}\right) \cdot \eta_{\text{num}}$
