# Hybrid Retrieval and Claim Verification Methodology

## 1. Lexical vs. Dense/Subword Semantic Retrieval Trade-offs
Information retrieval systems for technical documentation face a fundamental tension between exact lexical matching and semantic generalization:
- **Lexical Retrieval (Okapi BM25)**: Excels at matching exact error codes (`HTTP 409`, `ERR_MODBUS_CRC`), chemical formulas, function identifiers (`hmac.compare_digest`), and rare domain acronyms (`CUSUM`, `GLRT`, `EKF`). Parameterized by term-frequency saturation $k_1 = 1.5$ and document-length normalization $b = 0.75$, BM25 prevents long boilerplate documents from dominating short, high-precision passages.
- **Subword & N-Gram Semantic Vector Similarity**: Pure word-level BM25 fails when queries use morphological variations or paraphrases. By projecting passages into a log-weighted TF-IDF vector space over word unigrams, word bigrams, and character $4$-grams, the retriever captures subword roots and local phrase structure via cosine similarity.

## 2. Reciprocal Rank Fusion (RRF) and Score Calibration
To combine lexical and semantic signals without sensitivity to uncalibrated score scales, GroundCheck computes both a convex linear combination ($\alpha \cdot S_{\text{BM25}} + (1 - \alpha) \cdot S_{\text{sem}}$) and Reciprocal Rank Fusion:
$$\text{RRF}(d) = \sum_{m \in \{\text{lex}, \text{sem}\}} w_m \frac{1}{k_{\text{rrf}} + \text{rank}_m(d)}$$
with rank constant $k_{\text{rrf}} = 20$. Crucially, relative min-max normalization across a single query's results is avoided because dividing by the maximum score of an unrelated query artificially inflates garbage results to $1.0$. Instead, GroundCheck normalizes against an absolute ideal saturation ceiling and applies a query-term coverage multiplier.

## 3. Claim-Level Groundedness Auditing and Numeric Fidelity
Generated answers are decomposed into atomic sentence claims. Each claim $c_i$ is audited against its explicitly cited passages $[d_j]$ along three axes:
1. **Content Token Entailment**: Measures the fraction of non-stopword normalized terms in $c_i$ present in passage $d_j$.
2. **Contiguous Phrase & Sentence Alignment**: Identifies the single best supporting sentence inside $d_j$ using unigram and bigram overlap.
3. **Numeric & Unit Fidelity**: Extracts all quantitative literals from $c_i$ and verifies exact presence in $d_j$. Claims introducing unverified numbers suffer a $65\%$ grounding penalty.

## 4. Epistemic Abstention and Conflict Surface Policy
When the highest calibrated passage score falls below the minimum evidence threshold ($\tau_{\text{min}} = 0.18$) or corpus-wide query term coverage falls below $34\%$, the synthesizer must abstain rather than extrapolate. When two relevant passages ($\ge \tau_{\text{min}}$) state incompatible numerical values for the same technical attribute and unit, the verifier flags a `conflicted_evidence` verdict and presents both cited excerpts side by side.
