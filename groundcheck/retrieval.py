"""
Hybrid Information Retrieval Engine for GroundCheck.
Combines exact lexical matching (Okapi BM25) with subword/word-ngram TF-IDF semantic
cosine similarity and Reciprocal Rank Fusion (RRF), plus calibrated absolute relevance scoring.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np

from groundcheck.models import ChunkModel, RetrievedPassage


STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your",
    "yours", "yourself", "yourselves", "explain", "describe", "tell", "does",
    "according", "document", "documents", "system",
}

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[-_./][a-zA-Z0-9]+)*")


def normalize_token(tok: str) -> str:
    """Lightweight morphological stemmer preserving technical acronyms and identifiers."""
    t = tok.lower().strip("-_./")
    if len(t) <= 3 or any(ch.isdigit() for ch in t):
        return t
    if t.endswith("ies") and len(t) > 5:
        return t[:-3] + "y"
    if t.endswith("ing") and len(t) > 6:
        return t[:-3]
    if t.endswith("ed") and len(t) > 4:
        return t[:-2]
    if t.endswith("es") and len(t) > 5 and not t.endswith("ses"):
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 4:
        return t[:-1]
    return t


def tokenize_lexical(text: str, remove_stopwords: bool = True) -> List[str]:
    """Extract normalized lexical tokens for BM25 indexing and exact term matching."""
    raw_tokens = _TOKEN_RE.findall(text.lower())
    out: List[str] = []
    for raw in raw_tokens:
        if remove_stopwords and raw in STOPWORDS:
            continue
        norm = normalize_token(raw)
        if len(norm) >= 2 and (not remove_stopwords or norm not in STOPWORDS):
            out.append(norm)
        # Also split hyphenated/slashed compounds so sub-parts match (e.g. "sha-256" -> "sha", "256")
        if any(sep in raw for sep in ("-", "_", "/")):
            for part in re.split(r"[-_/]", raw):
                p_norm = normalize_token(part)
                if len(p_norm) >= 2 and p_norm != norm and (not remove_stopwords or p_norm not in STOPWORDS):
                    out.append(p_norm)
    return out


def extract_semantic_features(text: str) -> Counter:
    """
    Extract word unigrams, word bigrams, and character 4-grams for semantic vector representation.
    This captures morphological variations, technical terminology, and phrase paraphrases.
    """
    tokens = tokenize_lexical(text, remove_stopwords=True)
    counts: Counter = Counter()
    for tok in tokens:
        counts[f"w:{tok}"] += 1.5
        # Character 4-grams inside content words capture semantic root similarity
        if len(tok) >= 4:
            padded = f"<{tok}>"
            for i in range(len(padded) - 3):
                counts[f"c:{padded[i:i+4]}"] += 0.35
    # Word bigrams capture local phrase structure
    for i in range(len(tokens) - 1):
        counts[f"b:{tokens[i]}_{tokens[i+1]}"] += 1.25
    return counts


class HybridRetriever:
    """
    Indexes a collection of ChunkModels and executes configurable BM25, Semantic, or Hybrid search.
    """

    def __init__(self, chunks: List[ChunkModel], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.n_docs = len(chunks)

        self.doc_tokens: List[List[str]] = []
        self.doc_token_counts: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.df_lexical: Counter = Counter()

        self.semantic_counters: List[Counter] = []
        self.df_semantic: Counter = Counter()

        self._build_index()

    def _build_index(self) -> None:
        if self.n_docs == 0:
            self.avgdl = 1.0
            return

        for chunk in self.chunks:
            # Weight section title and document title slightly higher in chunk representation
            enriched_text = f"{chunk.doc_title} {chunk.section_title}\n{chunk.text}"
            toks = tokenize_lexical(enriched_text, remove_stopwords=True)
            self.doc_tokens.append(toks)
            cnt = Counter(toks)
            self.doc_token_counts.append(cnt)
            self.doc_lengths.append(len(toks))
            for t in cnt.keys():
                self.df_lexical[t] += 1

            sem_cnt = extract_semantic_features(enriched_text)
            self.semantic_counters.append(sem_cnt)
            for f in sem_cnt.keys():
                self.df_semantic[f] += 1

        self.avgdl = max(1.0, sum(self.doc_lengths) / float(self.n_docs))

    def _compute_bm25_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if not query_tokens or self.n_docs == 0:
            return scores

        q_counts = Counter(query_tokens)
        for term, q_freq in q_counts.items():
            df = self.df_lexical.get(term, 0)
            if df == 0:
                continue
            # Robertson-Sparck Jones IDF with smoothing
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for idx in range(self.n_docs):
                tf = self.doc_token_counts[idx].get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lengths[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[idx] += idf * ((tf * (self.k1 + 1.0)) / denom) * (1.0 + 0.1 * (q_freq - 1))
        return scores

    def _compute_semantic_scores(self, query_text: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if not query_text.strip() or self.n_docs == 0:
            return scores

        q_feats = extract_semantic_features(query_text)
        if not q_feats:
            return scores

        # Compute query TF-IDF weights and norm
        q_weights: Dict[str, float] = {}
        q_norm_sq = 0.0
        for feat, val in q_feats.items():
            df = self.df_semantic.get(feat, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + self.n_docs / float(df))
            w = (1.0 + math.log(val)) * idf
            q_weights[feat] = w
            q_norm_sq += w * w

        if q_norm_sq <= 1e-12:
            return scores
        q_norm = math.sqrt(q_norm_sq)

        for idx in range(self.n_docs):
            doc_feats = self.semantic_counters[idx]
            if not doc_feats:
                continue
            dot = 0.0
            d_norm_sq = 0.0
            for feat, val in doc_feats.items():
                df = self.df_semantic.get(feat, 1)
                idf = math.log(1.0 + self.n_docs / float(df))
                dw = (1.0 + math.log(val)) * idf
                d_norm_sq += dw * dw
                if feat in q_weights:
                    dot += q_weights[feat] * dw
            if d_norm_sq > 1e-12 and dot > 0.0:
                scores[idx] = dot / (q_norm * math.sqrt(d_norm_sq))

        return scores

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        alpha: float = 0.55,
    ) -> List[RetrievedPassage]:
        """
        Retrieve top-K passages with calibrated relevance scores in [0.0, 1.0].
        - mode='bm25': Pure lexical BM25 ranking
        - mode='semantic': Pure subword/ngram TF-IDF cosine ranking
        - mode='hybrid': Blended lexical + semantic + Reciprocal Rank Fusion (RRF)
        """
        if self.n_docs == 0 or not query or not query.strip():
            return []

        q_tokens = tokenize_lexical(query, remove_stopwords=True)
        unique_q_tokens = list(dict.fromkeys(q_tokens))

        raw_bm25 = self._compute_bm25_scores(q_tokens)
        raw_sem = self._compute_semantic_scores(query)

        # Calibrated absolute normalization so unanswerable queries get genuinely low scores
        # (never divide by a tiny max score to make a 0.01 match look like 1.0!)
        ideal_bm25_denom = max(3.5, len(unique_q_tokens) * 1.65, float(np.max(raw_bm25)) if self.n_docs else 3.5)
        norm_bm25 = np.clip(raw_bm25 / ideal_bm25_denom, 0.0, 1.0)

        # Calibrate cosine similarity: typical strong passage match is ~0.35-0.65 cosine
        norm_sem = np.clip(raw_sem / max(0.42, float(np.max(raw_sem)) if self.n_docs else 0.42), 0.0, 1.0)
        # Apply absolute ceiling based on raw cosine so unrelated queries stay near 0
        for i in range(self.n_docs):
            if raw_sem[i] < 0.25:
                norm_sem[i] = min(norm_sem[i], raw_sem[i] * 2.2)

        # Compute RRF across both rankings
        bm25_order = np.argsort(-raw_bm25)
        sem_order = np.argsort(-raw_sem)
        bm25_ranks = np.empty(self.n_docs, dtype=np.int32)
        sem_ranks = np.empty(self.n_docs, dtype=np.int32)
        for rank_idx, doc_idx in enumerate(bm25_order, start=1):
            bm25_ranks[doc_idx] = rank_idx
        for rank_idx, doc_idx in enumerate(sem_order, start=1):
            sem_ranks[doc_idx] = rank_idx

        rrf_k = 20.0
        raw_rrf = np.zeros(self.n_docs, dtype=np.float64)
        for i in range(self.n_docs):
            if raw_bm25[i] > 0 or raw_sem[i] > 0:
                r_lex = 1.0 / (rrf_k + bm25_ranks[i]) if raw_bm25[i] > 0 else 0.0
                r_sem = 1.0 / (rrf_k + sem_ranks[i]) if raw_sem[i] > 0 else 0.0
                raw_rrf[i] = (alpha * r_lex + (1.0 - alpha) * r_sem) * (rrf_k + 1.0)

        # Exact query term coverage boost
        coverage_scores = np.zeros(self.n_docs, dtype=np.float64)
        matched_terms_per_doc: List[List[str]] = [[] for _ in range(self.n_docs)]
        if unique_q_tokens:
            for i in range(self.n_docs):
                body_tokens = set(tokenize_lexical(self.chunks[i].text, remove_stopwords=True))
                all_doc_tokens = set(self.doc_tokens[i])
                matched = [t for t in unique_q_tokens if t in all_doc_tokens]
                body_matched = [t for t in unique_q_tokens if t in body_tokens]
                matched_terms_per_doc[i] = matched
                # Weight matches in actual passage body higher than title-only matches
                coverage_scores[i] = (
                    0.75 * (len(body_matched) / float(len(unique_q_tokens)))
                    + 0.25 * (len(matched) / float(len(unique_q_tokens)))
                )

        mode_clean = mode.lower().strip()
        combined = np.zeros(self.n_docs, dtype=np.float64)
        for i in range(self.n_docs):
            if mode_clean == "bm25":
                base = 0.75 * norm_bm25[i] + 0.25 * coverage_scores[i]
            elif mode_clean == "semantic":
                base = 0.80 * norm_sem[i] + 0.20 * coverage_scores[i]
            else:
                linear_blend = alpha * norm_bm25[i] + (1.0 - alpha) * norm_sem[i]
                base = 0.65 * linear_blend + 0.25 * coverage_scores[i] + 0.10 * raw_rrf[i]

            # Penalize passages that have zero lexical term overlap if query has >=2 distinct terms
            if len(unique_q_tokens) >= 2 and len(matched_terms_per_doc[i]) == 0:
                base *= 0.35

            combined[i] = float(np.clip(base, 0.0, 1.0))

        top_indices = np.argsort(-combined)[: max(1, top_k)]
        results: List[RetrievedPassage] = []

        for rank_num, idx in enumerate(top_indices, start=1):
            c = self.chunks[int(idx)]
            score = round(float(combined[idx]), 4)
            if score >= 0.52:
                tier = "High"
            elif score >= 0.30:
                tier = "Moderate"
            elif score >= 0.18:
                tier = "Low"
            else:
                tier = "Marginal"

            results.append(
                RetrievedPassage(
                    rank=rank_num,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    filename=c.filename,
                    doc_title=c.doc_title,
                    page_number=c.page_number,
                    section_title=c.section_title,
                    text=c.text,
                    bm25_score=round(float(norm_bm25[idx]), 4),
                    semantic_score=round(float(norm_sem[idx]), 4),
                    rrf_score=round(float(raw_rrf[idx]), 4),
                    combined_score=score,
                    matched_terms=matched_terms_per_doc[int(idx)],
                    relevance_tier=tier,
                )
            )

        return results
