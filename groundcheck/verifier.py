"""
Groundedness Verification, Citation Auditing, and Contradiction Detection Engine.
Evaluates whether claims in an answer are genuinely supported by their cited passages,
detects conflicting specifications across passages, and calibrates epistemic certainty.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from groundcheck.models import (
    ClaimVerification,
    ConflictPair,
    RetrievedPassage,
    VerificationReport,
)
from groundcheck.retrieval import tokenize_lexical


_CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9_\-]+::ch-\d{3}|S\d+)\]")

_NEGATION_WORDS = {"not", "never", "no", "cannot", "disabled", "prohibited", "unsupported", "false"}


def clean_latex_and_markdown(text: str) -> str:
    """Normalize LaTeX math wrappers (`\text{ ms}`, `$`) so unit and numeric regexes match cleanly."""
    cleaned = re.sub(r"\\text\{\s*([^}]*)\s*\}", r" \1 ", text)
    cleaned = cleaned.replace("$", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Split markdown answer into evaluable sentences/bullet claims while keeping trailing
    `[doc_id::ch-XXX]` citations attached to their sentence.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    claims: List[str] = []
    for line in lines:
        if line.startswith("#"):
            continue
        cleaned = re.sub(r"^[-*•]\s+", "", line).strip()
        if not cleaned:
            continue
        # Only split on sentence punctuation when followed by an uppercase letter (not a `[` citation!)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)
        for p in parts:
            p_str = p.strip()
            if len(p_str) >= 12 and not _CITATION_PATTERN.fullmatch(p_str):
                claims.append(p_str)
    return claims


def extract_cited_ids(claim_text: str, passage_lookup: Dict[str, RetrievedPassage]) -> List[str]:
    """
    Extract cited chunk_ids from a claim. Supports both `[doc-id::ch-001]` and shorthand `[S1]`.
    """
    raw_matches = _CITATION_PATTERN.findall(claim_text)
    resolved: List[str] = []
    rank_map = {f"S{p.rank}": p.chunk_id for p in passage_lookup.values()}
    for m in raw_matches:
        if m in passage_lookup:
            if m not in resolved:
                resolved.append(m)
        elif m in rank_map:
            cid = rank_map[m]
            if cid not in resolved:
                resolved.append(cid)
    return resolved


def strip_citations(claim_text: str) -> str:
    """Remove citation brackets and markdown formatting for clean lexical/semantic comparison."""
    no_cites = _CITATION_PATTERN.sub("", claim_text)
    no_md = re.sub(r"[*_`~]+", "", no_cites)
    return clean_latex_and_markdown(no_md)


def compute_claim_passage_support(claim_text: str, passage_text: str) -> Tuple[float, Optional[str]]:
    """
    Measure how well `passage_text` entails/supports `claim_text` in [0.0, 1.0].
    Also returns the best supporting verbatim sentence from `passage_text`.
    """
    clean_claim = strip_citations(claim_text)
    claim_tokens = tokenize_lexical(clean_claim, remove_stopwords=True)
    if not claim_tokens:
        return 0.0, None

    norm_passage = clean_latex_and_markdown(passage_text)
    passage_tokens_set = set(tokenize_lexical(norm_passage, remove_stopwords=True))
    if not passage_tokens_set:
        return 0.0, None

    # 1. Token recall of claim content words inside the passage
    matched_tokens = [t for t in claim_tokens if t in passage_tokens_set]
    token_recall = len(matched_tokens) / float(len(claim_tokens))

    # 2. Numeric / identifier fidelity check: if a claim states numbers, verify they appear in the passage
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", clean_claim))
    passage_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", norm_passage))
    numeric_penalty = 1.0
    if claim_numbers:
        matched_nums = claim_numbers.intersection(passage_numbers)
        if len(matched_nums) < len(claim_numbers):
            numeric_penalty = 0.35 + 0.65 * (len(matched_nums) / float(len(claim_numbers)))

    # 3. Best sentence overlap inside the passage
    passage_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", passage_text) if len(s.strip()) >= 10
    ]
    best_sent = None
    best_sent_score = 0.0
    claim_bigrams = {
        (claim_tokens[i], claim_tokens[i + 1]) for i in range(len(claim_tokens) - 1)
    }

    for sent in passage_sentences:
        s_tokens = tokenize_lexical(clean_latex_and_markdown(sent), remove_stopwords=True)
        if not s_tokens:
            continue
        s_set = set(s_tokens)
        overlap = sum(1 for t in claim_tokens if t in s_set) / float(len(claim_tokens))
        s_bigrams = {(s_tokens[i], s_tokens[i + 1]) for i in range(len(s_tokens) - 1)}
        bg_overlap = (
            len(claim_bigrams.intersection(s_bigrams)) / float(len(claim_bigrams))
            if claim_bigrams
            else overlap
        )
        score = 0.70 * overlap + 0.30 * bg_overlap
        if score > best_sent_score:
            best_sent_score = score
            best_sent = sent

    combined_support = (0.55 * token_recall + 0.45 * best_sent_score) * numeric_penalty
    return round(float(min(1.0, max(0.0, combined_support))), 4), best_sent


def detect_passage_conflicts(
    question: str,
    passages: List[RetrievedPassage],
    min_relevance: float = 0.18,
) -> List[ConflictPair]:
    """
    Inspect relevant retrieved passages for conflicting numeric specifications on the queried topic.
    """
    relevant = [p for p in passages if p.combined_score >= min_relevance]
    if len(relevant) < 2:
        return []

    q_tokens = set(tokenize_lexical(question, remove_stopwords=True))
    conflicts: List[ConflictPair] = []

    passage_specs: List[Tuple[RetrievedPassage, str, Set[str], Set[Tuple[str, str]], bool]] = []
    for p in relevant:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", p.text) if len(s.strip()) >= 15]
        for sent in sentences:
            norm_sent = clean_latex_and_markdown(sent)
            s_tokens = set(tokenize_lexical(norm_sent, remove_stopwords=True))
            min_shared = 1 if len(q_tokens) <= 2 else 2
            if len(s_tokens.intersection(q_tokens)) < min_shared:
                continue

            num_units = set(
                re.findall(
                    r"\b(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|seconds|minutes|hours|%|percent|mb|gb|hz|bar)\b",
                    norm_sent.lower(),
                )
            )
            has_neg = bool(
                set(re.findall(r"[a-z]+", norm_sent.lower())).intersection(_NEGATION_WORDS)
            )
            if num_units:
                passage_specs.append((p, sent, s_tokens, num_units, has_neg))

    seen_pairs: Set[Tuple[str, str]] = set()
    for i in range(len(passage_specs)):
        p_a, sent_a, toks_a, nums_a, _ = passage_specs[i]
        for j in range(i + 1, len(passage_specs)):
            p_b, sent_b, toks_b, nums_b, _ = passage_specs[j]
            if p_a.chunk_id == p_b.chunk_id:
                continue
            pair_key = tuple(sorted([p_a.chunk_id, p_b.chunk_id]))
            if pair_key in seen_pairs:
                continue

            shared_subject = toks_a.intersection(toks_b).intersection(q_tokens)
            overall_shared = toks_a.intersection(toks_b)
            # Require strong subject overlap so distinct attributes don't false-trigger
            if len(shared_subject) < 2 or len(overall_shared) < 4:
                continue

            for val_a, unit_a in nums_a:
                for val_b, unit_b in nums_b:
                    norm_u_a = "ms" if unit_a in ("ms", "milliseconds") else (
                        "s" if unit_a in ("s", "sec", "seconds") else unit_a
                    )
                    norm_u_b = "ms" if unit_b in ("ms", "milliseconds") else (
                        "s" if unit_b in ("s", "sec", "seconds") else unit_b
                    )
                    if norm_u_a == norm_u_b and val_a != val_b:
                        seen_pairs.add(pair_key)
                        conflicts.append(
                            ConflictPair(
                                chunk_id_a=p_a.chunk_id,
                                source_a=f"{p_a.filename} ({p_a.section_title})",
                                excerpt_a=sent_a,
                                chunk_id_b=p_b.chunk_id,
                                source_b=f"{p_b.filename} ({p_b.section_title})",
                                excerpt_b=sent_b,
                                conflict_description=(
                                    f"Conflicting specification detected: '{p_a.filename}' states "
                                    f"{val_a} {unit_a}, whereas '{p_b.filename}' states {val_b} {unit_b}."
                                ),
                            )
                        )
                        break

    return conflicts


def verify_answer(
    question: str,
    answer_markdown: str,
    passages: List[RetrievedPassage],
    min_evidence_score: float = 0.18,
    explicit_abstention: bool = False,
) -> VerificationReport:
    """
    Audit every claim in `answer_markdown` against `passages` and produce a full `VerificationReport`.
    """
    passage_lookup = {p.chunk_id: p for p in passages}
    conflicts = detect_passage_conflicts(question, passages, min_relevance=min_evidence_score)
    raw_claims = split_into_sentences(answer_markdown)

    claim_verifications: List[ClaimVerification] = []
    total_citations_made = 0
    valid_citations_count = 0
    substantive_scores: List[float] = []
    supported_count = 0

    for idx, c_text in enumerate(raw_claims, start=1):
        lower_c = c_text.lower()
        is_abstention_line = explicit_abstention or any(
            phrase in lower_c
            for phrase in (
                "insufficient evidence",
                "cannot answer",
                "do not contain sufficient",
                "no supporting passages",
                "not mentioned in the uploaded",
                "abstains from generating",
            )
        )
        is_conflict_header = any(
            phrase in lower_c
            for phrase in (
                "conflicting evidence detected",
                "discrepancy between sources",
                "sources disagree",
                "report discrepant values",
            )
        )

        cited_ids = extract_cited_ids(c_text, passage_lookup)

        if is_abstention_line and not cited_ids:
            claim_verifications.append(
                ClaimVerification(
                    claim_index=idx,
                    claim_text=c_text,
                    cited_chunk_ids=[],
                    status="abstention_notice",
                    support_score=1.0,
                    best_supporting_chunk_id=None,
                    best_supporting_quote=None,
                    explanation="Transparent abstention notice communicating insufficient evidence in corpus.",
                )
            )
            continue

        if is_conflict_header and conflicts:
            claim_verifications.append(
                ClaimVerification(
                    claim_index=idx,
                    claim_text=c_text,
                    cited_chunk_ids=cited_ids,
                    status="conflicted",
                    support_score=1.0,
                    best_supporting_chunk_id=conflicts[0].chunk_id_a,
                    best_supporting_quote=conflicts[0].conflict_description,
                    explanation="Explicitly surfaces a verified discrepancy across retrieved source documents.",
                )
            )
            for cid in cited_ids:
                total_citations_made += 1
                valid_citations_count += 1
            supported_count += 1
            substantive_scores.append(1.0)
            continue

        candidate_passages = (
            [passage_lookup[cid] for cid in cited_ids if cid in passage_lookup]
            if cited_ids
            else passages
        )

        best_score = 0.0
        best_cid: Optional[str] = None
        best_quote: Optional[str] = None

        for cand in candidate_passages:
            score, quote = compute_claim_passage_support(c_text, cand.text)
            if score > best_score:
                best_score = score
                best_cid = cand.chunk_id
                best_quote = quote

        for cid in cited_ids:
            total_citations_made += 1
            if cid in passage_lookup:
                c_score, _ = compute_claim_passage_support(c_text, passage_lookup[cid].text)
                if c_score >= 0.38:
                    valid_citations_count += 1

        if best_score >= 0.55 and cited_ids:
            status = "supported"
            support_val = best_score
            explanation = f"Directly entailed by cited passage [{best_cid}] (support score: {best_score:.2f})."
            supported_count += 1
            substantive_scores.append(support_val)
        elif best_score >= 0.35:
            status = "partially_supported"
            support_val = best_score
            cite_note = f"cited [{best_cid}]" if cited_ids else f"uncited passage [{best_cid}]"
            explanation = f"Partially supported by {cite_note} (support score: {best_score:.2f})."
            substantive_scores.append(support_val)
        else:
            status = "unsupported"
            support_val = best_score
            explanation = "Claim lacks sufficient lexical or factual grounding in the retrieved passages."
            substantive_scores.append(support_val)

        claim_verifications.append(
            ClaimVerification(
                claim_index=idx,
                claim_text=c_text,
                cited_chunk_ids=cited_ids,
                status=status,
                support_score=round(support_val, 4),
                best_supporting_chunk_id=best_cid,
                best_supporting_quote=best_quote,
                explanation=explanation,
            )
        )

    top_passage_score = passages[0].combined_score if passages else 0.0

    if explicit_abstention or top_passage_score < min_evidence_score:
        verdict = "insufficient_evidence"
        overall_groundedness = 1.0 if explicit_abstention else 0.0
        citation_prec = 1.0 if total_citations_made == 0 else (valid_citations_count / float(total_citations_made))
        epistemic_note = (
            f"Abstained from answering: highest retrieved passage relevance ({top_passage_score:.2f}) "
            f"fell below the required evidence threshold ({min_evidence_score:.2f}) or lacked query term coverage."
        )
    elif conflicts:
        verdict = "conflicted_evidence"
        overall_groundedness = (
            sum(substantive_scores) / float(len(substantive_scores)) if substantive_scores else 0.0
        )
        citation_prec = (
            valid_citations_count / float(total_citations_made) if total_citations_made > 0 else 1.0
        )
        epistemic_note = (
            f"Detected {len(conflicts)} conflicting specification(s) across retrieved documents. "
            "Both competing passages are cited below so you can inspect the discrepancy directly."
        )
    else:
        overall_groundedness = (
            sum(substantive_scores) / float(len(substantive_scores)) if substantive_scores else 0.0
        )
        citation_prec = (
            valid_citations_count / float(total_citations_made) if total_citations_made > 0 else 0.0
        )
        if overall_groundedness >= 0.62 and citation_prec >= 0.70:
            verdict = "grounded"
            epistemic_note = (
                f"Answer is strongly grounded ({overall_groundedness * 100:.1f}% claim support, "
                f"{citation_prec * 100:.1f}% citation precision) across {len(passages)} retrieved passages."
            )
        elif overall_groundedness >= 0.35:
            verdict = "partially_grounded"
            epistemic_note = (
                f"Answer has moderate grounding ({overall_groundedness * 100:.1f}% claim support). "
                "Verify individual partially-supported claims against the source excerpts."
            )
        else:
            verdict = "insufficient_evidence"
            epistemic_note = "Generated claims could not be verified against the retrieved passages."

    return VerificationReport(
        verdict=verdict,
        overall_groundedness_score=round(float(overall_groundedness), 4),
        citation_precision=round(float(citation_prec), 4),
        supported_claim_count=supported_count,
        total_substantive_claims=len(substantive_scores),
        has_conflict=len(conflicts) > 0,
        conflicts=conflicts,
        claims=claim_verifications,
        epistemic_note=epistemic_note,
    )
