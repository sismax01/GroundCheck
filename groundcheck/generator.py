"""
Evidence-Grounded Answer Synthesis Engine for GroundCheck.
Supports two transparent modes:
1. `extractive` (Default): Deterministic, zero-cost local synthesis that selects the highest-salience
   sentences from retrieved passages, attaches verbatim `[chunk_id]` citations, surfaces conflicts
   honestly, and abstains when evidence is insufficient.
2. `llm`: Calls an OpenAI-compatible `/chat/completions` endpoint with strict system instructions
   enforcing inline `[chunk_id]` citations and honest abstention.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

import httpx

from groundcheck.config import Settings
from groundcheck.models import ConflictPair, RetrievedPassage
from groundcheck.retrieval import tokenize_lexical
from groundcheck.verifier import clean_latex_and_markdown, detect_passage_conflicts


def _score_sentence_for_question(
    sentence: str,
    q_tokens: List[str],
    passage_score: float,
) -> float:
    """Rank an individual sentence within a retrieved passage for answering `question`."""
    norm_s = clean_latex_and_markdown(sentence)
    s_tokens = tokenize_lexical(norm_s, remove_stopwords=True)
    if len(s_tokens) < 5 or not q_tokens:
        return 0.0
    q_set = set(q_tokens)
    s_set = set(s_tokens)
    matched = q_set.intersection(s_set)
    if not matched:
        return 0.0

    coverage = len(matched) / float(len(q_set))
    density = len(matched) / float(max(6, len(s_tokens)))

    has_spec = 1.15 if re.search(r"\b\d+(?:\.\d+)?\b", norm_s) else 1.0
    return (0.65 * coverage + 0.20 * passage_score + 0.15 * min(1.0, density * 4.0)) * has_spec


def synthesize_extractive_answer(
    question: str,
    passages: List[RetrievedPassage],
    min_evidence_score: float = 0.18,
) -> Tuple[str, bool]:
    """
    Deterministically synthesize a grounded, cited answer from `passages`.
    Maximizes marginal query-term coverage across up to 4 selected sentences so multi-part
    questions are answered completely.
    Returns `(answer_markdown, is_explicit_abstention)`.
    """
    q_tokens = tokenize_lexical(question, remove_stopwords=True)
    if not passages or not q_tokens:
        return (
            "Insufficient evidence in the uploaded corpus to answer this question. "
            "No relevant passages matched the query terms.",
            True,
        )

    qualifying = [p for p in passages if p.combined_score >= min_evidence_score and len(p.matched_terms) > 0]

    all_matched_terms: Set[str] = set()
    for p in qualifying:
        body_toks = set(tokenize_lexical(clean_latex_and_markdown(p.text), remove_stopwords=True))
        all_matched_terms.update(body_toks.intersection(set(q_tokens)))

    unique_q = set(q_tokens)
    corpus_term_coverage = len(all_matched_terms) / float(max(1, len(unique_q)))

    if not qualifying or (len(unique_q) >= 2 and corpus_term_coverage < 0.34):
        top_score = passages[0].combined_score if passages else 0.0
        return (
            f"Insufficient evidence in the indexed documents to answer this question reliably "
            f"(highest passage relevance score: {top_score:.2f} vs threshold {min_evidence_score:.2f}; "
            f"query term coverage: {corpus_term_coverage * 100:.0f}%). "
            "GroundCheck abstains from generating unsupported claims when primary source evidence is absent.",
            True,
        )

    conflicts: List[ConflictPair] = detect_passage_conflicts(
        question, qualifying, min_relevance=min_evidence_score
    )

    lines: List[str] = []
    if conflicts:
        lines.append(
            "Conflicting evidence detected across the retrieved technical documents regarding this question:"
        )
        for c in conflicts:
            lines.append(f"- {c.excerpt_a} [{c.chunk_id_a}]")
            lines.append(f"- {c.excerpt_b} [{c.chunk_id_b}]")
        lines.append(
            f"Because [{conflicts[0].chunk_id_a}] and [{conflicts[0].chunk_id_b}] report discrepant values, "
            "verify which specification version applies to your deployment environment."
        )
        return "\n".join(lines), False

    candidates: List[Tuple[float, Set[str], str, RetrievedPassage]] = []
    for p in qualifying[:5]:
        raw_sents = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", p.text) if len(s.strip()) >= 28
        ]
        for sent in raw_sents:
            if sent.startswith("#") or set(sent).issubset({"|", "-", ":", " "}):
                continue
            cleaned_sent = re.sub(r"^[-*•]\s+", "", sent).strip()
            # Skip standalone section header lines that lack verbs/punctuation
            if len(cleaned_sent.split()) < 6:
                continue
            if cleaned_sent.endswith(":"):
                continue
            if not cleaned_sent.endswith((".", "!", "?")):
                cleaned_sent += "."
            sc = _score_sentence_for_question(cleaned_sent, q_tokens, p.combined_score)
            if sc > 0.16:
                s_q_matches = set(
                    tokenize_lexical(clean_latex_and_markdown(cleaned_sent), remove_stopwords=True)
                ).intersection(unique_q)
                candidates.append((sc, s_q_matches, cleaned_sent, p))

    if not candidates:
        return (
            "Insufficient evidence: although passages shared isolated keywords, no passage contained "
            "a substantive statement addressing the question.",
            True,
        )

    # Greedy submodular / maximal marginal coverage selection across up to 4 bullets
    selected_bullets: List[str] = []
    covered_q_terms: Set[str] = set()
    seen_fingerprints: Set[Tuple[str, ...]] = set()
    remaining = list(candidates)

    while remaining and len(selected_bullets) < 4:
        best_idx = -1
        best_gain = -1.0
        for idx, (base_sc, s_q_matches, sent_text, _) in enumerate(remaining):
            fp = tuple(sorted(set(tokenize_lexical(clean_latex_and_markdown(sent_text), remove_stopwords=True))))
            if not fp or fp in seen_fingerprints:
                continue
            new_terms = s_q_matches - covered_q_terms
            marginal_bonus = 0.55 * (len(new_terms) / float(max(1, len(unique_q))))
            gain = base_sc + marginal_bonus
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx < 0:
            break

        base_sc, s_q_matches, sent_text, passage = remaining.pop(best_idx)
        # Stop adding bullets once all matched query terms are covered and we have at least 2 bullets
        if len(selected_bullets) >= 3 and len(s_q_matches - covered_q_terms) == 0 and base_sc < 0.32:
            break

        fp = tuple(sorted(set(tokenize_lexical(clean_latex_and_markdown(sent_text), remove_stopwords=True))))
        seen_fingerprints.add(fp)
        covered_q_terms.update(s_q_matches)
        selected_bullets.append(f"- {sent_text} [{passage.chunk_id}]")

    return "\n".join(selected_bullets), False


def generate_llm_answer(
    question: str,
    passages: List[RetrievedPassage],
    settings: Settings,
    min_evidence_score: float = 0.18,
) -> Tuple[str, bool, str]:
    """
    Call an external OpenAI-compatible chat completion API to synthesize a cited answer.
    """
    if not settings.llm_api_key:
        raise RuntimeError(
            "External LLM mode requested, but GROUNDCHECK_LLM_API_KEY is not configured. "
            "Set GROUNDCHECK_LLM_API_KEY in your .env file or switch to 'extractive' mode."
        )

    qualifying = [p for p in passages if p.combined_score >= min_evidence_score]
    if not qualifying:
        return (
            "Insufficient evidence in the indexed documents to answer this question reliably.",
            True,
            "external_llm",
        )

    context_blocks = []
    for p in qualifying[:6]:
        context_blocks.append(
            f"Passage ID: [{p.chunk_id}]\n"
            f"Document: {p.filename} (Page {p.page_number or 1}, Section: {p.section_title})\n"
            f"Content:\n{p.text}"
        )
    context_str = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are GroundCheck, a rigorous evidence-grounded technical research assistant.\n"
        "Rules:\n"
        "1. Answer ONLY using facts explicitly stated in the provided passages.\n"
        "2. Every factual sentence MUST end with an inline citation in the exact format [doc_id::ch-XXX].\n"
        "3. If the passages conflict, explicitly report both conflicting values with their respective citations.\n"
        "4. If the passages do not contain enough evidence to answer, state clearly: "
        "'Insufficient evidence in the uploaded corpus to answer this question.' Do not guess or extrapolate."
    )
    user_prompt = f"Question: {question}\n\nRetrieved Evidence Passages:\n{context_str}"

    url = f"{settings.llm_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }

    try:
        with httpx.Client(timeout=float(settings.llm_timeout_seconds)) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"External LLM API request failed ({url}): {exc}") from exc

    is_abstention = "insufficient evidence" in content.lower()
    return content, is_abstention, f"external_llm ({settings.llm_model})"
