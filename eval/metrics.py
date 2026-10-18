"""Deterministic (non-LLM) metrics: ranking, short-answer matching and abstention detection."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Iterable

# ======================================================================================
# Ranking metrics (document level) - pytrec_eval conventions: linear gain, log2(rank+1) discount
# ======================================================================================
RANK_METRIC_NAMES = (
    [f"hit@{k}" for k in (1, 3, 5, 8)]
    + [f"recall@{k}" for k in (5, 10, 100)]
    + ["precision@5", "mrr", "map", "ndcg@5", "ndcg@10", "evidence_all@5", "evidence_all@10"]
)


def rank_metrics(ranked: list[str], rel: dict[str, int]) -> dict[str, float]:
    """Metrics for one query. ``ranked`` is a deduplicated doc-id list (best first)."""
    relevant = {d for d, g in rel.items() if g > 0}
    n_rel = len(relevant)
    out: dict[str, float] = {}
    if n_rel == 0:
        return {name: float("nan") for name in RANK_METRIC_NAMES}

    ranked = ranked[:100]
    is_rel = [1 if d in relevant else 0 for d in ranked]
    for k in (1, 3, 5, 8):
        out[f"hit@{k}"] = float(any(is_rel[:k]))
    for k in (5, 10, 100):
        out[f"recall@{k}"] = sum(is_rel[:k]) / n_rel
    out["precision@5"] = sum(is_rel[:5]) / 5
    out["mrr"] = next((1.0 / (i + 1) for i, r in enumerate(is_rel) if r), 0.0)

    hits, ap = 0, 0.0
    for i, r in enumerate(is_rel, start=1):
        if r:
            hits += 1
            ap += hits / i
    out["map"] = ap / n_rel  # AP@100 normalised by |R| (trec_eval map_cut_100)

    gains = [float(rel.get(d, 0)) if rel.get(d, 0) > 0 else 0.0 for d in ranked]
    ideal = sorted((float(g) for g in rel.values() if g > 0), reverse=True)
    for k in (5, 10):
        dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))
        idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal[:k], start=1))
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    for k in (5, 10):
        out[f"evidence_all@{k}"] = float(sum(is_rel[:k]) == n_rel)
    return out


# ======================================================================================
# Short-answer metrics (SQuAD-style normalisation)
# ======================================================================================
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: Iterable[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(pred: str, golds: Iterable[str]) -> float:
    best = 0.0
    p_tok = normalize_answer(pred).split()
    for g in golds:
        g_tok = normalize_answer(g).split()
        common = Counter(p_tok) & Counter(g_tok)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec, rec = n_same / len(p_tok), n_same / len(g_tok)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def contains_answer(pred: str, golds: Iterable[str]) -> float:
    """1.0 if any normalised gold string occurs in the normalised prediction (lenient EM)."""
    p = f" {normalize_answer(pred)} "
    return float(any(f" {normalize_answer(g)} " in p for g in golds if normalize_answer(g)))


def rgb_correct(pred: str, answer: list) -> float:  # type: ignore[type-arg]
    """RGB accuracy rule: every answer slot must be matched by one of its accepted variants.

    ``answer`` is a list whose items are either a string or a list of accepted variants.
    """
    p = pred.lower().replace(" ", "")
    for slot in answer:
        variants = slot if isinstance(slot, list) else [slot]
        if not any(str(v).lower().replace(" ", "") in p for v in variants):
            return 0.0
    return 1.0


# ======================================================================================
# Abstention detection
# ======================================================================================
REFUSAL_TEMPLATE = "Based on the provided context, there is insufficient evidence to answer."
_ABSTAIN_PHRASES = (
    "insufficient evidence", "insufficient information", "not enough information",
    "not enough evidence", "cannot answer", "can't answer", "cannot be answered",
    "unable to answer", "no information", "does not contain", "do not contain",
    "does not provide", "do not provide", "not mentioned", "could not find",
    "cannot find", "unable to find", "not in the provided", "do not have enough",
    "does not specify", "not specified", "no relevant information", "not available in the",
    "not address", "do not mention", "does not mention",
)  # fmt: skip


def is_abstention(answer: str) -> bool:
    """The system's explicit refusal sentence, or a short answer built around a refusal phrase.

    Heuristic by design (documented in REPORT.md); the length cap keeps partial answers that
    merely note one missing detail from counting as refusals.
    """
    a = answer.strip().lower()
    if not a:
        return True  # an empty generation is not an answer
    if REFUSAL_TEMPLATE.lower() in a:
        return True
    return len(a.split()) <= 60 and any(p in a for p in _ABSTAIN_PHRASES)
