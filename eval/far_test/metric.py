"""Scoring of the far-reference test.

A question has *parts* (for example "where the character is named" and "where he is only 'he'"). Each part
is a set of acceptable pages: a fact that is repeated on several pages can be found on any of them. A
question counts as answered at depth K when every part has at least one acceptable page among the first K
ranked pages. This replaces "earliest and latest gold page", which punishes finding a valid alternative.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

Parts = Sequence[frozenset[str]]


def parts_found(ranked_pages: Sequence[str], parts: Parts, k: int) -> tuple[bool, ...]:
    """For each part: is one of its pages among the first ``k`` ranked pages?"""
    if not parts:
        raise ValueError("a question with no parts cannot be scored")
    top = set(ranked_pages[:k])
    return tuple(bool(part & top) for part in parts)


def complete_at_k(ranked_pages: Sequence[str], parts: Parts, k: int) -> bool:
    return all(parts_found(ranked_pages, parts, k))


def rrf(
    ranked_lists: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> list[str]:
    """Reciprocal Rank Fusion of ranked page lists: score = sum of weight / (k + rank).

    Ties are broken by first appearance (the earlier list, then the earlier rank), so the order is stable.
    """
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(f"{len(weights)} weights for {len(ranked_lists)} ranked lists")
    votes = weights if weights is not None else [1.0] * len(ranked_lists)
    score: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for weight, ranked in zip(votes, ranked_lists, strict=True):
        for rank, page in enumerate(ranked, start=1):
            score[page] = score.get(page, 0.0) + weight / (k + rank)
            first_seen.setdefault(page, len(first_seen))
    return sorted(score, key=lambda page: (-score[page], first_seen[page]))


def paired_counts(base: Sequence[bool], arm: Sequence[bool]) -> tuple[int, int, int]:
    """(wins, losses, ties) of ``arm`` against ``base`` on the same questions."""
    wins = sum(1 for b, a in zip(base, arm, strict=True) if a and not b)
    losses = sum(1 for b, a in zip(base, arm, strict=True) if b and not a)
    return wins, losses, len(base) - wins - losses


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial test of wins against losses (ties carry no information)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(f, increasing: bool) -> float:  # type: ignore[no-untyped-def]
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (f(mid) < 0) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def exact_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) confidence interval for a proportion."""
    lower = (
        0.0
        if successes == 0
        else _bisect(lambda p: 1 - _binom_cdf(successes - 1, n, p) - alpha / 2, increasing=True)
    )
    upper = (
        1.0
        if successes == n
        else _bisect(lambda p: _binom_cdf(successes, n, p) - alpha / 2, increasing=False)
    )
    return lower, upper
