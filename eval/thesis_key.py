"""Corrected page key of the thesis set (``pdf-quantum``).

The key lists nearly every correct page twice: once as the thesis's printed page number and once as the PDF page
(the front matter is 20 pages long; see ``eval/pdf_sets.py``). A gold page that has a partner exactly 20 pages
later is therefore dropped and the later (PDF) page is kept. The rule was written down before any deep result
existed. Stored rows are never changed; the corrected scores are computed on copies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from eval.metrics import rank_metrics
from eval.pdf_sets import page_number

PRINTED_TO_PDF = 20
Group = dict[str, dict[str, Any]]


def corrected_relevant(relevant: Sequence[str]) -> list[str]:
    pages = {page_number(p) for p in relevant}
    return [p for p in relevant if page_number(p) + PRINTED_TO_PDF not in pages]


def correct_group(group: Group) -> Group:
    """Copies of the rows with the corrected gold pages and the metrics recomputed from the stored ranking."""
    fixed: Group = {}
    for qid, row in group.items():
        relevant = corrected_relevant(row["relevant"])
        fixed[qid] = {
            **row,
            "relevant": relevant,
            "metrics": rank_metrics(row["ranked"], {doc: 1 for doc in relevant}),
        }
    return fixed
