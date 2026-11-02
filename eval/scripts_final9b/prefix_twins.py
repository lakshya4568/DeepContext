"""Does the product's near-duplicate filter explain why mixed search can score below dense only?

``deduplicate_candidates`` (src/deep_context/retrieval/hybrid.py) keeps one chunk per first-120-characters
prefix (``content[:120].strip().lower()``). The benchmark labels every document id on its own, so a relevant
document that is dropped because a twin ranks above it counts as a miss.

Table 1 (every set): what the saved rows show, dense only against mixed search with keyword vote 0.02.
Table 2 (CovidQA): of the relevant documents dense search returned and mixed search did not, how many had a twin
(same first 120 characters) ranked above them in the dense list. CovidQA documents have no title and are one chunk
each, so the document text is the chunk text there.

    PYTHONPATH=.:src uv run --no-sync python eval/scripts_final9b/prefix_twins.py --arm fast \\
        > eval/results_final9b/NOTES_prefix_twins_fast.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.main_summaries import task_for

OUT = Path("eval/results_final9b")
DATASETS = [
    "ragbench-covidqa", "scifact", "multihoprag", "hotpotqa", "fiqa", "docsqa", "pdf-quantum", "pdf-got",
]  # fmt: skip


def product_key(text: str) -> str:
    """The key of the product's filter (hybrid.py, ``deduplicate_candidates``)."""
    return text[:120].strip().lower()


def load(path: Path, dataset: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            if r["dataset"].split("|")[0] == dataset and "error" not in r and r["relevant"]:
                rows[(r["config"], r["qid"])] = r
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def symptoms(dense: dict, mixed: dict, dataset: str) -> str:
    qids = sorted(
        {q for (c, q) in dense if c == "dense_only"} & {q for (c, q) in mixed if c == "rrf"}
    )
    if not qids:
        return f"| {dataset} | 0 | n/a | n/a | n/a | n/a |"
    d = [dense[("dense_only", q)] for q in qids]
    m = [mixed[("rrf", q)] for q in qids]
    return (
        f"| {dataset} | {len(qids)} | {mean([r['metrics']['recall@100'] for r in d]):.3f} "
        f"| {mean([r['metrics']['recall@100'] for r in m]):.3f} "
        f"| {mean([float(len(r['ranked'])) for r in d]):.1f} | {mean([float(len(r['ranked'])) for r in m]):.1f} |"
    )


def covid_attribution(dense: dict, mixed: dict) -> list[str]:
    corpus = task_for("ragbench-covidqa").corpus
    keys = {doc_id: product_key(d["text"]) for doc_id, d in corpus.items()}
    sharing = {k for k in keys.values() if sum(1 for v in keys.values() if v == k) > 1}
    n_sharing = sum(1 for v in keys.values() if v in sharing)
    lost = twin_above = 0
    for q in sorted(
        {q for (c, q) in dense if c == "dense_only"} & {q for (c, q) in mixed if c == "rrf"}
    ):
        d, m = dense[("dense_only", q)], mixed[("rrf", q)]
        for doc in d["relevant"]:
            if doc in d["ranked"] and doc not in m["ranked"]:
                lost += 1
                above = d["ranked"][: d["ranked"].index(doc)]
                twin_above += any(keys.get(x) == keys.get(doc) for x in above)
    return [
        f"CovidQA has {len(corpus)} documents; {n_sharing} of them ({100 * n_sharing / len(corpus):.0f}%) share "
        "their first 120 characters with at least one other document.",
        f"Relevant documents that dense search returned and mixed search did not: {lost}; "
        f"of these, a twin ranks above them in the dense list: {twin_above}.",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arm", choices=["fast", "deep"], default="fast")
    arm = ap.parse_args().arm
    print(f"# Near-duplicate filter and mixed search ({arm} ingestion)\n")
    print("## Table 1: dense only against mixed search (keyword vote 0.02), same questions\n")
    print(
        "| set | questions | Recall@100 dense | Recall@100 mixed | documents returned, dense | documents returned, mixed |"
    )
    print("|---|---|---|---|---|---|")
    covid: tuple[dict, dict] | None = None
    for ds in DATASETS:
        dense = load(OUT / f"R_{arm}_equal" / "raw_retrieval.jsonl", ds)
        mixed = load(OUT / f"R_{arm}_vote0.02" / "raw_retrieval.jsonl", ds)
        print(symptoms(dense, mixed, ds))
        if ds == "ragbench-covidqa":
            covid = (dense, mixed)
    print("\n## Table 2: CovidQA, where the lost documents went\n")
    if covid is not None and covid[0] and covid[1]:
        for line in covid_attribution(*covid):
            print(f"- {line}")


if __name__ == "__main__":
    main()
