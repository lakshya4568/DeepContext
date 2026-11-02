"""Tables of the final round: deep ingestion (9B summaries) against fast ingestion, from the saved rows.

    uv run --no-sync python -m eval.final_report --results-dir eval/results_final9b

Reads ``R_<arm>_<suffix>/raw_retrieval.jsonl`` (``eval.run_retrieval``) and ``A_<arm>/`` (``eval.run_generation``,
``eval.judge_correctness``, ``eval.judge_grounding``) for the arms ``fast`` and ``deep``. Writes ``RETRIEVAL.md``,
``ANSWERS.md``, ``table_retrieval.csv`` and ``table_answers.csv`` next to them. A run that does not exist is shown
as "not run", never as zero. The thesis (pdf-quantum) is given with the stored key and with the corrected key
(``eval/thesis_key.py``); only the corrected key enters the counts.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval import answers_report as AR
from eval.answers_deep_vs_fast import paired_difference
from eval.compare_runs import Group, load_groups
from eval.deep_vs_fast import Setting, compare_row, render_dataset, verdict
from eval.thesis_key import correct_group

RETRIEVAL_DATASETS = [
    "ragbench-covidqa", "scifact", "multihoprag", "hotpotqa", "fiqa", "fiqa-sub", "docsqa", "pdf-quantum", "pdf-got",
]  # fmt: skip
ANSWER_DATASETS = ["ragbench-covidqa", "multihoprag", "docsqa", "pdf-quantum", "pdf-got"]
THESIS = "pdf-quantum"
ARMS = ("fast", "deep")
SETTINGS = [  # (label, run folder suffix, config)
    ("dense only", "equal", "dense_only"),
    ("keyword only", "equal", "bm25_only"),
    ("mixed, equal votes", "equal", "rrf"),
    ("mixed, keyword vote 0.02", "vote0.02", "rrf"),
    ("mixed, keyword vote 0.02 + reranker", "rerank_vote0.02", "rrf_bge"),
]
METRICS = (("ndcg@10", "ndcg10"), ("hit@5", "hit5"), ("recall@100", "recall100"))
NOT_RUN = "not run"
BETTER, WORSE = "deep better", "deep worse"


class _Runs:
    """Each ``raw_retrieval.jsonl`` is read once."""

    def __init__(self, results_dir: Path) -> None:
        self._dir = results_dir
        self._groups: dict[Path, dict[tuple[str, str], Group]] = {}

    def group(self, arm: str, suffix: str, dataset: str, config: str) -> Group:
        path = self._dir / f"R_{arm}_{suffix}" / "raw_retrieval.jsonl"
        if path not in self._groups:
            self._groups[path] = load_groups(path) if path.exists() else {}
        return self._groups[path].get((dataset, config), {})


def _settings(runs: _Runs, dataset: str) -> list[Setting]:
    return [
        (
            label,
            runs.group("fast", suffix, dataset, config),
            runs.group("deep", suffix, dataset, config),
        )
        for label, suffix, config in SETTINGS
    ]


def _variants(runs: _Runs, dataset: str) -> list[tuple[str, list[Setting]]]:
    """(key name, settings); the thesis comes twice, with the stored key and with the corrected key."""
    settings = _settings(runs, dataset)
    if dataset != THESIS:
        return [("", settings)]
    fixed = [(label, correct_group(fast), correct_group(deep)) for label, fast, deep in settings]
    return [("stored", settings), ("corrected", fixed)]


def _mean(group: Group, metric: str) -> float:
    return sum(row["metrics"][metric] for row in group.values()) / len(group)


def render_fast_only(dataset: str, settings: Sequence[Setting], note: str) -> str:
    """The fast arm's own numbers for a dataset whose deep arm was not run."""
    lines = [
        f"### {dataset}{note}: fast only (deep: {NOT_RUN})\n",
        "| search setting | questions | fast nDCG@10 | fast Hit@5 | fast Recall@100 |",
        "|---|---|---|---|---|",
    ]
    for label, fast, _deep in settings:
        if not fast:
            lines.append(f"| {label} ({NOT_RUN}) | 0 | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {len(fast)} | {_mean(fast, 'ndcg@10'):.3f} | {_mean(fast, 'hit@5'):.3f} "
            f"| {_mean(fast, 'recall@100'):.3f} |"
        )
    return "\n".join(lines) + "\n"


def retrieval_markdown(results_dir: Path, datasets: Sequence[str]) -> str:
    runs = _Runs(results_dir)
    out: list[str] = []
    for dataset in datasets:
        for key, settings in _variants(runs, dataset):
            note = f" ({key} key)" if key else ""
            if any(fast for _l, fast, _d in settings) and not any(
                deep for _l, _f, deep in settings
            ):
                out.append(render_fast_only(dataset, settings, note))
                continue
            marked = [
                (label if fast and deep else f"{label} ({NOT_RUN})", fast, deep)
                for label, fast, deep in settings
            ]
            out.append(render_dataset(dataset, marked, note))
    return "\n".join(out)


def _retrieval_record(
    dataset: str, key: str, label: str, fast: Group, deep: Group
) -> dict[str, Any]:
    row = compare_row(fast, deep)
    record: dict[str, Any] = {
        "dataset": dataset, "key": key, "setting": label, "n": row["n"],
        "fast_ndcg10": row.get("fast"), "deep_ndcg10": row.get("deep"),
    }  # fmt: skip
    for metric, name in METRICS:
        delta = row.get(metric)
        record[f"diff_{name}"] = None if delta is None else delta.diff
        record[f"{name}_lo"] = None if delta is None else delta.lo
        record[f"{name}_hi"] = None if delta is None else delta.hi
    record["verdict"] = verdict(row["ndcg@10"], "deep") if row["n"] else NOT_RUN
    return record


def retrieval_records(results_dir: Path, datasets: Sequence[str]) -> list[dict[str, Any]]:
    """One record per dataset, key and search setting: both means, the paired differences and the verdict."""
    runs = _Runs(results_dir)
    return [
        _retrieval_record(dataset, key, label, fast, deep)
        for dataset in datasets
        for key, settings in _variants(runs, dataset)
        for label, fast, deep in settings
    ]


def _cell(record: dict[str, Any]) -> str:
    if record["verdict"] == NOT_RUN:
        return NOT_RUN
    mark = {BETTER: " ▲", WORSE: " ▼"}.get(record["verdict"], "")
    return f"{record['diff_ndcg10']:+.3f}{mark}"


def glance_markdown(records: Sequence[dict[str, Any]]) -> str:
    """One table, every dataset against every search setting; the counts leave out the stored thesis key."""
    labels = [label for label, _suffix, _config in SETTINGS]
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        cells.setdefault((r["dataset"], r["key"]), {})[r["setting"]] = _cell(r)
    lines = [
        "| set | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for (dataset, key), row in cells.items():
        name = f"{dataset} ({key} key)" if key else dataset
        lines.append(
            f"| {name} | " + " | ".join(row.get(label, NOT_RUN) for label in labels) + " |"
        )
    counted = [r for r in records if r["key"] != "stored" and r["verdict"] != NOT_RUN]
    better = sum(r["verdict"] == BETTER for r in counted)
    worse = sum(r["verdict"] == WORSE for r in counted)
    noun = "comparison" if len(counted) == 1 else "comparisons"
    return "\n".join(
        [
            "Deep minus fast, nDCG@10. ▲ deep better, ▼ deep worse (the 95% interval excludes zero); "
            "no mark: no clear difference.\n",
            *lines,
            "",
            f"{len(counted)} {noun}: {better} deep better, {worse} deep worse, "
            f"{len(counted) - better - worse} no clear difference.",
            "",
        ]
    )


def _answer_arms(results_dir: Path) -> dict[str, AR.Arm]:
    return {name: AR.load_arm(results_dir / f"A_{name}") for name in ARMS}


def answer_records(results_dir: Path, datasets: Sequence[str]) -> list[dict[str, Any]]:
    """One record per dataset and arm: the answer-level numbers, the deep arm with its paired differences."""
    arms = _answer_arms(results_dir)
    records: list[dict[str, Any]] = []
    for dataset in datasets:
        for name, arm in arms.items():
            record: dict[str, Any] = {"dataset": dataset, "arm": name, **AR.summarise(arm, dataset)}
            correct = None if name == ARMS[0] else paired_difference(arms[ARMS[0]], arm, dataset)
            faithful = (
                None
                if name == ARMS[0]
                else AR.paired_field(arms[ARMS[0]], arm, dataset, "faithfulness")
            )
            refused = (
                None
                if name == ARMS[0]
                else AR.paired_field(arms[ARMS[0]], arm, dataset, "abstained")
            )
            record |= {
                "diff_correctness": None if correct is None else correct.diff,
                "correctness_lo": None if correct is None else correct.lo,
                "correctness_hi": None if correct is None else correct.hi,
                "verdict": "baseline" if name == ARMS[0] else AR.verdict_words(correct),
                "diff_faithfulness": None if faithful is None else faithful.diff,
                "faithfulness_lo": None if faithful is None else faithful.lo,
                "faithfulness_hi": None if faithful is None else faithful.hi,
                "diff_false_abstention": None if refused is None else refused.diff,
                "false_abstention_lo": None if refused is None else refused.lo,
                "false_abstention_hi": None if refused is None else refused.hi,
                "faithfulness_verdict": "baseline"
                if name == ARMS[0]
                else AR.verdict_words(faithful),
            }
            records.append(record)
    return records


def answers_markdown(results_dir: Path, datasets: Sequence[str]) -> str:
    missing = [
        f"- {name}: {NOT_RUN}"
        for name in ARMS
        if not (results_dir / f"A_{name}" / "raw_generation.jsonl").exists()
    ]
    return "\n".join([*missing, "", AR.render(_answer_arms(results_dir), datasets)])


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def answers_glance_markdown(records: Sequence[dict[str, Any]]) -> str:
    """One row per dataset, the fast and the deep arm side by side (the full tables are in ANSWERS.md)."""
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        arms.setdefault(record["dataset"], {})[record["arm"]] = record
    lines = [
        "| set | answerable questions | correct, fast | correct, deep | deep minus fast (points) | verdict "
        "| wrongly says it does not know, fast / deep (deep minus fast, points) | rightly says it does not know, fast / deep "
        "| claims backed, fast / deep | citation precision, fast / deep |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset, pair in arms.items():
        fast, deep = pair["fast"], pair["deep"]
        score = lambda r: "n/a" if r["correctness"] is None else f"{100 * r['correctness']:.1f}"  # noqa: E731
        diff = (
            "n/a"
            if deep["diff_correctness"] is None
            else f"{100 * deep['diff_correctness']:+.1f} "
            f"[{100 * deep['correctness_lo']:+.1f}, {100 * deep['correctness_hi']:+.1f}]"
        )
        refusal_diff = (
            "n/a"
            if deep["diff_false_abstention"] is None
            else f"{100 * deep['diff_false_abstention']:+.1f} "
            f"[{100 * deep['false_abstention_lo']:+.1f}, {100 * deep['false_abstention_hi']:+.1f}]"
        )
        if deep["n_unanswerable"]:
            rightly = (
                f"{_share(fast['correct_abstention'])} / {_share(deep['correct_abstention'])} "
                f"(of {deep['n_unanswerable']})"
            )
        else:
            rightly = "no such questions"
        lines.append(
            f"| {dataset} | {deep['n_answerable']} | {score(fast)} | {score(deep)} | {diff} | {deep['verdict']} "
            f"| {_share(fast['false_abstention'])} / {_share(deep['false_abstention'])} ({refusal_diff}) | {rightly} "
            f"| {_share(fast['claims_backed'])} / {_share(deep['claims_backed'])} "
            f"| {_share(fast['citation_precision'])} / {_share(deep['citation_precision'])} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results_final9b"))
    parser.add_argument("--datasets", nargs="+", default=RETRIEVAL_DATASETS)
    parser.add_argument("--answer-datasets", nargs="+", default=ANSWER_DATASETS)
    args = parser.parse_args(argv)
    out: Path = args.results_dir

    records = retrieval_records(out, args.datasets)
    (out / "RETRIEVAL.md").write_text(
        "\n".join(
            [
                "# Deep ingestion against fast ingestion: retrieval\n",
                glance_markdown(records),
                "## Per dataset\n",
                retrieval_markdown(out, args.datasets),
            ]
        ),
        encoding="utf-8",
    )
    write_csv(out / "table_retrieval.csv", records)
    (out / "ANSWERS.md").write_text(
        "\n".join(
            [
                '# Deep ingestion against fast ingestion: answers, grounding, "I don\'t know"\n',
                answers_markdown(out, args.answer_datasets),
            ]
        ),
        encoding="utf-8",
    )
    write_csv(out / "table_answers.csv", answer_records(out, args.answer_datasets))
    print(f"wrote RETRIEVAL.md, ANSWERS.md, table_retrieval.csv, table_answers.csv in {out}")


if __name__ == "__main__":
    main()
