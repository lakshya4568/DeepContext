"""Deep ingestion (the application's contextual summaries) against fast ingestion, on identical questions.

Each ``--pair`` names one search setting and where its fast and deep rows are::

    uv run --no-sync python -m eval.deep_vs_fast --out eval/results_main/DEEP_VS_FAST.md \\
        --pair "dense only=eval/results_main/M0_stock:dense_only|eval/results_main/D1_deep:dense_only"

The mean of each run has no interval here; the difference is a paired bootstrap over the questions both runs
answered (95% CI, 10,000 resamples), as in every other table of this harness. For the thesis the table is given
twice: with the stored key and with the corrected key (``eval/thesis_key.py``).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.compare_runs import METRICS, Delta, Group, compare_metric, load_groups
from eval.thesis_key import correct_group

Setting = tuple[str, Group, Group]  # (label, fast rows, deep rows)
Names = tuple[str, str]  # (name of the first run, name of the second run)
DEFAULT_NAMES: Names = ("fast", "deep")
DEFAULT_TITLE = "Deep ingestion against fast ingestion"


def compare_row(fast: Group, deep: Group) -> dict[str, Any]:
    keys = sorted(set(fast) & set(deep))
    row: dict[str, Any] = {"n": len(keys)}
    if keys:
        row["fast"] = sum(fast[k]["metrics"]["ndcg@10"] for k in keys) / len(keys)
        row["deep"] = sum(deep[k]["metrics"]["ndcg@10"] for k in keys) / len(keys)
    for metric, _label in METRICS:
        row[metric] = compare_metric(fast, deep, metric)
    return row


def verdict(d: Delta | None, other: str = DEFAULT_NAMES[1]) -> str:
    """Plain words from the 95% interval of the nDCG@10 difference (second run minus first)."""
    if d is None:
        return "n/a"
    if d.lo > 0:
        return f"{other} better"
    if d.hi < 0:
        return f"{other} worse"
    return "no clear difference"


def _delta(d: Delta | None) -> str:
    return "n/a" if d is None else f"{d.diff:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}]"


def render_dataset(
    dataset: str, settings: Sequence[Setting], note: str = "", names: Names = DEFAULT_NAMES
) -> str:
    first, second = names
    diff = f"{second} - {first}"
    lines = [
        f"### {dataset}{note}\n",
        f"| search setting | questions | {first} nDCG@10 | {second} nDCG@10 | {diff}, nDCG@10 | verdict | {diff}, Hit@5 | {diff}, Recall@100 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, fast, deep in settings:
        row = compare_row(fast, deep)
        if not row["n"]:
            lines.append(f"| {label} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {row['n']} | {row['fast']:.3f} | {row['deep']:.3f} | {_delta(row['ndcg@10'])} | "
            f"{verdict(row['ndcg@10'], second)} | {_delta(row['hit@5'])} | {_delta(row['recall@100'])} |"
        )
    return "\n".join(lines) + "\n"


def _parse_pair(spec: str) -> tuple[str, tuple[Path, str], tuple[Path, str]]:
    label, rest = spec.split("=", 1)
    fast, deep = rest.split("|", 1)
    (fast_dir, fast_cfg), (deep_dir, deep_cfg) = (s.rsplit(":", 1) for s in (fast, deep))
    return label, (Path(fast_dir), fast_cfg), (Path(deep_dir), deep_cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--pair", action="append", required=True, help="LABEL=FASTDIR:CONFIG|DEEPDIR:CONFIG"
    )
    parser.add_argument("--datasets", nargs="+", default=["docsqa", "pdf-quantum", "pdf-got"])
    parser.add_argument("--base-name", default=DEFAULT_NAMES[0], help="name of the first run")
    parser.add_argument("--other-name", default=DEFAULT_NAMES[1], help="name of the second run")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    names: Names = (args.base_name, args.other_name)

    loaded: dict[Path, dict[tuple[str, str], Group]] = {}

    def rows(directory: Path, dataset: str, config: str) -> Group:
        if directory not in loaded:
            loaded[directory] = load_groups(directory / "raw_retrieval.jsonl")
        return loaded[directory].get((dataset, config), {})

    pairs = [_parse_pair(spec) for spec in args.pair]
    out = [f"# {args.title}\n"]
    for dataset in args.datasets:
        settings = [
            (label, rows(fd, dataset, fc), rows(dd, dataset, dc))
            for label, (fd, fc), (dd, dc) in pairs
        ]
        note = " (stored key)" if dataset == "pdf-quantum" else ""
        out.append(render_dataset(dataset, settings, note, names))
        if dataset == "pdf-quantum":
            fixed = [
                (label, correct_group(fast), correct_group(deep)) for label, fast, deep in settings
            ]
            out.append(render_dataset(dataset, fixed, " (corrected key)", names))
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
