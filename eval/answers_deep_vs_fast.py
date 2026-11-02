"""Answer-level comparison of the arms (fast, deep with a small summariser, deep with a bigger one).

Each arm is a results directory with ``raw_generation.jsonl`` (``eval.run_generation``) and ``rubric_scores.jsonl``
(``eval.judge_correctness``). For answerable questions the rubric score is correct 1.0, partial 0.5, incorrect or
abstained 0.0; the table also gives how often the answer model declined an answerable question (false abstention)
and an unanswerable one (correct abstention). The difference to the first arm is a paired bootstrap over the
answerable questions both arms answered.

    uv run --no-sync python -m eval.answers_deep_vs_fast --out eval/results_main/ANSWERS_DEEP_VS_FAST.md \\
        --arm "fast=eval/results_main/G_fast" --arm "deep 9B=eval/results_main/G_deep9b"
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.compare_runs import Delta
from eval.deep_vs_fast import verdict
from eval.stats import paired_test

Arm = dict[
    tuple[str, str], dict[str, Any]
]  # (dataset, question id) -> score, abstained, unanswerable


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_arm(directory: Path) -> Arm:
    arm: Arm = {}
    for row in _read(directory / "raw_generation.jsonl"):
        if "error" in row:
            continue
        arm[(row["dataset"], row["qid"])] = {
            "score": None,
            "abstained": bool(row.get("abstained")),
            "unanswerable": bool(row.get("unanswerable")),
        }
    for row in _read(directory / "rubric_scores.jsonl"):
        record = arm.get((row["dataset"], row["qid"]))
        if record is not None:
            record["score"] = row.get("score")
    return arm


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise(arm: Arm, dataset: str) -> dict[str, Any]:
    rows = [r for (d, _q), r in arm.items() if d == dataset]
    answerable = [r for r in rows if not r["unanswerable"]]
    unanswerable = [r for r in rows if r["unanswerable"]]
    return {
        "n_answerable": len(answerable),
        "correctness": _mean([r["score"] for r in answerable if r["score"] is not None]),
        "false_abstention": _mean([float(r["abstained"]) for r in answerable]),
        "n_unanswerable": len(unanswerable),
        "correct_abstention": _mean([float(r["abstained"]) for r in unanswerable]),
    }


def paired_difference(base: Arm, arm: Arm, dataset: str) -> Delta | None:
    keys = sorted(
        k
        for k, r in base.items()
        if k[0] == dataset and not r["unanswerable"] and r["score"] is not None
        and k in arm and arm[k]["score"] is not None
    )  # fmt: skip
    if not keys:
        return None
    b = [base[k]["score"] for k in keys]
    a = [arm[k]["score"] for k in keys]
    res = paired_test(b, a)
    return Delta(
        len(keys),
        sum(b) / len(b),
        sum(a) / len(a),
        res.mean_diff,
        res.ci_lo,
        res.ci_hi,
        res.p_bootstrap,
    )


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def render(arms: Mapping[str, Arm], datasets: Sequence[str]) -> str:
    names = list(arms)
    base = arms[names[0]]
    out = ["# Answers: fast against deep ingestion\n"]
    for dataset in datasets:
        out += [
            f"### {dataset}\n",
            f"| arm | answerable | rubric correctness | difference to {names[0]} | verdict | false abstention | unanswerable | correct abstention |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name in names:
            s = summarise(arms[name], dataset)
            d = None if name == names[0] else paired_difference(base, arms[name], dataset)
            diff = (
                "-"
                if name == names[0]
                else ("n/a" if d is None else f"{d.diff:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}]")
            )
            who = "-" if name == names[0] else verdict(d)
            out.append(
                f"| {name} | {s['n_answerable']} | {_pct(s['correctness'])} | {diff} | {who} | "
                f"{_pct(s['false_abstention'])} | {s['n_unanswerable']} | {_pct(s['correct_abstention'])} |"
            )
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="LABEL=RESULTS_DIR (the first is the baseline)",
    )
    parser.add_argument("--datasets", nargs="+", default=["docsqa", "pdf-quantum", "pdf-got"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    arms = {
        label: load_arm(Path(path)) for label, path in (spec.split("=", 1) for spec in args.arm)
    }
    args.out.write_text(render(arms, args.datasets), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
