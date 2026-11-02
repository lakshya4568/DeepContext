"""Answer-level report of the final round: correctness, saying "I don't know", grounding and citations.

Each arm is a results directory with ``raw_generation.jsonl`` (``eval.run_generation``), ``rubric_scores.jsonl``
(``eval.judge_correctness``) and, when it was run, ``grounding_scores.jsonl`` (``eval.judge_grounding``). The first
arm is the baseline; the differences are paired bootstraps (95% interval, 10,000 resamples) over the questions both
arms have.

    uv run --no-sync python -m eval.answers_report --out eval/results_final9b/ANSWERS.md \\
        --arm "fast=eval/results_final9b/A_fast" --arm "deep 9B=eval/results_final9b/A_deep" \\
        --datasets multihoprag ragbench-covidqa docsqa pdf-quantum pdf-got

Populations: correctness and the two "I don't know" rates use all questions of the set (answerable and not);
citation shares use the answerable questions whose answer was not a refusal; grounding uses the answerable
questions whose answer the judge split into at least one claim.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.answers_deep_vs_fast import paired_difference
from eval.compare_runs import Delta
from eval.stats import paired_test
from eval.strict_prompt import check_citations

Arm = dict[tuple[str, str], dict[str, Any]]  # (dataset, question id) -> facts about the answer
DEFAULT_DATASETS = ["multihoprag", "ragbench-covidqa", "docsqa", "pdf-quantum", "pdf-got"]


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _record(row: dict[str, Any]) -> dict[str, Any]:
    seen_ids = row.get("context_ids_seen", [])
    relevant = set(row.get("relevant", []))
    gold = {n for n, ext in enumerate(seen_ids, start=1) if ext in relevant} if relevant else None
    check = check_citations(str(row.get("answer", "")), len(row.get("contexts_seen", [])), gold)
    return {
        "score": None,
        "abstained": bool(row.get("abstained")),
        "unanswerable": bool(row.get("unanswerable")),
        "valid_citations": check.valid,
        "invalid_citations": check.invalid,
        "cites_gold": check.cites_gold,
        "n_claims": None,
        "n_supported": None,
        "n_cited_claims": None,
        "n_citation_right": None,
        "faithfulness": None,
        "fully_grounded": None,
    }


def load_arm(directory: Path) -> Arm:
    arm: Arm = {}
    for row in _read(directory / "raw_generation.jsonl"):
        if "error" not in row:
            arm[(row["dataset"], row["qid"])] = _record(row)
    for row in _read(directory / "rubric_scores.jsonl"):
        record = arm.get((row["dataset"], row["qid"]))
        if record is not None:
            record["score"] = row.get("score")
    for row in _read(directory / "grounding_scores.jsonl"):
        record = arm.get((row["dataset"], row["qid"]))
        if record is not None and row.get("n_claims"):
            for key in (
                "n_claims", "n_supported", "n_cited_claims", "n_citation_right",
                "faithfulness", "fully_grounded",
            ):  # fmt: skip
                record[key] = row.get(key)
    return arm


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _share(flags: Sequence[bool]) -> float | None:
    return _mean([float(f) for f in flags])


def summarise(arm: Arm, dataset: str) -> dict[str, Any]:
    rows = [r for (d, _q), r in arm.items() if d == dataset]
    answerable = [r for r in rows if not r["unanswerable"]]
    unanswerable = [r for r in rows if r["unanswerable"]]
    answers = [r for r in answerable if not r["abstained"]]
    graded = [r for r in answerable if r["n_claims"]]
    cited = sum(r["n_cited_claims"] for r in graded)
    gold_known = [r for r in answers if r["cites_gold"] is not None]
    return {
        "n_answerable": len(answerable),
        "correctness": _mean([r["score"] for r in answerable if r["score"] is not None]),
        "false_abstention": _share([r["abstained"] for r in answerable]),
        "n_unanswerable": len(unanswerable),
        "correct_abstention": _share([r["abstained"] for r in unanswerable]),
        "n_answers": len(answers),
        "with_valid_citation": _share([bool(r["valid_citations"]) for r in answers]),
        "with_invented_citation": _share([bool(r["invalid_citations"]) for r in answers]),
        "cites_gold": _share([bool(r["cites_gold"]) for r in gold_known]),
        "n_graded": len(graded),
        "claims_backed": (
            sum(r["n_supported"] for r in graded) / sum(r["n_claims"] for r in graded)
            if graded
            else None
        ),
        "answers_fully_backed": _share([bool(r["fully_grounded"]) for r in graded]),
        "citation_precision": (
            sum(r["n_citation_right"] for r in graded) / cited if cited else None
        ),
    }


def paired_field(base: Arm, arm: Arm, dataset: str, field: str) -> Delta | None:
    """Paired difference of a per-answer number (for example ``faithfulness``) on answerable questions."""
    keys = sorted(
        k
        for k, r in base.items()
        if k[0] == dataset and not r["unanswerable"] and r[field] is not None
        and k in arm and arm[k][field] is not None
    )  # fmt: skip
    if not keys:
        return None
    b = [float(base[k][field]) for k in keys]
    a = [float(arm[k][field]) for k in keys]
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


def verdict_words(d: Delta | None) -> str:
    if d is None:
        return "n/a"
    if d.lo > 0:
        return "better"
    if d.hi < 0:
        return "worse"
    return "no clear difference"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _points(d: Delta | None) -> str:
    return "n/a" if d is None else f"{100 * d.diff:+.1f} [{100 * d.lo:+.1f}, {100 * d.hi:+.1f}]"


def render(arms: Mapping[str, Arm], datasets: Sequence[str]) -> str:
    names = list(arms)
    first = names[0]
    out: list[str] = []
    for dataset in datasets:
        out += [
            f"### {dataset}: answers\n",
            f"| arm | answerable | correct (score out of 100) | difference to {first} (points) | verdict "
            "| wrongly says it does not know (answerable) | unanswerable | rightly says it does not know |",
            "|---|---|---|---|---|---|---|---|",
        ]
        grounding = [
            f"### {dataset}: grounding and citations\n",
            f"| arm | answers with claims | claims backed by the documents | answers with every claim backed "
            f"| difference to {first}, per answer (points) | verdict | citation precision "
            "| answers with a valid citation | invented citation number | cites a relevant document |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name in names:
            s = summarise(arms[name], dataset)
            if name == first:
                diff, who, g_diff, g_who = "-", "-", "-", "-"
            else:
                d = paired_difference(arms[first], arms[name], dataset)
                g = paired_field(arms[first], arms[name], dataset, "faithfulness")
                diff, who, g_diff, g_who = (
                    _points(d),
                    verdict_words(d),
                    _points(g),
                    verdict_words(g),
                )
            score = "n/a" if s["correctness"] is None else f"{100 * s['correctness']:.1f}"
            out.append(
                f"| {name} | {s['n_answerable']} | {score} | {diff} | {who} | {_pct(s['false_abstention'])} "
                f"| {s['n_unanswerable']} | {_pct(s['correct_abstention'])} |"
            )
            grounding.append(
                f"| {name} | {s['n_graded']} | {_pct(s['claims_backed'])} | {_pct(s['answers_fully_backed'])} "
                f"| {g_diff} | {g_who} | {_pct(s['citation_precision'])} | {_pct(s['with_valid_citation'])} "
                f"| {_pct(s['with_invented_citation'])} | {_pct(s['cites_gold'])} |"
            )
        out += ["", *grounding, ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--arm", action="append", required=True, help="LABEL=RESULTS_DIR (first = baseline)"
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    arms = {
        label: load_arm(Path(path)) for label, path in (spec.split("=", 1) for spec in args.arm)
    }
    args.out.write_text(render(arms, args.datasets), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
