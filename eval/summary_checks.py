"""Checks of the generated summaries, per dataset: missing, empty, cut at the token cap, length, unsupported names.

Each ``--set LABEL=JOBS|RESULTS`` names a prompt file (``{key, prompt}``, from ``eval.main_summaries export``) and the
file the GPU wrote (``{key, summary, tokens}``, from ``eval/vm/summarize_vllm.py``). A prompt with no result is
*missing* (not written yet), not empty. *Cut* means the summary used the whole token cap. An *unsupported name* is a
capitalised word that appears nowhere in the text the writer was shown (``eval.summary_names``): memory of the model,
not of the document.

    uv run --no-sync python -m eval.summary_checks --cap 180 --out eval/results_final9b/SUMMARY_CHECKS.md \\
        --set "fiqa=jobs9b_fiqa.jsonl|results9b_fiqa.jsonl" --set "hotpotqa=jobs9b_hotpotqa.jsonl|results9b_hotpotqa.jsonl"
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.summary_names import shown_text, unsupported_names

DEFAULT_CAP = 180  # --max-tokens of eval/vm/summarize_vllm.py


def _read_by_key(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["key"]: row for row in (json.loads(line) for line in fh if line.strip())}


def check_set(
    jobs: Mapping[str, str], results: Mapping[str, Mapping[str, Any]], cap: int
) -> dict[str, Any]:
    """Counts for one dataset; ``jobs`` maps key to prompt, results for other keys are ignored."""
    written = {key: results[key] for key in jobs if key in results}
    with_text = {key: row for key, row in written.items() if row["summary"]}
    tokens = [int(row.get("tokens", 0)) for row in written.values()]
    unsupported = sum(
        bool(unsupported_names(str(row["summary"]), shown_text(jobs[key])))
        for key, row in with_text.items()
    )
    return {
        "prompts": len(jobs),
        "summarised": len(written),
        "missing": len(jobs) - len(written),
        "empty": len(written) - len(with_text),
        "capped": sum(t >= cap for t in tokens),
        "mean_tokens": sum(tokens) / len(tokens) if tokens else None,
        "unsupported": unsupported,
        "unsupported_share": unsupported / len(with_text) if with_text else None,
    }


def render(rows: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "| set | prompts | summarised | missing | empty | cut at the token cap | mean tokens "
        "| with an unsupported name |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, s in rows.items():
        mean = "n/a" if s["mean_tokens"] is None else f"{s['mean_tokens']:.1f}"
        if s["summarised"] == 0:
            share = "n/a (not run)"
        elif s["unsupported_share"] is None:
            share = "n/a"
        else:
            share = f"{s['unsupported']} ({100 * s['unsupported_share']:.0f}%)"
        lines.append(
            f"| {label} | {s['prompts']} | {s['summarised']} | {s['missing']} | {s['empty']} "
            f"| {s['capped']} | {mean} | {share} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--set", action="append", required=True, help="LABEL=JOBS|RESULTS")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows: dict[str, dict[str, Any]] = {}
    for spec in args.set:
        label, paths = spec.split("=", 1)
        jobs_path, results_path = (Path(p) for p in paths.split("|", 1))
        jobs = {key: row["prompt"] for key, row in _read_by_key(jobs_path).items()}
        rows[label] = check_set(jobs, _read_by_key(results_path), args.cap)
    table = render(rows)
    print(table)
    if args.out is not None:
        args.out.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
