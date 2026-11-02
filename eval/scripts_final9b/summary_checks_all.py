"""Summary checks for every dataset of the final round, from the files the GPU wrote (``eval.summary_checks`` does the counting).

The prompts of each dataset are rebuilt exactly as the export made them (``eval.main_summaries.requests_for_dataset``), so
"missing" means a chunk of that dataset's corpus without a summary. FiQA is checked as ``fiqa-sub`` (the corpus both arms use);
its summaries come from two files (the main run, stopped at about 45k, and the short extra pass).

    PYTHONPATH=.:src SUMMARY_MODEL=Qwen/Qwen3.5-9B EVAL_STORE=stock uv run --no-sync python eval/scripts_final9b/summary_checks_all.py
"""

from __future__ import annotations

from pathlib import Path

from eval.main_summaries import requests_for_dataset
from eval.summary_checks import DEFAULT_CAP, _read_by_key, check_set, render

CACHE = Path("eval/cache/summaries")
OUT = Path("eval/results_final9b/SUMMARY_CHECKS.md")
SETS: dict[str, list[str]] = {
    "ragbench-covidqa": ["results9b_ragbench-covidqa.jsonl"],
    "multihoprag": ["results9b_multihoprag.jsonl"],
    "scifact": ["results9b_scifact.jsonl"],
    "fiqa-sub": ["results9b_fiqa.jsonl", "results9b_fiqa-sub-extra.jsonl"],
    "hotpotqa": ["results9b_hotpotqa.jsonl"],
}


def main() -> None:
    rows = {}
    for name, files in SETS.items():
        results: dict[str, dict] = {}
        for file in files:
            results |= _read_by_key(CACHE / file)
        if not results:
            continue  # a set whose summaries were never written is left out of the table (it is "not run")
        jobs = {request.key: request.prompt for request in requests_for_dataset(name)}
        rows[name] = check_set(jobs, results, DEFAULT_CAP)
        print(name, rows[name], flush=True)
    table = render(rows)
    OUT.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
