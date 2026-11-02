"""The keyword-only-summary arm (deepkw) against fast and against deep, on the same questions.

Writes ``eval/results_final9b/DEEPKW_VS_FAST.md`` and ``DEEPKW_VS_DEEP.md``: first an "at a glance" table (difference of nDCG@10, deepkw minus the other arm; a mark when the 95% interval
excludes 0), then one table per dataset with the five search settings (a setting that was not run shows n/a). The thesis is counted with the corrected key, as in the main report.

Deepkw embeds the plain chunk (as fast does) and puts the 9B summary only into the keyword index (as deep does), so its dense-only rows equal fast's and its keyword-only rows equal deep's;
only the mixed settings are new.

    PYTHONPATH=.:src uv run --no-sync python eval/scripts_final9b/deepkw_compare.py
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from eval.compare_runs import Group, load_groups
from eval.deep_vs_fast import Setting, compare_row, render_dataset, verdict
from eval.final_report import RETRIEVAL_DATASETS, SETTINGS, THESIS
from eval.thesis_key import correct_group

RESULTS = Path("eval/results_final9b")
OTHER = "deepkw"
BASES = ("fast", "deep")
NOT_RUN = "not run"


def _group(
    cache: dict[Path, dict[tuple[str, str], Group]],
    arm: str,
    suffix: str,
    dataset: str,
    config: str,
) -> Group:
    path = RESULTS / f"R_{arm}_{suffix}" / "raw_retrieval.jsonl"
    if path not in cache:
        cache[path] = load_groups(path) if path.exists() else {}
    return cache[path].get((dataset, config), {})


def _settings(
    cache: dict[Path, dict[tuple[str, str], Group]], base: str, dataset: str
) -> list[Setting]:
    return [
        (label, _group(cache, base, suffix, dataset, config), _group(cache, OTHER, suffix, dataset, config))
        for label, suffix, config in SETTINGS
    ]  # fmt: skip


def _cell(setting: Setting) -> tuple[str, str]:
    """(text for the table, verdict word) of one dataset and search setting."""
    _label, first, second = setting
    row = compare_row(first, second)
    delta = row["ndcg@10"] if row["n"] else None
    if delta is None:
        return NOT_RUN, NOT_RUN
    word = verdict(delta, OTHER)
    mark = {f"{OTHER} better": " ▲", f"{OTHER} worse": " ▼"}.get(word, "")
    return f"{delta.diff:+.3f}{mark}", word


def glance(
    cache: dict[Path, dict[tuple[str, str], Group]], base: str, datasets: Sequence[str]
) -> str:
    labels = [label for label, _suffix, _config in SETTINGS]
    lines = [
        f"### {OTHER} minus {base}, nDCG@10 (▲ {OTHER} clearly better, ▼ clearly worse: the 95% interval excludes 0)\n",
        "| set | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    counts = {f"{OTHER} better": 0, f"{OTHER} worse": 0, "no clear difference": 0}
    for dataset in datasets:
        settings = _settings(cache, base, dataset)
        if dataset == THESIS:
            settings = [
                (label, correct_group(first), correct_group(second))
                for label, first, second in settings
            ]
        cells = []
        for setting in settings:
            text, word = _cell(setting)
            cells.append(text)
            if word in counts:
                counts[word] += 1
        lines.append(
            f"| {dataset}{' (corrected key)' if dataset == THESIS else ''} | "
            + " | ".join(cells)
            + " |"
        )
    total = sum(counts.values())
    lines.append(
        f"\n{total} comparisons: {counts[f'{OTHER} better']} {OTHER} better, {counts[f'{OTHER} worse']} {OTHER} worse, {counts['no clear difference']} no clear difference.\n"
    )
    return "\n".join(lines)


def build(base: str, datasets: Sequence[str]) -> str:
    cache: dict[Path, dict[tuple[str, str], Group]] = {}
    out = [
        f"# {OTHER} (summary only in the keyword index) against {base}\n",
        glance(cache, base, datasets),
    ]
    for dataset in datasets:
        settings = _settings(cache, base, dataset)
        if not any(first and second for _label, first, second in settings):
            continue
        out.append(
            render_dataset(
                dataset, settings, " (stored key)" if dataset == THESIS else "", (base, OTHER)
            )
        )
        if dataset == THESIS:
            fixed = [
                (label, correct_group(first), correct_group(second))
                for label, first, second in settings
            ]
            out.append(render_dataset(dataset, fixed, " (corrected key)", (base, OTHER)))
    return "\n".join(out)


def main() -> None:
    datasets = [d for d in RETRIEVAL_DATASETS if d not in ("fiqa",)]  # whole FiQA has no deep arm
    for base in BASES:
        path = RESULTS / f"DEEPKW_VS_{base.upper()}.md"
        path.write_text(build(base, datasets), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
