"""Builds eval/results_final9b/REPORT.md from the hand-written template and the tables the result files give.

Every table and every count in the report is produced from the files in ``eval/results_final9b/``; the template holds only
the words around them. Placeholders in the template: ``{{GLANCE}}``, ``{{COUNTS}}``, ``{{ANSWERS}}``, ``{{FAST_ONLY}}``,
``{{SUMMARY_CHECKS}}``, ``{{DEEPKW}}`` (the two "at a glance" tables of ``DEEPKW_VS_FAST.md`` and ``DEEPKW_VS_DEEP.md``), plus ``{{STATUS}}`` and ``{{COST}}``, which are read from
``report_status.md`` and ``report_cost.md`` in the results directory (written by hand). Run the summary checks (``summary_checks_all.py``) and ``deepkw_compare.py`` first; this script regenerates the other tables.

    PYTHONPATH=.:src SUMMARY_MODEL=Qwen/Qwen3.5-9B EVAL_STORE=stock uv run --no-sync python eval/scripts_final9b/summary_checks_all.py
    PYTHONPATH=.:src uv run --no-sync python eval/scripts_final9b/deepkw_compare.py
    PYTHONPATH=.:src uv run --no-sync python eval/scripts_final9b/build_report.py
"""

from __future__ import annotations

from pathlib import Path

from eval import final_report as F

RESULTS = Path("eval/results_final9b")
TEMPLATE = Path(__file__).with_name("report_template.md")


def _fast_only_tables() -> str:
    """The per-dataset tables of the datasets whose deep arm was not run."""
    text = (RESULTS / "RETRIEVAL.md").read_text(encoding="utf-8")
    found = [b for b in text.split("### ")[1:] if ": fast only" in b.split("\n", 1)[0]]
    return (
        "\n".join("### " + b.rstrip() + "\n" for b in found)
        or "(none: both arms were run everywhere)"
    )


def _deepkw_glance() -> str:
    """The "at a glance" table of each deepkw comparison file (made by deepkw_compare.py): the part before the first per-dataset table."""
    glances = []
    for name in ("DEEPKW_VS_FAST.md", "DEEPKW_VS_DEEP.md"):
        path = RESULTS / name
        if not path.exists():
            glances.append(f"({name} not made: deepkw_compare.py was not run)")
            continue
        sections = path.read_text(encoding="utf-8").split("\n### ")
        glances.append("### " + sections[1].strip())
    return "\n\n".join(glances)


def _optional(name: str, default: str) -> str:
    path = RESULTS / name
    return path.read_text(encoding="utf-8") if path.exists() else default


def blocks() -> dict[str, str]:
    F.main(
        ["--results-dir", str(RESULTS)]
    )  # regenerate RETRIEVAL.md, ANSWERS.md and the CSV files first
    glance = F.glance_markdown(F.retrieval_records(RESULTS, F.RETRIEVAL_DATASETS))
    legend, table, counts = (part.strip() for part in glance.split("\n\n") if part.strip())
    return {
        "GLANCE": f"{legend}\n\n{table}",
        "COUNTS": counts,
        "ANSWERS": F.answers_glance_markdown(F.answer_records(RESULTS, F.ANSWER_DATASETS)),
        "FAST_ONLY": _fast_only_tables(),
        "SUMMARY_CHECKS": (RESULTS / "SUMMARY_CHECKS.md").read_text(encoding="utf-8"),
        "DEEPKW": _deepkw_glance(),
        "STATUS": _optional("report_status.md", "(status not written)"),
        "COST": _optional("report_cost.md", "(cost and time not written)"),
    }


def main() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    for name, block in blocks().items():
        text = text.replace("{{" + name + "}}", block.strip())
    (RESULTS / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"wrote {RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
