"""LLM-rubric answer correctness (reference-based), validated against Claude's labels.

Why this exists: RAGAS ``FactualCorrectness(mode="f1")`` decomposes both texts into claims and
scores claim overlap, which penalises a correct answer for every extra true claim it makes
(smoke run: judge mean 0.27 vs Claude 0.88, r = 0.09). This metric instead asks the judge model
one question per answer: *does the answer convey what the reference says?*

Rubric (verdict -> score): correct 1.0, partial 0.5, incorrect 0.0, abstained 0.0. For items the
corpus cannot answer, declining (or correcting a false premise) is the correct behaviour and
scores 1.0, while a confident made-up answer scores 0.0.

The judge (``eval.config.JUDGE_MODEL``, not the generator) runs at temperature 0, sees only the
question, the reference and the answer (never the retrieved evidence), and every call is cached on
disk. The ``reason`` field is generated *before* the verdict so the verdict follows brief reasoning.

Usage:
    uv run --no-sync python -m eval.judge_correctness --results-dir eval/results/smoke --validate
    uv run --no-sync python -m eval.judge_correctness --results-dir eval/results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from eval import config as C
from eval.strict_clients import get_runtime, stage_scope

log = logging.getLogger("eval.judge_correctness")

RUBRIC_VERSION = "v1"
VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "abstained": 0.0}
OUT_NAME = "rubric_scores.jsonl"

PROMPT = """You are grading an answer produced by a question-answering system against a reference answer.

Question:
{question}

Reference answer (ground truth):
{reference}

System answer:
{answer}
{unanswerable_note}
Decide whether the system answer conveys what the reference answer says. Judge only the content of the
answer against the reference; ignore style, length, formatting and citation markers. Extra true detail
beyond the reference is fine as long as it does not contradict the reference. If the reference is a
short label (for example "yes"/"no" or an entity name), the answer must reach the same conclusion.

Verdicts:
- "correct": every key point of the reference is conveyed and nothing contradicts it.
- "partial": some but not all key points of the reference are conveyed, or the answer is right but
  leaves out a required part (for example one of two requested items).
- "incorrect": the answer contradicts the reference, gives a different answer, or lacks the key point.
- "abstained": the answer declines to answer or says the information is not available, without
  giving the reference's content.

Respond with JSON only: {{"reason": "<at most 30 words>", "verdict": "correct|partial|incorrect|abstained"}}"""

UNANSWERABLE_NOTE = """
Important: the corpus does NOT contain the answer to this question (or the question rests on a false
premise). The reference above describes the expected behaviour. The correct behaviour is to decline,
say the information is not available, or point out the false premise. Grade "correct" if the system
answer does that; grade "incorrect" if it states a confident answer as if the information existed.
"""


def build_prompt(question: str, reference: str, answer: str, answerable: bool = True) -> str:
    return PROMPT.format(
        question=question.strip(),
        reference=reference.strip() or "(none)",
        answer=(answer or "").strip() or "(empty answer)",
        unanswerable_note="" if answerable else UNANSWERABLE_NOTE,
    )


def parse_verdict(text: str) -> tuple[str | None, str]:
    """Return ``(verdict, reason)``; verdict is None if the judge output is unusable."""
    try:
        d = json.loads(text)
        verdict = str(d.get("verdict", "")).strip().lower()
        reason = str(d.get("reason", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r'"verdict"\s*:\s*"([a-z]+)"', text.lower())
        verdict = m.group(1) if m else ""
        reason = ""
    return (verdict if verdict in VERDICT_SCORE else None), reason


async def grade(
    question: str,
    reference: str,
    answer: str,
    *,
    answerable: bool = True,
    model: str = C.JUDGE_MODEL,
    thinking_budget: int | None = 0,
) -> dict[str, Any]:
    rt = get_runtime()
    prompt = build_prompt(question, reference, answer, answerable)
    with stage_scope("judge"):
        res = await rt.generate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            json_mode=True,
            thinking_budget=thinking_budget,
            stage="judge",
        )
    verdict, reason = parse_verdict(res.text)
    # Unanswerable items: declining is the correct behaviour, whatever the judge called it.
    score = None if verdict is None else VERDICT_SCORE[verdict]
    if not answerable and verdict == "abstained":
        score = 1.0
    return {"verdict": verdict, "score": score, "reason": reason}


async def grade_all(
    rows: list[dict[str, Any]], model: str, thinking_budget: int | None, concurrency: int
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(r: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await grade(
                r["query"],
                r["reference"],
                r["answer"],
                answerable=r["answerable"],
                model=model,
                thinking_budget=thinking_budget,
            )

    return await asyncio.gather(*(one(r) for r in rows))


def _label(model: str, thinking_budget: int | None) -> str:
    return f"{model}@think{thinking_budget}" if thinking_budget is not None else model


def validate(rdir: Path, model: str, thinking_budget: int | None, concurrency: int) -> dict:
    """Score the 50 audit items and compare with Claude's correctness labels."""
    items = {
        (d["dataset"], d["config"], d["qid"]): d
        for d in map(json.loads, (rdir / "claude_audit_items.jsonl").open())
    }
    labels = {
        (d["dataset"], d["config"], d["qid"]): d
        for d in map(json.loads, (rdir / "claude_labels.jsonl").open())
    }
    keys = sorted(set(items) & set(labels))
    rows = [
        {
            "query": items[k]["query"],
            "reference": items[k]["reference"],
            "answer": items[k]["answer"],
            "answerable": True,
        }
        for k in keys
    ]
    graded = asyncio.run(grade_all(rows, model, thinking_budget, concurrency))
    get_runtime().cache.flush()
    usable = [(k, g) for k, g in zip(keys, graded, strict=True) if g["score"] is not None]
    c = np.array([labels[k]["correctness"] for k, _ in usable], dtype=float)
    r = np.array([g["score"] for _, g in usable], dtype=float)
    f1 = np.array(
        [items[k]["judge_scores"]["factual_correctness_f1"] for k, _ in usable], dtype=float
    )
    out: dict[str, Any] = {
        "labeller": "Claude (Anthropic model) - NOT human labels",
        "judge": _label(model, thinking_budget),
        "rubric_version": RUBRIC_VERSION,
        "n_items": len(keys),
        "n_scored": len(usable),
        "claude_mean": float(c.mean()),
        "rubric_mean": float(r.mean()),
        "mean_abs_diff": float(np.mean(np.abs(c - r))),
        "exact_agreement": float(np.mean(c == r)),
        "binary_agreement_at_0.5": float(np.mean((c >= 0.5) == (r >= 0.5))),
        "rubric_more_lenient_share": float(np.mean(r > c + 0.25)),
        "rubric_stricter_share": float(np.mean(r < c - 0.25)),
        "confusion_claude_rows_x_rubric_cols": {
            f"claude={cv}": {
                f"rubric={rv}": int(np.sum((c == cv) & (r == rv))) for rv in (0.0, 0.5, 1.0)
            }
            for cv in (0.0, 0.5, 1.0)
        },  # fmt: skip
    }
    if c.std() > 0 and r.std() > 0:
        out["pearson_r"] = float(np.corrcoef(c, r)[0, 1])
        out["spearman_rho"] = float(sp_stats.spearmanr(c, r)[0])
        out["weighted_kappa_linear"] = _weighted_kappa(c, r)
    out["ragas_factual_correctness_f1_for_comparison"] = {
        "mean": float(f1.mean()),
        "mean_abs_diff_vs_claude": float(np.mean(np.abs(c - f1))),
        "pearson_r": float(np.corrcoef(c, f1)[0, 1]) if f1.std() > 0 and c.std() > 0 else None,
    }
    disagreements = [
        {"key": list(k), "claude": labels[k]["correctness"], "rubric": g["score"],
         "rubric_verdict": g["verdict"], "rubric_reason": g["reason"], "claude_note": labels[k].get("note")}
        for k, g in usable if abs(labels[k]["correctness"] - g["score"]) > 1e-9
    ]  # fmt: skip
    out["disagreements"] = disagreements
    return out


def _weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Linearly weighted Cohen's kappa over the ordinal levels {0, 0.5, 1}."""
    levels = np.array([0.0, 0.5, 1.0])
    ia, ib = np.searchsorted(levels, a), np.searchsorted(levels, b)
    k = len(levels)
    obs = np.zeros((k, k))
    for x, y in zip(ia, ib, strict=True):
        obs[x, y] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom > 0 else float("nan")


def score_results(
    rdir: Path, model: str, thinking_budget: int | None, concurrency: int, batch: int,
    datasets: list[str] | None, configs: list[str] | None,
) -> None:  # fmt: skip
    label = _label(model, thinking_budget)
    raw = [json.loads(x) for x in (rdir / "raw_generation.jsonl").open()]
    out_path = rdir / OUT_NAME
    done: set[tuple[str, str, str]] = set()
    if out_path.exists():
        for x in out_path.open():
            d = json.loads(x)
            if d.get("judge_model") == label and d.get("rubric_version") == RUBRIC_VERSION:
                done.add((d["run_tag"], d["config"], d["qid"]))
    todo = [
        r for r in raw
        if not r.get("error")
        and (r["run_tag"], r["config"], r["qid"]) not in done
        and (datasets is None or r["dataset"] in datasets)
        and (configs is None or r["config"] in configs)
    ]  # fmt: skip
    log.info("%d rows to grade with rubric %s, judge=%s", len(todo), RUBRIC_VERSION, label)
    t0 = time.perf_counter()
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        graded = asyncio.run(
            grade_all(
                [
                    {
                        "query": r["query"],
                        "reference": r["reference"],
                        "answer": r["answer"],
                        "answerable": not r["unanswerable"],
                    }
                    for r in chunk
                ],
                model,
                thinking_budget,
                concurrency,
            )
        )
        with out_path.open("a") as f:
            for r, g in zip(chunk, graded, strict=True):
                if g["score"] is None:
                    log.error(
                        "unparseable verdict for %s/%s/%s - not saved",
                        r["dataset"],
                        r["config"],
                        r["qid"],
                    )
                    continue
                f.write(json.dumps({
                    "run_tag": r["run_tag"], "dataset": r["dataset"], "config": r["config"],
                    "qid": r["qid"], "judge_model": label, "rubric_version": RUBRIC_VERSION,
                    "generator_model": C.GENERATOR_MODEL, "answerable": not r["unanswerable"],
                    **g,
                }) + "\n")  # fmt: skip
        get_runtime().cache.flush()
        log.info(
            "graded %d/%d (%.0fs)", min(i + batch, len(todo)), len(todo), time.perf_counter() - t0
        )
    log.info("usage this process:\n%s", get_runtime().usage_report())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--model", default=C.JUDGE_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument(
        "--validate", action="store_true", help="grade Claude's audit items and compare"
    )
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr
    )
    for noisy in ("httpx", "google_genai", "deep_context", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.model == C.GENERATOR_MODEL:
        raise SystemExit("Judge model must differ from the generator model.")
    rdir = Path(a.results_dir)
    if a.validate:
        out = validate(rdir, a.model, a.thinking_budget, a.concurrency)
        (rdir / "rubric_validation.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=1))
        print(f"{len(out['disagreements'])} disagreements -> {rdir / 'rubric_validation.json'}")
        return
    score_results(rdir, a.model, a.thinking_budget, a.concurrency, a.batch, a.datasets, a.configs)


if __name__ == "__main__":
    main()
