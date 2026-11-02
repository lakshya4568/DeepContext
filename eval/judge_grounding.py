"""Grounding check: is every claim of an answer backed by the documents the model was shown?

For each answer the judge model (``eval.config.JUDGE_MODEL``, never the generator) sees the question, the numbered
documents exactly as the answer model saw them (``contexts_seen``) and the answer. It splits the answer into factual
claims and says, per claim, whether the documents state it (``supported``), which documents do, and which documents
the answer cited next to it. From that:

* ``faithfulness``: share of claims the documents support (a claim that adds anything the documents do not say is
  unsupported, even when it is true in the real world);
* ``citation_precision``: among claims that carry a citation, the share where a cited document is one that supports it;
* ``n_uncited_claims``: claims with no citation at all.

Exact refusals and empty answers have no claims and are not graded. The judge runs at temperature 0 and every call is
cached on disk, so a second run costs nothing.

    uv run --no-sync python -m eval.judge_grounding --results-dir eval/results_final9b/A_deep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deep_context.retrieval.quality_gates import REFUSAL_TEMPLATE
from eval import config as C
from eval.strict_clients import ApiExhaustedError, get_runtime, stage_scope

log = logging.getLogger("eval.judge_grounding")

VERSION = "g1"
OUT_NAME = "grounding_scores.jsonl"
VERDICTS = ("supported", "unsupported")

PROMPT = """You are checking whether an answer is backed by the documents a question-answering system was given.

Documents the system saw:
{documents}

Question:
{question}

Answer to check:
{answer}

Task:
1. Split the answer into its separate factual claims. Ignore sentences that only say information is missing, and ignore the citation markers themselves.
2. For each claim decide: "supported" if the documents state it or directly imply it; "unsupported" if it adds anything the documents do not say, even if it is true in the real world.
3. For each claim list the numbers of the documents that support it ("support") and the numbers of the documents the answer cites next to it ("cited", an empty list when it cites nothing).

Respond with JSON only: {{"claims": [{{"claim": "<the claim in at most 15 words>", "verdict": "supported|unsupported", "support": [1], "cited": [1]}}]}}
If the answer contains no factual claim, respond with {{"claims": []}}."""


@dataclass(frozen=True)
class Claim:
    claim: str
    supported: bool
    support: list[int]
    cited: list[int]


def build_prompt(question: str, answer: str, documents: Sequence[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {text.strip()}" for i, text in enumerate(documents, start=1))
    return PROMPT.format(
        documents=numbered or "(no documents were shown)",
        question=question.strip(),
        answer=answer.strip(),
    )


def _load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _ints(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    numbers: list[int] = []
    for item in value:
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            continue
    return numbers


def parse_claims(text: str) -> list[Claim] | None:
    """The judge's claims, or None when its output is unusable (the row is then retried, not scored)."""
    data = _load_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        return None
    claims: list[Claim] = []
    for item in data["claims"]:
        if not isinstance(item, dict):
            return None
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            return None
        claims.append(
            Claim(
                str(item.get("claim", "")),
                verdict == "supported",
                _ints(item.get("support")),
                _ints(item.get("cited")),
            )
        )
    return claims


def summarise(claims: Sequence[Claim]) -> dict[str, Any]:
    supported = [c for c in claims if c.supported]
    cited = [c for c in claims if c.cited]
    right = [c for c in cited if c.supported and set(c.cited) & set(c.support)]
    n = len(claims)
    return {
        "n_claims": n,
        "n_supported": len(supported),
        "faithfulness": len(supported) / n if n else None,
        "fully_grounded": len(supported) == n if n else None,
        "n_cited_claims": len(cited),
        "n_citation_right": len(right),
        "citation_precision": len(right) / len(cited) if cited else None,
        "n_uncited_claims": n - len(cited),
    }


def needs_grading(row: dict[str, Any]) -> bool:
    """A real answer: not an error row, not empty and not the exact refusal sentence."""
    answer = str(row.get("answer", "")).strip()
    return not row.get("error") and bool(answer) and REFUSAL_TEMPLATE not in answer


async def grade(
    question: str, answer: str, documents: Sequence[str], *, model: str, thinking_budget: int | None
) -> dict[str, Any]:
    rt = get_runtime()
    with stage_scope("judge"):
        res = await rt.generate(
            model=model,
            messages=[{"role": "user", "content": build_prompt(question, answer, documents)}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=2048,
            json_mode=True,
            thinking_budget=thinking_budget,
            stage="judge",
        )
    claims = parse_claims(res.text)
    if claims is None:
        return {"n_claims": None}
    return {**summarise(claims), "claims": [asdict(c) for c in claims]}


async def grade_all(
    rows: list[dict[str, Any]], model: str, concurrency: int, thinking_budget: int | None = 0
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(r: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            try:
                return await grade(
                    r["query"],
                    r["answer"],
                    r["documents"],
                    model=model,
                    thinking_budget=thinking_budget,
                )
            except (
                ApiExhaustedError
            ) as exc:  # recorded as unusable; the row is retried on the next run
                log.warning("grading failed after retries (%s): %s", r.get("qid"), str(exc)[:120])
                return {"n_claims": None}

    return await asyncio.gather(*(one(r) for r in rows))


def _flush_cache() -> None:
    get_runtime().cache.flush()


def score_results(
    rdir: Path,
    model: str,
    *,
    concurrency: int,
    batch: int,
    datasets: list[str] | None = None,
    configs: list[str] | None = None,
    flush: Callable[[], None] = _flush_cache,
) -> None:
    """Grade every answer of ``rdir/raw_generation.jsonl`` that has no score yet; append to ``grounding_scores.jsonl``."""
    raw = [json.loads(x) for x in (rdir / "raw_generation.jsonl").open() if x.strip()]
    out_path = rdir / OUT_NAME
    done: set[tuple[str, str, str]] = set()
    if out_path.exists():
        for line in out_path.open():
            d = json.loads(line)
            if d.get("judge_model") == model and d.get("grounding_version") == VERSION:
                done.add((d["run_tag"], d["config"], d["qid"]))
    todo = [
        r for r in raw
        if needs_grading(r)
        and (r["run_tag"], r["config"], r["qid"]) not in done
        and (datasets is None or r["dataset"] in datasets)
        and (configs is None or r["config"] in configs)
    ]  # fmt: skip
    log.info("%d answers to grade with grounding %s, judge=%s", len(todo), VERSION, model)
    t0 = time.perf_counter()
    loop = (
        asyncio.new_event_loop()
    )  # one loop for every batch: the judge client of batch 1 is reused later
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        jobs = [
            {
                "qid": r["qid"],
                "query": r["query"],
                "answer": r["answer"],
                "documents": r["contexts_seen"],
            }
            for r in chunk
        ]
        graded = loop.run_until_complete(grade_all(jobs, model, concurrency))
        with out_path.open("a") as f:
            for r, g in zip(chunk, graded, strict=True):
                if g.get("n_claims") is None:
                    log.error(
                        "unusable judgement for %s/%s/%s - not saved",
                        r["dataset"],
                        r["config"],
                        r["qid"],
                    )
                    continue
                f.write(json.dumps({
                    "run_tag": r["run_tag"], "dataset": r["dataset"], "config": r["config"],
                    "qid": r["qid"], "judge_model": model, "grounding_version": VERSION,
                    "generator_model": C.GENERATOR_MODEL, "unanswerable": r.get("unanswerable", False),
                    **g,
                }) + "\n")  # fmt: skip
        flush()
        log.info(
            "graded %d/%d (%.0fs)", min(i + batch, len(todo)), len(todo), time.perf_counter() - t0
        )
    loop.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--model", default=C.JUDGE_MODEL)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr
    )
    for noisy in ("httpx", "google_genai", "deep_context", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.model == C.GENERATOR_MODEL:
        raise SystemExit("Judge model must differ from the generator model.")
    score_results(
        Path(a.results_dir), a.model, concurrency=a.concurrency, batch=a.batch,
        datasets=a.datasets, configs=a.configs,
    )  # fmt: skip
    log.info("usage this process:\n%s", get_runtime().usage_report())


if __name__ == "__main__":
    main()
