"""Why did deep ingestion not help? Free analyses of the saved search rows and of the summaries (no GPU, no Gemini call).

``ranks``      how the best relevant document moved between the fast and the deep ranking, and whether deep creates "hub" documents
               (documents that fill the top 10 of many different questions)
``summaries``  what the summaries contain: how much is the anchor, how much repeats the passage, how much comes from the
               parent context instead of the passage
``title``      does the document title, which the summary repeats in its first words, push documents up whose title matches the
               question (dense search, candidate documents of each question, cached and stored vectors only)
``chunks``     for the best chunk of each relevant document: does the question-to-chunk cosine fall more when the summary adds more new
               words, takes words from the parent's opening instead of the chunk, or is long next to the chunk
``vectors``    bug check and embedding space on the FiQA subset: is every stored summary the summary of its own chunk, is every
               stored vector the vector of the text it should be (read back from the cache, no Gemini call), and how far does the
               summary move the vector away from the questions

    PYTHONPATH=.:src SUMMARY_MODEL=Qwen/Qwen3.5-9B EVAL_STORE=stock uv run --no-sync python eval/scripts_final9b/why_deep.py ranks
    PYTHONPATH=.:src SUMMARY_MODEL=Qwen/Qwen3.5-9B EVAL_STORE=stock uv run --no-sync python eval/scripts_final9b/why_deep.py summaries
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

RESULTS = Path("eval/results_final9b")
CACHE = Path("eval/cache/summaries")
SETS = ("ragbench-covidqa", "scifact", "multihoprag", "fiqa-sub")
STOP = set(
    "the a an and or of to in on for with by from at as is are was were be been it its this that these those which who whom what when where "
    "how not no but if then than so such can could may might will would shall should has have had do does did into over under about after before "
    "their there they them he she his her we our you your i also more most other some any each both between during through while".split()
)


def _rows(folder: str, dataset: str, config: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    path = RESULTS / folder / "raw_retrieval.jsonl"
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if (
                r["dataset"] == dataset
                and r["config"] == config
                and "error" not in r
                and r["relevant"]
            ):
                out[r["qid"]] = r
    return out


def _best_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return i
    return None


def _gini(counts: list[int]) -> float:
    xs = sorted(counts)
    n, total = len(xs), sum(xs)
    return (
        0.0
        if not total
        else sum((2 * i - n - 1) * x for i, x in enumerate(xs, start=1)) / (n * total)
    )


def ranks() -> None:
    from eval.main_summaries import task_for

    print("## How the best relevant document moved (dense only and mixed 0.02; fast to deep)\n")
    print(
        "| set | setting | questions | deep ranks the relevant document higher | same place | lower | median move of the changed (places) | in the top 10 in fast, not in deep | in the top 10 in deep, not in fast |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    hub_lines: list[str] = []
    for dataset in SETS:
        for label, folder_f, folder_d, config in (
            ("dense only", "R_fast_equal", "R_deep_equal", "dense_only"),
            ("mixed 0.02", "R_fast_vote0.02", "R_deep_vote0.02", "rrf"),
        ):
            fast, deep = _rows(folder_f, dataset, config), _rows(folder_d, dataset, config)
            qids = sorted(set(fast) & set(deep))
            up = same = down = lost = gained = 0
            moves: list[int] = []
            for q in qids:
                rel = set(fast[q]["relevant"])
                bf, bd = _best_rank(fast[q]["ranked"], rel), _best_rank(deep[q]["ranked"], rel)
                bf, bd = bf or 10_000, bd or 10_000
                if bd < bf:
                    up += 1
                elif bd > bf:
                    down += 1
                else:
                    same += 1
                if bd != bf:
                    moves.append(min(bd, 1000) - min(bf, 1000))
                lost += bf <= 10 < bd
                gained += bd <= 10 < bf
            med = statistics.median(moves) if moves else 0
            n = len(qids)
            print(
                f"| {dataset} | {label} | {n} | {up} ({100 * up / n:.0f}%) | {same} ({100 * same / n:.0f}%) | {down} ({100 * down / n:.0f}%) | {med:+.0f} | {lost} | {gained} |"
            )
            if config == "dense_only":
                counts_f, counts_d = Counter(), Counter()
                for q in qids:
                    counts_f.update(fast[q]["ranked"][:10])
                    counts_d.update(deep[q]["ranked"][:10])
                slots = 10 * n
                top_f, top_d = counts_f.most_common(1)[0], counts_d.most_common(1)[0]
                hub_lines.append(
                    f"| {dataset} | {len(counts_f)} | {len(counts_d)} | {top_f[1]} | {top_d[1]} | "
                    f"{sum(c for _d, c in counts_f.most_common(max(1, len(counts_f) // 100))) / slots:.1%} | "
                    f"{sum(c for _d, c in counts_d.most_common(max(1, len(counts_d) // 100))) / slots:.1%} | "
                    f"{_gini(list(counts_f.values())):.2f} | {_gini(list(counts_d.values())):.2f} |"
                )
    print(
        "\n## Hub documents: who fills the top 10 of the dense ranking (all questions together)\n"
    )
    print(
        "| set | different documents in the top 10s, fast | deep | most frequent document appears in this many top 10s, fast | deep | share of all top-10 places held by the busiest 1% of documents, fast | deep | Gini of appearances, fast | deep |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(hub_lines))

    # intruders: documents that deep puts in the top 10 of a question and fast does not (dense only), against the documents it drops
    print(
        "\n## Length of the documents that deep brings into the top 10 and the ones it drops (dense only, characters of the document text)\n"
    )
    print(
        "| set | all documents (median) | brought in by deep (median) | dropped by deep (median) |"
    )
    print("|---|---|---|---|")
    for dataset in SETS:
        corpus = task_for(dataset).corpus
        length = {d: len(v.get("text", "")) for d, v in corpus.items()}
        fast, deep = (
            _rows("R_fast_equal", dataset, "dense_only"),
            _rows("R_deep_equal", dataset, "dense_only"),
        )
        brought: list[int] = []
        dropped: list[int] = []
        for q in set(fast) & set(deep):
            f10, d10 = set(fast[q]["ranked"][:10]), set(deep[q]["ranked"][:10])
            brought += [length[d] for d in d10 - f10 if d in length]
            dropped += [length[d] for d in f10 - d10 if d in length]
        print(
            f"| {dataset} | {statistics.median(length.values()):.0f} | {statistics.median(brought):.0f} | {statistics.median(dropped):.0f} |"
        )


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'\-]{2,}", text.lower()) if w not in STOP]


def _prompts(dataset: str, jobs_dir: Path | None) -> dict[str, str]:
    """key -> prompt of a dataset: from the exported job files when given, else rebuilt from the corpus (slow)."""
    if jobs_dir is None:
        from eval.main_summaries import requests_for_dataset

        return {r.key: r.prompt for r in requests_for_dataset(dataset)}
    name = "fiqa" if dataset == "fiqa-sub" else dataset
    out: dict[str, str] = {}
    for line in (jobs_dir / f"jobs9b_{name}.jsonl").open(encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            out[d["key"]] = d["prompt"]
    return out


def summaries(jobs_dir: Path | None) -> None:
    print("## What the summaries contain (random sample of up to 3,000 per set)\n")
    print(
        "| set | mean words in the summary | of which the opening anchor ('In <title> (<section>),') | mean words in the passage | share of the summary's content words found in the passage | found only in the parent context | found nowhere in the prompt | summaries that start with a bare number as title |"
    )
    print("|---|---|---|---|---|---|---|---|")
    files = {
        "ragbench-covidqa": ["results9b_ragbench-covidqa.jsonl"],
        "scifact": ["results9b_scifact.jsonl"],
        "multihoprag": ["results9b_multihoprag.jsonl"],
        "fiqa-sub": ["results9b_fiqa.jsonl", "results9b_fiqa-sub-extra.jsonl"],
    }
    for dataset in SETS:
        results: dict[str, str] = {}
        for file in files[dataset]:
            for line in (CACHE / file).open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    results[d["key"]] = d["summary"]
        prompts = _prompts(dataset, jobs_dir)
        reqs = [(key, prompts[key]) for key in prompts if key in results]
        random.Random(7).shuffle(reqs)
        n = s_words = a_words = p_words = in_p = in_c = nowhere = total = bare = 0
        for key, prompt in reqs[:3000]:
            passage = prompt.split("<passage>", 1)[-1].split("</passage>", 1)[0]
            document = prompt.split("<document>", 1)[-1].split("</document>", 1)[0]
            context = document.split("Context:", 1)[-1] if "Context:" in document else ""
            summary = results[key]
            anchor = re.match(r"In (.*?) \([^)]*\),\s*", summary)
            body = summary[anchor.end() :] if anchor else summary
            n += 1
            s_words += len(summary.split())
            a_words += len(anchor.group(0).split()) if anchor else 0
            p_words += len(passage.split())
            bare += bool(anchor and re.fullmatch(r"\d+", anchor.group(1).strip()))
            pset, cset = set(_words(passage)), set(_words(context))
            for w in _words(body):
                total += 1
                if w in pset:
                    in_p += 1
                elif w in cset:
                    in_c += 1
                else:
                    nowhere += 1
        print(
            f"| {dataset} | {s_words / n:.1f} | {a_words / n:.1f} | {p_words / n:.0f} | {in_p / total:.0%} | {in_c / total:.0%} | {nowhere / total:.0%} | {bare / n:.0%} |"
        )


def vectors(dataset: str) -> None:
    import asyncio

    import asyncpg
    import numpy as np
    from eval import config as C
    from eval import pg
    from eval.main_summaries import task_for
    from eval.strict_clients import _sha, get_runtime

    rt = get_runtime()

    def cached(text: str, *, query: bool = False, title: str | None = None) -> np.ndarray | None:
        formatted = rt.format_embedding_text(text, is_query=query, title=title, task=None)
        hit = rt.cache.get(_sha("emb", C.EMBEDDING_MODEL, C.EMBEDDING_DIM, formatted))
        return None if hit is None else np.frombuffer(hit, dtype=np.float32)

    def unit(m: np.ndarray) -> np.ndarray:
        return m / np.linalg.norm(m, axis=-1, keepdims=True)

    async def run() -> None:
        conn = await asyncpg.connect(pg.dsn_for("deepcontext_eval"))
        try:
            names = [
                r["nspname"]
                for r in await conn.fetch(
                    "select nspname from pg_namespace where nspname like $1",
                    dataset.replace("-", "_") + "_%",
                )
            ]
            fast_s = next(n for n in names if n.endswith("_main"))
            deep_s = next(n for n in names if n.endswith("_maindeep9b"))
            sample = await conn.fetch(
                f"select c.id, c.content, c.summary_text, c.embedding::real[] as de, d.title as title, d.metadata->>'external_id' as ext "
                f'from "{deep_s}".chunks c join "{deep_s}".documents d on d.id = c.document_id '
                f"where c.level = 'child' and c.summary_text is not null order by random() limit 600"
            )
            print(f"{dataset}: deep schema {deep_s}: sampled {len(sample)} child chunks\n")
            # 1. is each summary the summary of its own chunk?
            rng = random.Random(3)
            own, other = [], []
            contents = [r["content"] for r in sample]
            for r in sample:
                sw = set(_words(re.sub(r"^In .*?\([^)]*\),\s*", "", r["summary_text"])))
                if not sw:
                    continue
                own.append(len(sw & set(_words(r["content"]))) / len(sw))
                other.append(len(sw & set(_words(rng.choice(contents)))) / len(sw))
            print(
                f"1. summary vs chunk word overlap: own chunk {statistics.mean(own):.0%}, a random other chunk {statistics.mean(other):.0%}"
            )
            # 2. is the stored vector the vector of the text it should be?
            missing = 0
            cos_deep_expected: list[float] = []
            for r in sample:
                title = r["title"] if r["title"] != r["ext"] else None
                expected = cached(f"{r['summary_text']}\n\n{r['content']}", title=title)
                if expected is None:
                    missing += 1
                    continue
                stored = np.asarray(r["de"], dtype=np.float32)
                cos_deep_expected.append(float(unit(stored) @ unit(expected)))
            print(
                f"2. stored deep vector against the cached vector of (summary + chunk): {len(cos_deep_expected)} compared, {missing} not in the cache, "
                f"lowest cosine {min(cos_deep_expected):.5f}"
            )
            ids = [r["id"] for r in sample]
            fast_rows = await conn.fetch(
                f"select c.id, c.content, c.embedding::real[] as fe, d.title as title, d.metadata->>'external_id' as ext "
                f'from "{fast_s}".chunks c join "{fast_s}".documents d on d.id = c.document_id where c.id = any($1::uuid[])',
                ids,
            )
            cos_fast_expected = []
            for r in fast_rows:
                title = r["title"] if r["title"] != r["ext"] else None
                expected = cached(r["content"], title=title)
                if expected is not None:
                    cos_fast_expected.append(
                        float(unit(np.asarray(r["fe"], dtype=np.float32)) @ unit(expected))
                    )
            print(
                f"   stored fast vector against the cached vector of the chunk: {len(cos_fast_expected)} compared, lowest cosine {min(cos_fast_expected):.5f}\n"
            )
            # 3. how far does the summary move the vector, and does the space get flatter?
            fast_by_id = {r["id"]: np.asarray(r["fe"], dtype=np.float32) for r in fast_rows}
            pairs = [
                (np.asarray(r["de"], dtype=np.float32), fast_by_id[r["id"]])
                for r in sample
                if r["id"] in fast_by_id
            ]
            D, F = unit(np.stack([a for a, _ in pairs])), unit(np.stack([b for _, b in pairs]))
            same = (D * F).sum(1)
            idx = np.random.default_rng(5).permutation(len(D))
            print(
                f"3. cosine between the deep and the fast vector of the SAME chunk: mean {same.mean():.3f} (1.0 would mean the summary changed nothing)"
            )
            print(
                f"   mean cosine between two DIFFERENT random chunks: fast {(F * F[idx]).sum(1).mean():.3f}, deep {(D * D[idx]).sum(1).mean():.3f}"
            )
            # 4. the questions: relevant document against random chunks, in both spaces
            task = task_for(dataset)
            rand = await conn.fetch(
                f"select id from \"{deep_s}\".chunks where level = 'child' order by random() limit 3000"
            )
            rid = [r["id"] for r in rand]
            fr = {
                r["id"]: np.asarray(r["e"], dtype=np.float32)
                for r in await conn.fetch(
                    f'select id, embedding::real[] as e from "{fast_s}".chunks where id = any($1::uuid[])',
                    rid,
                )
            }
            dr = {
                r["id"]: np.asarray(r["e"], dtype=np.float32)
                for r in await conn.fetch(
                    f'select id, embedding::real[] as e from "{deep_s}".chunks where id = any($1::uuid[])',
                    rid,
                )
            }
            RF, RD = unit(np.stack([fr[i] for i in rid])), unit(np.stack([dr[i] for i in rid]))
            margins_f, margins_d, beaten_f, beaten_d = [], [], [], []
            used = 0
            for qid, query in task.queries.items():
                if used >= 250:
                    break
                qv = cached(query, query=True)
                rel = list(task.qrels[qid])
                if qv is None or not rel:
                    continue
                rows = await conn.fetch(
                    f'select c.id, c.embedding::real[] as de from "{deep_s}".chunks c join "{deep_s}".documents d on d.id = c.document_id '
                    f"where c.level = 'child' and d.metadata->>'external_id' = any($1::text[])",
                    rel,
                )
                if not rows:
                    continue
                fids = [r["id"] for r in rows]
                frows = {
                    r["id"]: np.asarray(r["e"], dtype=np.float32)
                    for r in await conn.fetch(
                        f'select id, embedding::real[] as e from "{fast_s}".chunks where id = any($1::uuid[])',
                        fids,
                    )
                }
                q = unit(qv)
                sim_f = max(float(unit(frows[i]) @ q) for i in fids if i in frows)
                sim_d = max(float(unit(np.asarray(r["de"], dtype=np.float32)) @ q) for r in rows)
                rf, rd = RF @ q, RD @ q
                margins_f.append(sim_f - rf.mean())
                margins_d.append(sim_d - rd.mean())
                beaten_f.append(float((rf > sim_f).mean()))
                beaten_d.append(float((rd > sim_d).mean()))
                used += 1
            print(
                f"\n4. {used} questions: the best chunk of a relevant document against 3,000 random chunks"
            )
            print(
                f"   cosine to the relevant document minus the average cosine to random chunks: fast {statistics.mean(margins_f):.3f}, deep {statistics.mean(margins_d):.3f}"
            )
            print(
                f"   share of the random chunks that score higher than the relevant document: fast {statistics.mean(beaten_f):.2%}, deep {statistics.mean(beaten_d):.2%}"
            )
        finally:
            await conn.close()

    asyncio.run(run())


def title_boost(dataset: str) -> None:
    import asyncio

    import asyncpg
    import numpy as np
    from eval import config as C
    from eval import pg
    from eval.main_summaries import task_for
    from eval.strict_clients import _sha, get_runtime

    rt = get_runtime()
    task = task_for(dataset)
    titles = {d: v.get("title", "") for d, v in task.corpus.items()}

    def query_vector(query: str) -> np.ndarray | None:
        formatted = rt.format_embedding_text(query, is_query=True, title=None, task=None)
        hit = rt.cache.get(_sha("emb", C.EMBEDDING_MODEL, C.EMBEDDING_DIM, formatted))
        return None if hit is None else np.frombuffer(hit, dtype=np.float32)

    def overlap(query: str, title: str) -> float:
        t = set(_words(title))
        return len(t & set(_words(query))) / len(t) if t else 0.0

    async def run() -> None:
        conn = await asyncpg.connect(pg.dsn_for("deepcontext_eval"))
        try:
            names = [
                r["nspname"]
                for r in await conn.fetch(
                    "select nspname from pg_namespace where nspname like $1",
                    dataset.replace("-", "_") + "_%",
                )
            ]
            fast_s = next(n for n in names if n.endswith("_main"))
            deep_s = next(n for n in names if n.endswith("_maindeep9b"))
            fast, deep = (
                _rows("R_fast_equal", dataset, "dense_only"),
                _rows("R_deep_equal", dataset, "dense_only"),
            )
            qids = sorted(set(fast) & set(deep))
            random.Random(11).shuffle(qids)
            records: list[tuple[float, float, bool]] = []  # (delta score, title overlap, relevant)
            for qid in qids[:200]:
                qv = query_vector(task.queries[qid])
                if qv is None:
                    continue
                q = qv / np.linalg.norm(qv)
                rel = set(fast[qid]["relevant"])
                pool = list(rel | set(fast[qid]["ranked"][:10]) | set(deep[qid]["ranked"][:10]))
                rows = await conn.fetch(
                    f"select c.id, d.metadata->>'external_id' as ext, c.embedding::real[] as de from \"{deep_s}\".chunks c "
                    f"join \"{deep_s}\".documents d on d.id = c.document_id where c.level = 'child' and d.metadata->>'external_id' = any($1::text[])",
                    pool,
                )
                frows = {
                    r["id"]: np.asarray(r["e"], dtype=np.float32)
                    for r in await conn.fetch(
                        f'select id, embedding::real[] as e from "{fast_s}".chunks where id = any($1::uuid[])',
                        [r["id"] for r in rows],
                    )
                }
                best_f: dict[str, float] = {}
                best_d: dict[str, float] = {}
                for r in rows:
                    ext = r["ext"]
                    f, d = frows[r["id"]], np.asarray(r["de"], dtype=np.float32)
                    best_f[ext] = max(best_f.get(ext, -1.0), float(f / np.linalg.norm(f) @ q))
                    best_d[ext] = max(best_d.get(ext, -1.0), float(d / np.linalg.norm(d) @ q))
                for ext in best_f:
                    records.append(
                        (
                            best_d[ext] - best_f[ext],
                            overlap(task.queries[qid], titles.get(ext, "")),
                            ext in rel,
                        )
                    )
            delta = np.array([r[0] for r in records])
            ov = np.array([r[1] for r in records])
            rel_flag = np.array([r[2] for r in records])
            print(
                f"{dataset}: {len(records)} (question, document) pairs from {min(200, len(qids))} questions"
            )
            print(
                f"   change in the question-to-document cosine (deep minus fast): relevant documents {delta[rel_flag].mean():+.4f}, other candidates {delta[~rel_flag].mean():+.4f}"
            )
            if ov.std() > 0:
                print(
                    f"   correlation of that change with the share of the document's title words that the question uses: {np.corrcoef(delta, ov)[0, 1]:+.2f}"
                )
                for lo, hi, label in (
                    (0, 1e-9, "no title word in the question"),
                    (1e-9, 0.34, "up to a third of the title words"),
                    (0.34, 1.01, "more than a third of the title words"),
                ):
                    sel = (ov >= lo) & (ov < hi)
                    if sel.sum():
                        print(
                            f"   {label}: {sel.sum()} pairs, mean change {delta[sel].mean():+.4f}"
                        )
        finally:
            await conn.close()

    asyncio.run(run())


def chunks(dataset: str) -> None:
    import asyncio

    import asyncpg
    import numpy as np
    from eval import config as C
    from eval import pg
    from eval.main_summaries import task_for
    from eval.strict_clients import _sha, get_runtime

    rt = get_runtime()
    task = task_for(dataset)

    def query_vector(query: str) -> np.ndarray | None:
        formatted = rt.format_embedding_text(query, is_query=True, title=None, task=None)
        hit = rt.cache.get(_sha("emb", C.EMBEDDING_MODEL, C.EMBEDDING_DIM, formatted))
        return None if hit is None else np.frombuffer(hit, dtype=np.float32)

    async def run() -> None:
        conn = await asyncpg.connect(pg.dsn_for("deepcontext_eval"))
        try:
            names = [
                r["nspname"]
                for r in await conn.fetch(
                    "select nspname from pg_namespace where nspname like $1",
                    dataset.replace("-", "_") + "_%",
                )
            ]
            fast_s = next(n for n in names if n.endswith("_main"))
            deep_s = next(n for n in names if n.endswith("_maindeep9b"))
            qids = [q for q in task.queries if task.qrels.get(q)]
            random.Random(13).shuffle(qids)
            rows_out: list[
                tuple[float, float, float, float]
            ] = []  # delta, new share, bleed share, dilution
            for qid in qids[:250]:
                qv = query_vector(task.queries[qid])
                if qv is None:
                    continue
                q = qv / np.linalg.norm(qv)
                rows = await conn.fetch(
                    f"select c.id, c.content, c.summary_text, p.content as parent, d.metadata->>'external_id' as ext, c.embedding::real[] as de "
                    f'from "{deep_s}".chunks c join "{deep_s}".documents d on d.id = c.document_id '
                    f'left join "{deep_s}".chunks p on p.id = c.parent_chunk_id '
                    f"where c.level = 'child' and d.metadata->>'external_id' = any($1::text[])",
                    list(task.qrels[qid]),
                )
                if not rows:
                    continue
                frows = {
                    r["id"]: np.asarray(r["e"], dtype=np.float32)
                    for r in await conn.fetch(
                        f'select id, embedding::real[] as e from "{fast_s}".chunks where id = any($1::uuid[])',
                        [r["id"] for r in rows],
                    )
                }
                best: dict[str, tuple[float, Any]] = {}
                for r in rows:
                    f = frows[r["id"]]
                    sim = float(f / np.linalg.norm(f) @ q)
                    if r["ext"] not in best or sim > best[r["ext"]][0]:
                        best[r["ext"]] = (sim, r)
                for sim_f, r in best.values():
                    d = np.asarray(r["de"], dtype=np.float32)
                    delta = float(d / np.linalg.norm(d) @ q) - sim_f
                    body = re.sub(r"^In .*?\([^)]*\),\s*", "", r["summary_text"] or "")
                    sw = _words(body)
                    if not sw:
                        continue
                    cset, pset = set(_words(r["content"])), set(_words((r["parent"] or "")[:800]))
                    new = [w for w in sw if w not in cset]
                    bleed = [w for w in new if w in pset]
                    rows_out.append(
                        (
                            delta,
                            len(new) / len(sw),
                            len(bleed) / len(sw),
                            len(body.split()) / (len(body.split()) + len(r["content"].split())),
                        )
                    )
            a = np.array(rows_out)
            print(
                f"{dataset}: {len(a)} (question, best chunk of a relevant document) pairs; mean change of the cosine {a[:, 0].mean():+.4f}"
            )
            for col, name in (
                (1, "share of the summary's words that are not in the chunk"),
                (2, "share taken from the parent's opening instead of the chunk"),
                (3, "summary length next to chunk length"),
            ):
                if a[:, col].std() > 0:
                    lo, hi = np.quantile(a[:, col], [1 / 3, 2 / 3])
                    parts = [
                        a[a[:, col] <= lo, 0].mean(),
                        a[(a[:, col] > lo) & (a[:, col] <= hi), 0].mean(),
                        a[a[:, col] > hi, 0].mean(),
                    ]
                    print(
                        f"   {name}: correlation with the change {np.corrcoef(a[:, 0], a[:, col])[0, 1]:+.2f}; lowest / middle / highest third: {parts[0]:+.4f} / {parts[1]:+.4f} / {parts[2]:+.4f}"
                    )
        finally:
            await conn.close()

    asyncio.run(run())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("what", choices=["ranks", "summaries", "vectors", "title", "chunks"])
    parser.add_argument("--dataset", default="fiqa-sub", help="for 'vectors'")
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=None,
        help="folder with jobs9b_<dataset>.jsonl (else the prompts are rebuilt)",
    )
    args = parser.parse_args()
    {
        "ranks": ranks,
        "summaries": lambda: summaries(args.jobs_dir),
        "vectors": lambda: vectors(args.dataset),
        "title": lambda: title_boost(args.dataset),
        "chunks": lambda: chunks(args.dataset),
    }[args.what]()


if __name__ == "__main__":
    main()
