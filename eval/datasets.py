"""Dataset loaders for the public benchmarks (BEIR, MultiHop-RAG, RAGBench, RGB).

Every loader returns a ``Task`` with the same shape so the runners are dataset-agnostic.
Subsampling is deterministic (seed 42) and always keeps *all* relevant documents of the selected
queries plus random distractors; the exact procedure is recorded in ``Task.meta`` and surfaces in
REPORT.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval import config as C

log = logging.getLogger("eval.datasets")

# Verified 2026-10-18 from the cited sources (see REPORT.md §2).
BEIR_LICENSES = {
    "scifact": "claims/annotations CC BY 4.0; abstracts ODC-By 1.0 (S2ORC) — allenai/scifact LICENSE.md",
    "fiqa": "HF BeIR card: cc-by-sa-4.0 (packaging); FiQA-2018 upstream terms not stated — UNVERIFIED",
    "hotpotqa": "CC BY-SA 4.0 — hotpotqa.github.io",
}


@dataclass
class Task:
    name: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]  # relevant docs per query (rel > 0)
    answers: dict[str, str] = field(default_factory=dict)  # reference answers (QA tasks)
    qtypes: dict[str, str] = field(default_factory=dict)
    unanswerable: set[str] = field(default_factory=set)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def qids(self) -> list[str]:
        return list(self.queries)


# ======================================================================================
# Sampling helpers
# ======================================================================================
def stratified_sample(groups: dict[str, list[str]], n: int, seed: int = C.SAMPLE_SEED) -> list[str]:
    """Proportional (largest-remainder) deterministic sample of ``n`` ids across groups."""
    total = sum(len(v) for v in groups.values())
    if n >= total:
        return sorted(i for v in groups.values() for i in v)
    quotas = {g: n * len(v) / total for g, v in groups.items()}
    alloc = {g: int(q) for g, q in quotas.items()}
    leftover = n - sum(alloc.values())
    for g in sorted(quotas, key=lambda g: (quotas[g] - alloc[g], g), reverse=True)[:leftover]:
        alloc[g] += 1
    rng = random.Random(seed)
    picked: list[str] = []
    for g in sorted(groups):
        picked += rng.sample(sorted(groups[g]), alloc[g])
    return picked


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    log.info("Downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed, documented dataset URLs
    tmp.rename(dest)


# ======================================================================================
# BEIR
# ======================================================================================
def _beir_source(name: str, split: str):  # type: ignore[no-untyped-def]
    """Return (queries, qrels, corpus_iter_factory, info) for a BEIR dataset.

    Read from the Hugging Face BEIR mirror (``BeIR/<name>`` + ``BeIR/<name>-qrels``): the
    official UKP host served ~15 KB/s during this run. The mirror carries the same BEIR release;
    the HF commit hashes are recorded in the task metadata for reproducibility.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    out = C.DATA_DIR / "beir_hf"
    corpus_p = hf_hub_download(f"BeIR/{name}", "corpus/corpus-00000-of-00001.parquet",
                               repo_type="dataset", local_dir=out / name)  # fmt: skip
    queries_p = hf_hub_download(f"BeIR/{name}", "queries/queries-00000-of-00001.parquet",
                                repo_type="dataset", local_dir=out / name)  # fmt: skip
    qrels_p = hf_hub_download(f"BeIR/{name}-qrels", f"{split}.tsv", repo_type="dataset",
                              local_dir=out / f"{name}-qrels")  # fmt: skip
    api = HfApi()
    queries = {str(r["_id"]): r["text"] for r in pq.read_table(queries_p).to_pylist()}
    qrels: dict[str, dict[str, int]] = {}
    for i, line in enumerate(Path(qrels_p).open()):
        if i == 0 and line.lower().startswith("query-id"):
            continue
        qid, did, score = line.rstrip("\n").split("\t")[:3]
        if int(float(score)) > 0:
            qrels.setdefault(qid, {})[did] = int(float(score))

    def corpus_iter():  # type: ignore[no-untyped-def]
        pf = pq.ParquetFile(corpus_p)
        for batch in pf.iter_batches(batch_size=50_000, columns=["_id", "title", "text"]):
            d = batch.to_pydict()
            yield from zip((str(x) for x in d["_id"]), d["title"], d["text"], strict=True)

    info = {
        "source": f"HF mirror BeIR/{name} @ {api.dataset_info(f'BeIR/{name}').sha[:10]} "
        f"+ BeIR/{name}-qrels @ {api.dataset_info(f'BeIR/{name}-qrels').sha[:10]}",
    }
    return queries, qrels, corpus_iter, info


def load_beir(
    name: str,
    *,
    n_queries: int | None = None,
    max_docs: int | None = None,
    split: str = "test",
    seed: int = C.SAMPLE_SEED,
) -> Task:
    """Load a BEIR dataset; optionally sample queries and cap the corpus (relevant docs kept)."""
    suffix = "" if split == "test" else f"_{split}"
    cache = C.DATA_DIR / f"{name}_sub_q{n_queries}_d{max_docs}_s{seed}{suffix}.json"
    if cache.exists():
        return _task_from_json(json.loads(cache.read_text()))

    queries_all, qrels_all, corpus_iter, info = _beir_source(name, split)
    qids = sorted(q for q in qrels_all if q in queries_all)
    n_total_queries = len(qids)
    if n_queries is not None and n_queries < len(qids):
        qids = sorted(random.Random(seed).sample(qids, n_queries))
    qrels = {q: qrels_all[q] for q in qids}
    relevant = {d for q in qids for d in qrels[q]}

    rng = random.Random(seed)
    corpus: dict[str, dict[str, str]] = {}
    reservoir: list[tuple[str, dict[str, str]]] = []
    seen_distractors = 0
    cap = None if max_docs is None else max(0, max_docs - len(relevant))
    n_corpus_total = 0
    for did, title, text in corpus_iter():
        n_corpus_total += 1
        doc = {"title": title or "", "text": text or ""}
        if did in relevant or cap is None:
            corpus[did] = doc
            continue
        seen_distractors += 1  # Algorithm R reservoir sampling (seeded)
        if len(reservoir) < cap:
            reservoir.append((did, doc))
        else:
            j = rng.randint(0, seen_distractors - 1)
            if j < cap:
                reservoir[j] = (did, doc)
    for did, doc in reservoir:
        corpus[did] = doc
    missing = relevant - set(corpus)
    task = Task(
        name=name,
        corpus=corpus,
        queries={q: queries_all[q] for q in qids},
        qrels=qrels,
        meta={
            **info,
            "license": BEIR_LICENSES.get(name, "see source"),
            "split": split,
            "queries_available": n_total_queries,
            "queries_used": len(qids),
            "corpus_total_docs": n_corpus_total,
            "corpus_used_docs": len(corpus),
            "relevant_docs_in_used_corpus": len(relevant) - len(missing),
            "relevant_docs_missing_from_corpus": len(missing),
            "sampling": (
                "full corpus"
                if max_docs is None
                else f"seed={seed}: all {len(relevant)} relevant docs + "
                f"{len(reservoir)} uniformly sampled distractors (Algorithm R)"
            )
            + (
                f"; {len(qids)}/{n_total_queries} queries sampled (seed={seed})"
                if len(qids) < n_total_queries
                else "; all queries"
            ),
        },
    )
    cache.write_text(json.dumps(_task_to_json(task)))
    return task


# ======================================================================================
# MultiHop-RAG
# ======================================================================================
def load_multihoprag(n_queries: int | None = None, seed: int = C.SAMPLE_SEED) -> Task:
    from huggingface_hub import hf_hub_download

    d = C.DATA_DIR / "multihoprag"
    q_path = hf_hub_download(
        "yixuantt/MultiHopRAG", "MultiHopRAG.json", repo_type="dataset", local_dir=d
    )
    c_path = hf_hub_download(
        "yixuantt/MultiHopRAG", "corpus.json", repo_type="dataset", local_dir=d
    )
    raw_q = json.loads(Path(q_path).read_text())
    raw_c = json.loads(Path(c_path).read_text())
    title_to_id = {doc["title"]: f"mh{i:04d}" for i, doc in enumerate(raw_c)}
    corpus = {
        title_to_id[doc["title"]]: {
            "title": doc["title"],
            "text": f"{doc['source']} ({doc['published_at'][:10]}). {doc['body']}",
        }
        for doc in raw_c
    }
    groups: dict[str, list[str]] = {}
    for i, q in enumerate(raw_q):
        groups.setdefault(q["question_type"], []).append(f"q{i:04d}")
    qids = stratified_sample(groups, n_queries or len(raw_q), seed)
    queries, qrels, answers, qtypes = {}, {}, {}, {}
    unanswerable: set[str] = set()
    for qid in sorted(qids):
        q = raw_q[int(qid[1:])]
        queries[qid] = q["query"]
        answers[qid] = q["answer"]
        qtypes[qid] = q["question_type"]
        qrels[qid] = {title_to_id[e["title"]]: 1 for e in q["evidence_list"]}
        if q["question_type"] == "null_query":
            unanswerable.add(qid)
    return Task(
        name="multihoprag",
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        answers=answers,
        qtypes=qtypes,
        unanswerable=unanswerable,
        meta={
            "source": "HF yixuantt/MultiHopRAG (MultiHopRAG.json + corpus.json)",
            "license": "ODC-By (per HF dataset card)",
            "queries_available": len(raw_q),
            "queries_used": len(queries),
            "corpus_total_docs": len(raw_c),
            "corpus_used_docs": len(corpus),
            "sampling": (
                f"stratified by question_type, seed={seed}" if n_queries else "all queries"
            )
            + "; full 609-article corpus",
            "null_queries_used": len(unanswerable),
        },
    )


# ======================================================================================
# RAGBench
# ======================================================================================
def load_ragbench(
    subset: str = "covidqa",
    split: str = "test",
    n_queries: int | None = None,
    seed: int = C.SAMPLE_SEED,
) -> Task:
    from datasets import load_dataset

    ds = load_dataset("galileo-ai/ragbench", subset, split=split)
    ids = [f"r{i:04d}" for i in range(len(ds))]
    qids = (
        sorted(random.Random(seed).sample(ids, n_queries))
        if n_queries and n_queries < len(ids)
        else ids
    )
    corpus: dict[str, dict[str, str]] = {}
    queries, qrels, answers = {}, {}, {}
    for qid in qids:
        row = ds[int(qid[1:])]
        queries[qid] = row["question"]
        answers[qid] = row["response"]
        doc_ids = []
        for text in row["documents"]:
            did = "d" + hashlib.sha1(text.encode()).hexdigest()[:16]
            corpus.setdefault(did, {"title": "", "text": text})
            doc_ids.append(did)
        rel_idx = {int(re.match(r"\d+", k).group()) for k in row["all_relevant_sentence_keys"]}  # type: ignore[union-attr]
        qrels[qid] = {doc_ids[i]: 1 for i in rel_idx if i < len(doc_ids)}
    keep = [q for q in qids if qrels[q]]  # queries with at least one annotated relevant passage
    return Task(
        name=f"ragbench-{subset}",
        corpus=corpus,
        queries={q: queries[q] for q in keep},
        qrels={q: qrels[q] for q in keep},
        answers={q: answers[q] for q in keep},
        meta={
            "source": f"HF galileo-ai/ragbench ({subset}, {split})",
            "license": "CC BY 4.0 (per HF dataset card)",
            "queries_available": len(ds),
            "queries_used": len(keep),
            "queries_dropped_no_relevant_passage": len(qids) - len(keep),
            "corpus_used_docs": len(corpus),
            "sampling": f"all {split} rows" if not n_queries else f"seed={seed} sample",
            "reference_caveat": (
                "RAGBench 'response' is an LLM-generated answer (not human gold); "
                "reference-based RAGAS metrics are therefore only a proxy."
            ),
        },
    )


# ======================================================================================
# RGB (generation-only: passages are supplied, no retrieval)
# ======================================================================================
RGB_URL = "https://raw.githubusercontent.com/chen700564/RGB/master/data/{f}.json"


def load_rgb(
    kind: str = "en_refine", n: int | None = None, seed: int = C.SAMPLE_SEED
) -> list[dict[str, Any]]:
    """Rows of the RGB benchmark (JSON lines). ``kind``: en_refine | en_int | en_fact."""
    path = C.DATA_DIR / "rgb" / f"{kind}.json"
    _download(RGB_URL.format(f=kind), path)
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if n and n < len(rows):
        rows = sorted(random.Random(seed).sample(rows, n), key=lambda r: int(r["id"]))
    return rows


# ======================================================================================
# (de)serialisation for cached subsets
# ======================================================================================
def _task_to_json(t: Task) -> dict[str, Any]:
    return {
        "name": t.name, "corpus": t.corpus, "queries": t.queries, "qrels": t.qrels,
        "answers": t.answers, "qtypes": t.qtypes, "unanswerable": sorted(t.unanswerable),
        "meta": t.meta,
    }  # fmt: skip


def _task_from_json(d: dict[str, Any]) -> Task:
    return Task(
        name=d["name"], corpus=d["corpus"], queries=d["queries"], qrels=d["qrels"],
        answers=d.get("answers", {}), qtypes=d.get("qtypes", {}),
        unanswerable=set(d.get("unanswerable", [])), meta=d.get("meta", {}),
    )  # fmt: skip


# ======================================================================================
# Dispatcher
# ======================================================================================
PUBLIC_DATASETS = ("scifact", "fiqa", "hotpotqa", "multihoprag", "ragbench-covidqa")


def build_task(
    name: str, n_queries: int | None = None, max_docs: int | None = None, split: str = "test"
) -> Task:
    if name in ("scifact", "fiqa", "hotpotqa"):
        return load_beir(name, n_queries=n_queries, max_docs=max_docs, split=split)
    if name == "multihoprag":
        return load_multihoprag(n_queries=n_queries)
    if name.startswith("ragbench-"):
        return load_ragbench(name.split("-", 1)[1], n_queries=n_queries)
    if name == "docsqa":
        from eval.docs_qa import load_docs_qa

        return load_docs_qa()
    if name.startswith("pdf-"):
        from eval.pdf_sets import load_pdf_set

        return load_pdf_set(name)
    raise ValueError(f"Unknown dataset '{name}'.")
