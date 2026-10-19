"""Assemble REPORT.md from the CSV/JSON artifacts in the results directory.

Prose is limited to method descriptions and caveats; every number comes from a results file.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path

import pandas as pd

from eval import config as C
from eval.make_tables import CONFIG_ORDER, fmt_ci

HEADLINE = [("hit@5", "Hit@5"), ("ndcg@10", "nDCG@10")]


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=C.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - report generation must not fail on git state
        return "unknown"


def ver(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "n/a"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append(
            "| " + " | ".join("" if (isinstance(v, float) and v != v) else str(v) for v in r) + " |"
        )
    return "\n".join(lines) + "\n"


def ci_cell(r: pd.Series, m: str, digits: int = 3) -> str:
    if m not in r or pd.isna(r[m]):
        return "–"
    return fmt_ci(r[m], r[f"{m}_lo"], r[f"{m}_hi"], digits)


def read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def headline(rt: pd.DataFrame, rg: pd.DataFrame, sy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, tag), g in rt.groupby(["dataset", "run_tag"], sort=False):
        for cfg in [c for c in CONFIG_ORDER if c != C.ENGINE_CONFIG]:
            src = g[g["config"] == (cfg if cfg != "rrf_bge_two_pass" else "rrf_bge")]
            if src.empty:
                continue
            r = src.iloc[0]
            rec = {
                "dataset": ds,
                "config": cfg,
                "n": int(r["n_queries"]),
                "Hit@5": ci_cell(r, "hit@5"),
                "nDCG@10": ci_cell(r, "ndcg@10"),
            }
            gen = (
                rg[(rg["dataset"] == ds) & (rg["run_tag"] == tag) & (rg["config"] == cfg)]
                if not rg.empty
                else pd.DataFrame()
            )
            if not gen.empty:
                q = gen.iloc[0]
                rec.update(
                    {
                        "Faithfulness": ci_cell(q, "faithfulness"),
                        "Context Recall": ci_cell(q, "context_recall"),
                        "Factual F1": ci_cell(q, "factual_correctness_f1"),
                    }
                )
            else:
                rec.update({"Faithfulness": "–", "Context Recall": "–", "Factual F1": "–"})
            s = (
                sy[(sy["dataset"] == ds) & (sy["config"] == cfg)]
                if not sy.empty
                else pd.DataFrame()
            )
            rec["p95 latency (s)"] = (
                f"{s.iloc[0]['total_p95_s']:.2f}"
                if not s.empty and "total_p95_s" in s and pd.notna(s.iloc[0].get("total_p95_s"))
                else "–"
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--judge", default=C.JUDGE_MODEL)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    rd = Path(args.results_dir)
    rt, rg = read_csv(rd / "table_retrieval.csv"), read_csv(rd / "table_ragas.csv")
    rb, sy = read_csv(rd / "table_robustness.csv"), read_csv(rd / "table_system.csv")
    sg, vis = read_csv(rd / "significance.csv"), read_csv(rd / "table_visibility.csv")
    rgb = read_csv(rd / "table_rgb.csv")
    meta = (
        json.loads((rd / "dataset_meta.json").read_text())
        if (rd / "dataset_meta.json").exists()
        else {}
    )

    hl = headline(rt, rg, sy)
    hl.to_csv(rd / "table_headline.csv", index=False)

    out: list[str] = []
    banner = f"**{args.label}** — " if args.label else ""
    out.append(
        f"# DeepContext public-benchmark report\n\n{banner}generated from files in `{rd}`; every number below is read from them.\n"
    )
    out.append(
        f"- Branch `{git('rev-parse', '--abbrev-ref', 'HEAD')}` @ `{git('rev-parse', '--short', 'HEAD')}` (base `feat/rag-quality-generation`), working tree {'dirty' if git('status', '--porcelain') else 'clean'} at report time"
    )
    out.append(
        f"- Judge: `{args.judge}` (temperature 0) — generator: `{C.GENERATOR_MODEL}` (different model; both are Google models, so family bias is possible)"
    )
    out.append(
        f"- Embeddings: `{C.EMBEDDING_MODEL}` {C.EMBEDDING_DIM}-d via Vertex AI (`{C.EMBEDDING_LOCATION}` endpoint); reranker: {C.RERANKER_NAME}\n"
    )

    out.append("## 1. Setup and versions\n")
    pk = [
        "ragas",
        "beir",
        "datasets",
        "pandas",
        "numpy",
        "scipy",
        "google-genai",
        "langchain-core",
        "langchain-community",
        "huggingface-hub",
        "pyarrow",
    ]
    out.append(
        f"Python {platform.python_version()} on {platform.platform()} ({platform.machine()}); "
        + ", ".join(f"{p} {ver(p)}" for p in pk)
        + ".\n"
    )
    backends = sorted({m.get("index", {}).get("backend", "?") for m in meta.values()})
    storages = sorted(
        {m.get("index", {}).get("storage", "?").split(", schema")[0] for m in meta.values()}
    )
    n_schemas = len(meta)
    pg_note = (
        "On PostgreSQL the lexical channel is the product's Postgres full-text search (`ts_rank` over the first 8 "
        "query terms), **not BM25**, and the dense channel is pgvector HNSW (approximate). `bm25_only` rows "
        "therefore mean *Postgres full-text only*.\n"
    )
    sqlite_note = "Lexical channel = SQLite FTS5 `bm25()`; dense channel = exact cosine.\n"
    out.append(
        "Storage backend: "
        + ", ".join(backends)
        + " — "
        + "; ".join(storages)
        + f" (one schema per index; {n_schemas} indexes, see `dataset_meta.json`). "
        + (pg_note if "postgres" in backends else sqlite_note)
    )
    out.append(
        "Ingestion: *fast* path only (`DocumentParser` + `ParentChildChunker`, child 300–600 tokens, parent 1000–2500, 15% overlap); no Docling, no local summariser, no model loaded on the host. Title (when present) is prepended to the chunk text and passed in the embedding prefix.\n"
    )
    out.append(
        "### Harness deviations from the stock code (all inside `eval/`, `src/` untouched)\n"
    )
    out.append("""1. **Strict transport** (`eval/strict_clients.py`): Vertex AI (ADC) instead of API keys; no mock/hash embeddings, no Gemini→Groq→NVIDIA→mock failover, no silent heuristic-reranker fallback — a failure raises and the row is retried on the next run. The stock client would otherwise have produced hash embeddings in this environment.
2. **Models**: Gemini 3.x only. `gemini-embedding-2` is served from the `global` endpoint (404 in `us-central1`) and accepts one content per request; Gemini 3.x chat models are also called via `global`.
3. **Thinking flag**: the stock Gemini branch of `LLMClient.complete` ignores `enable_thinking=False`. The harness honours it (`thinking_budget=0`); the table below quantifies what the stock behaviour does to the two-pass extractor on the generator model. Set `EVAL_HONOR_ENABLE_THINKING=0` to reproduce the stock behaviour.
4. **Timeouts**: per-call caller timeouts (5–15 s) are replaced by 180 s so results do not depend on API load; the whole pipeline (rewriter, classifier, generator) is routed to the generator model by configuration.
5. **PostgreSQL schema**: on pgvector 0.8.x the stock `PostgresStore.initialize()` fails on a fresh database ("column does not have dimensions": HNSW index on an untyped `vector` column). The harness subclass runs the *same* DDL (extracted from the stock source) with `vector(768)` columns and without the ALTER-back-to-untyped block; all queries are the stock ones.
6. **HNSW search width**: the stock store never sets `hnsw.ef_search`, so pgvector's default (40) would cap a `LIMIT 100` dense query at 40 rows; the eval database sets `hnsw.ef_search = 100` (the value the project docs describe).
7. **Isolation**: all benchmark data lives in a dedicated database (`deepcontext_eval`), one schema per index; the application's own database is never written. The model-call cache (`eval/cache/llm_cache.sqlite`) is a local file, not corpus data.
8. **One-pass baseline** = the product's `PromptAssembler` prompt + one Gemini call, with the (empty) memory lookup stubbed to avoid a no-op embedding call.
9. **Latency**: cached model calls replay their recorded first-call latency; **latency percentiles come only from the separate live pass (`raw_latency.jsonl`)**.
10. RAGAS `ResponseRelevancy` uses `strictness=1` (a temperature-0 judge returns identical candidates, so the default 3 only triples cost).
""")
    tf = rd / "thinking_flag_check.json"
    if tf.exists():
        d = json.loads(tf.read_text())
        rows = []
        for mode in ("honoured (harness default)", "ignored (stock client)"):
            for ds, v in d.get(mode, {}).items():
                rows.append(
                    {
                        "enable_thinking flag": mode,
                        "dataset": ds,
                        "queries": v["queries"],
                        "generation calls": v["generation_calls"],
                        "calls hitting MAX_TOKENS": v["calls_hitting_max_tokens"],
                        "mean thinking tokens": round(v["think_tokens_mean"], 1),
                        "two-pass refusals": v["two_pass_refusals"],
                    }
                )
        out.append(
            f"\n**Thinking-flag check on `{d.get('generator_model')}`** (`thinking_flag_check.json`):\n"
        )
        out.append(md_table(pd.DataFrame(rows)))
    out.append("## 2. Datasets\n")
    if meta:
        rows = []
        for tag, m in meta.items():
            rows.append(
                {
                    "run": tag,
                    "queries used / available": f"{m.get('queries_used', '?')} / {m.get('queries_available', '?')}",
                    "corpus docs used / total": f"{m.get('corpus_used_docs', '?')} / {m.get('corpus_total_docs', m.get('corpus_used_docs', '?'))}",
                    "license": m.get("license", "see §8"),
                    "source": m.get("source", "–"),
                    "sampling": m.get("sampling", "–"),
                }
            )
        out.append(md_table(pd.DataFrame(rows)))
    out.append(
        "RGB (generation-only): `en_refine` / `en_int` / `en_fact` from github.com/chen700564/RGB (CC BY-NC-SA 4.0, non-commercial research use).\n"
    )

    out.append("## 3. Headline table (mean [95% bootstrap CI, 10k resamples, seed 0])\n")
    out.append(
        "`rrf_bge_two_pass` shares the retrieval of `rrf_bge`. Generation metrics are blank where no generation run exists for that dataset/config. p95 latency is end-to-end from the live pass.\n"
    )
    out.append(md_table(hl))

    out.append("## 4. Retrieval (document level)\n")
    cols = [
        ("hit@1", "Hit@1"),
        ("hit@3", "Hit@3"),
        ("hit@5", "Hit@5"),
        ("hit@8", "Hit@8"),
        ("recall@5", "R@5"),
        ("recall@10", "R@10"),
        ("recall@100", "R@100"),
        ("precision@5", "P@5"),
        ("mrr", "MRR"),
        ("map", "MAP@100"),
        ("ndcg@5", "nDCG@5"),
        ("ndcg@10", "nDCG@10"),
        ("evidence_all@5", "AllEvid@5"),
        ("evidence_all@10", "AllEvid@10"),
    ]
    for (ds, tag), g in rt.groupby(["dataset", "run_tag"], sort=False):
        out.append(f"\n### {ds} — `{tag}`\n")
        out.append(
            md_table(
                pd.DataFrame(
                    [
                        {
                            "config": r["config"],
                            "n": int(r["n_queries"]),
                            **{lbl: ci_cell(r, m) for m, lbl in cols},
                        }
                        for _, r in g.iterrows()
                    ]
                )
            )
        )
    out.append(
        "\nSee `ablation.md` for per-dataset tables with the best value per column in bold.\n"
    )

    out.append(
        "## 5. Generation quality (RAGAS, judged on the evidence the generator actually saw)\n"
    )
    out.append(
        "RAGAS `retrieved_contexts` = the slice of each retrieved parent that fits the prompt's 12,000-character evidence budget (`table_visibility.csv`). Reference-based metrics inherit the quality of the references (RAGBench references are model-generated; MultiHop-RAG references are 1–5 word answers).\n"
    )
    for (ds, tag), g in rg.groupby(["dataset", "run_tag"], sort=False) if not rg.empty else []:
        out.append(f"\n### {ds} — `{tag}`\n")
        cols2 = [
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "ResponseRelevancy"),
            ("context_precision", "CtxPrecision(ref)"),
            ("context_recall", "CtxRecall"),
            ("factual_correctness_f1", "FactualCorr F1"),
            ("exact_match", "EM"),
            ("token_f1", "token F1"),
            ("contains_answer", "contains-answer"),
        ]
        out.append(
            md_table(
                pd.DataFrame(
                    [
                        {
                            "config": r["config"],
                            "answerable": int(r["n_answerable"]),
                            "judged": int(r["n_scored"]),
                            **{lbl: ci_cell(r, m) for m, lbl in cols2 if m in r and pd.notna(r[m])},
                        }
                        for _, r in g.iterrows()
                    ]
                )
            )
        )

    out.append("\n## 6. Robustness\n")
    if not rb.empty:
        out.append(
            "Abstention = the system's refusal sentence, an empty output, or a short answer built on a refusal phrase (heuristic, `eval/metrics.py`).\n"
        )
        out.append(
            md_table(
                pd.DataFrame(
                    [
                        {
                            "dataset": r["dataset"],
                            "config": r["config"],
                            "unanswerable n": int(r["n_unanswerable"]),
                            "abstention on unanswerable ↑": ci_cell(
                                r, "abstention_rate_on_unanswerable"
                            )
                            if r["n_unanswerable"]
                            else "–",
                            "answerable n": int(r["n_answerable"]),
                            "false abstention ↓": ci_cell(r, "false_abstention_rate"),
                            "false abst. | gold in context": f"{r['false_abstention_given_gold_in_context']:.3f}"
                            if pd.notna(r.get("false_abstention_given_gold_in_context"))
                            else "–",
                            "anachronism gate fired": f"{r['anachronism_gate_fire_rate']:.3f}"
                            if pd.notna(r.get("anachronism_gate_fire_rate"))
                            else "–",
                        }
                        for _, r in rb.iterrows()
                    ]
                )
            )
        )
    if not vis.empty:
        out.append("\n### Evidence visibility (12,000-character prompt budget)\n")
        out.append(
            md_table(
                vis[
                    [
                        "dataset",
                        "config",
                        "n",
                        "evidence_chars_total_mean",
                        "evidence_chars_seen_mean",
                        "fraction_of_evidence_seen",
                        "gold_in_context_rate",
                        "gold_in_seen_context_rate",
                        "gold_docs_seen_fraction",
                    ]
                ].round(3)
            )
        )
    if not rgb.empty:
        out.append("\n### RGB (generation only; 5 passages per query)\n")
        keep = [c for c in rgb.columns if not c.endswith(("_lo", "_hi"))]
        out.append(md_table(rgb[keep].round(3)))

    out.append("\n## 7. System: tokens, cost, latency\n")
    out.append(
        "Tokens are from the main run; $ uses list prices in `eval/config.py` (Gemini API standard tier; Vertex billing may differ). The hosted reranker's price is not published in the repo, so its tokens are reported but excluded from $.\n"
    )
    if not sy.empty:
        cols3 = [
            c
            for c in [
                "dataset",
                "config",
                "n_generated",
                "live_n",
                "total_p50_s",
                "total_p95_s",
                "retrieval_p50_s",
                "retrieval_p95_s",
                "rerank_p50_s",
                "rerank_p95_s",
                "generation_p50_s",
                "generation_p95_s",
                "gen_in_tokens_mean",
                "gen_out_tokens_mean",
                "rerank_tokens_mean",
                "generator_cost_per_1k_queries_usd",
            ]
            if c in sy.columns
        ]
        out.append(md_table(sy[cols3].round(2)))

    out.append("\n## 8. Statistics\n")
    out.append(
        "Paired tests on identical queries (baseline → variant; positive = variant better, except abstention rows): paired bootstrap (10k, seed 0) and Wilcoxon signed-rank; effect size = Cohen's d_z.\n"
    )
    if not sg.empty:
        show = sg.copy()
        for c in ("mean_baseline", "mean_variant", "mean_diff", "ci_lo", "ci_hi", "cohens_dz"):
            show[c] = show[c].round(4)
        show["p_bootstrap"] = show["p_bootstrap"].map(lambda x: f"{x:.4f}")
        show["p_wilcoxon"] = show["p_wilcoxon"].map(lambda x: f"{x:.4f}")
        out.append(md_table(show.drop(columns=["run_tag"])))
    out.append(
        "\nNo multiple-comparison correction is applied; with this many comparisons treat p-values as descriptive.\n"
    )

    ca = rd / "claude_audit_summary.json"
    if ca.exists():
        d = json.loads(ca.read_text())
        out.append("## 8b. Judge cross-check by Claude (model labels, not human labels)\n")
        out.append(
            f"Claude (an Anthropic model) independently labelled {d['n_labelled']} sampled answers (read in full against the evidence "
            "the generator saw; labels in `claude_labels.jsonl`, merged into `judge_audit.csv`). "
            "`human_label` is still blank.\n"
        )
        rows = []
        for m in ("faithfulness", "correctness"):
            v = d[m]
            rows.append(
                {
                    "metric (Claude label vs Gemini judge score)": m
                    if m == "faithfulness"
                    else "correctness vs RAGAS FactualCorrectness F1",
                    "n": v["n"],
                    "Claude mean": round(v["claude_mean"], 3),
                    "judge mean": round(v["gemini_judge_mean"], 3),
                    "mean |diff|": round(v["mean_abs_diff"], 3),
                    "Pearson r": round(v.get("pearson_r", float("nan")), 3),
                    "Spearman ρ": round(v.get("spearman_rho", float("nan")), 3),
                    "binary agreement @0.5": round(v["binary_agreement_at_0.5"], 3),
                    "judge stricter (>0.25)": round(v["judge_stricter_share"], 3),
                    "judge more lenient (>0.25)": round(v["judge_more_lenient_share"], 3),
                }
            )
        out.append(md_table(pd.DataFrame(rows)))
        out.append("\n")

    out.append("## 9. Limitations, leakage and comparability (read before citing)\n")
    out.append("""- **Judge validity.** The primary judge (`gemini-3.7-flash`) is a different model from the generator (`gemini-3.8-flash`) but from the same vendor/family, so family and self-preference bias are possible. An independent cross-check by Claude (Anthropic) on a sample is reported in `claude_audit_summary.json` / `judge_audit.csv` (`claude_label` column) when present. These are **model labels, not human labels**: the `human_label` column is intentionally blank and no human agreement figure is claimed.
- **Data leakage.** BEIR (SciFact/FiQA/HotpotQA), HotpotQA's Wikipedia text, MultiHop-RAG news articles (2023) and RGB are public and may be in the training data of both the generator and the judge; the embedding model and reranker may also have seen them. The documentation-QA set is derived from this repository's own docs and cannot be leaked into any model trained before it existed, but it was authored by an AI assistant (quotes machine-verified; answers not human-reviewed).
- **Comparability with published numbers.** Corpus subsampling (HotpotQA, FiQA/others where capped) makes the task easier than the BEIR leaderboard task; BM25 here is SQLite FTS5 (no stemming/stopwords) not Lucene/Pyserini; chunking splits long documents and scores are max-chunk-per-document; query counts differ from published protocols. Do **not** put these next to leaderboard numbers without these qualifiers.
- **RAGBench** questions are written against their own source passages; pooling passages from many questions makes some questions under-specified, which inflates false abstention and depresses retrieval metrics. References are LLM-generated.
- **MultiHop-RAG / long documents.** Both generators cap the evidence sent to the model at 12,000 characters, so on long articles most retrieved context never reaches the LLM (see §6). This is a property of the shipped system and is measured, not tuned.
- **Hard-coded anachronism gate** (`quality_gates.is_anachronism`) is specific to the GoT corpus; it is only exercised by the `engine_as_shipped` rows (fire rate in §6).
- **Small samples.** Smoke runs (n=20 per dataset) are pipeline checks, not evidence of differences between configs.
- The SLM contextual-summary ingestion variant (Qwen3-0.6B) was **not run** (not on the base branch; deferred by request). No numbers are reported for it.
""")
    out.append(
        "## 10. Reproduce\n\n```bash\nuv venv --python 3.12 .venv && uv sync --extra dev\nuv pip install --python .venv/bin/python -r eval/requirements.txt\ngcloud auth application-default login   # Vertex AI ADC\n./run_all.sh\n```\n"
    )
    (rd / "REPORT.md").write_text("\n".join(out) + "\n")
    print(f"wrote {rd / 'REPORT.md'}")


if __name__ == "__main__":
    main()
