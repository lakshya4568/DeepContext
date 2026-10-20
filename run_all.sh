#!/usr/bin/env bash
# Reproduce the FINAL DeepContext benchmark (fixed search) end to end. Resumable: every model call is
# cached on disk and every raw row is skipped once written, so you can Ctrl-C and restart.
#
# Prerequisites
#   * uv, Python 3.12            (uv venv --python 3.12 .venv)
#   * uv sync --extra dev && uv pip install --python .venv/bin/python -r eval/requirements.txt
#   * Google Cloud ADC + Vertex AI:  gcloud auth application-default login
#       env (or .env): GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION (generation; default us-central1)
#   * A bge-reranker-v2-m3 server: either EcoHash (ECOHASH_API_KEY) or a self-hosted one
#     (POST <url>/rerank {query, documents}) -> EVAL_RERANK_URL=https://host
#   * PostgreSQL 15+ with pgvector (brew services start postgresql@15). Data goes to a dedicated
#     database `deepcontext_eval` (created automatically; one schema per index) using the server and
#     credentials of POSTGRES_DSN - the application database is never written.
#   * Models: Gemini 3.x only (generator gemini-3.8-flash, judge gemini-3.7-flash, via Vertex AI).
#
# Search settings (the fixed search; set the three LEGACY values to reproduce the original code):
#   LEXICAL_RANKING=bm25 DEDUP_MODE=content_hash RRF_WEIGHT_LEXICAL=0.1     (fixed; 0.1 chosen on dev questions)
#   LEXICAL_RANKING=ts_rank DEDUP_MODE=prefix120 RRF_WEIGHT_LEXICAL=1.0     (original; use RESULTS=eval/results_orig)
#
# NOTE: `uv run --no-sync` is deliberate: a plain `uv run` would re-sync to uv.lock and could
# downgrade the pinned eval packages (ragas needs langchain 0.3.x).
set -euo pipefail
cd "$(dirname "$0")"

RESULTS="${RESULTS:-eval/results_final}"
PY="${PY:-uv run --no-sync python}"
JUDGE="${EVAL_JUDGE_MODEL:-gemini-3.7-flash}"   # must differ from the generator (gemini-3.8-flash)
export LEXICAL_RANKING="${LEXICAL_RANKING:-bm25}"
export DEDUP_MODE="${DEDUP_MODE:-content_hash}"
export RRF_WEIGHT_LEXICAL="${RRF_WEIGHT_LEXICAL:-0.1}"
export RRF_WEIGHT_DENSE="${RRF_WEIGHT_DENSE:-1.0}"
N_GEN="${N_GEN:-300}"            # MultiHop-RAG queries sampled
N_ANS="${N_ANS:-100}"            # answered per large dataset (hash-selected; all docs/PDF questions are answered)
N_RAGAS="${N_RAGAS:-50}"         # RAGAS-judged per large dataset (same queries for every config)
N_LAT="${N_LAT:-15}"             # queries in the live latency pass
HOTPOT_QUERIES="${HOTPOT_QUERIES:-300}"; HOTPOT_DOCS="${HOTPOT_DOCS:-30000}"
CFGS="dense_only bm25_only rrf rrf_bge rrf_bge_two_pass engine_as_shipped"   # all six configurations
CFGS_LARGE="dense_only bm25_only rrf rrf_bge rrf_bge_two_pass"               # FiQA/HotpotQA: no generation
RAGAS_CFGS="rrf_bge rrf_bge_two_pass engine_as_shipped"                      # RAGAS-judged subset (cost)
LOGS="$RESULTS/logs"; mkdir -p "$LOGS"

echo "== 0. sanity: gold-quote verification + offline unit tests (relevant files only)"
$PY -m eval.docs_qa
$PY -m pytest tests/test_eval_harness.py tests/test_search_gaps.py -q -p no:cacheprovider

echo "== 1. retrieval ablation (rank metrics) - raw_retrieval.jsonl"
for ds in docsqa pdf-quantum pdf-got ragbench-covidqa scifact; do
  $PY -m eval.run_retrieval --dataset "$ds" --configs $CFGS --concurrency 6 --results-dir "$RESULTS"
done
$PY -m eval.run_retrieval --dataset multihoprag --n-queries "$N_GEN" --configs $CFGS --concurrency 6 --results-dir "$RESULTS"
$PY -m eval.run_retrieval --dataset fiqa --configs $CFGS_LARGE --concurrency 6 --results-dir "$RESULTS"
$PY -m eval.run_retrieval --dataset hotpotqa --n-queries "$HOTPOT_QUERIES" --max-docs "$HOTPOT_DOCS" \
    --configs $CFGS_LARGE --concurrency 6 --results-dir "$RESULTS"

echo "== 2. generation (six configs) - raw_generation.jsonl"
export EVAL_LLM_CONCURRENCY="${EVAL_LLM_CONCURRENCY:-12}"
for ds in docsqa pdf-quantum pdf-got; do
  $PY -m eval.run_generation --dataset "$ds" --configs $CFGS --concurrency 8 --results-dir "$RESULTS"
done
$PY -m eval.run_generation --dataset ragbench-covidqa --per-dataset "$N_ANS" --configs $CFGS --concurrency 8 --results-dir "$RESULTS"
$PY -m eval.run_generation --dataset multihoprag --n-queries "$N_GEN" --per-dataset "$N_ANS" --configs $CFGS --concurrency 8 --results-dir "$RESULTS"

echo "== 3. RGB robustness (generation only; passages are given, no retrieval) - raw_rgb.jsonl"
for kind in en_refine en_int en_fact; do $PY -m eval.run_rgb --kind "$kind" --results-dir "$RESULTS"; done

echo "== 3b. stock enable_thinking bug check on the generator model - thinking_flag_check.json"
$PY -m eval.check_thinking_flag --results-dir "$RESULTS"

echo "== 4. LLM-rubric correctness (all six configs; judge=$JUDGE) - rubric_scores.jsonl"
EVAL_REQUEST_TIMEOUT_S=60 $PY -m eval.judge_correctness --results-dir "$RESULTS" --model "$JUDGE" --concurrency 8

echo "== 5. RAGAS ($RAGAS_CFGS, $N_RAGAS queries per dataset; judge=$JUDGE) - ragas_scores.jsonl"
EVAL_REQUEST_TIMEOUT_S=120 EVAL_LLM_CONCURRENCY=16 \
  $PY -m eval.score_ragas --results-dir "$RESULTS" --model "$JUDGE" --configs $RAGAS_CFGS --per-dataset "$N_RAGAS" --batch 48

echo "== 6. live latency pass (uncached, sequential, machine otherwise idle) - raw_latency.jsonl"
unset EVAL_LLM_CONCURRENCY
$PY -m eval.run_latency --dataset multihoprag --n-queries "$N_GEN" --n "$N_LAT" --results-dir "$RESULTS"
$PY -m eval.run_latency --dataset ragbench-covidqa --n "$N_LAT" --results-dir "$RESULTS"
$PY -m eval.run_latency --dataset docsqa --n "$N_LAT" --results-dir "$RESULTS"

echo "== 7. tables, significance tests, gap analysis, failure analysis, judge audit, report"
$PY -m eval.make_tables      --results-dir "$RESULTS" --judge "$JUDGE" --rubric-judge "$JUDGE@think0"
$PY -m eval.gap_analysis     --results-dir "$RESULTS"
$PY -m eval.failure_analysis --results-dir "$RESULTS" --judge "$JUDGE"
# Optional cross-check: Claude labels (claude_labels.jsonl, written by an Anthropic model reading
# claude_audit_items.jsonl) -> $PY -m eval.claude_audit --results-dir "$RESULTS"
$PY -m eval.make_report      --results-dir "$RESULTS" --judge "$JUDGE"
echo "done -> $RESULTS/REPORT.md"
