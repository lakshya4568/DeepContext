#!/usr/bin/env bash
# Reproduce the DeepContext public-benchmark run end to end (resumable: every model call and every
# raw row is cached/skipped on re-run, so you can Ctrl-C and restart safely).
#
# Prerequisites
#   * uv, Python 3.12            (uv venv --python 3.12 .venv)
#   * uv sync --extra dev && uv pip install --python .venv/bin/python -r eval/requirements.txt
#   * Google Cloud ADC + Vertex AI:  gcloud auth application-default login
#       env (or .env): GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION (generation; default us-central1)
#   * ECOHASH_API_KEY               (hosted bge-reranker-v2-m3)
#   Keys/projects are read from the environment / .env; nothing is printed or committed.
#
# NOTE: `uv run --no-sync` is deliberate: a plain `uv run` would re-sync to uv.lock and could
# downgrade the pinned eval packages (ragas needs langchain 0.3.x).
set -euo pipefail
cd "$(dirname "$0")"

RESULTS="${RESULTS:-eval/results}"
PY="${PY:-uv run --no-sync python}"
JUDGE="${EVAL_JUDGE_MODEL:-gemini-2.5-pro}"
JUDGE_THINK="${EVAL_JUDGE_THINKING_BUDGET:-128}"      # tokens; see REPORT.md (cost vs. agreement)
N_GEN="${N_GEN:-300}"                                 # queries per dataset for LLM-judged metrics
N_LAT="${N_LAT:-30}"                                  # queries in the live latency pass
HOTPOT_QUERIES="${HOTPOT_QUERIES:-1000}"              # HotpotQA test queries (of 7,405)
HOTPOT_DOCS="${HOTPOT_DOCS:-200000}"                  # corpus cap: all relevant docs + random distractors
mkdir -p "$RESULTS"

echo "== 0. sanity: gold-quote verification + offline unit tests (relevant file only)"
$PY -m eval.docs_qa
$PY -m pytest tests/test_eval_harness.py -q -p no:cacheprovider

echo "== 1. retrieval ablation (dense / bm25 / rrf / rrf+bge) - raw_retrieval.jsonl"
$PY -m eval.run_retrieval --dataset scifact                                     --results-dir "$RESULTS"
$PY -m eval.run_retrieval --dataset fiqa                                        --results-dir "$RESULTS"
$PY -m eval.run_retrieval --dataset hotpotqa --n-queries "$HOTPOT_QUERIES" --max-docs "$HOTPOT_DOCS" --results-dir "$RESULTS"
$PY -m eval.run_retrieval --dataset multihoprag --n-queries "$N_GEN"            --results-dir "$RESULTS"
$PY -m eval.run_retrieval --dataset ragbench-covidqa                            --results-dir "$RESULTS"
$PY -m eval.run_retrieval --dataset docsqa                                      --results-dir "$RESULTS"

echo "== 2. generation (one-pass / two-pass / engine-as-shipped) - raw_generation.jsonl"
CFGS="dense_only bm25_only rrf rrf_bge rrf_bge_two_pass engine_as_shipped"
$PY -m eval.run_generation --dataset multihoprag --n-queries "$N_GEN" --configs $CFGS --results-dir "$RESULTS"
$PY -m eval.run_generation --dataset ragbench-covidqa                 --configs $CFGS --results-dir "$RESULTS"
$PY -m eval.run_generation --dataset docsqa                           --configs $CFGS --results-dir "$RESULTS"

echo "== 3. RGB robustness (generation only) - raw_rgb.jsonl"
$PY -m eval.run_rgb --kind en_refine --results-dir "$RESULTS"
$PY -m eval.run_rgb --kind en_int    --results-dir "$RESULTS"
$PY -m eval.run_rgb --kind en_fact   --results-dir "$RESULTS"

echo "== 4. RAGAS judging (judge=$JUDGE, thinking budget=$JUDGE_THINK) - ragas_scores.jsonl"
$PY -m eval.score_ragas --results-dir "$RESULTS" --model "$JUDGE" --thinking-budget "$JUDGE_THINK"

echo "== 5. live latency pass (uncached, sequential) - raw_latency.jsonl"
$PY -m eval.run_latency --dataset multihoprag --n-queries "$N_GEN" --n "$N_LAT" --results-dir "$RESULTS"
$PY -m eval.run_latency --dataset ragbench-covidqa                 --n "$N_LAT" --results-dir "$RESULTS"
$PY -m eval.run_latency --dataset docsqa                           --n "$N_LAT" --results-dir "$RESULTS"

echo "== 6. tables, significance tests, failure analysis, judge audit, report"
JLABEL="$JUDGE@think$JUDGE_THINK"
$PY -m eval.make_tables      --results-dir "$RESULTS" --judge "$JLABEL"
$PY -m eval.failure_analysis --results-dir "$RESULTS" --judge "$JLABEL"
$PY -m eval.make_report      --results-dir "$RESULTS" --judge "$JLABEL"
echo "done -> $RESULTS/REPORT.md"
