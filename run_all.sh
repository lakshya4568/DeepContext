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
#   * PostgreSQL 15+ with pgvector (brew services start postgresql@15). Data goes to a dedicated
#     database `deepcontext_eval` (created automatically; one schema per index) using the server and
#     credentials of POSTGRES_DSN - the application database is never written. EVAL_BACKEND=sqlite
#     switches to SQLite (different lexical/dense behaviour; do not mix backends in one report).
#   * Models: Gemini 3.x only (generator gemini-3.8-flash, judge gemini-3.7-flash, via Vertex AI).
#   Keys/projects are read from the environment / .env; nothing is printed or committed.
#
# NOTE: `uv run --no-sync` is deliberate: a plain `uv run` would re-sync to uv.lock and could
# downgrade the pinned eval packages (ragas needs langchain 0.3.x).
set -euo pipefail
cd "$(dirname "$0")"

RESULTS="${RESULTS:-eval/results}"
PY="${PY:-uv run --no-sync python}"
JUDGE="${EVAL_JUDGE_MODEL:-gemini-3.7-flash}"      # must differ from the generator (gemini-3.8-flash)
JUDGE_THINK="${EVAL_JUDGE_THINKING_BUDGET:-}"       # optional thinking-token budget; empty = model default
N_GEN="${N_GEN:-300}"                                 # queries per dataset for LLM-judged metrics
N_LAT="${N_LAT:-30}"                                  # queries in the live latency pass
HOTPOT_QUERIES="${HOTPOT_QUERIES:-1000}"              # HotpotQA test queries (of 7,405)
HOTPOT_DOCS="${HOTPOT_DOCS:-200000}"                  # corpus cap: all relevant docs + random distractors
mkdir -p "$RESULTS"

echo "== 0. sanity: gold-quote verification + offline unit tests (relevant file only)"
$PY -m eval.docs_qa
$PY -m pytest tests/test_eval_harness.py -q -p no:cacheprovider

LOGS="$RESULTS/logs"; mkdir -p "$LOGS"
CFGS="dense_only bm25_only rrf rrf_bge rrf_bge_two_pass engine_as_shipped"   # all six configurations
RAGAS_CFGS="rrf_bge rrf_bge_two_pass engine_as_shipped"                       # RAGAS-judged subset (cost)
GEN_CONC="${GEN_CONC:-24}"                                                    # concurrent model calls in the generation/judging lane
RET_CONC="${RET_CONC:-12}"                                                    # concurrent queries in the retrieval lane (engine config calls the LLM)

# Two independent lanes (separate processes; every model call is cached and every raw row is skipped
# once written, so either lane can be interrupted and re-run):
#   lane R: retrieval ablation, all six configurations, small corpora first, then FiQA (57k docs) and
#           HotpotQA ($HOTPOT_DOCS docs / $HOTPOT_QUERIES queries). rrf_bge_two_pass shares rrf_bge retrieval.
#   lane G: generation -> RGB -> thinking-flag check -> rubric correctness -> RAGAS.
# The generation datasets (docsqa, ragbench-covidqa, multihoprag) are full-corpus indexes. On a fresh
# database both lanes may build the same index at once; that is safe (deterministic ids, idempotent
# inserts, cached embeddings) but wasteful, so run lane R's small datasets first if you care.
lane_retrieval() {
  export EVAL_LLM_CONCURRENCY="$RET_CONC"
  for ds in docsqa pdf-quantum pdf-got ragbench-covidqa scifact; do
    $PY -m eval.run_retrieval --dataset "$ds" --configs $CFGS --concurrency "$RET_CONC" --results-dir "$RESULTS"
  done
  $PY -m eval.run_retrieval --dataset multihoprag --n-queries "$N_GEN" --configs $CFGS --concurrency "$RET_CONC" --results-dir "$RESULTS"
  $PY -m eval.run_retrieval --dataset fiqa --configs $CFGS --concurrency "$RET_CONC" --results-dir "$RESULTS"
  $PY -m eval.run_retrieval --dataset hotpotqa --n-queries "$HOTPOT_QUERIES" --max-docs "$HOTPOT_DOCS" \
      --configs $CFGS --concurrency "$RET_CONC" --results-dir "$RESULTS"
}
lane_generation() {
  export EVAL_LLM_CONCURRENCY="$GEN_CONC"
  echo "== 2. generation (six configs) - raw_generation.jsonl"
  $PY -m eval.run_generation --dataset multihoprag --n-queries "$N_GEN" --configs $CFGS --concurrency 12 --results-dir "$RESULTS"
  $PY -m eval.run_generation --dataset ragbench-covidqa                 --configs $CFGS --concurrency 12 --results-dir "$RESULTS"
  $PY -m eval.run_generation --dataset docsqa                           --configs $CFGS --concurrency 12 --results-dir "$RESULTS"
  # custom long-document case study: the project's two PDFs with their existing 36-query sets
  $PY -m eval.run_generation --dataset pdf-quantum                      --configs $CFGS --concurrency 12 --results-dir "$RESULTS"
  $PY -m eval.run_generation --dataset pdf-got                          --configs $CFGS --concurrency 12 --results-dir "$RESULTS"
  echo "== 3. RGB robustness (generation only) - raw_rgb.jsonl"
  for kind in en_refine en_int en_fact; do $PY -m eval.run_rgb --kind "$kind" --results-dir "$RESULTS"; done
  echo "== 3b. stock enable_thinking bug check on the generator model - thinking_flag_check.json"
  $PY -m eval.check_thinking_flag --results-dir "$RESULTS"
  echo "== 3c. LLM-rubric correctness (all six configs, judge=$JUDGE) - rubric_scores.jsonl"
  $PY -m eval.judge_correctness --results-dir "$RESULTS" --model "$JUDGE" --concurrency "$GEN_CONC"
  echo "== 4. RAGAS judging of $RAGAS_CFGS (judge=$JUDGE, thinking budget=${JUDGE_THINK:-default}) - ragas_scores.jsonl"
  TB_ARGS=(); [ -n "$JUDGE_THINK" ] && TB_ARGS=(--thinking-budget "$JUDGE_THINK")
  $PY -m eval.score_ragas --results-dir "$RESULTS" --model "$JUDGE" --configs $RAGAS_CFGS --batch 48 "${TB_ARGS[@]}"
}
lane_retrieval >> "$LOGS/lane_retrieval.log" 2>&1 & PID_R=$!
lane_generation >> "$LOGS/lane_generation.log" 2>&1 & PID_G=$!
FAIL=0
wait "$PID_R" || { echo "lane R (retrieval) FAILED - see $LOGS/lane_retrieval.log"; FAIL=1; }
wait "$PID_G" || { echo "lane G (generation/judging) FAILED - see $LOGS/lane_generation.log"; FAIL=1; }
[ "$FAIL" = 0 ] || exit 1

echo "== 5. live latency pass (uncached, sequential, machine otherwise idle) - raw_latency.jsonl"
$PY -m eval.run_latency --dataset multihoprag --n-queries "$N_GEN" --n "$N_LAT" --results-dir "$RESULTS"
$PY -m eval.run_latency --dataset ragbench-covidqa                 --n "$N_LAT" --results-dir "$RESULTS"
$PY -m eval.run_latency --dataset docsqa                           --n "$N_LAT" --results-dir "$RESULTS"

echo "== 6. tables, significance tests, failure analysis, judge audit, report"
JLABEL="$JUDGE"; [ -n "$JUDGE_THINK" ] && JLABEL="$JUDGE@think$JUDGE_THINK"
$PY -m eval.make_tables      --results-dir "$RESULTS" --judge "$JLABEL"
$PY -m eval.failure_analysis --results-dir "$RESULTS" --judge "$JLABEL"
# Optional cross-check: Claude labels (claude_labels.jsonl, written by an Anthropic model reading
# claude_audit_items.jsonl) -> python -m eval.claude_audit --results-dir "$RESULTS"
$PY -m eval.make_report      --results-dir "$RESULTS" --judge "$JLABEL"
echo "done -> $RESULTS/REPORT.md"
