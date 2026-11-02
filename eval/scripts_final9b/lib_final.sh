#!/bin/zsh
# Shared settings for the final 9B round. Source this file; it sets the environment for one arm.
#   fast : no summaries,           index suffix _main
#   deep : 9B summaries (cached),  index suffix _maindeep9b
cd "${0:A:h}/../.." || return 1
export PYTHONPATH=.:src EVAL_STORE=stock
OUT=eval/results_final9b
arm_env() {
  case $1 in
    fast) export EVAL_INGEST=fast EVAL_INDEX_SUFFIX=_main ;;
    deep) export EVAL_INGEST=deep EVAL_INDEX_SUFFIX=_maindeep9b SUMMARY_MODEL=Qwen/Qwen3.5-9B ;;
    deepkw) export EVAL_INGEST=deepkw EVAL_INDEX_SUFFIX=_maindeepkw SUMMARY_MODEL=Qwen/Qwen3.5-9B ;;   # summary only in the keyword index
    *) echo "unknown arm $1"; return 1 ;;
  esac
}
# The subset each set was loaded with in the fast benchmark (results_final/REPORT.md section 2).
flags() {
  case $1 in
    hotpotqa)    echo "--n-queries 300 --max-docs 30000" ;;
    multihoprag) echo "--n-queries 300" ;;
    *)           echo "" ;;
  esac
}
# Self-hosted servers on the GPU VM, reached through the ssh tunnel (ports chosen by convention).
gpu_env() {
  export EVAL_RERANK_URL=http://127.0.0.1:18001 EVAL_SELFHOSTED_LLM_URL=http://127.0.0.1:18000 EVAL_GENERATOR_MODEL=qwen3.5-9b
}
