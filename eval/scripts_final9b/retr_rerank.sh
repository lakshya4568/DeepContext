#!/bin/zsh
# Mixed search (keyword vote 0.02) + the self-hosted bge reranker. Needs the GPU tunnel (EVAL_RERANK_URL).
# Usage: retr_rerank.sh <fast|deep> <dataset> [<dataset> ...]
source "${0:A:h}/lib_final.sh"
arm=$1; shift
arm_env $arm || exit 2
gpu_env
for ds in "$@"; do
  f=$(flags $ds)
  echo "== $(date +%H:%M:%S) $arm $ds reranker, vote 0.02"
  RRF_WEIGHT_LEXICAL=0.02 uv run --no-sync python -m eval.run_retrieval --dataset $ds ${=f} --configs rrf_bge \
    --concurrency 4 --results-dir $OUT/R_${arm}_rerank_vote0.02 2>&1 | grep -E "index:|rows$|Traceback|Error|error" | cut -c1-220
done
echo "== $(date +%H:%M:%S) DONE retr_rerank $arm: $*"
