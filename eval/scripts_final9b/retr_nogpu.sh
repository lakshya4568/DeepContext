#!/bin/zsh
# First-stage retrieval, no GPU: dense, keyword, mixed (equal votes), mixed (keyword vote 0.02).
# Usage: retr_nogpu.sh <fast|deep> <dataset> [<dataset> ...]
source "${0:A:h}/lib_final.sh"
arm=$1; shift
arm_env $arm || exit 2
for ds in "$@"; do
  f=$(flags $ds)
  echo "== $(date +%H:%M:%S) $arm $ds equal votes"
  uv run --no-sync python -m eval.run_retrieval --dataset $ds ${=f} --configs dense_only bm25_only rrf \
    --concurrency 4 --results-dir $OUT/R_${arm}_equal 2>&1 | grep -E "index:|rows$|Traceback|Error|error" | cut -c1-220
  echo "== $(date +%H:%M:%S) $arm $ds vote 0.02"
  RRF_WEIGHT_LEXICAL=0.02 uv run --no-sync python -m eval.run_retrieval --dataset $ds ${=f} --configs rrf \
    --concurrency 4 --results-dir $OUT/R_${arm}_vote0.02 2>&1 | grep -E "index:|rows$|Traceback|Error|error" | cut -c1-220
done
echo "== $(date +%H:%M:%S) DONE retr_nogpu $arm: $*"
