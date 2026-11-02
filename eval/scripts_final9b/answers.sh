#!/bin/zsh
# Answers by the self-hosted 9B with the strict documents-only prompt. Needs the GPU tunnel (9B + reranker).
# Usage: answers.sh <fast|deep> <dataset> [<dataset> ...]   (optional: LIMIT=3 for a smoke test)
# One answers.sh at a time: two writers must not append to the same results file.
source "${0:A:h}/lib_final.sh"
arm=$1; shift
arm_env $arm || exit 2
gpu_env
export RRF_WEIGHT_LEXICAL=0.02
lim=""; [ -n "$LIMIT" ] && lim="--max-gen $LIMIT"
lock="$OUT/.answers.lock"
until mkdir "$lock" 2> /dev/null; do sleep 15; done
trap 'rmdir "$lock" 2> /dev/null' EXIT
for ds in "$@"; do
  f=$(flags $ds)
  echo "== $(date +%H:%M:%S) answers $arm $ds"
  uv run --no-sync python -m eval.run_generation --dataset $ds ${=f} ${=lim} --configs rrf_bge_strict --concurrency 12 \
    --results-dir $OUT/A_${arm} 2>&1 | grep -E "answers$|finished|index:|Traceback|Error|error" | cut -c1-220
done
echo "== $(date +%H:%M:%S) DONE answers $arm: $*"
