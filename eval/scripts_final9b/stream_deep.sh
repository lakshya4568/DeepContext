#!/bin/zsh
# Phase 1, Mac side, no GPU. While the GPU writes summaries (and does nothing else), copy what is written, store it, and embed
# and store every window of the deep index whose summaries are complete: side by side, in batches. When a dataset's summaries
# are finished: the last copy, the rest of the index and the first-stage searches (dense, keyword, mixed).
# The reranker and the answers are phase 2 (phase2_gpu.sh) and wait until the GPU has no summaries to write.
# Needs VM_IP and VM_ID. STREAM_INTERVAL = seconds between two rounds (default 420).
# Usage: VM_IP=<ip> VM_ID=<instance id> stream_deep.sh <dataset> [<dataset> ...]      (in the order the GPU works)
here="${0:A:h}"
source "$here/lib_final.sh"
arm_env deep
interval=${STREAM_INTERVAL:-420}
for ds in "$@"; do
  until eval/vm/vmssh.sh "test -f ~/summaries_${ds}.done" 2> /dev/null; do
    echo "== $(date +%H:%M:%S) $ds: copy what is written, store it, embed the finished windows"
    SKIP_STATUS=1 zsh "$here/fetch_import.sh" $ds
    uv run --no-sync python "$here/build_index.py" --dataset $ds 2>&1 | grep -E "stopped|finished|Traceback|Error" | cut -c1-200
    sleep $interval
  done
  echo "== $(date +%H:%M:%S) $ds: summaries finished"
  zsh "$here/fetch_import.sh" $ds || { echo "FAILED: fetch or import of $ds"; exit 1; }
  zsh "$here/retr_nogpu.sh" deep $ds
done
echo "== $(date +%H:%M:%S) DONE stream_deep: $*"
