#!/bin/zsh
# FiQA is the longest summary job (about 72,000 prompts), so its deep index is built in pieces while the GPU is still
# writing: every 25 minutes the summaries written so far are copied and stored, and an index build is attempted (it embeds
# and stores each 1,500-document window whose summaries are all there and stops at the first window that is not finished).
# When the GPU is finished: the last copy, the full index, the searches and the reranker.
# Needs VM_IP, VM_ID and the tunnel.
# Usage: VM_IP=<ip> VM_ID=<instance id> pipeline_fiqa.sh
here="${0:A:h}"
source "$here/lib_final.sh"
arm_env deep
ds=fiqa
until eval/vm/vmssh.sh "test -f ~/summaries_${ds}.done" 2> /dev/null; do
  echo "== $(date +%H:%M:%S) fiqa: copy what is written so far, build what can be built"
  SKIP_STATUS=1 zsh "$here/fetch_import.sh" $ds
  uv run --no-sync python "$here/build_index.py" --dataset $ds 2>&1 | grep -E "stopped|finished|Traceback|Error" | cut -c1-200
  sleep 1500
done
echo "== $(date +%H:%M:%S) fiqa: summaries finished"
zsh "$here/fetch_import.sh" $ds || { echo "FAILED: fetch or import of $ds"; exit 1; }
zsh "$here/retr_nogpu.sh" deep $ds
zsh "$here/retr_rerank.sh" deep $ds
echo "== $(date +%H:%M:%S) DONE pipeline_fiqa"
