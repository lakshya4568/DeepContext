#!/bin/zsh
# After FiQA was stopped at about 45,000 summaries: store what the 9B wrote, find the summaries the FiQA subset ("fiqa-sub")
# still lacks (the relevant documents outside the first 36,000, plus any chunk of the first 36,000 that was still in flight),
# have the 9B write exactly those, and store them. The GPU does nothing else meanwhile.
# Needs VM_IP and VM_ID; the 9B server must be up on the VM and no summary writer may be running.
# Usage: VM_IP=<ip> VM_ID=<instance id> extra_pass.sh
here="${0:A:h}"
source "$here/lib_final.sh"
export SUMMARY_MODEL=Qwen/Qwen3.5-9B
label=fiqa-sub-extra
jobs="eval/cache/summaries/jobs9b_${label}.jsonl"
SKIP_STATUS=1 zsh "$here/fetch_import.sh" fiqa || { echo "FAILED: fetch or import of fiqa"; exit 1; }
uv run --no-sync python -m eval.main_summaries export --datasets fiqa-sub --out "$jobs" 2>&1 | grep -E "wrote|Traceback|Error" | cut -c1-200
[ -s "$jobs" ] || { echo "nothing is missing: no extra pass needed"; exit 0; }
eval/vm/vmscp.sh up "$jobs" || { echo "FAILED: upload"; exit 1; }
eval/vm/vmssh.sh 'rm -f ~/summaries_fiqa-sub-extra.done; setsid nohup bash ~/run_summaries.sh fiqa-sub-extra > ~/slm/summaries_extra.out 2>&1 < /dev/null &'
echo "== $(date +%H:%M:%S) extra pass started on the VM"
for i in {1..180}; do
  eval/vm/vmssh.sh 'test -f ~/summaries_fiqa-sub-extra.done' 2> /dev/null && break
  sleep 10
done
eval/vm/vmssh.sh 'test -f ~/summaries_fiqa-sub-extra.done' 2> /dev/null || { echo "FAILED: the extra pass did not finish in 30 minutes"; exit 1; }
echo "== $(date +%H:%M:%S) extra pass finished"
SKIP_STATUS=1 zsh "$here/fetch_import.sh" $label || { echo "FAILED: fetch or import of $label"; exit 1; }
SUMMARY_MODEL=Qwen/Qwen3.5-9B uv run --no-sync python -m eval.main_summaries status --datasets fiqa-sub 2>&1 | grep -E "child chunks|Traceback|Error" | cut -c1-200
echo "== $(date +%H:%M:%S) DONE extra_pass"
