#!/bin/zsh
# Writes the 9B summary prompts, one file per dataset, biggest first (the order the GPU works in), so the Mac can
# build the indexes of the early datasets while the GPU is still writing summaries for the later ones.
# Usage: export_jobs.sh <out_dir>      (no GPU, no model call; chunks each corpus the way the fast run did)
source "${0:A:h}/lib_final.sh"
out=${1:?usage: export_jobs.sh <out_dir>}
mkdir -p "$out"
export SUMMARY_MODEL=Qwen/Qwen3.5-9B
for ds in fiqa hotpotqa scifact multihoprag ragbench-covidqa; do
  echo "== $(date +%H:%M:%S) export $ds"
  uv run --no-sync python -m eval.main_summaries export --datasets $ds --out "$out/jobs9b_$ds.jsonl" 2>&1 \
    | grep -E "prompts|wrote|Traceback|Error|error" | cut -c1-200
done
echo "== $(date +%H:%M:%S) DONE export_jobs"
