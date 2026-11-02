#!/bin/zsh
# Copies the 9B summaries of one dataset from the GPU VM and stores them in the summary cache (safe to repeat; a
# partly written file only adds the summaries that exist). SKIP_STATUS=1 skips the count of missing summaries.
# Usage: VM_IP=<ip> VM_ID=<instance id> fetch_import.sh <dataset>
source "${0:A:h}/lib_final.sh"
ds=${1:?usage: fetch_import.sh <dataset>}
dir=eval/cache/summaries
file="$dir/results9b_${ds}.jsonl"
mkdir -p "$dir"
copied=0
for attempt in 1 2 3 4 5 6; do   # a short network break must not stop the run
  if eval/vm/vmscp.sh down "results9b_${ds}.jsonl" "$dir/"; then copied=1; break; fi
  echo "copy failed (attempt $attempt of 6), trying again in 40 s"
  sleep 40
done
[ "$copied" -eq 1 ] || exit 1
# a copy made while the GPU is still writing can end in half a line: drop it
if [ -s "$file" ] && [ -n "$(tail -c1 "$file")" ]; then sed -i '' '$d' "$file"; fi
uv run --no-sync python -m eval.main_summaries import "$file" 2>&1 \
  | grep -E "stored|Traceback|Error" | cut -c1-200
[ -n "$SKIP_STATUS" ] && exit 0
SUMMARY_MODEL=Qwen/Qwen3.5-9B uv run --no-sync python -m eval.main_summaries status --datasets $ds 2>&1 \
  | grep -v 'Settings(' | cut -c1-200 | tail -4
