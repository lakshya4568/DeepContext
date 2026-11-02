#!/usr/bin/env bash
# Writes the 9B summaries for the datasets named, in that order, one results file per dataset. Resumable: a prompt
# whose key is already in the results file is skipped. A marker file ~/summaries_<dataset>.done appears when a
# dataset is complete, so the Mac can fetch and import it while the next one runs.
# Run on the VM (vLLM serving qwen3.5-9b on port 8000, see start_9b.sh):
#   setsid nohup bash ~/run_summaries.sh fiqa hotpotqa scifact multihoprag ragbench-covidqa \
#     > ~/slm/summaries.out 2>&1 < /dev/null &
# Needs ~/jobs9b_<dataset>.jsonl, and ~/deepctx/{summarize_vllm.py,src/deep_context}.
set -uo pipefail
cd "$HOME/deepctx"
# shellcheck disable=SC1091
source "$HOME/slm/.venv/bin/activate"
export SUMMARY_MODEL=Qwen/Qwen3.5-9B PYTHONPATH=src
for ds in "$@"; do
  rm -f "$HOME/summaries_${ds}.done"
  echo "== $(date -u +%FT%TZ) summaries $ds"
  if python summarize_vllm.py "$HOME/jobs9b_${ds}.jsonl" "$HOME/results9b_${ds}.jsonl" \
      --url http://127.0.0.1:8000 --served-model qwen3.5-9b --model-name Qwen/Qwen3.5-9B; then
    touch "$HOME/summaries_${ds}.done"
  else
    echo "FAILED: summaries $ds"
    exit 1
  fi
done
echo "== $(date -u +%FT%TZ) all summaries done"
