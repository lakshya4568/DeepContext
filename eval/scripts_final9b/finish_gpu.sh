#!/bin/zsh
# Last GPU step: wait until phase 3 (HotpotQA deep) has ended, make sure no answer failed (a failed answer would need the GPU
# again), then stop the VM so that it stops costing GPU time. The VM and its disk are kept (stopping is not deleting).
# Usage: finish_gpu.sh
here="${0:A:h}"
source "$here/lib_final.sh"
while pgrep -f "[p]hase3_hotpot.sh" > /dev/null; do sleep 30; done
echo "== $(date +%H:%M:%S) phase 3 has ended"
failed=$(python3 - <<'PY'
import json
from pathlib import Path

n = 0
for arm in ("A_fast", "A_deep"):
    path = Path("eval/results_final9b") / arm / "raw_generation.jsonl"
    if path.exists():
        n += sum(1 for line in path.open(encoding="utf-8") if line.strip() and "error" in json.loads(line))
print(n)
PY
)
if [ "$failed" -gt 0 ]; then
  echo "== $failed answer rows have an error: the VM is NOT stopped, so that they can be run again"
  exit 2
fi
echo "== no failed answers: stopping the VM"
CLOUDSDK_METRICS_REQUEST_ATTRIBUTION="agent-skills/1.0 (skill:gcloud)" gcloud compute instances stop rag-slm-gpu-c --project=gen-lang-client-0308909719 --zone=asia-southeast1-c --quiet
CLOUDSDK_METRICS_REQUEST_ATTRIBUTION="agent-skills/1.0 (skill:gcloud)" gcloud compute instances describe rag-slm-gpu-c --project=gen-lang-client-0308909719 --zone=asia-southeast1-c --format="value(status,lastStopTimestamp)" --quiet
echo "== $(date +%H:%M:%S) DONE finish_gpu"
