#!/bin/zsh
# Phase 3: HotpotQA deep, after phase 2 (the owner chose to run it, 2026-11-02 about 17:45). One GPU task at a time:
#   1. wait until phase2_all.sh has finished
#   2. unload the reranker on the VM, let the 9B write the HotpotQA summaries; the Mac embeds and stores batches side by side
#      and, when the summaries are finished, completes the deep index and runs the first-stage searches (stream_deep.sh)
#   3. load the reranker again and run the deep reranker setting
# Time limits (clock time on the Mac, HHMM): not started after START_BY (default 1910); summaries stopped at STOP_AT
# (default 2015), because the VM switches itself off at about 2045 and the last steps need about 25 minutes.
# Needs VM_IP, VM_ID and the tunnel.
# Usage: VM_IP=<ip> VM_ID=<instance id> phase3_hotpot.sh
here="${0:A:h}"
source "$here/lib_final.sh"
start_by=${START_BY:-1910}
stop_at=${STOP_AT:-2015}
while pgrep -f "[p]hase2_all.sh" > /dev/null; do sleep 30; done
echo "== $(date +%H:%M:%S) phase 2 is finished"
if [ "$(date +%H%M)" -gt "$start_by" ]; then
  echo "== too late to start HotpotQA deep (limit $start_by): not run"
  exit 3
fi
echo "== $(date +%H:%M:%S) unload the reranker, start the HotpotQA summaries on the VM"
eval/vm/vmssh.sh 'pkill -f "[r]erank_server"; rm -f ~/summaries_hotpotqa.done; setsid nohup bash ~/run_summaries.sh hotpotqa > ~/slm/summaries.out 2>&1 < /dev/null &'
sleep 30
zsh "$here/stream_deep.sh" hotpotqa &
stream_pid=$!
until eval/vm/vmssh.sh 'test -f ~/summaries_hotpotqa.done' 2> /dev/null; do
  if [ "$(date +%H%M)" -ge "$stop_at" ]; then
    echo "== $(date +%H:%M:%S) time limit reached: HotpotQA summaries are not finished, deep HotpotQA is not run"
    kill $stream_pid 2> /dev/null
    eval/vm/vmssh.sh 'pkill -f "[r]un_summaries.sh"; pkill -f "[s]ummarize_vllm.py"; true'
    exit 4
  fi
  sleep 30
done
echo "== $(date +%H:%M:%S) HotpotQA summaries finished on the VM; waiting for the Mac to finish the index and first-stage searches"
wait $stream_pid
echo "== $(date +%H:%M:%S) load the reranker for the deep HotpotQA searches"
eval/vm/vmssh.sh bash -s <<'REMOTE'
mkdir -p ~/slm/logs
setsid nohup bash -c 'cd ~/slm && source .venv/bin/activate && exec uvicorn --app-dir $HOME rerank_server:app --host 127.0.0.1 --port 8001' > ~/slm/logs/rerank.log 2>&1 < /dev/null &
REMOTE
for i in {1..120}; do
  curl -sf http://127.0.0.1:18001/health > /dev/null && break
  sleep 5
done
curl -sf http://127.0.0.1:18001/health > /dev/null || { echo "FAILED: the reranker does not answer after 10 minutes"; exit 1; }
zsh "$here/retr_rerank.sh" deep hotpotqa
echo "== $(date +%H:%M:%S) DONE phase3_hotpot"
