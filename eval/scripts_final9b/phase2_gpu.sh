#!/bin/zsh
# Phase 2, after the 9B has finished writing summaries: the reranker and the answers, one task after the other.
#   1. load the reranker on the VM and wait until it answers
#   2. answers: fast for CovidQA and MultiHop-RAG, then deep for MultiHop-RAG (each resumes where it stopped)
#   3. the deep reranker setting for the datasets named (default: multihoprag scifact)
# FiQA is run by hand afterwards (retr_rerank.sh deep fiqa), once stream_deep.sh has finished its index and first-stage searches.
# Needs VM_IP, VM_ID and the tunnel (18000 = 9B, 18001 = reranker).
# Usage: VM_IP=<ip> VM_ID=<instance id> phase2_gpu.sh [<dataset> ...]
here="${0:A:h}"
source "$here/lib_final.sh"
datasets=("$@")
[ ${#datasets} -eq 0 ] && datasets=(multihoprag scifact)
echo "== $(date +%H:%M:%S) phase 2: load the reranker on the VM"
eval/vm/vmssh.sh bash -s <<'REMOTE'
mkdir -p ~/slm/logs
setsid nohup bash -c 'cd ~/slm && source .venv/bin/activate && exec uvicorn --app-dir $HOME rerank_server:app --host 127.0.0.1 --port 8001' > ~/slm/logs/rerank.log 2>&1 < /dev/null &
REMOTE
for i in {1..120}; do
  curl -sf http://127.0.0.1:18001/health > /dev/null && break
  sleep 5
done
curl -sf http://127.0.0.1:18001/health > /dev/null || { echo "FAILED: the reranker does not answer after 10 minutes"; exit 1; }
echo "== $(date +%H:%M:%S) the reranker answers"
zsh "$here/answers.sh" fast ragbench-covidqa multihoprag
zsh "$here/answers.sh" deep multihoprag
zsh "$here/retr_rerank.sh" deep $datasets
echo "== $(date +%H:%M:%S) DONE phase 2: answers; deep reranker for $datasets"
