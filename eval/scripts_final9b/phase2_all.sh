#!/bin/zsh
# Phase 2 with the FiQA subset, one GPU task after the other:
#   phase2_gpu.sh (reranker, answers, deep reranker for MultiHop-RAG and SciFact), then the reranker setting for the FiQA subset,
#   fast arm first, then the deep arm as soon as the Mac has finished the deep subset index and its first-stage searches
#   (the marker file results_final9b/.deep_fiqa-sub_first_stage is written by `retr_nogpu.sh deep fiqa-sub` run through
#   deep_fiqa_sub_first_stage.sh).
# Needs VM_IP, VM_ID and the tunnel.
# Usage: VM_IP=<ip> VM_ID=<instance id> phase2_all.sh
here="${0:A:h}"
source "$here/lib_final.sh"
zsh "$here/phase2_gpu.sh"
zsh "$here/retr_rerank.sh" fast fiqa-sub
echo "== $(date +%H:%M:%S) waiting for the deep FiQA subset index and first-stage searches"
until [ -f "$OUT/.deep_fiqa-sub_first_stage" ]; do sleep 20; done
zsh "$here/retr_rerank.sh" deep fiqa-sub
echo "== $(date +%H:%M:%S) DONE phase2_all"
