#!/bin/zsh
# After each dataset's 9B summaries are finished on the GPU VM: copy them, build the deep index, run the searches and the
# reranker, and (sets with reference answers) the answers. One dataset at a time, in the order given.
# Needs VM_IP, VM_ID and the tunnel (local 18000 and 18001).
# Usage: VM_IP=<ip> VM_ID=<instance id> pipeline_deep.sh <dataset> [<dataset> ...]
here="${0:A:h}"
source "$here/lib_final.sh"
for ds in "$@"; do
  echo "== $(date +%H:%M:%S) waiting for the summaries of $ds"
  until eval/vm/vmssh.sh "test -f ~/summaries_${ds}.done" 2> /dev/null; do sleep 60; done
  echo "== $(date +%H:%M:%S) $ds: summaries finished"
  zsh "$here/fetch_import.sh" $ds || { echo "FAILED: fetch or import of $ds"; exit 1; }
  zsh "$here/retr_nogpu.sh" deep $ds
  zsh "$here/retr_rerank.sh" deep $ds
  case $ds in
    ragbench-covidqa|multihoprag) zsh "$here/answers.sh" deep $ds ;;
  esac
done
echo "== $(date +%H:%M:%S) DONE pipeline_deep: $*"
