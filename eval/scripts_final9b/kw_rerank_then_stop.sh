#!/bin/zsh
# End of the GPU day, in this order (one GPU task at a time):
#   1. wait until the first-stage searches of the keyword-only-summary arm (deepkw) have ended
#   2. wait until the deep HotpotQA index and its first-stage searches are done (stream_deep.sh), then run deepkw for HotpotQA (no GPU, no new embedding)
#   3. wait until phase 3 has ended (it leaves the reranker loaded), then run the deepkw reranker setting on every set, unless it is too late
#   4. hand over to finish_gpu.sh: if no answer row failed, stop the VM
# The VM switches itself off at about 20:45 (8 h limit), so the reranker runs are not started after LATEST (default 2015).
# Usage: kw_rerank_then_stop.sh <dataset> [<dataset> ...]      (the datasets for the deepkw reranker setting, HotpotQA last)
here="${0:A:h}"
source "$here/lib_final.sh"
latest=${LATEST:-2015}
while pgrep -f "[r]etr_nogpu.sh deepkw" > /dev/null; do sleep 20; done
echo "== $(date +%H:%M:%S) deepkw first-stage searches have ended"
while pgrep -f "[s]tream_deep.sh" > /dev/null; do sleep 20; done
echo "== $(date +%H:%M:%S) the deep HotpotQA first stage has ended: deepkw for HotpotQA"
zsh "$here/retr_nogpu.sh" deepkw hotpotqa
while pgrep -f "[p]hase3_hotpot.sh" > /dev/null; do sleep 20; done
echo "== $(date +%H:%M:%S) phase 3 has ended"
if [ "$(date +%H%M)" -gt "$latest" ]; then
  echo "== too late for the deepkw reranker setting (limit $latest): not run"
elif curl -sf http://127.0.0.1:18001/health > /dev/null; then
  zsh "$here/retr_rerank.sh" deepkw "$@"
else
  echo "== the reranker does not answer: the deepkw reranker setting is not run"
fi
exec zsh "$here/finish_gpu.sh"
