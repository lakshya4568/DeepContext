#!/bin/zsh
# Everything that needs the GPU tunnel but not the new summaries, most valuable first (run after a smoke test of answers.sh):
#   strict-prompt answers for docs, thesis, novel (both arms), the deep reranker setting for those three,
#   the fast reranker setting for all eight sets, fast answers for the two public sets with reference answers.
# Needs the tunnel (local 18000 = 9B, 18001 = reranker). One job at a time: the Mac has 8 GB.
here="${0:A:h}"
zsh "$here/answers.sh" fast docsqa pdf-quantum pdf-got
zsh "$here/answers.sh" deep docsqa pdf-quantum pdf-got
zsh "$here/retr_rerank.sh" deep docsqa pdf-quantum pdf-got
zsh "$here/retr_rerank.sh" fast docsqa pdf-quantum pdf-got ragbench-covidqa multihoprag scifact hotpotqa fiqa
zsh "$here/answers.sh" fast ragbench-covidqa multihoprag
echo "== $(date +%H:%M:%S) DONE chain_gpu_a"
