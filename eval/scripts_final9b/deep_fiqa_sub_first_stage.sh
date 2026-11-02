#!/bin/zsh
# Builds the deep index of the FiQA subset (summaries and embeddings come from the cache, only the last partial window and the
# extra relevant documents need new embeddings) and runs the first-stage searches; then writes the marker phase2_all.sh waits for.
# No GPU.
# Usage: deep_fiqa_sub_first_stage.sh
here="${0:A:h}"
source "$here/lib_final.sh"
rm -f "$OUT/.deep_fiqa-sub_first_stage"
zsh "$here/retr_nogpu.sh" deep fiqa-sub
touch "$OUT/.deep_fiqa-sub_first_stage"
