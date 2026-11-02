#!/bin/zsh
# Rubric correctness and grounding for one arm's answers (Gemini 3.7 flash as judge; no GPU needed).
# Usage: judge.sh <fast|deep>
source "${0:A:h}/lib_final.sh"
arm=$1
export EVAL_GENERATOR_MODEL=qwen3.5-9b
echo "== $(date +%H:%M:%S) rubric $arm"
uv run --no-sync python -m eval.judge_correctness --results-dir $OUT/A_${arm} 2>&1 | grep -E "graded|Traceback|Error|rows to grade" | cut -c1-200 | tail -5
echo "== $(date +%H:%M:%S) grounding $arm"
uv run --no-sync python -m eval.judge_grounding --results-dir $OUT/A_${arm} 2>&1 | grep -E "graded|Traceback|Error|answers to grade" | cut -c1-200 | tail -5
echo "== $(date +%H:%M:%S) DONE judge $arm"
