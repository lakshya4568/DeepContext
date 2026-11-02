#!/usr/bin/env bash
# Swap the summariser for Qwen3.5-9B (FP8) on the same port; the reranker keeps running.
# Run on the VM: setsid nohup bash ~/start_9b.sh > ~/slm/start9b.out 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/slm"
# shellcheck disable=SC1091
source .venv/bin/activate
rm -f servers9b.ready servers9b.failed
fail() { echo "FAILED: $1"; touch servers9b.failed; exit 1; }
pkill -f "[v]llm serve" || true
for i in $(seq 1 30); do pgrep -f "[v]llm serve" > /dev/null || break; sleep 2; done
sleep 5
echo "== $(date -u +%FT%TZ) starting vLLM with Qwen3.5-9B (fp8)"
setsid nohup vllm serve Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port 8000 --served-model-name qwen3.5-9b \
  --language-model-only --max-model-len 8192 --quantization fp8 \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.66}" --max-num-seqs 48 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > logs/vllm9b.log 2>&1 < /dev/null &
for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null; then echo "vLLM 9B ready after about $((i * 5))s"; break; fi
  if ! pgrep -f "[v]llm serve" > /dev/null; then tail -n 40 logs/vllm9b.log; fail "vllm exited"; fi
  sleep 5
done
curl -sf http://127.0.0.1:8000/health > /dev/null || { tail -n 40 logs/vllm9b.log; fail "vllm not ready"; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
touch servers9b.ready
echo "== $(date -u +%FT%TZ) 9B ready"
