#!/usr/bin/env bash
# Starts the vLLM (Qwen3.5-2B) server first, then the reranker, both on localhost.
# Run on the VM: setsid nohup bash ~/start_servers.sh > ~/slm/start.out 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/slm"
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p logs
rm -f servers.ready servers.failed
fail() { echo "FAILED: $1"; touch servers.failed; exit 1; }

if ! curl -sf http://127.0.0.1:8000/health > /dev/null; then
  echo "== $(date -u +%FT%TZ) starting vLLM"
  setsid nohup vllm serve Qwen/Qwen3.5-2B \
    --host 127.0.0.1 --port 8000 --served-model-name qwen3.5-2b \
    --language-model-only --max-model-len 8192 \
    --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.55}" --max-num-seqs 64 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > logs/vllm.log 2>&1 < /dev/null &
fi
for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null; then echo "vLLM ready after about $((i * 5))s"; break; fi
  if ! pgrep -f "vllm serve" > /dev/null; then tail -n 40 logs/vllm.log; fail "vllm exited"; fi
  sleep 5
done
curl -sf http://127.0.0.1:8000/health > /dev/null || { tail -n 40 logs/vllm.log; fail "vllm not ready in 20 min"; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

if ! curl -sf http://127.0.0.1:8001/health > /dev/null; then
  echo "== $(date -u +%FT%TZ) starting reranker"
  setsid nohup uvicorn --app-dir "$HOME" rerank_server:app --host 127.0.0.1 --port 8001 \
    > logs/rerank.log 2>&1 < /dev/null &
fi
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8001/health > /dev/null; then echo "reranker ready after about $((i * 3))s"; break; fi
  if ! pgrep -f "rerank_server:app" > /dev/null; then tail -n 40 logs/rerank.log; fail "reranker exited"; fi
  sleep 3
done
curl -sf http://127.0.0.1:8001/health > /dev/null || { tail -n 40 logs/rerank.log; fail "reranker not ready"; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
touch servers.ready
echo "== $(date -u +%FT%TZ) both servers ready"
