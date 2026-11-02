#!/usr/bin/env bash
# Installs uv, a Python venv with the nightly vLLM build (needed for Qwen3.5), and downloads the two models.
# Run on the VM:  nohup bash ~/setup.sh > ~/setup.out 2>&1 &
set -euo pipefail
mkdir -p "$HOME/slm"
cd "$HOME/slm"
rm -f setup.done setup.failed
trap 'echo "setup failed at line $LINENO" >> setup.log; touch setup.failed' ERR
exec >> setup.log 2>&1
echo "== $(date -u +%FT%TZ) setup start"

if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== $(date -u +%FT%TZ) installing vllm (nightly, as the Qwen3.5 model card says)"
uv pip install vllm --torch-backend="${TORCH_BACKEND:-auto}" --extra-index-url https://wheels.vllm.ai/nightly

python - <<'PY'
import torch, transformers, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda available", torch.cuda.is_available(), "| torch cuda", torch.version.cuda)
PY

echo "== $(date -u +%FT%TZ) downloading models"
hf download Qwen/Qwen3.5-2B
hf download BAAI/bge-reranker-v2-m3 --exclude "assets/*"
du -sh "$HOME/.cache/huggingface" || true

echo "== $(date -u +%FT%TZ) setup done"
touch setup.done
