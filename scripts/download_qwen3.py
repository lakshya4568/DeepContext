import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

model_name = "Qwen/Qwen3-0.6B"
token = os.getenv("HF_TOKEN", None)

print(f"[*] Downloading/Loading {model_name}...")
try:
    tok = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    print("[+] Tokenizer downloaded successfully.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading weights into {device} (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"[+] {model_name} loaded successfully on {device}!")

    # Test generation
    prompt = (
        "Summarize: DeepContext is a high-performance hybrid RAG system with HNSW vector search."
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    res = tok.decode(out[0], skip_special_tokens=True)
    print(f"[+] Test Generation Output:\n{res}")
    print("QWEN3_DOWNLOAD_SUCCESS=True")
except Exception as e:
    print(f"QWEN3_DOWNLOAD_ERROR: {e}")
