"""Self-hosted BAAI/bge-reranker-v2-m3 server with the contract that eval/strict_clients.py expects.

POST /rerank  {"query": str, "documents": [str, ...]}
          ->  {"results": [{"index": int, "relevance_score": float}, ...], "usage": {"total_tokens": int}}
relevance_score = sigmoid(logit), i.e. in [0, 1] (the cached scores from earlier runs are all in [0, 1]).
Bound to localhost only; reach it through an SSH port-forward.
"""

from __future__ import annotations

import os
import threading
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "BAAI/bge-reranker-v2-m3"
MAX_LEN = int(os.environ.get("RERANK_MAX_LEN", "1024"))
BATCH = int(os.environ.get("RERANK_BATCH", "32"))

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, torch_dtype=torch.float16)
model = model.to("cuda").eval()
lock = threading.Lock()  # one GPU batch at a time; requests queue instead of competing for memory

app = FastAPI(title="bge-reranker-v2-m3")


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "model": MODEL, "max_len": MAX_LEN, "batch": BATCH}


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict[str, object]:
    started = time.time()
    n = len(req.documents)
    scores = [0.0] * n
    total_tokens = 0
    order = sorted(range(n), key=lambda i: len(req.documents[i]))  # similar lengths per batch
    with lock, torch.inference_mode():
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            enc = tokenizer(
                [req.query] * len(idx),
                [req.documents[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to("cuda")
            total_tokens += int(enc["attention_mask"].sum().item())
            logits = model(**enc).logits.view(-1).float()
            probs = torch.sigmoid(logits).cpu().tolist()
            for i, p in zip(idx, probs):
                scores[i] = float(p)
    results = sorted(
        ({"index": i, "relevance_score": s} for i, s in enumerate(scores)),
        key=lambda r: -r["relevance_score"],
    )
    return {
        "model": MODEL,
        "results": results,
        "usage": {"total_tokens": total_tokens},
        "latency_s": round(time.time() - started, 4),
    }
