"""Gold questions for the documentation-QA benchmark (closed book: the repo's own docs only).

Every answerable item carries verbatim ``evidence`` quotes from the documentation on this branch.
``eval.docs_qa.verify_gold`` fails if any quote is missing, and relevance labels are derived by
locating the quotes in the section corpus, never assigned by hand. Quotes are matched after
removing markdown emphasis/backticks and collapsing whitespace.

Categories
  direct       single fact stated verbatim
  paraphrased  question shares few keywords with the evidence
  multi        needs two or more facts (same or different sections)
  unanswerable the documentation does not contain the answer -> the system should abstain
  false_prem   the question presumes something the docs contradict -> should correct or abstain
"""

from __future__ import annotations

from typing import Any

ABSTAIN_REF = "The documentation does not state this."

ITEMS: list[dict[str, Any]] = [
    # ------------------------------------------------------------------ direct
    dict(
        id="d01",
        cat="direct",
        q="Which environment variable overrides the reranker's RRF blend weight?",
        a="RERANKER_BLEND_RRF_WEIGHT",
        ev=["RERANKER_BLEND_RRF_WEIGHT=0.70 uv run python scripts/evaluate_rag.py"],
    ),
    dict(
        id="d02",
        cat="direct",
        q="What full-pipeline Hit@5 did the GoT benchmark show before and after the 0.70/0.30 blend change?",
        a="It fell from 87.1% to 61.3%.",
        ev=["full-pipeline Hit@5 from 87.1% to 61.3%"],
    ),
    dict(
        id="d03",
        cat="direct",
        q="What is the default value of AGENTIC_GRADE_THRESHOLD?",
        a="0.30",
        ev=["AGENTIC_GRADE_THRESHOLD (default 0.30)"],
    ),
    dict(
        id="d04",
        cat="direct",
        q="What is the default response-cache TTL?",
        a="300 seconds.",
        ev=["TTL-based expiry (CACHE_TTL, default 300s)"],
    ),
    dict(
        id="d05",
        cat="direct",
        q="Which cache backend is used when CACHE_URL is not set?",
        a="An in-memory backend (automatic fallback).",
        ev=["Redis-backed when CACHE_URL is set; automatic in-memory fallback otherwise."],
    ),
    dict(
        id="d06",
        cat="direct",
        q="What is the default value of AGENTIC_MAX_REWRITES?",
        a="2",
        ev=["AGENTIC_MAX_REWRITES=2 # bounded corrective loop"],
    ),
    dict(
        id="d07",
        cat="direct",
        q="What daily token quota does Groq's on-demand free tier enforce?",
        a="200,000 tokens per day (TPD).",
        ev=["Groq's on-demand free tier enforces a 200,000 tokens/day (TPD) quota"],
    ),
    dict(
        id="d08",
        cat="direct",
        q="What character budget was set for planner synthesis context to avoid Groq TPM overflows?",
        a="12,000 characters (about 3,000 tokens).",
        ev=["budgeted synthesis context to 12,000 chars (~3,000 tokens)"],
    ),
    dict(
        id="d09",
        cat="direct",
        q="Which model does the reranker upgrade guide propose as the real trained cross-encoder?",
        a="BAAI/bge-reranker-v2-m3",
        ev=["a real trained cross-encoder (BAAI/bge-reranker-v2-m3)"],
    ),
    dict(
        id="d10",
        cat="direct",
        q="According to the reranker upgrade guide, what Hit@1 value shows where the heuristic reranker failed?",
        a="32.3%",
        ev=["This is where your heuristic failed (32.3%)."],
    ),
    dict(
        id="d11",
        cat="direct",
        q="When was the full RLM paper submitted according to the critical assessment?",
        a="31 December 2025.",
        ev=["the full paper was submitted 31 December 2025"],
    ),
    dict(
        id="d12",
        cat="direct",
        q="By how much did the purpose-trained RLM-Qwen3-8B beat its base model?",
        a="By 28.3%.",
        ev=["(RLM-Qwen3-8B) beat its base model by 28.3%"],
    ),
    dict(
        id="d13",
        cat="direct",
        q="What share of an evaluation set does the Rag eval guide suggest should be adversarial or unanswerable?",
        a="10–20%.",
        ev=["10–20% adversarial or unanswerable questions"],
    ),
    dict(
        id="d14",
        cat="direct",
        q="How many questions does the Rag eval guide suggest for an initial prototype?",
        a="30–50 questions.",
        ev=["30–50 questions for an initial prototype"],
    ),
    dict(
        id="d15",
        cat="direct",
        q="How should a grounded RAG system behave when retrieval returns insufficient or low-confidence context?",
        a="Use the product's safe fallback behavior instead of fabricating an answer.",
        ev=[
            "If retrieval returns insufficient or low-confidence context, use the product’s safe fallback behavior instead of fabricating an answer."
        ],
    ),
    # ------------------------------------------------------------------ paraphrased
    dict(
        id="p01",
        cat="paraphrased",
        q="Why was the retrieval regression easy to miss even though answers looked better?",
        a="Generation-side faithfulness improved (43% to 86.2%) over the same period, which hid the retrieval drop.",
        ev=["Generation-side faithfulness improved over the same period (43% -> 86.2%)"],
    ),
    dict(
        id="p02",
        cat="paraphrased",
        q="Which mechanism keeps chunks that both lexical and dense search ranked highly from being dropped after reranking?",
        a="protect_consensus, which protects the top-10 dual hits.",
        ev=["protect_consensus (top-10 dual hits)"],
    ),
    dict(
        id="p03",
        cat="paraphrased",
        q="What stops a stale answer from being served after a source document is removed?",
        a="Deleting documents invalidates the rag cache namespace.",
        ev=[
            "Deleting documents (DELETE /v1/documents[/{id}]) invalidates the rag namespace so stale citations cannot survive re-ingestion."
        ],
    ),
    dict(
        id="p04",
        cat="paraphrased",
        q="What does the corrective loop do once its rewrite budget is spent and the documents are still weak?",
        a="It abstains with the standard insufficiency refusal instead of answering from weak context.",
        ev=[
            "abstain — safe fallback returning the standard insufficiency refusal, never answering from weak context."
        ],
    ),
    dict(
        id="p05",
        cat="paraphrased",
        q="Which answers are allowed into the whole-answer cache?",
        a="Only answers that passed the evidence-support gate; RLM-engine answers are never cached.",
        ev=[
            "/v1/query caches only answers that passed the evidence-support gate (RLM-engine answers are never cached)"
        ],
    ),
    dict(
        id="p06",
        cat="paraphrased",
        q="How does the corrective RAG grader decide a retrieved document is relevant?",
        a="Deterministic term-overlap scoring against AGENTIC_GRADE_THRESHOLD (default 0.30).",
        ev=["deterministic term-overlap scoring against AGENTIC_GRADE_THRESHOLD (default 0.30)"],
    ),
    dict(
        id="p07",
        cat="paraphrased",
        q="Why is a high faithfulness score alone not enough to trust a RAG system?",
        a="An answer can be fully supported by the retrieved context and still be wrong if the retriever missed necessary information.",
        ev=[
            "an answer can be completely supported by the retrieved context while still being wrong because the retriever missed necessary information"
        ],
    ),
    dict(
        id="p08",
        cat="paraphrased",
        q="How does the cache keep per-user personalization without putting who is asking into its keys?",
        a="User IDs are deliberately excluded from key material; personalization is resolved inside the pipeline via embedding/reranker preferences that are part of the key.",
        ev=["User IDs are deliberately excluded from key material"],
    ),
    # ------------------------------------------------------------------ multi
    dict(
        id="m01",
        cat="multi",
        q="After the blend regression, which config setting now holds the RRF weight and what is its default?",
        a="reranker_blend_rrf_weight, default 0.60.",
        ev=["added reranker_blend_rrf_weight (default 0.60)"],
    ),
    dict(
        id="m02",
        cat="multi",
        q="Which two layers does the Rag eval guide say a RAG evaluation should be split into, and which Python library does it suggest starting with?",
        a="Retrieval quality and answer quality; Ragas.",
        ev=[
            "evaluate your RAG system in two layers: retrieval quality and answer quality",
            "I recommend starting with Ragas for metric-based evaluation",
        ],
    ),
    dict(
        id="m03",
        cat="multi",
        q="What stays unchanged in the BGE upgrade's blend and consensus handling, and what does the decision rule say to do if BGE wins on Hit@1 and nDCG@5?",
        a="The RRF blend and protect_consensus stay unchanged; make local_cross_encoder the default.",
        ev=[
            "Blend logic | _blend_with_rrf (0.60 RRF + 0.40 raw) | Same — unchanged",
            "Beats heuristic on Hit@1 and nDCG@5 | Make local_cross_encoder the default",
        ],
    ),
    dict(
        id="m04",
        cat="multi",
        q="Which reranking strategy uses the hosted BGE model, and what does it return?",
        a="The ecohash strategy: BGE-reranker-v2-m3 cross-encoder via the EcoHash API with calibrated probabilities.",
        ev=[
            "ecohash (Hosted Neural): BGE-reranker-v2-m3 cross-encoder via EcoHash API with calibrated probabilities."
        ],
    ),
    dict(
        id="m05",
        cat="multi",
        q="Name the memory store for immutable constraints and the one for session history.",
        a="memory_policy holds immutable runtime safety and behavior constraints; memory_episode holds session history.",
        ev=[
            "memory_policy: Immutable runtime safety and behavior constraints.",
            "memory_episode: Session history, outcome summaries, and interaction traces.",
        ],
    ),
    dict(
        id="m06",
        cat="multi",
        q="Which request fields feed the cache key, and which identifier is deliberately left out?",
        a="query, tenant, permissions, document_ids, top_k, model, embedding model/dim and reranker feed the key; user IDs are excluded.",
        ev=[
            "top_k, model, embedding_model, embedding_dim, reranker",
            "User IDs are deliberately excluded from key material",
        ],
    ),
    dict(
        id="m07",
        cat="multi",
        q="What did Prime Intellect's ablation find about RLM runtime, and what happened on math-python with GPT-5-mini?",
        a="RLM always increases the time to finish the job, and it made GPT-5-mini worse at math-python.",
        ev=[
            "In all cases, the RLM increases the time required to finish the job significantly",
            "made GPT-5-mini worse at math-python",
        ],
    ),
    dict(
        id="m08",
        cat="multi",
        q="Name two of the compounding bugs that caused the Page 616 miss.",
        a="Score inversion in the cross-encoder reranker (negative BM25 scores) and punctuation/curly quotes breaking FTS.",
        ev=["Score Inversion in Cross-Encoder Reranker", "Punctuation & Curly Quotes in FTS"],
    ),
    dict(
        id="m09",
        cat="multi",
        q="What do the repo rules say about editing uv.lock and about which tools may manage Python dependencies?",
        a="Never manually edit uv.lock (commit lockfile changes when resolution changes); use uv, not pip, poetry, pipenv or conda.",
        ev=[
            "Never manually edit uv.lock.",
            "Do not use pip, poetry, pipenv, conda, or system-level package installation unless the user explicitly requests an exception.",
        ],
    ),
    # ------------------------------------------------------------------ unanswerable
    dict(
        id="u01",
        cat="unanswerable",
        q="What is the monthly infrastructure cost of running the platform on AWS?",
        a=ABSTAIN_REF,
        ev=[],
    ),
    dict(
        id="u02",
        cat="unanswerable",
        q="Which company is the primary commercial customer of the platform?",
        a=ABSTAIN_REF,
        ev=[],
    ),
    dict(
        id="u03",
        cat="unanswerable",
        q="What uptime SLA does the project guarantee?",
        a=ABSTAIN_REF,
        ev=[],
    ),
    dict(
        id="u04",
        cat="unanswerable",
        q="What Hit@5 did the system achieve on the BEIR SciFact benchmark?",
        a=ABSTAIN_REF,
        ev=[],
    ),
    dict(
        id="u05",
        cat="unanswerable",
        q="How many contributors have committed to the repository?",
        a=ABSTAIN_REF,
        ev=[],
    ),
    dict(
        id="u06",
        cat="unanswerable",
        q="Which Kubernetes version is required to deploy the platform?",
        a=ABSTAIN_REF,
        ev=[],
    ),
    dict(
        id="u07",
        cat="unanswerable",
        q="What is the p99 latency of the hosted EcoHash reranker in production?",
        a=ABSTAIN_REF,
        ev=[],
    ),
    dict(
        id="u08",
        cat="unanswerable",
        q="Which encryption algorithm protects the API keys stored in the .env file?",
        a=ABSTAIN_REF,
        ev=[],
    ),
    # ------------------------------------------------------------------ false premise
    dict(
        id="f01",
        cat="false_prem",
        q="Why did the platform migrate from pgvector to Qdrant?",
        a="The documentation does not describe a migration to Qdrant; pgvector remains the vector store.",
        ev=[],
    ),
    dict(
        id="f02",
        cat="false_prem",
        q="By how much did Hit@5 improve after the blend was changed to 0.70/0.30?",
        a="It did not improve; it fell from 87.1% to 61.3%.",
        ev=["full-pipeline Hit@5 from 87.1% to 61.3%"],
    ),
    dict(
        id="f03",
        cat="false_prem",
        q="Why does the corrective RAG loop depend on LangChain?",
        a="It does not; the cache, scheduler and agentic loop are hand-built with no LangGraph, Airflow or Celery dependencies.",
        ev=["All three are hand-built (no LangGraph / Airflow / Celery dependencies)"],
    ),
    dict(
        id="f04",
        cat="false_prem",
        q="When did the local ONNX BGE model replace the heuristic reranker as the default strategy?",
        a="It did not; cross_encoder (the heuristic) is documented as the default.",
        ev=["cross_encoder (Default): Exact quote match + token Jaccard overlap + lexical boost."],
    ),
]
