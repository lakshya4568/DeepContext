# DeepContext RAG Quality Next Steps

Branch: `feat/rag-quality-generation`. `main` is never modified by this work.

## Status as of this commit

### Fixed: reranker blend regression (root cause identified and reverted)

A later commit on this branch (`50fff08`) changed the reranker consensus blend from:

```text
0.60 * norm_rrf + 0.40 * norm_raw, consensus_boost = 0.15 for top-3
```

to:

```text
0.70 * norm_rrf + 0.30 * norm_raw, consensus_boost = 0.20 (top-3) / 0.08 (top-6)
```

On the identical 755-page GoT corpus and identical 36-query `tests/eval_dataset.json`,
this dropped full-pipeline Hit@5 from **87.1% to 61.3%**, with these category-level
losses:

| Category | Before (0.60/0.40) | After (0.70/0.30) | Delta |
|---|---:|---:|---:|
| Direct Factual Hit@5 | 100.0% | 75.0% | -25 |
| Citation/Appendix Hit@5 | 100.0% | 75.0% | -25 |
| Multi-Hop Hit@5 | 87.5% | 62.5% | -25 |
| Paraphrased Hit@5 | 66.7% | 16.7% | -50 |

Generation-side faithfulness *improved* over the same period (43% -> 86.2%), which is
why the retrieval regression was not obvious from faithfulness/abstention numbers alone.
Always check Hit@5/nDCG@5 alongside faithfulness after any reranker change.

**Fix applied in this commit:**

- `src/deep_context/retrieval/reranker.py`: blend weight and consensus boost tiers are
  now read from `settings` instead of being hardcoded, and the docstring on
  `_blend_with_rrf` records this regression history so it isn't silently reintroduced.
- `src/deep_context/core/config.py`: added `reranker_blend_rrf_weight` (default `0.60`),
  `reranker_consensus_top1_count` / `top2_count`, and `reranker_consensus_boost_tier1` /
  `tier2`, all overridable via environment variables for A/B testing without code edits.
- `tests/test_reranker_blend_regression.py`: locks the default weight at `0.60` and
  asserts the raw (lexical/semantic) signal can still outrank a weak-RRF candidate.

**To A/B test a different weight, do not edit the code — set an env var:**

```bash
RERANKER_BLEND_RRF_WEIGHT=0.70 uv run python scripts/evaluate_rag.py
RERANKER_BLEND_RRF_WEIGHT=0.60 uv run python scripts/evaluate_rag.py
```

Compare Hit@5, nDCG@5, and factual F1 across both runs before changing the default.

## Already shipped on this branch (prior commits)

- Two-pass grounded generation (`src/deep_context/generation/grounded_answer.py`),
  wired into `routes_rag.py`, `cli/main.py`, and `scripts/evaluate_rag.py`.
- Anachronism refusal, hop-coverage retry, and consensus-hit protection
  (`src/deep_context/retrieval/quality_gates.py`).
- Corpus-agnostic query rewriting (no hardcoded book/paper vocabulary).
- Document-agnostic appendix/chapter intent boosting in `hybrid.py`.
- Evidence verifier overlap threshold raised 0.35 -> 0.50.

## Still open

1. **Citation/appendix retrieval on scientific PDFs.** The generic `appendix a`..`appendix g`
   / `chapter 1`..`chapter 8` string matching in `hybrid.py` depends on those exact phrases
   appearing in both the query and the chunk's `section_path` or first 250 characters. Confirm
   the ingestion parser actually populates `section_path` for arbitrary PDFs before trusting
   this boost on a new corpus.
2. **Paraphrase category remains the weakest** (Hit@1 near 0% across every corpus tested so
   far). The rewriter was deliberately de-hardcoded to stop overfitting `eval_dataset.json`;
   expect lower paraphrase scores until a retrieve-rewrite-retrieve loop (grounding rewrites in
   the corpus's own vocabulary rather than a hardcoded glossary) is implemented.
3. **Multi-hop factual F1 is the lowest F1 across every run.** Verify `hop_coverage()` in
   `quality_gates.py` is actually triggering a second retrieval pass for multi-hop queries in
   the wired API/CLI/eval paths, not only inside `RetrievalEngine.retrieve`.
4. **Always re-run the full 36-query benchmark after any reranker or rewriter change**, and
   record Hit@5, nDCG@5, factual F1, and faithfulness together. A generation-only improvement
   can mask a retrieval regression, as this commit's history demonstrates.
