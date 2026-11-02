# Weighted RRF (patch for `main`, not applied to `main`)

`weighted_rrf.patch` lets each ranked list in Reciprocal Rank Fusion have its own vote:

- `reciprocal_rank_fusion(ranked_lists, k=60, weights=None)`: score = sum of `weight / (k + rank)`. No weights means equal
  votes of 1.0, so the old behaviour is unchanged. A wrong number of weights raises `ValueError`.
- Two settings, `RRF_WEIGHT_LEXICAL` and `RRF_WEIGHT_DENSE`, default 1.0 each (`core/config.py`). `HybridRetriever.retrieve_candidates`
  and `hybrid_search` pass them.
- `tests/test_weighted_rrf.py`: 8 tests (plain RRF unchanged, weights, zero vote, wrong length, and the retriever defaulting to
  equal votes and using the configured votes).

## Check on a pristine copy of `main`

`git archive main src tests/test_weighted_rrf.py`-style copy, `git apply -p1 weighted_rrf.patch`, then pytest with main's own
setting `asyncio_mode = "auto"` (it lives in `pyproject.toml`, which a bare `src` copy does not have): **8 passed**. Without that
setting the two async retriever tests fail with "async def functions are not natively supported", which is a missing setting
of the copy and not a defect of the patch.

## Why

On `main` as shipped, equal votes make the weak keyword list drag the fused ranking below dense-only search
(see `eval/MAIN_REBASE_RESULTS.md`). A keyword vote of 0.02 brings it back to dense-only level on the three custom sets
(nDCG@10 docs 0.831, thesis 0.490, novel 0.805, against 0.776, 0.414, 0.449 as shipped).

## Decision for the owner

The patch keeps the default at 1.0 / 1.0. Changing the shipped default is a retrieval-configuration change, so it needs your
decision. The evidence is first-stage retrieval on three sets (novel held out); the keyword channel adds nothing on top of dense
search there, so a low vote is a safe tie-breaker, and 0.02 was chosen on dev sets by a rule written down in advance.
