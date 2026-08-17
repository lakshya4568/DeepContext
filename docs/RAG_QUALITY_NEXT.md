# DeepContext RAG Quality Next Steps

Implemented on branch `feat/rag-quality-generation` against `main` @ `25b1181`.

## Shipped in this branch

- Two-pass grounded generation in `src/deep_context/generation/grounded_answer.py`
- Anachronism refusal, hop coverage, and consensus-hit protection in `src/deep_context/retrieval/quality_gates.py`
- Wider rerank pool + hop retry in `engine.py`
- BM25/dense ranks + section-based appendix boost in `hybrid.py`
- Corpus-agnostic rewrite prompts in `rewriter.py`
- Search-task Gemini rerank + RRF blend + bypass in `reranker.py`
- Partial-answer vs refuse contract in `prompt_assembler.py`
- Verifier overlap 0.35 -> 0.50 in `checker.py`

## Still wire manually

Replace the one-shot `llm_client.complete(...)` call in:

- `src/deep_context/api/routes_rag.py` (`query_platform` and the stream generator)
- `src/deep_context/cli/main.py` (`query_cmd`)
- `scripts/evaluate_rag.py` generation block

with:

```python
from deep_context.generation.grounded_answer import generate_grounded_answer

result = await generate_grounded_answer(query, retrieval_res.parent_chunks)
answer = result.answer
```

Then re-run `uv run pytest` and `uv run python scripts/evaluate_rag.py`.
