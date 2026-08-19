<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# how can I evaluate my rag model, some python modules that can help me to evaluate my rag model best of them

The best practical setup is to evaluate your RAG system in **two layers**: retrieval quality and answer quality. For Python, I recommend starting with **Ragas** for metric-based evaluation, then adding **DeepEval** if you want pytest/CI integration and broader LLM testing. Ragas provides metrics for evaluating RAG components systematically rather than relying only on manual “vibe checks.”[^1]

## What to evaluate

### 1. Retrieval quality

Your retriever should return the right chunks, preferably near the top.


| Metric | What it measures | Useful for |
| :-- | :-- | :-- |
| Context precision | Whether relevant chunks are ranked above irrelevant chunks | Testing reranking, top-$k$, chunk ordering |
| Context recall | Whether all information needed for the answer was retrieved | Testing embeddings, chunking, search coverage |
| Context relevancy | How relevant the retrieved context is to the query | Detecting noisy or oversized retrieval results |
| Hit rate / Recall@k | Whether at least one relevant document appears in the top $k$ | Basic retriever benchmarking |
| MRR | How early the first relevant document appears | Comparing ranking strategies |
| nDCG@k | Ranking quality when documents have different relevance levels | Advanced retrieval evaluation |

### 2. Generation quality

Your generator should answer using the retrieved context without inventing information.


| Metric | What it measures |
| :-- | :-- |
| Faithfulness | Whether answer claims are supported by retrieved context |
| Answer relevancy | Whether the answer actually addresses the question |
| Correctness | Whether the answer matches a reference answer |
| Completeness | Whether important parts of the expected answer are included |
| Citation correctness | Whether cited passages support the associated claims |
| Citation completeness | Whether factual claims have citations |

Faithfulness alone is insufficient: an answer can be completely supported by the retrieved context while still being wrong because the retriever missed necessary information. Context recall and answer correctness help detect this problem.[^2]

## Recommended Python modules

### 1. Ragas — best starting point

Install:

```bash
pip install ragas datasets
```

Ragas is particularly convenient for evaluating:

- Faithfulness.
- Answer relevancy.
- Context precision.
- Context recall.
- Context relevancy.
- Noise sensitivity.

It supports both reference-free evaluation and evaluation using reference answers. Its main advantage is that you can evaluate a complete RAG pipeline with a relatively small amount of code.[^3]

A typical evaluation dataset contains:

```python
dataset = {
    "question": ["What is the refund policy?"],
    "answer": ["Customers can request a refund within 30 days."],
    "contexts": [["Refunds are available within 30 days of purchase."]],
    "ground_truth": ["Customers may request a refund within 30 days of purchase."],
}
```

A current-style Ragas evaluation can look like this:

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

data = {
    "user_input": ["What is the refund policy?"],
    "response": ["Customers can request a refund within 30 days."],
    "retrieved_contexts": [["Refunds are available within 30 days of purchase."]],
    "reference": ["Customers may request a refund within 30 days of purchase."],
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

print(result)
```

The exact imports can vary between Ragas versions, so check the version-specific documentation if an API has changed.

### 2. DeepEval — best for testing and CI/CD

Install:

```bash
pip install deepeval
```

DeepEval is a good choice when you want to treat your RAG pipeline like a testable software component. It supports metrics such as:

- `FaithfulnessMetric`.
- `AnswerRelevancyMetric`.
- `ContextualRelevancyMetric`.
- `ContextualPrecisionMetric`.
- `ContextualRecallMetric`.

It also integrates naturally with pytest and can be used in continuous integration pipelines.[^4]

Example:

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
)

test_case = LLMTestCase(
    input="What is the refund policy?",
    actual_output="Customers can request a refund within 30 days.",
    retrieval_context=["Refunds are available within 30 days of purchase."],
    expected_output="Customers may request a refund within 30 days of purchase.",
)

metrics = [
    FaithfulnessMetric(threshold=0.8),
    AnswerRelevancyMetric(threshold=0.8),
    ContextualRelevancyMetric(threshold=0.8),
    ContextualRecallMetric(threshold=0.8),
]

evaluate(test_cases=[test_case], metrics=metrics)
```

DeepEval describes its contextual metrics as retriever-focused and faithfulness/answer relevancy as generator-focused, which makes it useful for locating the failing component in your pipeline.[^5]

### 3. TruLens — best for observability

TruLens is useful when you want to inspect individual RAG steps:

- Query transformation.
- Retriever calls.
- Reranking.
- Prompt construction.
- LLM generation.
- Final response.

Use it when your main problem is not only “what score did I get?” but also “why did this response fail?” It is better suited to trace-level debugging and runtime observability than to a minimal offline evaluation setup.[^6]

### 4. Arize Phoenix — best for tracing and debugging

Phoenix is useful for:

- Inspecting retrieval traces.
- Finding irrelevant retrieved chunks.
- Comparing embedding and reranking behavior.
- Monitoring latency and token usage.
- Reviewing production failures.

A common architecture is to use Ragas or DeepEval for quantitative evaluation and Phoenix for tracing and debugging.

### 5. Promptfoo — best for prompt/model comparisons

Promptfoo is useful for testing multiple:

- Prompt templates.
- LLM providers.
- Model versions.
- RAG configurations.
- Safety rules.

For example, you can compare:

```text
chunk_size=300, top_k=5
chunk_size=500, top_k=5
chunk_size=500, top_k=10
```

This is useful when selecting a configuration rather than evaluating one fixed pipeline.

## My recommendation

For your AI/RAG projects, use this stack:


| Requirement | Recommended tool |
| :-- | :-- |
| Offline RAG metrics | Ragas |
| Automated regression tests | DeepEval |
| Runtime traces | Phoenix |
| Prompt/model experiments | Promptfoo |
| Custom retrieval metrics | `scikit-learn`, `ranx`, or your own code |

If you want only one library, choose **Ragas**. If you want a production-oriented evaluation workflow, combine **Ragas + DeepEval + Phoenix**.

## Build a proper evaluation dataset

Do not evaluate only five manually selected questions. Create a dataset with roughly:

- 30–50 questions for an initial prototype.
- 100–300 questions for a more reliable comparison.
- 10–20% adversarial or unanswerable questions.

Include different categories:

```json
{
  "question": "What is the cancellation policy?",
  "reference": "Users can cancel within 14 days.",
  "relevant_doc_ids": ["policy_03"],
  "difficulty": "easy",
  "category": "policy"
}
```

Include:

- Direct fact questions.
- Multi-hop questions requiring multiple chunks.
- Questions containing ambiguous terms.
- Questions with paraphrased wording.
- Questions whose answer is not present in the knowledge base.
- Questions targeting outdated or conflicting documents.
- Questions requiring citations.

For unanswerable questions, test whether your model says something like:

```text
I could not find enough information in the provided documents to answer that reliably.
```

This is often more important than maximizing a single score.

## Use separate evaluation splits

Create three datasets:

1. **Development set** — used while changing chunking, embeddings, prompts, and $k$.
2. **Validation set** — used for choosing the best configuration.
3. **Test set** — kept unchanged until the final evaluation.

Do not repeatedly optimize against the same test questions. Otherwise, your system may overfit to those questions and appear better than it really is.

## Add non-LLM metrics

LLM-as-a-judge metrics are useful, but they can be inconsistent and expensive. Also measure deterministic metrics where possible.

```python
from sklearn.metrics import precision_score, recall_score

# 1 means relevant, 0 means irrelevant
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 1, 1, 0, 0]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
```

For retrieval, track:

```text
Recall@1
Recall@3
Recall@5
MRR
nDCG@5
Hit rate
```

For the complete system, track:

```text
Faithfulness
Answer relevancy
Answer correctness
Citation precision
Citation recall
p50 latency
p95 latency
Token cost per query
Abstention accuracy
```


## Interpret failures correctly

| Observed result | Likely problem |
| :-- | :-- |
| Low context recall | Poor embeddings, bad chunking, insufficient $k$, missing documents |
| Low context precision | Too many irrelevant chunks, weak reranker, excessive $k$ |
| High context scores but low faithfulness | Prompting or generation problem |
| High faithfulness but low correctness | Retrieved context is incomplete or incorrect |
| High correctness but low answer relevancy | Answer is overly verbose or poorly focused |
| Good offline scores but poor user feedback | Evaluation dataset does not represent real usage |
| Good retrieval but hallucinated citations | Citation extraction or answer-generation problem |
| Good scores on easy questions only | Dataset lacks difficult, multi-hop, and unanswerable cases |

## Practical baseline

Start with this minimal workflow:

```text
1. Create 100 question-answer-context examples.
2. Store question, retrieved chunks, answer, and reference answer.
3. Run Ragas:
   - faithfulness
   - answer relevancy
   - context precision
   - context recall
4. Run DeepEval threshold tests in CI.
5. Log failed examples and inspect their traces.
6. Change only one variable at a time:
   - chunk size
   - overlap
   - embedding model
   - top-k
   - reranker
   - prompt
7. Re-run the same evaluation set.
8. Confirm improvements on an untouched test set.
```

A sensible initial target is not a universal score such as “everything must be above 0.9.” Establish a baseline first, then improve the metrics most related to your application’s failure cost. For a document question-answering system, prioritize **context recall, faithfulness, answer correctness, and citation correctness**.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://docs.ragas.io/en/stable/

[^2]: https://redis.io/blog/get-better-rag-responses-with-ragas/

[^3]: https://docs.ragas.io/en/v0.1.21/concepts/metrics/

[^4]: https://deepeval.com/docs/getting-started-rag

[^5]: https://deepeval.com/guides/guides-rag-evaluation

[^6]: https://www.datasumi.com/blog/rag-evaluation-frameworks

[^7]: https://deepeval.com/docs/metrics-ragas

[^8]: https://deepeval.com/docs/metrics-introduction

[^9]: https://www.patronus.ai/llm-testing/rag-evaluation-metrics

[^10]: https://www.neurealm.com/blogs/harnessing-the-power-of-llm-evaluation-with-ragas-a-comprehensive-guide/

[^11]: https://www.confident-ai.com/blog/how-to-evaluate-rag-applications-in-ci-cd-pipelines-with-deepeval

[^12]: https://www.llamaindex.ai/blog/evaluating-rag-with-deepeval-and-llamaindex

[^13]: https://deepeval.com/blog/top-5-llm-evaluation-frameworks

[^14]: https://qaskills.sh/blog/ragas-faithfulness-answer-relevancy-guide

[^15]: https://www.sapotacorp.vn/blog/rag-evaluation-ragas-faithfulness-relevance

