"""RAGAS judge wiring: LLM + embeddings backed by the strict, cached Vertex runtime.

The judge is a different model from the generator (``eval.config.JUDGE_MODEL``), runs at
temperature 0 (the caller's temperature is ignored), and every judgement is cached on disk, so
re-scoring is free and deterministic.
"""

from __future__ import annotations

import math
import typing as t
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings  # noqa: F401 - base of BaseRagasEmbeddings
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from eval import config as C
from eval.strict_clients import VertexRuntime, get_runtime, stage_scope

METRIC_COLUMNS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
    "factual_correctness(mode=f1)": "factual_correctness_f1",
}


@dataclass
class VertexJudgeLLM(BaseRagasLLM):
    model: str = C.JUDGE_MODEL
    thinking_budget: int | None = None
    runtime: VertexRuntime = field(default_factory=get_runtime, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.multiple_completion_supported = False  # n>1 handled by looping below

    def generate_text(self, *a: t.Any, **k: t.Any) -> LLMResult:  # pragma: no cover
        raise RuntimeError("Synchronous generation is not used by the harness.")

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = 0.01,  # noqa: ARG002 - judge is always temperature 0
        stop: list[str] | None = None,  # noqa: ARG002
        callbacks: t.Any = None,  # noqa: ARG002
    ) -> LLMResult:
        gens = []
        for i in range(n):
            with stage_scope("judge"):
                res = await self.runtime.generate(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt.to_string()}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=8192,
                    json_mode=True,
                    thinking_budget=self.thinking_budget,
                    candidate_idx=i,
                    stage="judge",
                )
            gens.append(
                Generation(text=res.text, generation_info={"finish_reason": res.finish_reason})
            )
        return LLMResult(generations=[gens])

    def is_finished(self, response: LLMResult) -> bool:
        for gen_list in response.generations:
            for g in gen_list:
                fr = (g.generation_info or {}).get("finish_reason")
                if fr not in (None, "STOP", "FinishReason.STOP"):
                    return False
        return True


class VertexEmbeddings(BaseRagasEmbeddings):
    """gemini-embedding-2 (768-d, global endpoint) for ResponseRelevancy; raw text, cached."""

    def __init__(self, runtime: VertexRuntime | None = None) -> None:
        super().__init__()
        self.runtime = runtime or get_runtime()
        self.run_config = RunConfig()

    async def aembed_query(self, text: str) -> list[float]:
        v = await self.runtime.embed_formatted(
            text.strip() or " ", model=C.EMBEDDING_MODEL, dim=C.EMBEDDING_DIM
        )
        return v.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [await self.aembed_query(x) for x in texts]

    def embed_query(self, text: str) -> list[float]:
        v = self.runtime.embed_formatted_sync(
            text.strip() or " ", model=C.EMBEDDING_MODEL, dim=C.EMBEDDING_DIM
        )
        return v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(x) for x in texts]


def build_metrics(llm: VertexJudgeLLM, emb: VertexEmbeddings) -> list[t.Any]:
    return [
        Faithfulness(llm=llm),
        # strictness=1: temperature-0 judge returns identical candidates, so the default 3 would
        # triple the cost for no information.
        ResponseRelevancy(llm=llm, embeddings=emb, strictness=1),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        FactualCorrectness(llm=llm, mode="f1"),
    ]


def score_rows(
    rows: list[dict[str, t.Any]], model: str = C.JUDGE_MODEL, thinking_budget: int | None = None
) -> list[dict[str, float]]:
    """Score ``rows`` (user_input / response / retrieved_contexts / reference) with RAGAS."""
    llm = VertexJudgeLLM(model=model, thinking_budget=thinking_budget)
    emb = VertexEmbeddings()
    ds = EvaluationDataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
            }
            for r in rows
        ]
    )
    result = evaluate(
        ds,
        metrics=build_metrics(llm, emb),
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(
            timeout=600, max_retries=3, max_wait=30, max_workers=C.LLM_CONCURRENCY * 2
        ),
        raise_exceptions=False,
        show_progress=False,
        allow_nest_asyncio=False,
    )
    df = result.to_pandas()
    out = []
    for _, rec in df.iterrows():
        scores = {}
        for col, short in METRIC_COLUMNS.items():
            v = rec.get(col)
            scores[short] = (
                float(v)
                if v is not None and not (isinstance(v, float) and math.isnan(v))
                else float("nan")
            )
        out.append(scores)
    return out
