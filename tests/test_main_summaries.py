"""Deep-ingestion arm: the application's ChunkSummarizer runs off-line, its output is cached, then embedded."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from eval import main_summaries as MS
from eval import pg
from eval.pg_cache import MemoryKV
from eval.vm import summarize_prompts as VM

from deep_context.core.types import Chunk, Document
from deep_context.ingestion.chunker import ParentChildChunker
from deep_context.ingestion.parser import DocumentParser
from deep_context.ingestion.summarizer import ChunkSummarizer

TEXT = "\n\n".join(f"Part {i}. " + " ".join(f"word{i}x{j}" for j in range(220)) for i in range(6))

Staged = tuple[Document, list[Chunk], list[Chunk]]


def _staged(ext_id: str = "d1", title: str = "Doc one") -> Staged:
    doc_id = pg.doc_uuid(ext_id)
    parents, children = ParentChildChunker().chunk_sections(
        doc_id, DocumentParser.parse(TEXT, doc_type="text")
    )
    return Document(id=doc_id, title=title, doc_type="text"), parents, children


def _requests(staged: Staged, summarizer: ChunkSummarizer) -> list[MS.SummaryRequest]:
    doc, parents, children = staged
    return MS.build_requests(summarizer, children, {p.id: p for p in parents}, title=doc.title)


def _cache_for(
    staged: list[Staged], summarizer: ChunkSummarizer, *, skip_last: bool = False
) -> MS.SummaryCache:
    cache = MS.SummaryCache(MemoryKV())
    requests = [r for s in staged for r in _requests(s, summarizer)]
    for r in requests[:-1] if skip_last else requests:
        cache.put_many({r.key: MS.SummaryRecord(f"Summary {r.key[:6]}.", 9, summarizer.model_name)})
    return cache


# --------------------------------------------------------------------------- requests and keys
def test_summary_key_depends_on_model_budget_and_prompt() -> None:
    keys = {
        MS.summary_key("m", 180, "p"),
        MS.summary_key("m2", 180, "p"),
        MS.summary_key("m", 90, "p"),
        MS.summary_key("m", 180, "p2"),
    }
    assert len(keys) == 4
    assert MS.summary_key("m", 180, "p") == MS.summary_key("m", 180, "p")


def test_a_request_holds_the_prompt_the_applications_summariser_builds() -> None:
    staged = _staged()
    summarizer = ChunkSummarizer()
    doc, parents, children = staged
    requests = _requests(staged, summarizer)
    assert len(requests) == len(children)
    child = children[0]
    expected = summarizer._build_prompt(
        child.content,
        section_path=child.section_path,
        document_title=doc.title,
        parent_context=parents[0].content,
    )
    assert requests[0].prompt == expected
    assert requests[0].key == MS.summary_key(
        summarizer.model_name, summarizer.max_summary_tokens, expected
    )


def test_a_child_with_an_unknown_parent_gets_no_parent_context() -> None:
    doc, _parents, children = _staged()
    summarizer = ChunkSummarizer()
    (request,) = MS.build_requests(summarizer, children[:1], {}, title=doc.title)
    assert request.prompt == summarizer._build_prompt(
        children[0].content,
        section_path=children[0].section_path,
        document_title=doc.title,
        parent_context=None,
    )


# --------------------------------------------------------------------------- cache
def test_cache_round_trips_records_omits_missing_keys_and_keeps_the_first_write() -> None:
    cache = MS.SummaryCache(MemoryKV())
    record = MS.SummaryRecord("A summary.", 7, "m")
    assert cache.put_many({"k1": record}) == 1
    assert cache.get_many(["k1", "k2"]) == {"k1": record}
    assert cache.put_many({"k1": MS.SummaryRecord("Other.", 1, "m")}) == 0
    assert cache.get_many(["k1"]) == {"k1": record}


# --------------------------------------------------------------------------- attach
def test_attach_sets_summaries_on_the_children_and_leaves_parents_alone() -> None:
    staged = [_staged()]
    summarizer = ChunkSummarizer()
    report = MS.attach_summaries(staged, _cache_for(staged, summarizer), summarizer)
    _doc, parents, children = staged[0]
    assert report == MS.AttachReport(n_children=len(children), n_fallback=0)
    assert all((c.summary_text or "").startswith("Summary ") for c in children)
    assert {c.summary_tokens for c in children} == {9}
    assert {c.summary_model for c in children} == {summarizer.model_name}
    assert all(p.summary_text is None for p in parents)


def test_an_empty_model_output_falls_back_to_the_first_25_words_as_the_application_does() -> None:
    staged = [_staged()]
    summarizer = ChunkSummarizer()
    cache = MS.SummaryCache(MemoryKV())
    for r in _requests(staged[0], summarizer):
        cache.put_many({r.key: MS.SummaryRecord("", 0, summarizer.model_name)})
    report = MS.attach_summaries(staged, cache, summarizer)
    children = staged[0][2]
    long_child = max(children, key=lambda c: len(c.content.split()))
    assert len(long_child.content.split()) > 25
    assert long_child.summary_text == " ".join(long_child.content.split()[:25]) + "..."
    assert long_child.summary_tokens == 25
    short_child = min(children, key=lambda c: len(c.content.split()))
    assert len(short_child.content.split()) <= 25
    assert short_child.summary_text == " ".join(short_child.content.split())  # no ellipsis
    assert report.n_fallback == report.n_children == len(children)


def test_attach_refuses_when_a_summary_is_missing_and_changes_nothing() -> None:
    staged = [_staged()]
    summarizer = ChunkSummarizer()
    cache = _cache_for(staged, summarizer, skip_last=True)
    children = staged[0][2]
    with pytest.raises(MS.MissingSummariesError) as err:
        MS.attach_summaries(staged, cache, summarizer)
    assert (err.value.n_missing, err.value.n_total) == (1, len(children))
    assert "eval.main_summaries export" in str(err.value)
    assert all(c.summary_text is None for c in children)


def test_contextual_text_is_what_the_applications_pipeline_embeds() -> None:
    assert MS.contextual_text(Chunk(content="body", summary_text="sum")) == "sum\n\nbody"
    assert MS.contextual_text(Chunk(content="body")) == "body"
    assert MS.contextual_text(Chunk(content="body", summary_text="")) == "body"


# --------------------------------------------------------------------------- job files
def test_export_lists_each_uncached_prompt_once(tmp_path: Path) -> None:
    cache = MS.SummaryCache(MemoryKV())
    cache.put_many({"k0": MS.SummaryRecord("done", 1, "m")})
    requests = [MS.SummaryRequest(f"k{i}", f"prompt {i}") for i in (0, 1, 1, 2)]
    out = tmp_path / "jobs.jsonl"
    report = MS.export_jobs(requests, cache, out)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert lines == [{"key": "k1", "prompt": "prompt 1"}, {"key": "k2", "prompt": "prompt 2"}]
    assert report == MS.ExportReport(total=4, unique=3, cached=1, written=2)


def test_import_fills_the_cache_and_a_second_import_adds_nothing(tmp_path: Path) -> None:
    rows = [
        {"key": "k1", "summary": "One.", "tokens": 3, "model": "m"},
        {"key": "k2", "summary": "", "tokens": 0, "model": "m"},
    ]
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    cache = MS.SummaryCache(MemoryKV())
    assert MS.import_results(results, cache) == 2
    assert cache.get_many(["k1", "k2"]) == {
        "k1": MS.SummaryRecord("One.", 3, "m"),
        "k2": MS.SummaryRecord("", 0, "m"),
    }
    assert MS.import_results(results, cache) == 0


# --------------------------------------------------------------------------- GPU-side loop
class _FakeSummarizer:
    model_name = "fake/model"

    def __init__(self, empty_for: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.empty_for = empty_for or set()

    def _generate_batch_sync(self, prompts: list[str]) -> list[tuple[str, int]]:
        self.calls.append(list(prompts))
        return [("", 0) if p in self.empty_for else (f"summary of {p}", 5) for p in prompts]


def test_the_gpu_loop_answers_every_job_once_in_batches_longest_prompt_first() -> None:
    jobs = [{"key": f"k{i}", "prompt": "x" * (10 + i)} for i in range(5)]
    fake = _FakeSummarizer()
    emitted: list[dict[str, Any]] = []
    assert VM.summarize_jobs(jobs, set(), fake, batch_size=2, emit=emitted.append) == 5
    assert sorted(r["key"] for r in emitted) == [f"k{i}" for i in range(5)]
    assert all(len(call) <= 2 for call in fake.calls)
    assert fake.calls[0][0] == "x" * 14
    assert {r["model"] for r in emitted} == {"fake/model"}


def test_the_gpu_loop_skips_finished_keys_so_a_run_can_resume() -> None:
    jobs = [{"key": f"k{i}", "prompt": f"p{i}"} for i in range(4)]
    emitted: list[dict[str, Any]] = []
    done = VM.summarize_jobs(jobs, {"k0", "k3"}, _FakeSummarizer(), 8, emitted.append)
    assert done == 2
    assert sorted(r["key"] for r in emitted) == ["k1", "k2"]


def test_the_gpu_loop_records_an_empty_generation_instead_of_dropping_it() -> None:
    jobs = [{"key": "k0", "prompt": "p0"}, {"key": "k1", "prompt": "p1"}]
    emitted: list[dict[str, Any]] = []
    fake = _FakeSummarizer(empty_for={"p1"})
    VM.summarize_jobs(jobs, set(), fake, 8, emitted.append)
    by_key = {r["key"]: r for r in emitted}
    assert (by_key["k1"]["summary"], by_key["k1"]["tokens"]) == ("", 0)
    assert by_key["k0"]["summary"] == "summary of p0"


def _stub_summarizer_module(monkeypatch: pytest.MonkeyPatch, *, loads: bool) -> None:
    import sys
    import types

    class StubSummarizer(_FakeSummarizer):
        def __init__(self, device: str = "auto") -> None:
            super().__init__()
            self._device = device

        async def _ensure_model_loaded(self) -> bool:
            return loads

    stub = types.ModuleType("deep_context.ingestion.summarizer")
    stub.ChunkSummarizer = StubSummarizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deep_context.ingestion.summarizer", stub)


def test_the_gpu_script_writes_results_and_a_second_run_adds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_summarizer_module(monkeypatch, loads=True)
    jobs, results = tmp_path / "jobs.jsonl", tmp_path / "results.jsonl"
    jobs.write_text(
        "".join(json.dumps({"key": f"k{i}", "prompt": f"p{i}"}) + "\n" for i in range(3))
    )
    assert VM.main([str(jobs), str(results), "--batch-size", "2"]) == 0
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert sorted(r["key"] for r in rows) == ["k0", "k1", "k2"]
    assert VM.main([str(jobs), str(results)]) == 0
    assert len(results.read_text().splitlines()) == 3


def test_the_gpu_script_stops_with_an_error_when_the_model_does_not_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_summarizer_module(monkeypatch, loads=False)
    jobs, results = tmp_path / "jobs.jsonl", tmp_path / "results.jsonl"
    jobs.write_text(json.dumps({"key": "k0", "prompt": "p0"}) + "\n")
    assert VM.main([str(jobs), str(results)]) == 2
    assert not results.exists()


# --------------------------------------------------------------------------- index (live PostgreSQL)
async def _deep_adapter(monkeypatch: pytest.MonkeyPatch, cache: MS.SummaryCache | None):  # type: ignore[no-untyped-def]
    from eval.adapter import DeepContextAdapter

    if not await pg.server_available():
        pytest.skip("PostgreSQL is not reachable")
    monkeypatch.setenv("EVAL_INGEST", "deep")
    adapter = DeepContextAdapter(f"test_deep_{uuid.uuid4().hex[:8]}", config="rrf", summaries=cache)
    seen: list[str] = []

    async def fake_embed_many(texts: list[str], *, model: str, dim: int) -> np.ndarray:
        seen.extend(texts)
        v = np.random.default_rng(0).normal(size=(len(texts), dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(adapter.runtime, "embed_many", fake_embed_many)
    return adapter, seen


async def test_deep_index_stores_the_summary_and_embeds_summary_plus_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = {"d1": {"title": "Doc one", "text": TEXT}}
    summarizer = ChunkSummarizer()
    staged = ParentChildChunker()
    cache = MS.SummaryCache(MemoryKV())
    adapter, seen = await _deep_adapter(monkeypatch, cache)
    doc, parents, children = adapter.stage_document(staged, "d1", corpus["d1"])  # type: ignore[misc]
    cache.put_many(
        {
            r.key: MS.SummaryRecord(f"Summary {r.key[:6]}.", 9, summarizer.model_name)
            for r in MS.build_requests(summarizer, children, {p.id: p for p in parents}, doc.title)
        }
    )
    try:
        stats = await adapter.index(corpus)
        assert stats.n_children == len(children)
        async with adapter.store._get_pool().acquire() as c:
            rows = await c.fetch(
                "SELECT summary_text, summary_model FROM chunks WHERE level = 'child'"
            )
            parent_rows = await c.fetch("SELECT summary_text FROM chunks WHERE level = 'parent'")
        assert len(rows) == len(children)
        assert all(r["summary_text"].startswith("Summary ") for r in rows)
        assert {r["summary_model"] for r in rows} == {summarizer.model_name}
        assert all(r["summary_text"] is None for r in parent_rows)
        assert len(seen) == len(children)
        assert all(t.startswith("title: Doc one | text: Summary ") and "\n\n" in t for t in seen)
        assert (await adapter._read_meta())["ingest"] == "deep"  # type: ignore[index]
        adapter.ingest = "fast"
        with pytest.raises(RuntimeError, match="ingest"):
            await adapter.index(corpus)
    finally:
        await adapter.close()
        await pg.drop_schema(adapter.schema)


async def test_deep_index_stops_before_embedding_or_storing_when_a_summary_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = {"d1": {"title": "Doc one", "text": TEXT}}
    adapter, seen = await _deep_adapter(monkeypatch, MS.SummaryCache(MemoryKV()))
    try:
        with pytest.raises(MS.MissingSummariesError):
            await adapter.index(corpus)
        assert seen == []
        async with adapter.store._get_pool().acquire() as c:
            assert await c.fetchval("SELECT count(*) FROM chunks") == 0
    finally:
        await adapter.close()
        await pg.drop_schema(adapter.schema)


def test_an_unknown_ingest_mode_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval.adapter import DeepContextAdapter

    monkeypatch.setenv("EVAL_INGEST", "deeper")
    with pytest.raises(ValueError, match="EVAL_INGEST"):
        DeepContextAdapter("test_x", config="rrf")
