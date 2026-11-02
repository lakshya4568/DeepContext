"""Variant B for one dataset: B0-B5 over the chunks of an existing fast-ingestion index, then B6.

The chunks are NOT re-made: they are read from the dataset's fast-ingestion schema (same ids, same text), so the
only difference between the fast index and the variant B index is the text each child vector embeds
(``summary + "\\n\\n" + child`` instead of the child alone). B6 writes a derived schema ``<fast schema>_vb``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import asyncpg

from eval import config as C
from eval import pg
from eval.variant_b import glossary as G
from eval.variant_b import summarize as Z
from eval.variant_b import synopsis as S
from eval.variant_b.llm import SlmClient
from eval.variant_b.profiles import (
    SHORT_DOC_DATASETS,
    Profile,
    group_and_order,
    profile_for_dataset,
)

log = logging.getLogger("eval.variant_b.pipeline")

SPAN_CHARS_STATIC = 8000
SPAN_CHARS_CAUSAL = (
    3000  # small passages keep a causal chunk from seeing events that come later in its own passage
)
SUMMARISED_OK = ("ok", "retried")  # fallback summaries are kept in the table but not embedded
ARTIFACT_FILES = (
    ("glossary", "glossary.json"),
    ("synopsis", "synopsis.json"),
    ("summary_report", "report.json"),
)


def find_artifacts(summaries_path: Path) -> dict[str, Path]:
    """The JSON files that belong to a summaries file.

    A mode run (``<dataset>/modes/<mode>/summaries.jsonl``) has its own report but shares the dataset's glossary and
    synopsis, which sit two levels up.
    """
    here = summaries_path.parent
    roots = [here, here.parent.parent] if here.parent.name == "modes" else [here]
    found: dict[str, Path] = {}
    for kind, name in ARTIFACT_FILES:
        for root in roots:
            if (root / name).exists():
                found[kind] = root / name
                break
    return found


async def store_artifacts(conn: asyncpg.Connection, summaries_path: Path) -> list[str]:
    """Keep the run's glossary, synopsis and summary report in the index's own schema (the connection's search path)."""
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS vb_artifacts (kind TEXT PRIMARY KEY, data JSONB NOT NULL, "
        "source TEXT NOT NULL, stored_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    stored = []
    for kind, path in find_artifacts(summaries_path).items():
        try:
            source = str(path.relative_to(C.REPO_ROOT))
        except ValueError:
            source = str(path)
        await conn.execute(
            "INSERT INTO vb_artifacts (kind, data, source) VALUES ($1, $2::jsonb, $3) ON CONFLICT (kind) "
            "DO UPDATE SET data = EXCLUDED.data, source = EXCLUDED.source, stored_at = now()",
            kind,
            path.read_text(),
            source,
        )
        stored.append(kind)
    return stored


# ---------------------------------------------------------------------------------------- data classes
@dataclass
class Span:
    doc: str  # external document id
    start: int  # character offset inside the document text
    pos: int  # position inside the source document (0-based, reading order)
    text: str


@dataclass
class Group:
    """One source document: a docs file, a whole PDF, or a single passage/article."""

    key: str
    doc_ids: list[str]  # external ids in reading order
    contents: dict[str, str]
    spans: list[Span]

    def spans_of(self, doc: str) -> list[Span]:
        return [s for s in self.spans if s.doc == doc]


@dataclass
class Child:
    id: str
    ext_doc_id: str
    ordinal: int
    text: str
    parent_text: str
    title: str
    group: str
    pos: int


@dataclass
class GroupContext:
    entries: list[G.Entry]
    synopsis: str
    notes: list[str]
    stats: dict[str, int]


@dataclass
class ChildResult:
    id: str
    ext_doc_id: str
    ordinal: int
    group: str
    pos: int
    summary: str
    status: str
    unsupported: list[str]
    words: int
    prompt_tokens: int
    completion_tokens: int
    calls: int
    reasons_first_try: list[str]
    glossary_entries: int
    had_parent_context: bool


def _check_ident(name: str) -> str:
    if not pg._IDENT.match(name):
        raise ValueError(f"unsafe schema name {name!r}")
    return name


def doc_content(d: dict[str, str]) -> str:
    """Same text the fast-ingestion adapter chunks (title + text unless the title is kept out of the content)."""
    title = (d.get("title") or "").strip()
    text = (d.get("text") or "").strip()
    return f"{title}\n\n{text}" if title and d.get("title_in_content") != "0" else text


# ---------------------------------------------------------------------------------------- planning
def build_groups(
    dataset: str, corpus: dict[str, dict[str, str]], profile: Profile
) -> dict[str, Group]:
    """Group corpus documents into source documents and cut each into passages (spans)."""
    span_chars = SPAN_CHARS_CAUSAL if profile.mode == "causal" else SPAN_CHARS_STATIC
    members: dict[str, list[tuple[int, str]]] = {}
    for ext_id in corpus:
        key, order = group_and_order(dataset, ext_id)
        members.setdefault(key, []).append((order, ext_id))
    groups: dict[str, Group] = {}
    for key, items in members.items():
        ordered = [ext for _, ext in sorted(items)]
        contents = {ext: doc_content(corpus[ext]) for ext in ordered}
        spans: list[Span] = []
        for ext in ordered:
            for start, piece in G.split_spans(contents[ext], span_chars):
                spans.append(Span(ext, start, len(spans), piece))
        groups[key] = Group(key, ordered, contents, spans)
    return groups


def span_position(group: Group, doc: str, child_text: str) -> int:
    """Position of the passage that contains the chunk (0 if the doc has no passage)."""
    spans = group.spans_of(doc)
    if not spans:
        return 0
    content = group.contents[doc]
    probe = child_text.strip()[:80]
    idx = content.find(probe)
    if idx == -1:
        half = len(child_text) // 2
        j = content.find(child_text[half : half + 60])
        idx = max(0, j - half) if j != -1 else 0
    chosen = spans[0]
    for s in spans:
        if s.start <= idx:
            chosen = s
    return chosen.pos


def child_ordinals(ext_doc_id: str, child_ids: list[str], ns: uuid.UUID) -> dict[str, int]:
    """The fast adapter names child n of a document ``uuid5(ns, '<ext>|child|n')``; recover n from the ids."""
    by_id = {str(uuid.uuid5(ns, f"{ext_doc_id}|child|{n}")): n for n in range(len(child_ids) + 4)}
    missing = [c for c in child_ids if c not in by_id]
    if missing:
        raise RuntimeError(
            f"{ext_doc_id}: {len(missing)} child ids do not follow the adapter's naming"
        )
    return {c: by_id[c] for c in child_ids}


async def load_children(source_schema: str, groups: dict[str, Group], dataset: str) -> list[Child]:
    from eval.adapter import _NS

    schema = _check_ident(source_schema)
    conn = await asyncpg.connect(pg.dsn_for(pg.EVAL_DB))
    try:
        docs = await conn.fetch(f'SELECT id::text AS id, title, metadata FROM "{schema}".documents')
        kids = await conn.fetch(
            f"SELECT c.id::text AS id, c.document_id::text AS document_id, c.content, "
            f'p.content AS parent_content FROM "{schema}".chunks c '
            f"LEFT JOIN \"{schema}\".chunks p ON p.id = c.parent_chunk_id WHERE c.level = 'child'"
        )
    finally:
        await conn.close()
    ext_by_store: dict[str, str] = {}
    title_by_ext: dict[str, str] = {}
    for d in docs:
        meta = (
            json.loads(d["metadata"])
            if isinstance(d["metadata"], str)
            else dict(d["metadata"] or {})
        )
        ext = meta["external_id"]
        ext_by_store[d["id"]] = ext
        title_by_ext[ext] = d["title"]
    key_of = {ext: g.key for g in groups.values() for ext in g.doc_ids}
    per_doc: dict[str, list[asyncpg.Record]] = {}
    for k in kids:
        per_doc.setdefault(ext_by_store[k["document_id"]], []).append(k)
    out: list[Child] = []
    for ext, rows in per_doc.items():
        ords = child_ordinals(ext, [r["id"] for r in rows], _NS)
        group = groups[key_of[ext]]
        for r in rows:
            out.append(
                Child(
                    id=r["id"],
                    ext_doc_id=ext,
                    ordinal=ords[r["id"]],
                    text=r["content"],
                    parent_text=r["parent_content"] or "",
                    title=title_by_ext.get(ext, ext),
                    group=group.key,
                    pos=span_position(group, ext, r["content"]),
                )
            )
    return out


# ---------------------------------------------------------------------------------------- B1-B3 per group
async def build_group_context(
    llm: SlmClient, profile: Profile, group: Group, *, short: bool, need_synopsis: bool = True
) -> GroupContext:
    stats: Counter[str] = Counter()
    results = await asyncio.gather(
        *(G.extract_span(llm, profile, s.text, s.pos) for s in group.spans)
    )
    raw: list[G.Entry] = []
    for entries, st in results:
        raw.extend(entries)
        stats.update(st)
    entries = G.merge_entries(raw)
    stats["entries_after_merge"] = len(entries)
    synopsis, notes = "", []
    if need_synopsis and not short and group.spans:
        if profile.mode == "causal":
            notes = await S.story_notes(llm, [s.text for s in group.spans])
        else:
            synopsis = await S.static_synopsis(
                llm, "\n\n".join(group.contents[d] for d in group.doc_ids)
            )
    return GroupContext(entries, synopsis, notes, dict(stats))


# ---------------------------------------------------------------------------------------- B4-B5 per group
def child_context(
    profile: Profile, gctx: GroupContext, child: Child, previous: list[str]
) -> tuple[Z.Context, int]:
    causal = profile.mode == "causal"
    window = Z.parent_window(child.parent_text, child.text)
    near = child.text + ("\n" + window if window != "none" else "")
    picked = G.select_slice(gctx.entries, near, child.pos, causal=causal)
    glossary = G.render_slice(picked, child.pos, causal=causal)
    synopsis = (
        gctx.notes[child.pos] if causal and child.pos < len(gctx.notes) else gctx.synopsis
    ) or "none"
    ctx = Z.Context(
        title=child.title,
        profile=profile,
        text=child.text,
        synopsis=synopsis,
        glossary=glossary,
        previous=tuple(previous[-2:]),
        parent_context=window,
        glossary_names=tuple(n for e in picked for n in e.names),
    )
    return ctx, len(picked)


def _to_result(child: Child, res: Z.Result, n_entries: int, had_parent: bool) -> ChildResult:
    return ChildResult(
        id=child.id, ext_doc_id=child.ext_doc_id, ordinal=child.ordinal, group=child.group, pos=child.pos,
        summary=res.summary, status=res.status, unsupported=res.check.unsupported, words=res.check.words,
        prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, calls=res.calls,
        reasons_first_try=res.reasons_first_try, glossary_entries=n_entries, had_parent_context=had_parent,
    )  # fmt: skip


async def summarise_group(
    llm: SlmClient,
    profile: Profile,
    gctx: GroupContext,
    children: list[Child],
    *,
    use_previous: bool | None = None,
    mode: Z.Mode = Z.SUMMARY_V1,
) -> list[ChildResult]:
    chain = mode.uses_previous if use_previous is None else use_previous
    ordered = sorted(children, key=lambda c: (c.pos, c.ordinal))
    results: list[ChildResult] = []
    if chain and len(ordered) > 1:  # each chunk sees the summaries of the two before it: sequential
        previous: list[str] = []
        for child in ordered:
            ctx, n = child_context(profile, gctx, child, previous)
            res = await Z.summarise(llm, ctx, mode)
            results.append(_to_result(child, res, n, ctx.parent_context != "none"))
            if res.status in SUMMARISED_OK:
                previous.append(res.summary)
        return results

    async def one(child: Child) -> ChildResult:
        ctx, n = child_context(profile, gctx, child, [])
        res = await Z.summarise(llm, ctx, mode)
        return _to_result(child, res, n, ctx.parent_context != "none")

    return list(await asyncio.gather(*(one(c) for c in ordered)))


# ---------------------------------------------------------------------------------------- run B0-B5
async def run_summaries(
    dataset: str,
    source_schema: str,
    corpus: dict[str, dict[str, str]],
    llm: SlmClient,
    out_dir: Path,
    *,
    mode: Z.Mode = Z.SUMMARY_V1,
    use_previous: bool | None = None,
    limit_groups: int | None = None,
    group_concurrency: int = 12,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    profile = profile_for_dataset(dataset)
    short = dataset in SHORT_DOC_DATASETS
    groups = build_groups(dataset, corpus, profile)
    children = await load_children(source_schema, groups, dataset)
    by_group: dict[str, list[Child]] = {}
    for c in children:
        by_group.setdefault(c.group, []).append(c)
    keys = sorted(by_group)
    if limit_groups is not None:
        keys = keys[:limit_groups]
    log.info("%s: %d groups, %d children, profile %s (%s)", dataset, len(keys), sum(len(by_group[k]) for k in keys), profile.name, profile.mode)  # fmt: skip

    sem = asyncio.Semaphore(group_concurrency)
    all_results: list[ChildResult] = []
    contexts: dict[str, GroupContext] = {}
    done = 0

    async def work(key: str) -> None:
        nonlocal done
        async with sem:
            gctx = await build_group_context(
                llm, profile, groups[key], short=short, need_synopsis="{synopsis}" in mode.user
            )
            contexts[key] = gctx
            res = await summarise_group(
                llm, profile, gctx, by_group[key], use_previous=use_previous, mode=mode
            )
            all_results.extend(res)
            done += 1
            log.info("  group %d/%d done: %s (%d children, %d glossary entries)", done, len(keys), key, len(res), len(gctx.entries))  # fmt: skip

    await asyncio.gather(*(work(k) for k in keys))
    llm.flush()

    out_dir.mkdir(parents=True, exist_ok=True)
    all_results.sort(key=lambda r: (r.group, r.pos, r.ordinal))
    (out_dir / "summaries.jsonl").write_text(
        "\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in all_results) + "\n"
    )
    (out_dir / "glossary.json").write_text(
        json.dumps(
            {k: [e.to_json() for e in v.entries] for k, v in contexts.items()}, ensure_ascii=False
        )
    )
    (out_dir / "synopsis.json").write_text(
        json.dumps({k: {"synopsis": v.synopsis, "notes": v.notes} for k, v in contexts.items()}, ensure_ascii=False)
    )  # fmt: skip
    report = summarise_report(
        dataset, source_schema, profile, llm, all_results, contexts, time.perf_counter() - t0
    )
    report["enrichment_mode"] = mode.name
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def reason_kind(reason: str) -> str:
    """Bucket a rejection reason from ``summarize.check_summary`` for the report."""
    if reason.startswith("not in the source"):
        return "terms_not_in_source"
    if "bullets" in reason or "lines" in reason or "sentences" in reason:
        return "format"
    if "leaves out" in reason:
        return "meta_comment"
    if reason.endswith("words"):
        return "too_long"
    return reason


def summarise_report(
    dataset: str,
    source_schema: str,
    profile: Profile,
    llm: SlmClient,
    results: list[ChildResult],
    contexts: dict[str, GroupContext],
    wall_s: float,
) -> dict[str, Any]:
    n = max(1, len(results))
    stats: Counter[str] = Counter()
    for c in contexts.values():
        stats.update(c.stats)
    raw = stats.get("raw", 0)
    first_try_failures = Counter(reason_kind(x) for res in results for x in res.reasons_first_try)
    return {
        "dataset": dataset,
        "source_schema": source_schema,
        "profile": profile.name,
        "mode": profile.mode,
        "summariser_model": llm.model,
        "children": len(results),
        "groups": len(contexts),
        "status": dict(Counter(r.status for r in results)),
        "first_try_ok_rate": round(sum(not r.reasons_first_try for r in results) / n, 3),
        "fallback_rate": round(sum(r.status == "fallback" for r in results) / n, 3),
        "first_try_failure_kinds": dict(first_try_failures),
        "mean_words": round(sum(r.words for r in results) / n, 1),
        "children_with_glossary_entries": sum(r.glossary_entries > 0 for r in results),
        "children_with_parent_context": sum(r.had_parent_context for r in results),
        "glossary": {
            "raw_entries": raw,
            "kept": stats.get("kept", 0),
            "kept_rate": round(stats.get("kept", 0) / raw, 3) if raw else None,
            "dropped_quote_not_in_text": stats.get("dropped_quote", 0),
            "dropped_name_not_in_text": stats.get("dropped_name", 0),
            "dropped_empty": stats.get("dropped_empty", 0),
            "json_parse_errors": stats.get("parse_error", 0),
            "quote_exact": stats.get("support_exact", 0),
            "quote_fuzzy": stats.get("support_fuzzy", 0),
            "entries_after_merge": stats.get("entries_after_merge", 0),
        },
        "tokens": {
            "prompt": sum(r.prompt_tokens for r in results),
            "completion": sum(r.completion_tokens for r in results),
            "summary_calls": sum(r.calls for r in results),
        },
        "llm_calls_made": llm.calls,
        "llm_cache_hits": llm.cache_hits,
        "wall_s": round(wall_s, 1),
    }


# ---------------------------------------------------------------------------------------- B6 index
def embed_text(
    summary: str | None,
    status: str,
    content: str,
    *,
    recipe: str = "prefix",
    min_words: int = 0,
) -> str:
    """What a child's vector embeds: ``generated text + blank line + child`` (``recipe="suffix"``: child first).

    The child alone is embedded when there is no usable generated text or the chunk has fewer than ``min_words`` words.
    """
    if not summary or status not in SUMMARISED_OK or len(content.split()) < min_words:
        return content
    return f"{content}\n\n{summary}" if recipe == "suffix" else f"{summary}\n\n{content}"


async def build_index(
    dataset: str,
    source_schema: str,
    summaries_path: Path,
    *,
    suffix: str = "_vb",
    recipe: str = "prefix",
    min_words: int = 0,
    window: int = 1000,
) -> dict[str, Any]:
    """Create ``<source_schema><suffix>``: copy documents and chunks, re-embed children, store summaries and metadata."""
    import numpy as np

    from eval.strict_clients import install_strict_patches, stage_scope

    t0 = time.perf_counter()
    src = _check_ident(source_schema)
    dst = _check_ident(f"{source_schema}{suffix}")
    rows = [json.loads(ln) for ln in summaries_path.read_text().splitlines() if ln.strip()]
    summ = {r["id"]: r for r in rows}

    await pg.ensure_database()
    await pg.ensure_schema(dst)
    store = pg.make_store_class()(pg.dsn_for(pg.EVAL_DB, dst))
    await store.initialize()
    runtime = install_strict_patches()
    try:
        pool = store._get_pool()
        conn = await asyncpg.connect(pg.dsn_for(pg.EVAL_DB))
        try:
            async with conn.transaction():
                await conn.execute(
                    f'INSERT INTO "{dst}".documents SELECT * FROM "{src}".documents ON CONFLICT DO NOTHING'
                )
                for level in ("parent", "child"):  # parents first: children reference them
                    await conn.execute(
                        f'INSERT INTO "{dst}".chunks (id, document_id, parent_chunk_id, level, content, token_count, '
                        f"section_path, page_number, created_at) SELECT id, document_id, parent_chunk_id, level, content, "
                        f'token_count, section_path, page_number, created_at FROM "{src}".chunks WHERE level = $1 '
                        f"ON CONFLICT DO NOTHING",
                        level,
                    )
            kids = await conn.fetch(
                f'SELECT c.id::text AS id, c.content, d.title, d.metadata FROM "{dst}".chunks c '
                f'JOIN "{dst}".documents d ON d.id = c.document_id '
                f"WHERE c.level = 'child' AND c.embedding IS NULL"
            )
        finally:
            await conn.close()
        missing = [k["id"] for k in kids if k["id"] not in summ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} child chunks have no row in {summaries_path}; run the summarise step first"
            )

        n_embedded = 0
        for w0 in range(0, len(kids), window):
            batch = kids[w0 : w0 + window]
            texts = []
            for k in batch:
                meta = (
                    json.loads(k["metadata"])
                    if isinstance(k["metadata"], str)
                    else dict(k["metadata"] or {})
                )
                title = k["title"] if k["title"] != meta.get("external_id") else None
                s = summ[k["id"]]
                texts.append(
                    runtime.format_embedding_text(
                        embed_text(
                            s["summary"],
                            s["status"],
                            k["content"],
                            recipe=recipe,
                            min_words=min_words,
                        ),
                        is_query=False,
                        title=title,
                        task=None,
                    )
                )
            with stage_scope("index"):
                vecs = await runtime.embed_many(texts, model=C.EMBEDDING_MODEL, dim=C.EMBEDDING_DIM)
            async with pool.acquire() as c:
                await c.executemany(
                    "UPDATE chunks SET embedding = $2 WHERE id = $1::uuid",
                    [
                        (k["id"], np.asarray(v, dtype=np.float32))
                        for k, v in zip(batch, vecs, strict=True)
                    ],
                )
            runtime.cache.flush()
            n_embedded += len(batch)
            log.info("  embedded %d/%d child chunks", n_embedded, len(kids))

        async with pool.acquire() as c:
            await c.execute(
                "CREATE TABLE IF NOT EXISTS chunk_summaries (chunk_id UUID PRIMARY KEY, summary_text TEXT NOT NULL, "
                "status TEXT NOT NULL, meta JSONB NOT NULL DEFAULT '{}')"
            )
            await c.executemany(
                "INSERT INTO chunk_summaries (chunk_id, summary_text, status, meta) VALUES ($1::uuid, $2, $3, $4::jsonb) "
                "ON CONFLICT (chunk_id) DO UPDATE SET summary_text = EXCLUDED.summary_text, status = EXCLUDED.status, "
                "meta = EXCLUDED.meta",
                [
                    (
                        r["id"],
                        r["summary"],
                        r["status"],
                        json.dumps({"words": r["words"], "group": r["group"], "pos": r["pos"]}),
                    )
                    for r in rows
                ],
            )
            stored = await store_artifacts(c, summaries_path)
            await c.execute("ANALYZE documents")
            await c.execute("ANALYZE chunks")
            await c.execute(
                "CREATE TABLE IF NOT EXISTS eval_index_meta (k TEXT PRIMARY KEY, v JSONB NOT NULL)"
            )
            src_meta = await _source_meta(src)
            meta = {
                **src_meta,
                "variant": "B",
                "recipe": recipe,
                "min_words": min_words,
                "source_schema": src,
                "summaries": len(rows),
                "summaries_embedded": sum(r["status"] in SUMMARISED_OK for r in rows),
                "embeds": "summary + child (child alone when the summary fell back)",
                "artifacts_in_postgres": stored,
            }
            await c.execute(
                "INSERT INTO eval_index_meta (k, v) VALUES ('meta', $1::jsonb) "
                "ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v",
                json.dumps(meta),
            )
    finally:
        await store.close()
        runtime.cache.flush()
    return {
        "schema": dst,
        "children_embedded": n_embedded,
        "summaries": len(rows),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


async def add_lexical_context(schema: str) -> int:
    """Contextual BM25: make the keyword-search column cover the generated text as well as the chunk.

    Only the generated ``tsv`` column changes (``content || ' ' || generated text``); ``content`` itself, and so everything
    a retriever returns or a reranker reads, stays the original chunk text. Returns how many chunks got extra text.
    """
    s = _check_ident(schema)
    conn = await asyncpg.connect(pg.dsn_for(pg.EVAL_DB))
    try:
        async with conn.transaction():
            await conn.execute(f'ALTER TABLE "{s}".chunks ADD COLUMN IF NOT EXISTS ctx_text TEXT')
            await conn.execute(
                f'UPDATE "{s}".chunks c SET ctx_text = cs.summary_text FROM "{s}".chunk_summaries cs '
                f"WHERE cs.chunk_id = c.id AND cs.status = ANY($1::text[])",
                list(SUMMARISED_OK),
            )
            await conn.execute(f'ALTER TABLE "{s}".chunks DROP COLUMN tsv')
            await conn.execute(
                f'ALTER TABLE "{s}".chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS '
                f"(to_tsvector('english', content || ' ' || coalesce(ctx_text, ''))) STORED"
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON "{s}".chunks USING GIN (tsv)'
            )
            await conn.execute(
                f"UPDATE \"{s}\".eval_index_meta SET v = v || '{{\"lexical_context\": true}}'::jsonb WHERE k = 'meta'"
            )
        n = await conn.fetchval(f'SELECT count(*) FROM "{s}".chunks WHERE ctx_text IS NOT NULL')
        await conn.execute(f'ANALYZE "{s}".chunks')
    finally:
        await conn.close()
    return int(n)


async def _source_meta(schema: str) -> dict[str, Any]:
    conn = await asyncpg.connect(pg.dsn_for(pg.EVAL_DB))
    try:
        row = await conn.fetchrow(
            f"SELECT v FROM \"{_check_ident(schema)}\".eval_index_meta WHERE k = 'meta'"
        )
    finally:
        await conn.close()
    if row is None:
        raise RuntimeError(
            f"{schema} has no eval_index_meta row (is it a finished fast-ingestion index?)"
        )
    return json.loads(row["v"]) if isinstance(row["v"], str) else dict(row["v"])
