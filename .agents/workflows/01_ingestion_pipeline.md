---
description: Ingestion Pipeline
---

# Workflow: Ingestion Pipeline

Turns a raw document (PDF, Markdown, code repo, HTML) into searchable chunks in `documents`/`chunks`
(`docs/DATA_MODEL.sql`). Implements FR2, FR5, FR6.

## Trigger

- A new document is added (upload, repo sync, scheduled crawl).
- An existing document changes (re-ingestion replaces its chunks; parent `document_id` stays stable so
  memory/citations referencing it don't dangle).

## Steps

```text
1. Detect format → choose parser
     PDF        → structure-aware PDF parser (preserve headings, page numbers, tables)
     Markdown   → heading-based structure parser
     Code       → AST-aware parser (function/class/module boundaries)
     HTML       → DOM-based structure parser (strip nav/boilerplate)

2. Extract structure
     → sections, headings, page numbers, table boundaries, code symbols
     → NEVER split a table away from its header/caption
     → NEVER split a function body from its signature

3. Decide retrieval_mode for this document (FR6)
     structured, long, hierarchy-meaningful (contracts, filings, manuals)
         → retrieval_mode = 'vectorless'; also build document_tree_nodes
     everything else (default)
         → retrieval_mode = 'hybrid'

4. Chunk (parent-child, structure-aware — FR2)
     parent: 1,000–2,500 tokens   (sent to the model at generation time)
     child:  300–600 tokens       (what's actually searched)
     overlap: 10–15%
     — for fact-heavy docs, bias smaller: child 200–400 / parent 800–1,500
     — for narrative/explanatory docs, bias larger: child 500–800 / parent 1,500–3,000
     — for code: one chunk per function/class; never split mid-function;
       attach file path, imports, and enclosing class/module as metadata

5. Attach metadata to every chunk
     document_id, section_path, page_number, permission_scope (inherited
     from the parent document), source timestamp

6. Embed child chunks
     → one embedding call per child chunk (batch where the API allows)
     → parent chunks are NOT embedded by default (they're never searched
       directly) — only embed a parent if you're doing late-chunking (below)

7. (Optional, per-document) Contextual retrieval prefix
     Before embedding, prepend a short generated one-liner locating the
     chunk: "Document: X. Section: Y. This chunk covers Z." This measurably
     helps retrieval on ambiguous chunks but costs one extra LLM call per
     chunk at ingestion time — enable it selectively for high-value or
     easily-confused corpora, not as a blanket default (cost scales with
     corpus size).

8. Write to Postgres
     INSERT INTO documents (...)
     INSERT INTO chunks (...) — parents first, then children with
     parent_chunk_id set
     (vectorless docs also populate document_tree_nodes, one row per
     section, with an LLM-generated `summary` per node used later for
     tree navigation — see 02_retrieval_pipeline.md)

9. Index
     — tsvector is a GENERATED column (auto-updates on insert)
     — vector index (ivfflat/hnsw) — see TECH_STACK.md §3 for index choice
       at your scale
```

## Alternative: late chunking

If you're working with an embedding model that supports a large context window, you can invert steps 4 and 6:
embed the **full document** first (or large windows of it), then derive per-chunk vectors from that
already-contextualized embedding rather than embedding isolated chunks. This preserves document-level context in
every chunk's vector, at the cost of requiring an embedding model that can actually take the whole document as
input. Treat this as a per-corpus decision, not a global default — most corpora are well served by standard
parent-child chunking (step 4) with contextual prefixes (step 7) for the chunks that need it.

## Failure modes to design for

- **Parser produces garbage on a malformed PDF** → fall back to a simpler extraction (page-level text dump) rather
  than failing the whole ingestion; flag the document as `metadata.parse_quality = 'degraded'` so retrieval/UI can
  warn rather than silently serve bad chunks.
- **A chunk exceeds the child token budget** (e.g. one giant unsplittable table) → allow an oversized child chunk
  rather than cutting a table in half; the token-budget numbers in step 4 are targets, not hard limits.
- **Re-ingestion race**: two ingestion runs for the same `document_id` overlapping → ingestion should run inside a
  transaction that deletes old chunks and inserts new ones atomically, not delete-then-insert as two statements.
