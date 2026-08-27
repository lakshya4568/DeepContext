# Production Database Indexing Guide
## Optimized PostgreSQL Indexes for pgvector & Hybrid Retrieval

This document defines the production database indexing architecture for the Deep Context Platform, optimized for high-throughput, low-latency hybrid retrieval (Dense Vector + BM25 Full-Text + Reciprocal Rank Fusion) using PostgreSQL 15+ and pgvector.

---

## 1. Index Architecture Overview

The `chunks` table utilizes four complementary indexes designed to maximize search speed and accuracy across both stages of hybrid retrieval:

```
                                  ┌──────────────────────────────┐
                                  │         chunks table         │
                                  └──────────────┬───────────────┘
                                                 │
            ┌──────────────────┬─────────────────┴────────────────┬──────────────────┐
            ▼                  ▼                                  ▼                  ▼
   [idx_chunks_embedding_hnsw] [idx_chunks_search_tsv]   [idx_chunks_parent_null] [idx_chunks_document_id]
   HNSW Cosine Vector Index    GIN Full-Text Index        Partial Index (Parents)  Composite (doc, id)
   (m=16, ef_c=200)            (Weighted B:text, C:summ)  (WHERE parent IS NULL)   (Document scoping)
```

| Index Name | Type / Method | Target Columns | Purpose |
|---|---|---|---|
| `idx_chunks_bm25` | **ParadeDB (Tantivy)** | `content`, `summary_text` | True Okapi BM25 index over chunk content and contextual summaries with non-linear term frequency saturation. |
| `idx_chunks_trgm` | **GIN (`gin_trgm_ops`)** | `content` | Fast substring and trigram similarity search for technical identifiers, product codes, SKUs, and typos. |
| `idx_chunks_embedding_hnsw` | **HNSW** (`vector_cosine_ops`) | `embedding` (768-dim) | High-speed approximate nearest neighbor search ($\approx 10\times$ faster than IVFFlat). |
| `idx_chunks_search_tsv` | **GIN** | `search_tsv` (tsvector) | Full-text FTS search over weighted summary ('A') + raw content ('B'). |
| `idx_chunks_parent_null` | **B-Tree** (Partial) | `(id, document_id)` WHERE `parent_chunk_id IS NULL` | Sub-millisecond parent chunk resolution without table scans over child chunks. |
| `idx_chunks_document_id` | **B-Tree** (Composite) | `(document_id, id)` | Fast filtering and chunk counting scoped by document. |

---

## 2. Automatic Full-Text Vector Generation (`search_tsv`)

A PostgreSQL `BEFORE INSERT OR UPDATE` trigger function (`update_chunks_tsv`) automatically computes the weighted `search_tsv` vector upon insertion:

```sql
CREATE OR REPLACE FUNCTION update_chunks_tsv() RETURNS trigger AS $$
BEGIN
  NEW.search_tsv := 
    setweight(to_tsvector('english', COALESCE(NEW.summary_text, '')), 'A') ||
    setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_chunks_tsv 
BEFORE INSERT OR UPDATE ON chunks
FOR EACH ROW EXECUTE FUNCTION update_chunks_tsv();
```

- **Weight 'A' (Highest rank priority)**: Assigned to the LLM-generated contextual summary (`summary_text`).
- **Weight 'B' (Standard rank priority)**: Assigned to the raw chunk content (`content`).

---

## 3. HNSW Parameter Tuning Guide

### Parameter Roles

1. **`m` (Build-time, Default: `16`)**:
   - Number of bidirectional links per graph node.
   - Higher values increase graph density and recall, at the cost of index build time and memory usage.
   - *Recommendation*: `16` for standard production; `24–32` for strict $>98\%$ recall requirements.

2. **`ef_construction` (Build-time, Default: `200`)**:
   - Size of the dynamic candidate list during graph construction.
   - Higher values produce higher-quality nearest-neighbor graphs.
   - *Recommendation*: `200` for 1024-dimensional vectors.

3. **`ef_search` (Query-time / Runtime, Default: `100`)**:
   - Size of the dynamic candidate list evaluated during similarity search.
   - Can be tuned per-query via `SET LOCAL hnsw.ef_search = <val>`.
   - *Trade-off*: Higher `ef_search` gives higher recall at the cost of higher query latency.

### Recommended Configuration Profiles

| Workload Scale | `m` | `ef_construction` | `ef_search` | Expected Recall@50 | Latency (p99) |
|---|---|---|---|---|---|
| **Small (<100K chunks)** | 16 | 200 | 100 | >95% | <25ms |
| **Medium (100K–1M chunks)** | 24 | 300 | 200 | >97% | <60ms |
| **Large (1M–10M chunks)** | 32 | 400 | 400 | >98% | <95ms |

### Memory Sizing

HNSW graph structures require approximately $2\times\text{--}4\times$ the memory of raw vector floats.
$$\text{Memory Size} \approx N_{\text{chunks}} \times \text{dim} \times 4\text{ bytes} \times 3$$
*Example*: $1,000,000$ chunks $\times 1024\text{ dim} \times 4 \times 3 \approx 12\text{ GB RAM}$.

---

## 4. Monitoring & Diagnostics

### Check Index Usage & Scans
```sql
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename = 'chunks';
```

### Check Index Disk Sizes
```sql
SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE tablename = 'chunks'
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Verify Query Execution Plan
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, document_id, content, 1 - (embedding <=> '[0.01, ...]'::vector) AS score
FROM chunks
WHERE level = 'child'
ORDER BY embedding <=> '[0.01, ...]'::vector
LIMIT 50;
```

---

## 5. Maintenance Operations

1. **Regular Statistics Updates (Weekly)**:
   ```sql
   VACUUM ANALYZE chunks;
   ```
2. **Online Index Rebuild (Zero-Downtime)**:
   ```sql
   REINDEX INDEX CONCURRENTLY idx_chunks_embedding_hnsw;
   REINDEX INDEX CONCURRENTLY idx_chunks_search_tsv;
   ```
