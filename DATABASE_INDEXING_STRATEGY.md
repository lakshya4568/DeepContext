# Production Database Indexing Strategy for RAG Chunks
## Optimized for pgvector + PostgreSQL Hybrid Retrieval

This guide defines the production database indexing architecture for the Deep Context Platform, optimized for high-throughput, low-latency hybrid retrieval (Dense Vector + BM25 Full-Text + Reciprocal Rank Fusion) using PostgreSQL 15+ and pgvector.

---

## 1. Index Design Goals

1. **Fast retrieval** — p99 latency <100ms for top-50 candidates per path
2. **High recall** — >95% recall@50 for hybrid search
3. **Memory efficient** — HNSW index fits in RAM (plan for 2-4x vector size)
4. **Scalable** — supports 100K–10M chunks without degradation
5. **Hybrid-ready** — BM25 + vector indexes work in parallel for RRF

---

## 2. Recommended Index Configuration

```sql
-- 1. Vector column with fixed dimension
ALTER TABLE chunks 
ALTER COLUMN embedding TYPE vector(1024);

-- 2. HNSW index for approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- 3. BM25 full-text search index (using PostgreSQL TSV)
ALTER TABLE chunks 
ADD COLUMN IF NOT EXISTS search_tsv TSVECTOR;

-- Auto-populate TSV from chunk content + summary
CREATE OR REPLACE FUNCTION update_chunks_tsv() RETURNS trigger AS $$
BEGIN
  NEW.search_tsv := 
    setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B') ||
    setweight(to_tsvector('english', COALESCE(NEW.summary_text, '')), 'C');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_chunks_tsv ON chunks;
CREATE TRIGGER trigger_update_chunks_tsv 
BEFORE INSERT OR UPDATE ON chunks
FOR EACH ROW EXECUTE FUNCTION update_chunks_tsv();

-- 4. GIN index for BM25/TSV queries
CREATE INDEX IF NOT EXISTS idx_chunks_search_tsv 
ON chunks USING GIN (search_tsv);

-- 5. Composite index for document-level filtering
CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
ON chunks (document_id, id);

-- 6. Partial index for parent chunks only (faster parent resolution)
CREATE INDEX IF NOT EXISTS idx_chunks_parent_null 
ON chunks (id, document_id) 
WHERE parent_chunk_id IS NULL;
```

---

## 3. HNSW Parameter Tuning

| Parameter | Default | Recommended | Effect |
|---|---|---|---|
| `m` | 16 | **16–32** | Connections per node. Higher = better recall, more memory |
| `ef_construction` | 64 | **200–400** | Candidates during build. Higher = slower build, better quality |
| `ef_search` (runtime) | 40 | **100–400** | Search width. Higher = better recall, slower queries |

### Scale Recommendations
- **Small (<100K chunks)**: `m=16, ef_c=200, ef_s=100`
- **Medium (100K–1M chunks)**: `m=24, ef_c=300, ef_s=200`
- **Large (1M–10M chunks)**: `m=32, ef_c=400, ef_s=400`

---

## 4. Monitoring & Diagnostics

```sql
-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename = 'chunks';

-- Check index sizes
SELECT indexname, pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE tablename = 'chunks'
ORDER BY pg_relation_size(indexrelid) DESC;
```
