-- Migration: 002_add_optimized_indexes.sql
-- Optimized PostgreSQL indexes for DeepContext Hybrid Retrieval (BM25 + pgvector)

-- 1. Ensure embedding vector column dimension (1024 for BGE-M3 / NVIDIA NIM)
DO $$
BEGIN
    BEGIN
        ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024);
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;
END $$;

-- 2. Add search_tsv column if missing
ALTER TABLE chunks 
ADD COLUMN IF NOT EXISTS search_tsv TSVECTOR;

-- 3. Add created_at column if missing (for time-based queries)
ALTER TABLE chunks 
ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

-- 4. Create/update TSV trigger function (Content weighted B, Summary weighted C)
CREATE OR REPLACE FUNCTION update_chunks_tsv() RETURNS trigger AS $$
BEGIN
  NEW.search_tsv := 
    setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B') ||
    setweight(to_tsvector('english', COALESCE(NEW.summary_text, '')), 'C');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 5. Attach trigger to chunks table
DROP TRIGGER IF EXISTS trigger_update_chunks_tsv ON chunks;
CREATE TRIGGER trigger_update_chunks_tsv 
BEFORE INSERT OR UPDATE ON chunks
FOR EACH ROW EXECUTE FUNCTION update_chunks_tsv();

-- 6. Backfill search_tsv for existing chunks
UPDATE chunks 
SET search_tsv = 
  setweight(to_tsvector('english', COALESCE(content, '')), 'B') ||
  setweight(to_tsvector('english', COALESCE(summary_text, '')), 'C')
WHERE search_tsv IS NULL;

-- 7. Create HNSW index for vector search (optimized for 1024-dim / high-recall workloads)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- 8. Create GIN index for BM25 full-text search over combined content + summary
CREATE INDEX IF NOT EXISTS idx_chunks_search_tsv 
ON chunks USING GIN (search_tsv);

-- 9. Create composite index for document-level filtering
CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
ON chunks (document_id, id);

-- 10. Create partial index for parent chunks (faster parent resolution)
CREATE INDEX IF NOT EXISTS idx_chunks_parent_null 
ON chunks (id, document_id) 
WHERE parent_chunk_id IS NULL;

-- 11. Set runtime parameters for query optimization
DO $$
BEGIN
    BEGIN
        ALTER SYSTEM SET hnsw.ef_search = 100;
        PERFORM pg_reload_conf();
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;
END $$;

-- 12. Analyze tables for query planner
ANALYZE chunks;

-- 13. Add helpful comment documenting index strategy
DO $$
BEGIN
    EXECUTE 'COMMENT ON INDEX idx_chunks_embedding_hnsw IS ''HNSW index for approximate nearest neighbor search. m=16, ef_construction=200. Tune ef_search at runtime for recall/latency trade-off.''';
EXCEPTION WHEN OTHERS THEN
    NULL;
END $$;
