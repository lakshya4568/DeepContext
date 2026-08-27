-- Migration: 003_pg_bm25_and_trigram_indexes.sql
-- Idempotent schema synchronization, pg_trgm trigram index, and ParadeDB pg_search setup.

-- 1. Ensure required columns exist on documents and chunks
ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER NOT NULL DEFAULT 0;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_path TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_number INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_text TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_tokens INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_model TEXT DEFAULT 'qwen3-0.6b';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS generated_at TIMESTAMPTZ;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_tsv TSVECTOR;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_tsv TSVECTOR;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

-- 2. Try enabling pg_trgm for exact code tokens, SKUs, and hyphenated identifiers
DO $$
BEGIN
    BEGIN
        CREATE EXTENSION IF NOT EXISTS pg_trgm;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'pg_trgm extension could not be created: %', SQLERRM;
    END;
END $$;

-- 3. Create GIN trigram index on chunks content if pg_trgm is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_chunks_trgm ON chunks USING GIN (content gin_trgm_ops)';
        RAISE NOTICE 'Created idx_chunks_trgm index successfully.';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not create idx_chunks_trgm: %', SQLERRM;
END $$;

-- 4. Try enabling pg_search (ParadeDB BM25) if installed
DO $$
BEGIN
    BEGIN
        CREATE EXTENSION IF NOT EXISTS pg_search;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'pg_search extension not available in environment: %', SQLERRM;
    END;
END $$;

-- 5. Create ParadeDB BM25 index on chunks if pg_search is active
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search') THEN
        BEGIN
            CALL paradedb.create_bm25(
                index_name => 'idx_chunks_bm25',
                table_name => 'chunks',
                key_field => 'id',
                text_fields => '{content: {}, summary_text: {}}'
            );
            RAISE NOTICE 'Created ParadeDB BM25 index idx_chunks_bm25.';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Notice creating ParadeDB BM25 index: %', SQLERRM;
        END;
    END IF;
END $$;

ANALYZE chunks;
