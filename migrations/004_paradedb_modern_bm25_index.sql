-- Migration: 004_paradedb_modern_bm25_index.sql
-- Replaces the BM25 setup of migration 003. That migration called paradedb.create_bm25(), a procedure that no
-- longer exists in pg_search, so the index was never created and lexical search silently used the FTS fallback.
-- The index below uses the current syntax; its tokenizer mirrors the English tsvector analysis (stemming and
-- stop words) so the two channels differ only in the ranking function.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search') THEN
        EXECUTE $idx$
            CREATE INDEX IF NOT EXISTS idx_chunks_bm25 ON chunks USING paradedb (
                id,
                (content::pdb.simple('stemmer=english', 'stopwords_language=english', 'ascii_folding=true')),
                (summary_text::pdb.simple('stemmer=english', 'stopwords_language=english', 'ascii_folding=true'))
            ) WITH (key_field = 'id')
        $idx$;
    ELSE
        RAISE NOTICE 'pg_search is not installed; lexical search keeps using the FTS fallback.';
    END IF;
END $$;
