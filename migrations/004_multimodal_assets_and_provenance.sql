-- Migration: 004_multimodal_assets_and_provenance.sql
-- Multimodal assets table, equation and code metadata columns, and complete provenance tracking

CREATE TABLE IF NOT EXISTS document_assets (
    id TEXT PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    node_id UUID REFERENCES document_tree_nodes(id) ON DELETE SET NULL,
    asset_type TEXT NOT NULL DEFAULT 'image', -- 'image' | 'chart' | 'diagram' | 'equation_crop'
    mime_type TEXT NOT NULL DEFAULT 'image/png',
    width INTEGER,
    height INTEGER,
    byte_size INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    caption TEXT,
    ocr_text TEXT,
    description TEXT,
    embedding VECTOR(768),
    page_number INTEGER,
    bbox JSONB,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    permission_scope TEXT[] NOT NULL DEFAULT ARRAY['default'],
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_assets_document ON document_assets (document_id);
CREATE INDEX IF NOT EXISTS idx_assets_node ON document_assets (node_id);
CREATE INDEX IF NOT EXISTS idx_assets_tenant ON document_assets (tenant_id);
CREATE INDEX IF NOT EXISTS idx_assets_sha256 ON document_assets (sha256);

-- HNSW index for asset vector search
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE INDEX IF NOT EXISTS idx_assets_embedding_hnsw ON document_assets USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not create idx_assets_embedding_hnsw: %', SQLERRM;
END $$;

-- Ensure document_tree_nodes has all provenance, equation, code, and asset columns
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS source_uri TEXT;
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS char_span JSONB;
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS raw_ref TEXT;
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS parser TEXT;
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS parser_version TEXT;
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS extraction_method TEXT;
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS confidence REAL;
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS line_range JSONB;
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS equation_data JSONB;
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS code_data JSONB;
ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS asset_id TEXT;

ANALYZE document_assets;
ANALYZE document_tree_nodes;
