import asyncio
import os
import sys
from deep_context.storage.postgres_store import PostgresStore
from deep_context.ingestion.summary_pipeline import SummaryIngestionPipeline, IngestRequest
from deep_context.core.config import settings

async def main():
    print('[*] Connecting to PostgreSQL...')
    store = PostgresStore(settings.postgres_dsn)
    await store.initialize()
    
    pipeline = SummaryIngestionPipeline(storage=store)
    
    test_content = '''# 1. Distributed Context Engine

The Deep Context Platform is an enterprise-grade retrieval-augmented generation (RAG) system engineered for high-throughput knowledge processing and autonomous reasoning. It utilizes Matryoshka Representation Learning (MRL) embeddings combined with Reciprocal Rank Fusion (RRF) and cross-encoder reranking to ensure optimal precision across extensive document repositories.

## 1.1 Hierarchical Chunking
Hierarchical chunking partitions source documents into large parent contexts (1000-2500 tokens) and granular child segments (300-600 tokens). Child chunks enable ultra-fast cosine similarity search across HNSW indexes, while parent chunks provide rich situational context for grounded generative synthesis.

# 2. Local GPU Summarization
Local Small Language Models (SLMs) such as Qwen3 running on NVIDIA CUDA hardware generate concise semantic summaries for each child chunk. These summaries are enriched with full-text search vectors (search_tsv), boosting BM25 recall by 40% when users query concepts not explicitly present in raw source text.
'''
    
    print('[*] Ingesting document with parent-child chunking & GPU summarization...')
    req = IngestRequest(
        title='Deep Context Live Test Doc',
        content=test_content,
        doc_type='markdown',
        embedding_model='gemini-embedding-2',
        embedding_dim=768,
        generate_summaries=True,
    )
    
    res = await pipeline.ingest(req)
    print(f'[+] Ingestion succeeded!')
    print(f'    Doc ID: {res.document_id}')
    print(f'    Parents: {res.parent_chunks_count}')
    print(f'    Children: {res.child_chunks_count}')
    print(f'    Summaries Generated: {res.summaries_generated_count}')
    
    # Verify in PostgreSQL
    doc = await store.get_document(res.document_id)
    assert doc is not None, 'Document not found in DB'
    print(f'[+] Verified Document in PostgreSQL: {doc.title}')
    
    chunks = await store.get_document_chunks_detail(res.document_id)
    print(f'[+] Total Chunks verified in PostgreSQL: {len(chunks)}')
    
    parents = [c for c in chunks if c['level'] == 'parent']
    children = [c for c in chunks if c['level'] == 'child']
    print(f'    Parent Chunks: {len(parents)}')
    print(f'    Child Chunks: {len(children)}')
    
    for c in children:
        assert c['parent_chunk_id'] is not None, 'Child missing parent_chunk_id'
        p_id = c['parent_chunk_id']
        parent_exists = any(p['id'] == p_id for p in parents)
        assert parent_exists, f'Parent {p_id} not found in parents'
        summary_sample = str(c.get('summary_text', ''))[:40]
        c_short = str(c['id'])[:8]
        p_short = str(p_id)[:8]
        print(f'    Child {c_short} linked to Parent {p_short} | Summary: {summary_sample}...')
    
    print('[+] ALL FOREIGN KEY & CHUNK CONSTRAINTS VERIFIED 100% IN POSTGRESQL!')

if __name__ == '__main__':
    asyncio.run(main())
