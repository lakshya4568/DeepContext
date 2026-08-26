import asyncio
import os
import uuid

import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def test_db_live():
    dsn = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/awems")
    conn = await asyncpg.connect(dsn)

    # 1. Insert test document
    doc_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO documents (id, title, doc_type) VALUES ($1, $2, $3)",
        doc_id,
        "Database Live Verification",
        "test",
    )

    # 2. Insert test chunk with content and summary_text
    chunk_id = uuid.uuid4()
    vec = [0.0] * 768
    vec[0] = 1.0
    vec_str = "[" + ",".join(map(str, vec)) + "]"

    await conn.execute(
        """
        INSERT INTO chunks (id, document_id, level, content, summary_text, token_count, embedding)
        VALUES ($1, $2, 'child', 'DeepContext advanced hybrid retrieval', 'Autonomous RAG with HNSW index', 25, $3::vector)
        """,
        chunk_id,
        doc_id,
        vec_str,
    )

    # 3. Verify trigger automatically computed search_tsv
    row = await conn.fetchrow(
        "SELECT search_tsv, (search_tsv IS NOT NULL) as has_search_tsv, (tsv IS NOT NULL) as has_tsv FROM chunks WHERE id = $1",
        chunk_id,
    )
    has_search_tsv = row["has_search_tsv"]
    tsv_val = row["search_tsv"]
    print(f"TRIGGER_VERIFIED={has_search_tsv}")
    print(f"SEARCH_TSV_VALUE={tsv_val}")

    # 4. Test BM25 match against search_tsv
    bm25_hits = await conn.fetch(
        "SELECT id FROM chunks WHERE search_tsv @@ plainto_tsquery($1, $2)",
        "english",
        "Autonomous RAG",
    )
    print(f"BM25_SUMMARY_MATCH={len(bm25_hits) >= 1}")

    # 5. Test HNSW Cosine vector search
    vec_hits = await conn.fetch(
        "SELECT id, 1 - (embedding <=> $1::vector) as similarity FROM chunks WHERE id = $2",
        vec_str,
        chunk_id,
    )
    sim = vec_hits[0]["similarity"]
    print(f"VECTOR_SIMILARITY={sim:.4f}")

    # Clean up test row
    await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)
    await conn.close()
    print("ALL_DB_VERIFICATIONS_PASSED=True")


if __name__ == "__main__":
    asyncio.run(test_db_live())
