import asyncio

from deep_context.core.types import RetrievalFilters
from deep_context.retrieval.engine import retrieval_engine


async def main():
    print("[*] Testing live hybrid retrieval...")
    filters = RetrievalFilters(tenant_id="default")

    # 1. Test semantic query
    res = await retrieval_engine.retrieve(
        query="How does the local GPU summarization boost BM25 recall?",
        filters=filters,
        top_k=3,
    )
    print(f"[+] Retrieval sufficient: {res.sufficient}")
    print(f"    Parent chunks returned: {len(res.parent_chunks)}")
    print(f"    Citations count: {len(res.citations)}")
    assert res.sufficient is True
    assert len(res.parent_chunks) > 0
    print(f"    Top content preview: {res.parent_chunks[0]['content'][:120]}...")

    # 2. Test exact identifier matching (e.g. Qwen3-0.6B)
    res_id = await retrieval_engine.retrieve(
        query="What model is Qwen3-0.6B?",
        filters=filters,
        top_k=3,
    )
    print(f"[+] Identifier query sufficient: {res_id.sufficient}")
    assert res_id.sufficient is True
    assert len(res_id.parent_chunks) > 0
    print("[+] LIVE RETRIEVAL VERIFICATION PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(main())
