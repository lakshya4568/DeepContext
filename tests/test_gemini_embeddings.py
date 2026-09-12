"""Tests for Google Gemini Embeddings (gemini-embedding-2, gemini-embedding-001, MRL dimensions)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_context.core.llm_client import LLMClient


@pytest.mark.asyncio
async def test_gemini_mock_embeddings_mrl_dimensions() -> None:
    client = LLMClient()
    setattr(client, "_refresh_gemini_client", lambda: None)
    client._gemini_client = None

    # Test 768-dim (Recommended MRL)
    emb_768 = await client.get_embeddings(
        ["Architecture of Distributed Deep Context"],
        model="gemini-embedding-2",
        dim=768,
        is_query=True,
    )
    assert len(emb_768) == 1
    assert len(emb_768[0]) == 768

    # Test 1536-dim
    emb_1536 = await client.get_embeddings(
        ["Parent-child chunk hierarchical resolution"],
        model="gemini-embedding-2",
        dim=1536,
        is_query=False,
        title="Architecture Doc",
    )
    assert len(emb_1536) == 1
    assert len(emb_1536[0]) == 1536

    # Test 3072-dim
    emb_3072 = await client.get_embeddings(
        ["Needle-in-a-Haystack benchmark"],
        model="gemini-embedding-2",
        dim=3072,
    )
    assert len(emb_3072) == 1
    assert len(emb_3072[0]) == 3072


@pytest.mark.asyncio
async def test_gemini_embedding_2_live_client_call_formatting() -> None:
    client = LLMClient()

    # Mock genai client
    mock_genai_client = MagicMock()
    mock_resp = MagicMock()

    # Mock return values for 768-dim
    fake_emb_1 = MagicMock()
    fake_emb_1.values = [0.1] * 768
    fake_emb_2 = MagicMock()
    fake_emb_2.values = [0.2] * 768
    mock_resp.embeddings = [fake_emb_1, fake_emb_2]

    mock_genai_client.aio.models.embed_content = AsyncMock(return_value=mock_resp)
    client._gemini_client = mock_genai_client
    setattr(client, "_refresh_gemini_client", lambda: mock_genai_client)

    # Test batch document call
    texts = ["First chunk content", "Second chunk content"]
    embs = await client.get_embeddings(
        texts,
        model="gemini-embedding-2",
        dim=768,
        title="Technical Specification",
        is_query=False,
    )

    assert len(embs) == 2
    assert len(embs[0]) == 768
    assert len(embs[1]) == 768

    # Verify call args
    mock_genai_client.aio.models.embed_content.assert_called_once()
    _, kwargs = mock_genai_client.aio.models.embed_content.call_args
    assert kwargs["model"] == "gemini-embedding-2"
    assert kwargs["config"].output_dimensionality == 768

    # Verify asymmetric document task formatting
    contents = kwargs["contents"]
    assert len(contents) == 2
    assert "title: Technical Specification | text: First chunk content" in str(
        contents[0].parts[0].text
    )


@pytest.mark.asyncio
async def test_gemini_embedding_2_query_formatting() -> None:
    client = LLMClient()

    mock_genai_client = MagicMock()
    mock_resp = MagicMock()
    fake_emb = MagicMock()
    fake_emb.values = [0.05] * 768
    mock_resp.embeddings = [fake_emb]
    mock_genai_client.aio.models.embed_content = AsyncMock(return_value=mock_resp)
    client._gemini_client = mock_genai_client
    setattr(client, "_refresh_gemini_client", lambda: mock_genai_client)

    emb = await client.get_embedding(
        "How does RLM handle recursive memory?",
        model="gemini-embedding-2",
        dim=768,
        is_query=True,
    )
    assert len(emb) == 768

    _, kwargs = mock_genai_client.aio.models.embed_content.call_args
    contents = kwargs["contents"]
    assert "task: search result | query: How does RLM handle recursive memory?" in str(
        contents[0].parts[0].text
    )


def test_gemini_candidates_generation() -> None:
    client = LLMClient()
    # 3.x models prioritize global endpoint where they are hosted on Vertex AI
    c_37 = client._get_gemini_candidates("gemini-3.7-flash")
    assert c_37[0] == ("gemini-3.7-flash", "global")
    assert ("gemini-2.5-flash", "global") in c_37

    c_38 = client._get_gemini_candidates("gemini-3.8-flash")
    assert c_38[0] == ("gemini-3.8-flash", "global")
    assert ("gemini-2.5-flash", "global") in c_38

    # 2.5 flash has candidate coverage across regional and global
    c_25 = client._get_gemini_candidates("gemini-2.5-flash")
    assert ("gemini-2.5-flash", None) in c_25
    assert ("gemini-2.5-flash", "global") in c_25


@pytest.mark.asyncio
async def test_gemini_complete_404_fallback() -> None:
    client = LLMClient()

    mock_client = MagicMock()

    async def mock_gen(model, contents, config):
        if "nonexistent" in model:
            raise Exception("404 NOT_FOUND. Publisher model was not found")
        resp = MagicMock()
        resp.candidates = []
        resp.text = "Hello from fallback 2.5 flash"
        return resp

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_gen)
    setattr(client, "_refresh_gemini_client", lambda location=None: mock_client)

    content, _ = await client.complete(
        [{"role": "user", "content": "Hi"}],
        model="gemini-nonexistent-model",
    )
    assert "Hello from fallback 2.5 flash" in content
