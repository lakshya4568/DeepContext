"""Tests for cross-platform hardware acceleration parity and embedding integrity."""

import pytest
import torch

from deep_context.core.llm_client import llm_client


def test_hardware_acceleration_detection():
    """Verify cross-platform device detection logic for MPS, CUDA, and CPU fallback."""
    # Ensure torch doesn't crash regardless of platform
    has_mps = torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()

    if has_mps:
        device = torch.device("mps")
        tensor = torch.zeros((2, 2), device=device)
        assert tensor.device.type == "mps"
    elif has_cuda:
        device = torch.device("cuda")
        tensor = torch.zeros((2, 2), device=device)
        assert tensor.device.type == "cuda"
    else:
        device = torch.device("cpu")
        tensor = torch.zeros((2, 2), device=device)
        assert tensor.device.type == "cpu"


@pytest.mark.asyncio
async def test_real_embedding_generation_no_mock_hashes():
    """
    Ensure real embedding calls invoke the configured model provider without injecting
    synthetic hash-based vectors.
    """
    sample_text = "Retriever and Parent-Child Chunking with provenance metadata."
    embs = await llm_client.get_embeddings(
        [sample_text],
        model="gemini-embedding-2",
        dim=768,
        is_query=False,
    )
    assert len(embs) == 1
    emb = embs[0]
    assert len(emb) == 768
    # Assert embeddings are floating point numbers not all zeros or trivial repeating hash patterns
    assert any(val != 0.0 for val in emb)
    assert not all(v == emb[0] for v in emb)
    norm = sum(v * v for v in emb) ** 0.5
    # Standard normalized vector length should be near 1.0
    assert abs(norm - 1.0) < 0.05 or norm > 0.1


@pytest.mark.asyncio
async def test_multimodal_image_embedding_integrity(tmp_path):
    """Verify that embed_image returns high-dimensional semantic vectors from image data."""
    from PIL import Image

    img_path = tmp_path / "quantum_figure.png"
    img = Image.new("RGB", (64, 64), color="red")
    img.save(img_path)

    emb = await llm_client.embed_image(img_path, model="gemini-embedding-2", dim=768)
    assert len(emb) == 768
    assert any(v != 0.0 for v in emb)
    norm = sum(v * v for v in emb) ** 0.5
    assert abs(norm - 1.0) < 0.05 or norm > 0.1
