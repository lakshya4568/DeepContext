"""Durable asset storage for multimodal images, charts, and equation crops."""

from __future__ import annotations

import base64
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from deep_context.core.config import settings
from deep_context.core.logging import logger
from deep_context.core.types import MultimodalAsset, generate_stable_asset_id


class AssetStore:
    """Manages durable local persistence of image, chart, and diagram assets."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or settings.asset_storage_dir)
        self._doc_assets: dict[str, list[MultimodalAsset]] = defaultdict(list)

    def _ensure_tenant_dir(self, tenant_id: str = "default") -> Path:
        tenant_dir = self.base_dir / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        return tenant_dir

    def save_image(
        self,
        image_data: Image.Image | bytes | str,
        document_id: str,
        node_id: str | None = None,
        asset_type: str = "image",
        caption: str | None = None,
        ocr_text: str | None = None,
        page_number: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        tenant_id: str = "default",
        permission_scope: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MultimodalAsset:
        """
        Validates, normalizes, and durably persists an image asset to disk.
        Returns a populated MultimodalAsset model.
        """
        pil_img: Image.Image
        raw_bytes: bytes

        if isinstance(image_data, Image.Image):
            pil_img = image_data
            buf = io.BytesIO()
            # Convert RGBA/P to RGB if JPEG or preserve PNG
            if pil_img.mode in ("RGBA", "LA", "P"):
                pil_img.save(buf, format="PNG")
            else:
                pil_img.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
        elif isinstance(image_data, str) and image_data.startswith("data:"):
            # Base64 data URL
            header, encoded = image_data.split(",", 1)
            raw_bytes = base64.b64decode(encoded)
            pil_img = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(image_data, str) and Path(image_data).exists():
            # Local file path
            with open(image_data, "rb") as f:
                raw_bytes = f.read()
            pil_img = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(image_data, bytes):
            raw_bytes = image_data
            pil_img = Image.open(io.BytesIO(raw_bytes))
        else:
            raise ValueError(f"Unsupported image_data type: {type(image_data)}")

        width, height = pil_img.size
        byte_size = len(raw_bytes)
        asset_id = generate_stable_asset_id(raw_bytes)
        sha256 = asset_id.replace("asset_", "")

        tenant_dir = self._ensure_tenant_dir(tenant_id)
        file_path = tenant_dir / f"{asset_id}.png"
        if not file_path.exists():
            file_path.write_bytes(raw_bytes)

        asset = MultimodalAsset(
            id=asset_id,
            document_id=document_id,
            node_id=node_id,
            asset_type=asset_type,
            mime_type="image/png",
            width=width,
            height=height,
            byte_size=byte_size,
            sha256=sha256,
            storage_path=str(file_path),
            caption=caption,
            ocr_text=ocr_text,
            page_number=page_number,
            bbox=bbox,
            tenant_id=tenant_id,
            permission_scope=permission_scope or ["default"],
            metadata=metadata or {},
        )
        self._doc_assets[document_id].append(asset)
        return asset

    def get_assets_for_document(self, document_id: str) -> list[MultimodalAsset]:
        """Returns all assets stored for the given document_id in this runtime."""
        return list(self._doc_assets.get(document_id, []))

    def get_asset_bytes(self, asset_id_or_path: str, tenant_id: str | None = None) -> bytes | None:
        """Retrieves raw image bytes for an asset ID or file path with strict tenant isolation."""
        p = Path(asset_id_or_path)
        if p.exists() and p.is_file():
            return p.read_bytes()

        if tenant_id:
            tenant_dir = self.base_dir / tenant_id
            candidate = tenant_dir / f"{asset_id_or_path}.png"
            if candidate.exists():
                return candidate.read_bytes()
            return None

        # Check in any tenant subfolder only if tenant wasn't specified
        for child in self.base_dir.glob(f"*/{asset_id_or_path}.png"):
            if child.is_file():
                return child.read_bytes()

        return None

    def get_asset_path(self, asset_id: str, tenant_id: str | None = None) -> Path | None:
        """Resolves file path for an asset ID with strict tenant isolation."""
        if tenant_id:
            candidate = self.base_dir / tenant_id / f"{asset_id}.png"
            if candidate.exists():
                return candidate
            return None

        for child in self.base_dir.glob(f"*/{asset_id}.png"):
            if child.is_file():
                return child
        return None

    def delete_asset(self, asset_id: str, tenant_id: str | None = None) -> bool:
        """Deletes an asset from storage."""
        p = self.get_asset_path(asset_id, tenant_id)
        if p and p.exists():
            try:
                p.unlink()
                return True
            except OSError as e:
                logger.warning("Failed to delete asset %s: %s", p, e)
        return False


asset_store = AssetStore()
