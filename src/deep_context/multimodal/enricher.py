"""Optional Multimodal Enrichment Stage for Figures, Charts, and Diagrams."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from deep_context.core.logging import logger
from deep_context.core.types import (
    DocumentElementType,
    DocumentTree,
    FigureDataModel,
    MultimodalEnrichment,
)


class MultimodalEnricher:
    """Enriches figures, charts, and diagrams with optional vision/OCR descriptions while preserving strict separation from source text."""

    def __init__(
        self,
        vision_fn: Callable[[FigureDataModel, str | None], dict[str, Any]] | None = None,
    ):
        self._vision_fn = vision_fn

    async def enrich_tree(
        self,
        tree: DocumentTree,
        enabled: bool = False,
        model_name: str = "vision-enricher",
    ) -> DocumentTree:
        """
        Enriches all FIGURE and CHART nodes in the document tree.
        If disabled or on failure, the document tree remains intact with status marked appropriately.
        """
        now = datetime.now(timezone.utc)

        for node in tree.nodes:
            if node.node_type not in (
                DocumentElementType.FIGURE,
                DocumentElementType.CHART,
            ):
                continue

            fig = node.figure_data
            if not fig:
                fig = FigureDataModel(
                    figure_type="chart" if node.node_type == DocumentElementType.CHART else "image",
                    caption=node.text if node.text else None,
                )
                node.figure_data = fig

            if not enabled:
                if not fig.enrichment:
                    fig.enrichment = MultimodalEnrichment(
                        status="skipped",
                        model_name=model_name,
                        processed_at=now,
                    )
                continue

            # Process enrichment safely with try-except boundary
            try:
                if self._vision_fn:
                    res = self._vision_fn(fig, node.text)
                    fig.enrichment = MultimodalEnrichment(
                        status="completed",
                        model_name=res.get("model_name", model_name),
                        description=res.get("description"),
                        extracted_labels=res.get("extracted_labels", []),
                        extracted_values=res.get("extracted_values", []),
                        confidence=res.get("confidence", 0.9),
                        limitations=res.get("limitations"),
                        processed_at=now,
                    )
                else:
                    from deep_context.core.llm_client import llm_client
                    from deep_context.storage.asset_store import asset_store

                    desc = None
                    aid = fig.asset_id or node.asset_id
                    if aid:
                        p = asset_store.get_asset_path(aid)
                        if p and p.exists():
                            try:
                                desc = await llm_client.describe_image(
                                    p,
                                    prompt=f"Inspect this figure: {fig.caption or node.text or ''}",
                                )
                            except Exception as e_desc:
                                logger.debug("Vision description notice for %s: %s", aid, e_desc)

                    if desc:
                        fig.enrichment = MultimodalEnrichment(
                            status="completed",
                            model_name="gemini-2.5-flash",
                            description=desc,
                            confidence=0.95,
                            processed_at=now,
                        )
                    else:
                        fig.enrichment = MultimodalEnrichment(
                            status="unavailable",
                            model_name=model_name,
                            description=None,
                            limitations="Visual interpretation model not attached or image crop unavailable.",
                            processed_at=now,
                        )
            except Exception as e:
                logger.warning(
                    "Multimodal enrichment failed for node %s (%s): %s",
                    node.id,
                    node.node_type.value,
                    e,
                )
                fig.enrichment = MultimodalEnrichment(
                    status="failed",
                    model_name=model_name,
                    limitations=f"Enrichment error: {str(e)}",
                    processed_at=now,
                )

        return tree

    def _extract_figure_interpretation(
        self,
        fig: FigureDataModel,
        context_text: str | None,
        model_name: str,
    ) -> dict[str, Any]:
        """Rule-based extraction when no external vision model is provided."""
        caption = fig.caption or context_text or ""
        labels: list[str] = []
        values: list[dict[str, Any]] = []

        if fig.figure_type == "chart":
            description = (
                f"Chart diagram representing: {caption}" if caption else "Uncaptioned chart"
            )
            return {
                "model_name": model_name,
                "description": description,
                "extracted_labels": labels,
                "extracted_values": values,
                "confidence": 0.8 if caption else 0.4,
                "limitations": "Extracted from document caption/metadata; visual interpretation model not attached.",
            }

        return {
            "model_name": model_name,
            "description": f"Figure depicting: {caption}" if caption else "Extracted figure",
            "extracted_labels": [],
            "extracted_values": [],
            "confidence": 0.8 if caption else 0.4,
            "limitations": "Extracted from document caption/metadata; visual interpretation model not attached.",
        }


multimodal_enricher = MultimodalEnricher()
