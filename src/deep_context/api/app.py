"""FastAPI application for Deep Context Platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deep_context.api.routes_health import router as health_router
from deep_context.api.routes_memory import router as memory_router
from deep_context.api.routes_rag import router as rag_router
from deep_context.api.routes_rlm import router as rlm_router
from deep_context.core.logging import logger
from deep_context.storage import close_storage, get_storage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting Deep Context Platform service...")
    await get_storage()
    yield
    logger.info("Shutting down Deep Context Platform service...")
    await close_storage()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Deep Context Platform API",
        description="Agent backend: Hybrid RAG, Typed Long-Term Memory, and Recursive Language Model (RLM) Engine.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(rag_router)
    app.include_router(memory_router)
    app.include_router(rlm_router)

    return app


app = create_app()
