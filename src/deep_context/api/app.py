"""FastAPI application for Deep Context Platform."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deep_context.api.routes_health import router as health_router
from deep_context.api.routes_memory import router as memory_router
from deep_context.api.routes_ops import router as ops_router
from deep_context.api.routes_rag import router as rag_router
from deep_context.core.config import settings
from deep_context.core.logging import logger
from deep_context.scheduler import register_default_jobs, scheduler_loop
from deep_context.storage import close_storage, get_storage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting Deep Context Platform service...")
    await get_storage()
    scheduler_task = None
    if settings.scheduler_enabled:
        await register_default_jobs()
        scheduler_task = asyncio.create_task(scheduler_loop())
        logger.info("Internal scheduler started (SCHEDULER_ENABLED=true).")
    yield
    if scheduler_task is not None:
        scheduler_task.cancel()
    logger.info("Shutting down Deep Context Platform service...")
    await close_storage()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Deep Context Platform API",
        description="Agent backend: Agentic Hybrid RAG with Local Qwen3 Summarization and Typed Long-Term Memory.",
        version="0.2.0",
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
    app.include_router(ops_router)

    return app


app = create_app()
