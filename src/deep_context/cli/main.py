"""Rich Typer CLI for the Deep Context Platform."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deep_context.agentic.router import QueryRouter
from deep_context.core.config import settings
from deep_context.core.llm_client import llm_client
from deep_context.core.types import (
    IngestRequest,
    RetrievalFilters,
    RetrievalMode,
)
from deep_context.ingestion.pipeline import ingestion_pipeline
from deep_context.retrieval.engine import retrieval_engine
from deep_context.rlm.orchestrator import RLMOrchestrator
from deep_context.storage import close_storage, get_storage

app = typer.Typer(
    name="deep-context",
    help="Deep Context Platform: Hybrid RAG + Typed Memory + RLM Engine CLI",
    add_completion=False,
)
console = Console()

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".log": "text",
    ".csv": "text",
    ".json": "text",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".java": "code",
    ".go": "code",
    ".rs": "code",
    ".cpp": "code",
    ".c": "code",
    ".html": "markdown",
}

IGNORED_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}


@app.command("clear")
@app.command("reset")
def clear_cmd() -> None:
    """Clear and remove all previously ingested documents from the database."""

    async def _run() -> None:
        storage = await get_storage()
        cnt = await storage.delete_all_documents()
        console.print(
            Panel(
                f"[bold green]Successfully Cleared Knowledge Base![/bold green]\n"
                f"Removed [yellow]{cnt}[/yellow] previously ingested documents and all associated chunks.",
                title="Reset Complete",
            )
        )
        await close_storage()

    asyncio.run(_run())


@app.command("ingest")
def ingest_cmd(
    file_path: str = typer.Argument(
        ..., help="Path to document file (PDF, TXT, MD, Code) to ingest"
    ),
    title: str = typer.Option("", "--title", "-t", help="Document title"),
    doc_type: str = typer.Option(
        "auto", "--type", help="Format: auto, markdown, code, pdf, html, text"
    ),
    vectorless: bool = typer.Option(
        False, "--vectorless", help="Enable vectorless tree navigation mode"
    ),
    embedding_model: str = typer.Option(
        "",
        "--embedding-model",
        "-e",
        help="Embedding model (e.g. 'gemini-embedding-2', 'gemini-embedding-001', 'nvidia/nv-embedqa-e5-v5')",
    ),
    embedding_dim: int = typer.Option(
        0,
        "--embedding-dim",
        "-d",
        help="Embedding output dimensionality (e.g. 768, 1536, 3072, 1024)",
    ),
) -> None:
    """Ingest any single PDF, TXT, MD, or Code file into the knowledge base."""

    async def _run() -> None:
        p = Path(file_path)
        if not p.exists():
            console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
            raise typer.Exit(1)

        ext = p.suffix.lower()
        detected_type = (
            doc_type if doc_type != "auto" else SUPPORTED_EXTENSIONS.get(ext, "text")
        )

        if detected_type == "pdf":
            content = str(p.resolve())
        else:
            content = p.read_text(encoding="utf-8", errors="replace")

        doc_title = title or p.stem
        mode = RetrievalMode.VECTORLESS if vectorless else RetrievalMode.HYBRID
        target_model = embedding_model or settings.embedding_model
        target_dim = embedding_dim or (
            768 if "gemini" in target_model.lower() else settings.embedding_dim
        )

        req = IngestRequest(
            title=doc_title,
            content=content,
            doc_type=detected_type,
            source_uri=str(p.resolve()),
            retrieval_mode=mode,
            embedding_model=target_model,
            embedding_dim=target_dim,
        )

        with console.status(
            f"[bold green]Ingesting {doc_title} (Embedding with {target_model} [{target_dim}-dim])..."
        ):
            res = await ingestion_pipeline.ingest(req)

        console.print(
            Panel(
                f"[bold green]Document Ingested Successfully![/bold green]\n"
                f"ID: [cyan]{res.document_id}[/cyan]\n"
                f"Title: {res.title}\n"
                f"Type: [magenta]{detected_type.upper()}[/magenta]\n"
                f"Embedding: [yellow]{res.embedding_model or target_model} ({res.embedding_dim or target_dim}-dim)[/yellow]\n"
                f"Parent Chunks: {res.parent_chunks_count} | Child Chunks: {res.child_chunks_count}\n"
                f"Retrieval Mode: {res.retrieval_mode.value} | Tree Nodes: {res.tree_nodes_count}",
                title="Ingestion Result",
            )
        )
        await close_storage()

    asyncio.run(_run())


@app.command("ingest-all")
@app.command("sync")
def ingest_all_cmd(
    target_path: str = typer.Argument(
        "documents",
        help="Folder or glob path to ingest all files from (defaults to './documents')",
    ),
    vectorless: bool = typer.Option(
        False,
        "--vectorless",
        help="Enable vectorless tree navigation mode for all files",
    ),
    embedding_model: str = typer.Option(
        "",
        "--embedding-model",
        "-e",
        help="Embedding model (e.g. 'gemini-embedding-2')",
    ),
    embedding_dim: int = typer.Option(
        0,
        "--embedding-dim",
        "-d",
        help="Embedding output dimensionality (e.g. 768, 1536, 3072, 1024)",
    ),
) -> None:
    """Batch ingest all PDFs, TXT, MD, and Code files in a folder without manual setup."""

    async def _run() -> None:
        p = Path(target_path)
        files_to_process: list[Path] = []

        if not p.exists():
            if target_path == "documents":
                p.mkdir(parents=True, exist_ok=True)
                console.print(
                    f"[yellow]Created '[bold]{target_path}[/bold]' folder.[/yellow] Place your PDFs, TXT, MD files inside and rerun."
                )
                return
            console.print(f"[bold red]Error:[/bold red] Path not found: {target_path}")
            raise typer.Exit(1)

        if p.is_file():
            files_to_process.append(p)
        else:
            for file_entry in p.rglob("*"):
                if file_entry.is_file():
                    if any(part in IGNORED_DIRS for part in file_entry.parts):
                        continue
                    if file_entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                        files_to_process.append(file_entry)

        if not files_to_process:
            console.print(
                f"[yellow]No supported documents found in '{target_path}'.[/yellow]"
            )
            return

        target_model = embedding_model or settings.embedding_model
        target_dim = embedding_dim or (
            768 if "gemini" in target_model.lower() else settings.embedding_dim
        )

        console.print(
            f"[bold green]Found {len(files_to_process)} document(s) to process using {target_model} ({target_dim}-dim)[/bold green]"
        )

        table = Table(title="Batch Ingestion Summary")
        table.add_column("Filename", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Size", style="dim")
        table.add_column("Parents", style="green")
        table.add_column("Children", style="yellow")
        table.add_column("Status", style="bold white")

        mode = RetrievalMode.VECTORLESS if vectorless else RetrievalMode.HYBRID

        for file_item in files_to_process:
            ext = file_item.suffix.lower()
            detected_type = SUPPORTED_EXTENSIONS.get(ext, "text")
            file_size_kb = f"{file_item.stat().st_size / 1024:.1f} KB"

            if detected_type == "pdf":
                content = str(file_item.resolve())
            else:
                content = file_item.read_text(encoding="utf-8", errors="replace")

            req = IngestRequest(
                title=file_item.stem,
                content=content,
                doc_type=detected_type,
                source_uri=str(file_item.resolve()),
                retrieval_mode=mode,
                embedding_model=target_model,
                embedding_dim=target_dim,
            )

            try:
                res = await ingestion_pipeline.ingest(req)
                table.add_row(
                    file_item.name,
                    detected_type.upper(),
                    file_size_kb,
                    str(res.parent_chunks_count),
                    str(res.child_chunks_count),
                    "[green]✓ Ingested[/green]",
                )
            except Exception as e:
                table.add_row(
                    file_item.name,
                    detected_type.upper(),
                    file_size_kb,
                    "-",
                    "-",
                    f"[red]✗ Error: {e}[/red]",
                )

        console.print(table)
        await close_storage()

    asyncio.run(_run())


@app.command("retrieve")
def retrieve_cmd(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to return"),
    user_id: str = typer.Option(
        "", "--user", "-u", help="User ID for preference resolution"
    ),
    embedding_model: str = typer.Option(
        "",
        "--embedding-model",
        "-e",
        help="Embedding model (e.g. 'gemini-embedding-2')",
    ),
    embedding_dim: int = typer.Option(
        0, "--embedding-dim", "-d", help="Embedding dimension"
    ),
    reranker: str = typer.Option(
        "",
        "--reranker",
        "-r",
        help="Reranker strategy ('cross_encoder', 'ecohash', 'local_cross_encoder')",
    ),
) -> None:
    """Run hybrid retrieval (BM25 + Vector + RRF + Multi-Strategy Rerank)."""

    async def _run() -> None:
        filters = RetrievalFilters()
        with console.status("[bold green]Executing Hybrid Retrieval..."):
            res = await retrieval_engine.retrieve(
                query=query,
                filters=filters,
                top_k=top_k,
                embedding_model=embedding_model if embedding_model else None,
                embedding_dim=embedding_dim if embedding_dim else None,
                reranker=reranker if reranker else None,
                user_id=user_id if user_id else None,
            )

        table = Table(
            title=f"Retrieval Results for: '{query}' (Sufficient: {res.sufficient})"
        )
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Document", style="magenta", width=20)
        table.add_column("Section", style="green", width=25)
        table.add_column("Content Snippet", style="white")

        for idx, p in enumerate(res.parent_chunks, start=1):
            table.add_row(
                str(idx),
                p.get("document_title", "Doc"),
                p.get("section_path") or "",
                p.get("content", "")[:180].replace("\n", " ") + "...",
            )

        console.print(table)
        await close_storage()

    asyncio.run(_run())


@app.command("query")
def query_cmd(
    query: str = typer.Argument(..., help="User query"),
    user_id: str = typer.Option(
        "default_user", "--user", "-u", help="User ID for memory & preferences"
    ),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Reasoning model to use (e.g. 'qwen/qwen3.6-27b', 'gemini-2.5-flash', 'z-ai/glm-5.2')",
    ),
    embedding_model: str = typer.Option(
        "",
        "--embedding-model",
        "-e",
        help="Embedding model (e.g. 'gemini-embedding-2', 'gemini-embedding-001')",
    ),
    embedding_dim: int = typer.Option(
        0, "--embedding-dim", "-d", help="Embedding dimension (e.g. 768, 1536, 3072)"
    ),
    reranker: str = typer.Option(
        "",
        "--reranker",
        "-r",
        help="Reranker strategy ('cross_encoder', 'ecohash', 'local_cross_encoder')",
    ),
) -> None:
    """Run full intelligent grounded query answering with preference resolution."""

    async def _run() -> None:
        filters = RetrievalFilters()

        with console.status("[bold green]Classifying query..."):
            decision = await QueryRouter.route(query=query)

        console.print(
            f"[bold blue]Router Decision:[/bold blue] Path: [cyan]{decision.path.value}[/cyan] | Shape: [magenta]{decision.query_shape.value}[/magenta]"
        )

        with console.status("[bold green]Retrieving relevant chunks..."):
            retrieval_res = await retrieval_engine.retrieve(
                query=query,
                filters=filters,
                top_k=6,
                embedding_model=embedding_model if embedding_model else None,
                embedding_dim=embedding_dim if embedding_dim else None,
                reranker=reranker if reranker else None,
                user_id=user_id,
            )

        console.print(
            f"[dim]Retrieved {len(retrieval_res.parent_chunks)} parent chunks (sufficient: {retrieval_res.sufficient})[/dim]"
        )

        from deep_context.generation.grounded_answer import generate_grounded_answer

        model_label = model or settings.llm_model
        with console.status(
            f"[bold green]Generating grounded answer with {model_label}..."
        ):
            grounded_res = await generate_grounded_answer(
                query=query,
                retrieved_chunks=retrieval_res.parent_chunks,
                model=model if model else None,
            )
            answer = grounded_res.answer
            reasoning = grounded_res.reason

        if reasoning:
            console.print(
                Panel(
                    reasoning,
                    title=f"[dim]Model Reasoning ({model_label})[/dim]",
                    style="dim",
                )
            )

        from deep_context.core.llm_client import LLMClient

        active_notice = (
            getattr(llm_client, "last_rate_limit", None) or LLMClient.global_rate_limit
        )
        if active_notice:
            console.print(
                Panel(
                    f"[bold red]⚠️  API RATE LIMIT NOTICE[/bold red]\n"
                    f"• Provider: [yellow]{active_notice.get('provider', 'Groq').upper()}[/yellow] | Model: [cyan]{active_notice.get('model', 'qwen/qwen3.6-27b')}[/cyan]\n"
                    f"• Quota: [bold]{active_notice.get('quota_type', 'Tokens Per Day')}[/bold] ({active_notice.get('used', '200k')}/{active_notice.get('limit', '200k')})\n"
                    f"• Quota Reset In: [bold green]{active_notice.get('retry_after', 'a few minutes')}[/bold green]\n"
                    f"• Fallback: [italic]Displaying grounded retrieved document evidence directly.[/italic]",
                    title="[bold yellow]API Quota Exceeded[/bold yellow]",
                    border_style="yellow",
                )
            )

        console.print(Panel(answer, title="[bold green]Answer[/bold green]"))
        await close_storage()

    asyncio.run(_run())


@app.command("preferences")
def get_preferences_cmd(
    user_id: str = typer.Argument(
        "default_user", help="User ID to view preferences for"
    ),
) -> None:
    """View saved embedding, reranker, and model preferences for a user."""

    async def _run() -> None:
        storage = await get_storage()
        from deep_context.memory.stores import MemoryStoreManager

        mgr = MemoryStoreManager(storage)
        prefs = await mgr.get_embedding_preferences(user_id=user_id)

        table = Table(title=f"User Preferences for: '{user_id}'")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Embedding Model", str(prefs.get("embedding_model")))
        table.add_row("Embedding Dimension", str(prefs.get("embedding_dim")))
        table.add_row("Reranker Strategy", str(prefs.get("reranker")))
        table.add_row("LLM Model", str(prefs.get("llm_model")))

        console.print(table)
        await close_storage()

    asyncio.run(_run())


@app.command("set-preference")
def set_preference_cmd(
    user_id: str = typer.Option("default_user", "--user", "-u", help="User ID"),
    embedding_model: str = typer.Option(
        "",
        "--embedding-model",
        "-e",
        help="Preferred embedding model (e.g. 'gemini-embedding-2', 'gemini-embedding-001', 'nvidia/nv-embedqa-e5-v5')",
    ),
    embedding_dim: int = typer.Option(
        0,
        "--embedding-dim",
        "-d",
        help="Preferred embedding dimension (e.g. 768, 1536, 3072)",
    ),
    reranker: str = typer.Option(
        "",
        "--reranker",
        "-r",
        help="Preferred reranker ('cross_encoder', 'ecohash', 'local_cross_encoder')",
    ),
    llm_model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Preferred LLM model (e.g. 'qwen/qwen3.6-27b', 'gemini-2.5-flash')",
    ),
) -> None:
    """Save user preferences for embeddings and rerankers into durable memory_preference store."""

    async def _run() -> None:
        storage = await get_storage()
        from deep_context.memory.stores import MemoryStoreManager

        mgr = MemoryStoreManager(storage)
        await mgr.set_embedding_preferences(
            user_id=user_id,
            embedding_model=embedding_model if embedding_model else None,
            embedding_dim=embedding_dim if embedding_dim else None,
            reranker=reranker if reranker else None,
            llm_model=llm_model if llm_model else None,
        )

        console.print(
            Panel(
                f"[bold green]Preferences Updated Successfully for User '{user_id}'![/bold green]\n"
                + (
                    f"• Embedding Model: [cyan]{embedding_model}[/cyan]\n"
                    if embedding_model
                    else ""
                )
                + (
                    f"• Embedding Dim: [cyan]{embedding_dim}[/cyan]\n"
                    if embedding_dim
                    else ""
                )
                + (
                    f"• Reranker Strategy: [magenta]{reranker}[/magenta]\n"
                    if reranker
                    else ""
                )
                + (f"• LLM Model: [yellow]{llm_model}[/yellow]\n" if llm_model else ""),
                title="Preference Saved",
            )
        )
        await close_storage()

    asyncio.run(_run())


@app.command("rlm")
def rlm_cmd(
    task_spec: str = typer.Argument(..., help="Task specification for RLM engine"),
    corpus_file: str = typer.Option("", "--corpus", "-c", help="Path to corpus file"),
) -> None:
    """Execute an RLM recursive session with sandboxed REPL and subagent messaging."""

    async def _run() -> None:
        storage = await get_storage()
        corpus = ""
        if corpus_file and Path(corpus_file).exists():
            p = Path(corpus_file)
            if p.suffix.lower() == ".pdf":
                from deep_context.ingestion.parser import DocumentParser

                with console.status(
                    f"[bold cyan]Extracting text from {p.name}...[/bold cyan]"
                ):
                    parser = DocumentParser()
                    sections = parser.parse(str(p.resolve()), doc_type="pdf")
                    corpus = "\n\n".join(
                        f"=== {s.title} ===\n{s.content}" for s in sections
                    )
            else:
                corpus = p.read_text(encoding="utf-8", errors="ignore")
        if not corpus:
            corpus = f"Repository Context and Ingested Material for task: {task_spec}"

        orchestrator = RLMOrchestrator(storage)
        model_label = settings.llm_model
        with console.status(
            f"[bold green]Running RLM Recursive Session with {model_label}...[/bold green]"
        ):
            res = await orchestrator.run_session(task_spec=task_spec, corpus=corpus)

        console.print(
            Panel(
                f"[bold green]RLM Session Completed![/bold green]\n"
                f"Session ID: [cyan]{res.session_id}[/cyan]\n"
                f"Turns Used: {res.turns_used} | Subagents Spawned: {res.children_spawned}\n\n"
                f"{res.answer}",
                title="RLM Synthesis Output",
            )
        )
        await close_storage()

    asyncio.run(_run())


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host interface"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
) -> None:
    """Start the FastAPI HTTP service."""
    console.print(
        f"[bold green]Starting Deep Context Platform on http://{host}:{port}[/bold green]"
    )
    uvicorn.run("deep_context.api.app:app", host=host, port=port, reload=False)


@app.command("scheduler")
def scheduler_cmd(
    poll_interval: int = typer.Option(
        10, "--poll-interval", help="Seconds between scheduler ticks"
    ),
) -> None:
    """Run the internal job scheduler loop (ingestion & index maintenance)."""

    async def _run() -> None:
        from deep_context.scheduler import register_default_jobs, run_due_jobs_once

        await register_default_jobs()
        console.print(
            Panel(
                "[bold green]Internal Scheduler Running[/bold green]\n"
                "Press Ctrl+C to stop. Jobs are persisted in the 'jobs' table.",
                title="Scheduler",
            )
        )
        try:
            while True:
                executed = await run_due_jobs_once()
                if executed:
                    console.print(f"[dim]Executed jobs: {', '.join(executed)}[/dim]")
                await asyncio.sleep(poll_interval)
        except KeyboardInterrupt:
            console.print("[yellow]Scheduler stopped.[/yellow]")
        finally:
            await close_storage()

    asyncio.run(_run())


@app.command("jobs")
def jobs_cmd() -> None:
    """List all registered scheduled jobs and their state."""

    async def _run() -> None:
        from deep_context.scheduler import TASKS

        storage = await get_storage()
        rows = await storage.list_jobs()

        table = Table(title="Scheduled Jobs")
        table.add_column("Name", style="cyan")
        table.add_column("Schedule", style="magenta")
        table.add_column("Next Run", style="green")
        table.add_column("Status", style="bold")
        table.add_column("Retries", style="yellow")

        for j in rows:
            table.add_row(
                str(j["name"]),
                str(j["schedule_cron"]),
                str(j["next_run_at"]),
                str(j["status"]),
                f"{j['retries']}/{j['max_retries']}",
            )

        console.print(table)
        console.print(f"[dim]Registered task callables: {sorted(TASKS.keys())}[/dim]")
        await close_storage()

    asyncio.run(_run())


@app.command("agentic-query")
def agentic_query_cmd(
    query: str = typer.Argument(..., help="User query"),
    max_rewrites: int = typer.Option(
        2, "--max-rewrites", help="Maximum corrective rewrite attempts"
    ),
    top_k: int = typer.Option(
        6, "--top-k", "-k", help="Chunks to retrieve per attempt"
    ),
) -> None:
    """Run the corrective agentic RAG state machine (retrieve -> grade -> rewrite -> generate)."""

    async def _run() -> None:
        from deep_context.agentic.graph import run_agentic_rag

        with console.status("[bold green]Running corrective agentic RAG graph..."):
            state = await run_agentic_rag(
                query=query, top_k=top_k, max_rewrites=max_rewrites
            )

        trace_lines = "\n".join(
            f"• [cyan]{t.get('node')}[/cyan] {', '.join(f'{k}={v}' for k, v in t.items() if k != 'node')}"
            for t in state.trace
        )
        status_color = "red" if state.abstained else "green"
        console.print(
            Panel(
                f"[bold {status_color}]{'Abstained (insufficient evidence)' if state.abstained else 'Answer Generated'}[/bold {status_color}]\n"
                f"Grade: [magenta]{state.grade_result}[/magenta] | "
                f"Rewrites: {state.rewrite_count} | "
                f"Support: {'✓' if state.support_passed else '✗'} "
                f"({state.support_confidence:.0%})\n\n"
                f"{state.answer}",
                title="Agentic RAG Result",
            )
        )
        console.print(
            Panel(trace_lines, title="[dim]Execution Trace[/dim]", style="dim")
        )
        await close_storage()

    asyncio.run(_run())
