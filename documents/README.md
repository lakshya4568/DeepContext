# Sample Knowledge Base Document

Welcome to the **Deep Context Knowledge Base**.

## Universal Ingestion
You can drop any of the following file types directly into this `documents/` folder:
- **PDF Documents** (`.pdf`) — Supports 1 to 1000+ page PDFs with streaming page extraction and section preservation.
- **Markdown Files** (`.md`, `.markdown`) — Preserves heading hierarchies and tree nodes.
- **Plain Text & Logs** (`.txt`, `.text`, `.log`, `.csv`) — Paragraph and sentence bounded chunking.
- **Source Code** (`.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.cpp`, `.c`) — AST function and class boundary parsing without mid-function cuts.

## Fast Commands
- Ingest everything in this folder:
  ```bash
  uv run deep-context sync
  ```
- Or open the UI at `http://localhost:8000` and click **"📁 Ingest All from ./documents/"**.
