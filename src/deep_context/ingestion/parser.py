"""Structure-aware parsers for Markdown, Code, PDF, and Structured text."""

from __future__ import annotations

import ast
import io
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedSection:
    title: str
    content: str
    section_path: str
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def count_approx_tokens(text: str) -> int:
    """Rough approximation: 1 token ~= 4 characters or 0.75 words."""
    words = len(text.split())
    chars = len(text)
    return max(words, chars // 4)


class DocumentParser:
    """Structure-aware parser respecting section boundaries, code AST, PDF pages, and tables."""

    @classmethod
    def parse(
        cls,
        content: str | bytes,
        doc_type: str = "markdown",
    ) -> list[ParsedSection]:
        doc_type = doc_type.lower()
        if doc_type == "pdf" or doc_type.endswith(".pdf"):
            return cls.parse_pdf(content)
        elif isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = content

        if doc_type == "markdown" or doc_type.endswith(".md"):
            return cls.parse_markdown(text)
        elif doc_type in ("python", "code", "py") or doc_type.endswith(
            (".py", ".js", ".ts", ".java", ".go")
        ):
            return cls.parse_code(text, doc_type)
        else:
            return cls.parse_text(text)

    @classmethod
    def parse_pdf(cls, content: str | bytes) -> list[ParsedSection]:
        """
        Streaming page-by-page PDF parser supporting massive documents (e.g. 1000+ pages)
        using pypdf with page boundary preservation and token bounding.
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDF parsing. Install it with: uv add pypdf")

        if isinstance(content, str):
            # Check if content is a file path or raw text
            import os

            if os.path.exists(content):
                stream: io.BufferedReader | io.BytesIO = open(content, "rb")
            else:
                stream = io.BytesIO(content.encode("latin-1"))
        else:
            stream = io.BytesIO(content)

        sections: list[ParsedSection] = []
        try:
            reader = pypdf.PdfReader(stream)
            total_pages = len(reader.pages)

            for page_idx in range(total_pages):
                page = reader.pages[page_idx]
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue

                page_num = page_idx + 1

                # If page is long, break it into paragraph-bounded sections
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
                if not paragraphs:
                    paragraphs = [page_text]

                current_block: list[str] = []
                current_tokens = 0
                sec_sub_idx = 1

                for p in paragraphs:
                    p_tok = count_approx_tokens(p)
                    if current_tokens + p_tok > 600 and current_block:
                        block_text = "\n\n".join(current_block)
                        first_line = current_block[0].splitlines()[0][:50].strip()
                        sections.append(
                            ParsedSection(
                                title=f"Page {page_num}: {first_line}",
                                content=block_text,
                                section_path=f"Page {page_num} > Part {sec_sub_idx}",
                                page_number=page_num,
                                metadata={
                                    "page": page_num,
                                    "total_pages": total_pages,
                                },
                            )
                        )
                        current_block = [p]
                        current_tokens = p_tok
                        sec_sub_idx += 1
                    else:
                        current_block.append(p)
                        current_tokens += p_tok

                if current_block:
                    block_text = "\n\n".join(current_block)
                    first_line = current_block[0].splitlines()[0][:50].strip()
                    sections.append(
                        ParsedSection(
                            title=f"Page {page_num}: {first_line}",
                            content=block_text,
                            section_path=f"Page {page_num} > Part {sec_sub_idx}",
                            page_number=page_num,
                            metadata={"page": page_num, "total_pages": total_pages},
                        )
                    )
        finally:
            if hasattr(stream, "close"):
                stream.close()

        return sections or [
            ParsedSection(
                title="Empty PDF",
                content="(Empty PDF document)",
                section_path="Page 1",
                page_number=1,
            )
        ]

    @classmethod
    def parse_markdown(cls, content: str) -> list[ParsedSection]:
        """Parse markdown into logical sections based on headings (#, ##, ###)."""
        lines = content.split("\n")
        sections: list[ParsedSection] = []

        current_headings: list[str] = []
        current_content_lines: list[str] = []
        current_title = "Introduction"

        heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

        for line in lines:
            match = heading_pattern.match(line)
            if match:
                # Save previous section if not empty
                if current_content_lines:
                    section_text = "\n".join(current_content_lines).strip()
                    if section_text:
                        section_path = (
                            " > ".join(current_headings) if current_headings else current_title
                        )
                        sections.append(
                            ParsedSection(
                                title=current_title,
                                content=section_text,
                                section_path=section_path,
                            )
                        )
                    current_content_lines = []

                level = len(match.group(1))
                heading_text = match.group(2).strip()
                current_title = heading_text

                # Adjust headings stack for section_path hierarchy
                if level <= len(current_headings):
                    current_headings = current_headings[: level - 1]
                current_headings.append(heading_text)
                current_content_lines.append(line)
            else:
                current_content_lines.append(line)

        # Append last section
        if current_content_lines:
            section_text = "\n".join(current_content_lines).strip()
            if section_text:
                section_path = " > ".join(current_headings) if current_headings else current_title
                sections.append(
                    ParsedSection(
                        title=current_title,
                        content=section_text,
                        section_path=section_path,
                    )
                )

        if not sections:
            sections.append(
                ParsedSection(title="Document", content=content, section_path="Document")
            )

        return sections

    @classmethod
    def parse_code(cls, content: str, doc_type: str = "code") -> list[ParsedSection]:
        """Parse source code by function/class boundaries without splitting mid-function."""
        if doc_type in ("python", "py") or "def " in content or "class " in content:
            try:
                tree = ast.parse(content)
                sections: list[ParsedSection] = []
                lines = content.splitlines()

                for node in tree.body:
                    if isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    ):
                        start_line = node.lineno - 1
                        end_line = getattr(node, "end_lineno", len(lines))
                        node_code = "\n".join(lines[start_line:end_line])
                        node_name = node.name
                        kind = "class" if isinstance(node, ast.ClassDef) else "function"
                        sections.append(
                            ParsedSection(
                                title=f"{kind} {node_name}",
                                content=node_code,
                                section_path=f"code > {kind} {node_name}",
                                metadata={"symbol": node_name, "kind": kind},
                            )
                        )
                if sections:
                    return sections
            except Exception:
                pass  # Fall back to regex/block parsing

        # Fallback block parsing for code
        blocks = re.split(r"\n(?=(?:def |class |function |export |public |private ))", content)
        sections = []
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            first_line = block.strip().splitlines()[0][:60]
            sections.append(
                ParsedSection(
                    title=f"Block: {first_line}",
                    content=block.strip(),
                    section_path=f"code > block_{i + 1}",
                )
            )
        return sections or [ParsedSection(title="Code", content=content, section_path="Code")]

    @classmethod
    def parse_text(cls, content: str) -> list[ParsedSection]:
        """Parse plain text into paragraph-bounded sections."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        sections: list[ParsedSection] = []
        current_section: list[str] = []
        current_tokens = 0

        for p in paragraphs:
            p_tokens = count_approx_tokens(p)
            if current_tokens + p_tokens > 800 and current_section:
                sec_text = "\n\n".join(current_section)
                sections.append(
                    ParsedSection(
                        title=f"Section {len(sections) + 1}",
                        content=sec_text,
                        section_path=f"Document > Section {len(sections) + 1}",
                    )
                )
                current_section = [p]
                current_tokens = p_tokens
            else:
                current_section.append(p)
                current_tokens += p_tokens

        if current_section:
            sec_text = "\n\n".join(current_section)
            sections.append(
                ParsedSection(
                    title=f"Section {len(sections) + 1}",
                    content=sec_text,
                    section_path=f"Document > Section {len(sections) + 1}",
                )
            )

        return sections or [
            ParsedSection(title="Document", content=content, section_path="Document")
        ]
