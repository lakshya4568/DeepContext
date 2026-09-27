"""Structure-aware parsers for PDF, Markdown, HTML, Code, and Structured text.

Uses IBM Docling as the primary parsing engine for supported formats (PDF, Markdown, DOCX, HTML),
preserving hierarchy, tables, cells, figures, charts, captions, reading order, and source provenance.
Includes native resilient fallbacks and a separate syntax-aware parser for source code files.
"""

from __future__ import annotations

import ast
import io
import os
import re
import tempfile
import uuid
from typing import Any

from deep_context.core.logging import logger
from deep_context.core.types import (
    DocumentElementType,
    DocumentNode,
    DocumentTree,
    FigureDataModel,
    ParsedSection,
    Provenance,
    TableCellData,
    TableDataModel,
)
from deep_context.ingestion.cleaner import TextCleaner


def count_approx_tokens(text: str) -> int:
    """Rough approximation: 1 token ~= 4 characters or 0.75 words."""
    words = len(text.split())
    chars = len(text)
    return max(words, chars // 4)


class DocumentParser:
    """Structure-aware parser using IBM Docling as primary parser with native fallbacks and AST code parsing."""

    @classmethod
    def parse_tree(
        cls,
        content: str | bytes,
        doc_type: str = "markdown",
        source_uri: str | None = None,
        title: str = "",
        use_docling: bool = True,
    ) -> DocumentTree:
        """
        Builds a normalized parse tree representing document logical structure.
        Uses IBM Docling as the primary parser for PDF, Markdown, DOCX, and HTML.
        """
        if isinstance(content, str) and "\x00" in content:
            content = content.replace("\x00", "")

        doc_type_lower = doc_type.lower()

        # 1. Source-code syntax-aware parsing path (never sent to Docling)
        if doc_type_lower in ("python", "code", "py") or doc_type_lower.endswith(
            (".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".cpp", ".c")
        ):
            code_str = (
                content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
            )
            return cls._parse_code_tree(
                code_str, doc_type=doc_type_lower, source_uri=source_uri, title=title
            )

        # 2. PDF Parsing: Primary Docling with resilient pypdf fallback
        if doc_type_lower == "pdf" or doc_type_lower.endswith(".pdf"):
            if use_docling:
                try:
                    tree = cls._parse_with_docling(
                        content, suffix=".pdf", source_uri=source_uri, title=title
                    )
                    if tree and len(tree.nodes) > 1:
                        return tree
                except Exception as e:
                    logger.warning(
                        "Docling primary PDF parser encountered an issue (%s); falling back to native PDF parser.",
                        e,
                    )
            return cls._parse_pdf_tree_native(content, source_uri=source_uri, title=title)

        # 3. Primary Docling Parsing for Markdown, DOCX, and HTML
        if doc_type_lower in (
            "docx",
            "html",
            "htm",
            "markdown",
            "md",
        ) or doc_type_lower.endswith((".docx", ".html", ".htm", ".md", ".markdown")):
            suffix = ".md"
            if "docx" in doc_type_lower:
                suffix = ".docx"
            elif "htm" in doc_type_lower:
                suffix = ".html"

            if use_docling:
                try:
                    tree = cls._parse_with_docling(
                        content, suffix=suffix, source_uri=source_uri, title=title
                    )
                    if tree and len(tree.nodes) > 1:
                        return tree
                except Exception as e:
                    logger.warning(
                        "Docling primary parser for %s encountered an issue (%s); falling back to native parser.",
                        doc_type,
                        e,
                    )

        # 4. Native Fallbacks
        text_str = (
            content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        )
        if doc_type_lower in ("markdown", "md") or doc_type_lower.endswith((".md", ".markdown")):
            return cls._parse_markdown_tree_native(text_str, source_uri=source_uri, title=title)
        elif doc_type_lower in ("html", "htm") or doc_type_lower.endswith((".html", ".htm")):
            clean_md = cls._html_to_markdown_fallback(text_str)
            return cls._parse_markdown_tree_native(clean_md, source_uri=source_uri, title=title)
        else:
            return cls._parse_text_tree_native(text_str, source_uri=source_uri, title=title)

    @classmethod
    def parse(
        cls,
        content: str | bytes,
        doc_type: str = "markdown",
    ) -> list[ParsedSection]:
        """
        Backward-compatible parse method returning list[ParsedSection].
        Builds the canonical DocumentTree under the hood and exports backward-compatible sections.
        """
        tree = cls.parse_tree(content, doc_type=doc_type)
        return tree.to_parsed_sections()

    @classmethod
    def parse_pdf(cls, content: str | bytes, use_docling: bool = True) -> list[ParsedSection]:
        """Parse PDF into ParsedSections using Docling with native fallback."""
        tree = cls.parse_tree(content, doc_type="pdf", use_docling=use_docling)
        return tree.to_parsed_sections()

    @classmethod
    def parse_markdown(cls, content: str) -> list[ParsedSection]:
        """Parse markdown into ParsedSections."""
        tree = cls.parse_tree(content, doc_type="markdown", use_docling=False)
        return tree.to_parsed_sections()

    @classmethod
    def parse_code(cls, content: str, doc_type: str = "code") -> list[ParsedSection]:
        """Parse code into ParsedSections."""
        tree = cls._parse_code_tree(content, doc_type=doc_type)
        return tree.to_parsed_sections()

    @classmethod
    def parse_text(cls, content: str) -> list[ParsedSection]:
        """Parse plain text into ParsedSections."""
        tree = cls._parse_text_tree_native(content)
        return tree.to_parsed_sections()

    @classmethod
    def parse_html_fallback(cls, content: str) -> list[ParsedSection]:
        """Parse HTML into ParsedSections."""
        clean_md = cls._html_to_markdown_fallback(content)
        return cls.parse_markdown(clean_md)

    # -----------------------------------------------------------------------
    # Docling Primary Extraction
    # -----------------------------------------------------------------------

    @classmethod
    def _parse_with_docling(
        cls,
        content: str | bytes,
        suffix: str = ".pdf",
        source_uri: str | None = None,
        title: str = "",
    ) -> DocumentTree | None:
        """Parses document via IBM Docling into a rich, normalized DocumentTree."""
        import logging

        for name in ("RapidOCR", "rapidocr", "docling", "onnxruntime"):
            logging.getLogger(name).setLevel(logging.ERROR)

        from docling.document_converter import DocumentConverter

        tmp_path = None
        try:
            if isinstance(content, str) and os.path.exists(content):
                file_to_convert = content
                if not source_uri:
                    source_uri = os.path.basename(content)
            else:
                raw_bytes = (
                    content
                    if isinstance(content, bytes)
                    else content.encode("utf-8", errors="replace")
                )
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = tmp.name
                file_to_convert = tmp_path

            converter = DocumentConverter()
            result = converter.convert(file_to_convert)
            docling_doc = result.document

            doc_id = str(uuid.uuid4())
            doc_title = title or getattr(docling_doc, "name", "") or "Document"

            root_node = DocumentNode(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                parent_id=None,
                node_type=DocumentElementType.DOCUMENT,
                reading_order=0,
                text=doc_title,
                raw_text=doc_title,
                section_path=doc_title,
                provenance=Provenance(source_uri=source_uri),
                metadata={"parser": "ibm_docling", "format": suffix.lstrip(".")},
            )

            nodes: list[DocumentNode] = [root_node]
            heading_stack: list[tuple[int, DocumentNode]] = []
            reading_order = 1
            last_fig_or_table_node: DocumentNode | None = None

            for item, level in docling_doc.iterate_items():
                label_str = getattr(item, "label", "text")
                label_val = label_str.value if hasattr(label_str, "value") else str(label_str)

                # 1. Extract provenance (page_no, bbox, charspan)
                prov_item = (
                    item.prov[0] if getattr(item, "prov", None) and len(item.prov) > 0 else None
                )
                page_no = getattr(prov_item, "page_no", None) if prov_item else None
                bbox_tuple = None
                if prov_item and getattr(prov_item, "bbox", None):
                    b = prov_item.bbox
                    bbox_tuple = getattr(b, "as_tuple", lambda: (b.l, b.t, b.r, b.b))()

                char_span_tuple = getattr(prov_item, "charspan", None) if prov_item else None
                raw_ref = getattr(item, "self_ref", None)

                provenance = Provenance(
                    source_uri=source_uri,
                    page_number=page_no,
                    bbox=bbox_tuple,
                    char_span=char_span_tuple,
                    raw_ref=raw_ref,
                )

                # Determine parent heading and current section path
                parent_id = heading_stack[-1][1].id if heading_stack else root_node.id
                section_path = (
                    " > ".join([h[1].text for h in heading_stack]) if heading_stack else doc_title
                )

                # Process by element type
                if label_val in ("title", "section_header"):
                    item_text = getattr(item, "text", "") or ""
                    cleaned_text = TextCleaner.clean(item_text)
                    if not cleaned_text:
                        continue

                    h_level = 1 if label_val == "title" else (level or 2)
                    while heading_stack and heading_stack[-1][0] >= h_level:
                        heading_stack.pop()

                    curr_parent = heading_stack[-1][1].id if heading_stack else root_node.id
                    h_path = (
                        " > ".join([h[1].text for h in heading_stack] + [cleaned_text])
                        if heading_stack
                        else f"{doc_title} > {cleaned_text}"
                    )

                    h_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=curr_parent,
                        node_type=DocumentElementType.TITLE
                        if label_val == "title"
                        else DocumentElementType.HEADING,
                        reading_order=reading_order,
                        text=cleaned_text,
                        raw_text=item_text,
                        section_path=h_path,
                        provenance=provenance,
                        metadata={"level": h_level, "label": label_val},
                    )
                    nodes.append(h_node)
                    heading_stack.append((h_level, h_node))
                    reading_order += 1
                    last_fig_or_table_node = None

                elif label_val == "table":
                    table_data = cls._extract_docling_table(item, docling_doc)
                    table_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=parent_id,
                        node_type=DocumentElementType.TABLE,
                        reading_order=reading_order,
                        text=table_data.markdown if table_data else "",
                        raw_text=getattr(item, "text", "")
                        or (table_data.markdown if table_data else ""),
                        section_path=section_path,
                        provenance=provenance,
                        table_data=table_data,
                        metadata={"label": label_val},
                    )
                    nodes.append(table_node)
                    last_fig_or_table_node = table_node
                    reading_order += 1

                elif label_val in ("picture", "chart"):
                    is_chart = (
                        label_val == "chart" or "chart" in str(getattr(item, "label", "")).lower()
                    )
                    fig_data = cls._extract_docling_figure(item, is_chart=is_chart)
                    fig_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=parent_id,
                        node_type=DocumentElementType.CHART
                        if is_chart
                        else DocumentElementType.FIGURE,
                        reading_order=reading_order,
                        text=fig_data.caption or (f"[{label_val.capitalize()}]"),
                        raw_text=getattr(item, "text", "") or "",
                        section_path=section_path,
                        provenance=provenance,
                        figure_data=fig_data,
                        metadata={"label": label_val},
                    )
                    nodes.append(fig_node)
                    last_fig_or_table_node = fig_node
                    reading_order += 1

                elif label_val == "caption":
                    item_text = getattr(item, "text", "") or ""
                    cleaned_text = TextCleaner.clean(item_text)
                    if last_fig_or_table_node:
                        if (
                            last_fig_or_table_node.table_data
                            and not last_fig_or_table_node.table_data.caption
                        ):
                            last_fig_or_table_node.table_data.caption = cleaned_text
                        elif (
                            last_fig_or_table_node.figure_data
                            and not last_fig_or_table_node.figure_data.caption
                        ):
                            last_fig_or_table_node.figure_data.caption = cleaned_text

                    cap_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=parent_id,
                        node_type=DocumentElementType.CAPTION,
                        reading_order=reading_order,
                        text=cleaned_text,
                        raw_text=item_text,
                        section_path=section_path,
                        provenance=provenance,
                        metadata={"label": label_val},
                    )
                    nodes.append(cap_node)
                    reading_order += 1

                elif label_val == "footnote":
                    item_text = getattr(item, "text", "") or ""
                    cleaned_text = TextCleaner.clean(item_text)
                    fn_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=parent_id,
                        node_type=DocumentElementType.FOOTNOTE,
                        reading_order=reading_order,
                        text=cleaned_text,
                        raw_text=item_text,
                        section_path=section_path,
                        provenance=provenance,
                        metadata={"label": label_val},
                    )
                    nodes.append(fn_node)
                    reading_order += 1

                elif label_val == "list_item":
                    item_text = getattr(item, "text", "") or ""
                    cleaned_text = TextCleaner.clean(item_text)
                    li_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=parent_id,
                        node_type=DocumentElementType.LIST_ITEM,
                        reading_order=reading_order,
                        text=f"- {cleaned_text}",
                        raw_text=item_text,
                        section_path=section_path,
                        provenance=provenance,
                        metadata={"label": label_val},
                    )
                    nodes.append(li_node)
                    reading_order += 1

                elif label_val == "code":
                    item_text = getattr(item, "text", "") or ""
                    code_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=parent_id,
                        node_type=DocumentElementType.CODE,
                        reading_order=reading_order,
                        text=item_text,
                        raw_text=item_text,
                        section_path=section_path,
                        provenance=provenance,
                        metadata={"label": label_val},
                    )
                    nodes.append(code_node)
                    reading_order += 1

                else:
                    item_text = getattr(item, "text", "") or ""
                    cleaned_text = TextCleaner.clean(item_text)
                    if not cleaned_text:
                        continue

                    # If text starts with Figure or Table, attach as caption to prior figure/table
                    if last_fig_or_table_node and cleaned_text.lower().startswith(
                        ("figure", "fig.", "table")
                    ):
                        if (
                            last_fig_or_table_node.table_data
                            and not last_fig_or_table_node.table_data.caption
                        ):
                            last_fig_or_table_node.table_data.caption = cleaned_text
                        elif (
                            last_fig_or_table_node.figure_data
                            and not last_fig_or_table_node.figure_data.caption
                        ):
                            last_fig_or_table_node.figure_data.caption = cleaned_text

                    p_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=parent_id,
                        node_type=DocumentElementType.PARAGRAPH,
                        reading_order=reading_order,
                        text=cleaned_text,
                        raw_text=item_text,
                        section_path=section_path,
                        provenance=provenance,
                        metadata={"label": label_val},
                    )
                    nodes.append(p_node)
                    reading_order += 1

            # Populate children_ids
            node_map = {n.id: n for n in nodes}
            for n in nodes:
                if n.parent_id and n.parent_id in node_map:
                    node_map[n.parent_id].children_ids.append(n.id)

            return DocumentTree(
                document_id=doc_id,
                title=doc_title,
                source_uri=source_uri,
                doc_type=suffix.lstrip("."),
                nodes=nodes,
                root_node_id=root_node.id,
                raw_metadata={
                    "parser": "ibm_docling",
                    "format": suffix.lstrip("."),
                    "num_pages": docling_doc.num_pages(),
                },
            )

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @classmethod
    def _extract_docling_table(cls, item: Any, doc: Any) -> TableDataModel:
        """Extracts structured TableDataModel from Docling TableItem."""
        caption_text = None
        if getattr(item, "captions", None):
            cap_parts = [getattr(c, "text", str(c)) for c in item.captions]
            caption_text = " ".join(cap_parts).strip() or None

        md_content = ""
        try:
            md_content = item.export_to_markdown(doc=doc)
        except Exception:
            try:
                md_content = item.export_to_markdown()
            except Exception:
                md_content = ""

        html_content = None
        try:
            html_content = item.export_to_html(doc=doc)
        except Exception:
            pass

        cells: list[TableCellData] = []
        headers: list[str] = []
        num_rows = 0
        num_cols = 0
        rows: list[list[str]] = []

        if getattr(item, "data", None):
            data = item.data
            num_rows = getattr(data, "num_rows", 0)
            num_cols = getattr(data, "num_cols", 0)

            grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]

            for c in getattr(data, "table_cells", []):
                r = getattr(c, "start_row_offset_idx", 0)
                col = getattr(c, "start_col_offset_idx", 0)
                txt = getattr(c, "text", "") or ""
                is_hdr = getattr(c, "column_header", False)
                row_span = getattr(c, "row_span", 1)
                col_span = getattr(c, "col_span", 1)

                bbox_tup = None
                if getattr(c, "bbox", None):
                    b = c.bbox
                    bbox_tup = getattr(b, "as_tuple", lambda: (b.l, b.t, b.r, b.b))()

                cell_data = TableCellData(
                    row_idx=r,
                    col_idx=col,
                    row_span=row_span,
                    col_span=col_span,
                    text=txt,
                    is_header=is_hdr,
                    bbox=bbox_tup,
                )
                cells.append(cell_data)

                if r < num_rows and col < num_cols:
                    grid[r][col] = txt

                if is_hdr and txt not in headers:
                    headers.append(txt)

            rows = grid

        # Fallback markdown generation if export_to_markdown failed
        if not md_content and rows:
            header_line = "| " + " | ".join(headers or rows[0]) + " |"
            sep_line = "| " + " | ".join(["---"] * len(headers or rows[0])) + " |"
            data_lines = ["| " + " | ".join(r) + " |" for r in rows[1:]]
            md_content = "\n".join([header_line, sep_line] + data_lines)

        page_numbers: list[int] = []
        if getattr(item, "prov", None):
            for p in item.prov:
                p_no = getattr(p, "page_no", None)
                if p_no and p_no not in page_numbers:
                    page_numbers.append(p_no)

        return TableDataModel(
            num_rows=num_rows,
            num_cols=num_cols,
            headers=headers,
            rows=rows,
            cells=cells,
            caption=caption_text,
            markdown=md_content,
            html=html_content,
            page_numbers=page_numbers,
        )

    @classmethod
    def _extract_docling_figure(cls, item: Any, is_chart: bool = False) -> FigureDataModel:
        """Extracts FigureDataModel from Docling PictureItem or ChartItem."""
        caption_text = None
        if getattr(item, "captions", None):
            cap_parts = [getattr(c, "text", str(c)) for c in item.captions]
            caption_text = " ".join(cap_parts).strip() or None

        asset_id = getattr(item, "self_ref", None) or str(uuid.uuid4())
        return FigureDataModel(
            asset_id=asset_id,
            caption=caption_text,
            figure_type="chart" if is_chart else "image",
            chart_metadata={},
            ocr_text=getattr(item, "text", None),
        )

    # -----------------------------------------------------------------------
    # Native Resilient Fallbacks
    # -----------------------------------------------------------------------

    @classmethod
    def _parse_pdf_tree_native(
        cls,
        content: str | bytes,
        source_uri: str | None = None,
        title: str = "",
    ) -> DocumentTree:
        """Streaming page-by-page PDF parser using pypdf, with repeated header/footer suppression."""
        import pypdf

        if isinstance(content, str):
            if os.path.exists(content):
                stream: io.BufferedReader | io.BytesIO = open(content, "rb")
                if not source_uri:
                    source_uri = os.path.basename(content)
            else:
                stream = io.BytesIO(content.encode("latin-1"))
        else:
            stream = io.BytesIO(content)

        doc_id = str(uuid.uuid4())
        doc_title = title or (source_uri or "PDF Document")

        root_node = DocumentNode(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            parent_id=None,
            node_type=DocumentElementType.DOCUMENT,
            reading_order=0,
            text=doc_title,
            raw_text=doc_title,
            section_path=doc_title,
            provenance=Provenance(source_uri=source_uri),
            metadata={"parser": "pypdf_fallback"},
        )

        nodes: list[DocumentNode] = [root_node]
        reading_order = 1

        try:
            reader = pypdf.PdfReader(stream)
            total_pages = len(reader.pages)

            raw_pages = []
            for page_idx in range(total_pages):
                page = reader.pages[page_idx]
                page_text = (page.extract_text() or "").strip()
                raw_pages.append({"page_number": page_idx + 1, "text": page_text})

            # Suppress repeated headers and footers across pages
            cleaned_pages = TextCleaner.suppress_headers_footers(raw_pages)

            for p_info in cleaned_pages:
                page_num = p_info["page_number"]
                page_text = p_info["text"]
                if not page_text:
                    continue

                # Create Page Section Node
                page_section_node = DocumentNode(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    parent_id=root_node.id,
                    node_type=DocumentElementType.SECTION,
                    reading_order=reading_order,
                    text=f"Page {page_num}",
                    raw_text=f"Page {page_num}",
                    section_path=f"{doc_title} > Page {page_num}",
                    provenance=Provenance(source_uri=source_uri, page_number=page_num),
                    metadata={"page": page_num, "total_pages": total_pages},
                )
                nodes.append(page_section_node)
                root_node.children_ids.append(page_section_node.id)
                reading_order += 1

                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
                for para in paragraphs:
                    cleaned_para = TextCleaner.clean(para)
                    if not cleaned_para:
                        continue

                    # Detect simple markdown-like or ascii tables in text
                    is_table = "|" in cleaned_para and cleaned_para.count("\n") >= 2
                    n_type = (
                        DocumentElementType.TABLE if is_table else DocumentElementType.PARAGRAPH
                    )

                    table_data = None
                    if is_table:
                        lines = [ln.strip() for ln in cleaned_para.splitlines() if ln.strip()]
                        headers = [col.strip() for col in lines[0].split("|") if col.strip()]
                        table_data = TableDataModel(
                            num_rows=len(lines),
                            num_cols=len(headers),
                            headers=headers,
                            markdown=cleaned_para,
                            page_numbers=[page_num],
                        )

                    p_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=page_section_node.id,
                        node_type=n_type,
                        reading_order=reading_order,
                        text=cleaned_para,
                        raw_text=para,
                        section_path=page_section_node.section_path,
                        provenance=Provenance(source_uri=source_uri, page_number=page_num),
                        table_data=table_data,
                        metadata={"page": page_num},
                    )
                    nodes.append(p_node)
                    page_section_node.children_ids.append(p_node.id)
                    reading_order += 1

        finally:
            if hasattr(stream, "close"):
                stream.close()

        if len(nodes) == 1:
            empty_node = DocumentNode(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                parent_id=root_node.id,
                node_type=DocumentElementType.PARAGRAPH,
                reading_order=1,
                text="(Empty PDF document)",
                raw_text="",
                section_path=doc_title,
                provenance=Provenance(source_uri=source_uri, page_number=1),
            )
            nodes.append(empty_node)
            root_node.children_ids.append(empty_node.id)

        return DocumentTree(
            document_id=doc_id,
            title=doc_title,
            source_uri=source_uri,
            doc_type="pdf",
            nodes=nodes,
            root_node_id=root_node.id,
        )

    @classmethod
    def _parse_markdown_tree_native(
        cls,
        content: str,
        source_uri: str | None = None,
        title: str = "",
    ) -> DocumentTree:
        """Native structure-aware markdown parser building a DocumentTree."""
        doc_id = str(uuid.uuid4())
        doc_title = title or "Markdown Document"

        root_node = DocumentNode(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            parent_id=None,
            node_type=DocumentElementType.DOCUMENT,
            reading_order=0,
            text=doc_title,
            raw_text=doc_title,
            section_path=doc_title,
            provenance=Provenance(source_uri=source_uri),
            metadata={"parser": "native_markdown"},
        )

        nodes: list[DocumentNode] = [root_node]
        heading_stack: list[tuple[int, DocumentNode]] = []
        reading_order = 1

        heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
        lines = content.splitlines()

        current_para_lines: list[str] = []
        in_code_block = False
        code_block_lines: list[str] = []

        def flush_paragraph() -> None:
            nonlocal reading_order, current_para_lines
            if not current_para_lines:
                return
            raw_p = "\n".join(current_para_lines).strip()
            current_para_lines = []
            if not raw_p:
                return

            cleaned_p = TextCleaner.clean(raw_p)
            parent_id = heading_stack[-1][1].id if heading_stack else root_node.id
            sec_path = (
                " > ".join([h[1].text for h in heading_stack]) if heading_stack else doc_title
            )

            # Check if paragraph is a table
            if "|" in raw_p and raw_p.count("\n") >= 1:
                t_lines = [
                    line_item.strip() for line_item in raw_p.splitlines() if line_item.strip()
                ]
                hdrs = [c.strip() for c in t_lines[0].split("|") if c.strip()]
                t_data = TableDataModel(
                    num_rows=len(t_lines),
                    num_cols=len(hdrs),
                    headers=hdrs,
                    markdown=raw_p,
                )
                node = DocumentNode(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    parent_id=parent_id,
                    node_type=DocumentElementType.TABLE,
                    reading_order=reading_order,
                    text=raw_p,
                    raw_text=raw_p,
                    section_path=sec_path,
                    provenance=Provenance(source_uri=source_uri),
                    table_data=t_data,
                )
            elif raw_p.startswith("!"):
                # Figure in markdown: ![caption](url)
                m = re.match(r"!\[(.*?)\]\((.*?)\)", raw_p)
                cap = m.group(1) if m else raw_p
                asset_id = m.group(2) if m else None
                f_data = FigureDataModel(asset_id=asset_id, caption=cap, figure_type="image")
                node = DocumentNode(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    parent_id=parent_id,
                    node_type=DocumentElementType.FIGURE,
                    reading_order=reading_order,
                    text=cap or raw_p,
                    raw_text=raw_p,
                    section_path=sec_path,
                    provenance=Provenance(source_uri=source_uri),
                    figure_data=f_data,
                )
            else:
                node = DocumentNode(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    parent_id=parent_id,
                    node_type=DocumentElementType.PARAGRAPH,
                    reading_order=reading_order,
                    text=cleaned_p,
                    raw_text=raw_p,
                    section_path=sec_path,
                    provenance=Provenance(source_uri=source_uri),
                )

            nodes.append(node)
            reading_order += 1

        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    code_block_lines.append(line)
                    code_text = "\n".join(code_block_lines)
                    parent_id = heading_stack[-1][1].id if heading_stack else root_node.id
                    sec_path = (
                        " > ".join([h[1].text for h in heading_stack])
                        if heading_stack
                        else doc_title
                    )
                    c_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=parent_id,
                        node_type=DocumentElementType.CODE,
                        reading_order=reading_order,
                        text=code_text,
                        raw_text=code_text,
                        section_path=sec_path,
                        provenance=Provenance(source_uri=source_uri),
                    )
                    nodes.append(c_node)
                    reading_order += 1
                    in_code_block = False
                    code_block_lines = []
                else:
                    flush_paragraph()
                    in_code_block = True
                    code_block_lines = [line]
                continue

            if in_code_block:
                code_block_lines.append(line)
                continue

            h_match = heading_re.match(line)
            if h_match:
                flush_paragraph()
                h_level = len(h_match.group(1))
                h_text = h_match.group(2).strip()

                while heading_stack and heading_stack[-1][0] >= h_level:
                    heading_stack.pop()

                curr_parent = heading_stack[-1][1].id if heading_stack else root_node.id
                h_path = (
                    " > ".join([h[1].text for h in heading_stack] + [h_text])
                    if heading_stack
                    else f"{doc_title} > {h_text}"
                )

                h_node = DocumentNode(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    parent_id=curr_parent,
                    node_type=DocumentElementType.HEADING,
                    reading_order=reading_order,
                    text=h_text,
                    raw_text=line,
                    section_path=h_path,
                    provenance=Provenance(source_uri=source_uri),
                    metadata={"level": h_level},
                )
                nodes.append(h_node)
                heading_stack.append((h_level, h_node))
                reading_order += 1
            elif not line.strip():
                flush_paragraph()
            else:
                current_para_lines.append(line)

        flush_paragraph()

        # Link children_ids
        node_map = {n.id: n for n in nodes}
        for n in nodes:
            if n.parent_id and n.parent_id in node_map:
                node_map[n.parent_id].children_ids.append(n.id)

        return DocumentTree(
            document_id=doc_id,
            title=doc_title,
            source_uri=source_uri,
            doc_type="markdown",
            nodes=nodes,
            root_node_id=root_node.id,
        )

    # -----------------------------------------------------------------------
    # Syntax-Aware Code Parser Path
    # -----------------------------------------------------------------------

    @classmethod
    def _parse_code_tree(
        cls,
        content: str,
        doc_type: str = "code",
        source_uri: str | None = None,
        title: str = "",
    ) -> DocumentTree:
        """
        Syntax-aware parser for source code (Python AST and language block parser).
        Never passes code through Docling.
        """
        doc_id = str(uuid.uuid4())
        doc_title = title or (source_uri or "Source Code")

        root_node = DocumentNode(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            parent_id=None,
            node_type=DocumentElementType.DOCUMENT,
            reading_order=0,
            text=doc_title,
            raw_text=doc_title,
            section_path=doc_title,
            provenance=Provenance(source_uri=source_uri),
            metadata={"parser": "code_ast", "language": doc_type},
        )

        nodes: list[DocumentNode] = [root_node]
        reading_order = 1

        # Python AST parsing
        if (
            doc_type in ("python", "py")
            or (source_uri and source_uri.endswith(".py"))
            or "def " in content
            or "class " in content
        ):
            import textwrap

            clean_code = textwrap.dedent(content)
            try:
                tree = ast.parse(clean_code)
                lines = clean_code.splitlines()

                # Check module docstring
                module_doc = ast.get_docstring(tree)
                if module_doc:
                    doc_node = DocumentNode(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        parent_id=root_node.id,
                        node_type=DocumentElementType.PARAGRAPH,
                        reading_order=reading_order,
                        text=module_doc,
                        raw_text=module_doc,
                        section_path=f"{doc_title} > module_docstring",
                        provenance=Provenance(source_uri=source_uri, page_number=1),
                        metadata={"kind": "module_docstring"},
                    )
                    nodes.append(doc_node)
                    root_node.children_ids.append(doc_node.id)
                    reading_order += 1

                for item in tree.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start_line = item.lineno - 1
                        end_line = getattr(item, "end_lineno", len(lines))
                        code_segment = "\n".join(lines[start_line:end_line])
                        is_class = isinstance(item, ast.ClassDef)
                        kind = "class" if is_class else "function"

                        item_node = DocumentNode(
                            id=str(uuid.uuid4()),
                            document_id=doc_id,
                            parent_id=root_node.id,
                            node_type=DocumentElementType.CODE,
                            reading_order=reading_order,
                            text=code_segment,
                            raw_text=code_segment,
                            section_path=f"{doc_title} > {kind} {item.name}",
                            provenance=Provenance(source_uri=source_uri, page_number=1),
                            metadata={
                                "symbol": item.name,
                                "kind": kind,
                                "start_line": start_line + 1,
                                "end_line": end_line,
                            },
                        )
                        nodes.append(item_node)
                        root_node.children_ids.append(item_node.id)
                        reading_order += 1

                        # If class, parse inner methods as children
                        if is_class and isinstance(item, ast.ClassDef):
                            for sub in item.body:
                                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    sub_start = sub.lineno - 1
                                    sub_end = getattr(sub, "end_lineno", len(lines))
                                    method_code = "\n".join(lines[sub_start:sub_end])
                                    m_node = DocumentNode(
                                        id=str(uuid.uuid4()),
                                        document_id=doc_id,
                                        parent_id=item_node.id,
                                        node_type=DocumentElementType.CODE,
                                        reading_order=reading_order,
                                        text=method_code,
                                        raw_text=method_code,
                                        section_path=f"{doc_title} > class {item.name} > method {sub.name}",
                                        provenance=Provenance(source_uri=source_uri, page_number=1),
                                        metadata={
                                            "symbol": sub.name,
                                            "class_name": item.name,
                                            "kind": "method",
                                            "start_line": sub_start + 1,
                                            "end_line": sub_end,
                                        },
                                    )
                                    nodes.append(m_node)
                                    item_node.children_ids.append(m_node.id)
                                    reading_order += 1

                if len(nodes) > 1:
                    return DocumentTree(
                        document_id=doc_id,
                        title=doc_title,
                        source_uri=source_uri,
                        doc_type="code",
                        nodes=nodes,
                        root_node_id=root_node.id,
                    )
            except Exception:
                pass  # Fall through to block regex parsing

        # Fallback block parsing for code (JS/TS/Go/Java/C++)
        blocks = re.split(
            r"\n(?=(?:def |class |function |export |public |private |func |fn ))", content
        )
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            first_line = block.strip().splitlines()[0][:60]
            b_node = DocumentNode(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                parent_id=root_node.id,
                node_type=DocumentElementType.CODE,
                reading_order=reading_order,
                text=block.strip(),
                raw_text=block.strip(),
                section_path=f"{doc_title} > block_{i + 1}",
                provenance=Provenance(source_uri=source_uri),
                metadata={"block_index": i + 1, "first_line": first_line},
            )
            nodes.append(b_node)
            root_node.children_ids.append(b_node.id)
            reading_order += 1

        if len(nodes) == 1:
            raw_node = DocumentNode(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                parent_id=root_node.id,
                node_type=DocumentElementType.CODE,
                reading_order=1,
                text=content,
                raw_text=content,
                section_path=doc_title,
                provenance=Provenance(source_uri=source_uri),
            )
            nodes.append(raw_node)
            root_node.children_ids.append(raw_node.id)

        return DocumentTree(
            document_id=doc_id,
            title=doc_title,
            source_uri=source_uri,
            doc_type="code",
            nodes=nodes,
            root_node_id=root_node.id,
        )

    @classmethod
    def _parse_text_tree_native(
        cls,
        content: str,
        source_uri: str | None = None,
        title: str = "",
    ) -> DocumentTree:
        """Plain text parser creating paragraph-bounded DocumentTree."""
        doc_id = str(uuid.uuid4())
        doc_title = title or (source_uri or "Text Document")

        root_node = DocumentNode(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            parent_id=None,
            node_type=DocumentElementType.DOCUMENT,
            reading_order=0,
            text=doc_title,
            raw_text=doc_title,
            section_path=doc_title,
            provenance=Provenance(source_uri=source_uri),
        )

        nodes: list[DocumentNode] = [root_node]
        reading_order = 1

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        for p in paragraphs:
            cleaned = TextCleaner.clean(p)
            if not cleaned:
                continue
            node = DocumentNode(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                parent_id=root_node.id,
                node_type=DocumentElementType.PARAGRAPH,
                reading_order=reading_order,
                text=cleaned,
                raw_text=p,
                section_path=doc_title,
                provenance=Provenance(source_uri=source_uri),
            )
            nodes.append(node)
            root_node.children_ids.append(node.id)
            reading_order += 1

        return DocumentTree(
            document_id=doc_id,
            title=doc_title,
            source_uri=source_uri,
            doc_type="text",
            nodes=nodes,
            root_node_id=root_node.id,
        )

    @classmethod
    def _html_to_markdown_fallback(cls, content: str) -> str:
        """Fallback HTML cleaner."""
        clean_text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", content, flags=re.DOTALL)
        clean_text = re.sub(
            r"<h([1-6])[^>]*>(.*?)</h\1>", r"\n# \2\n", clean_text, flags=re.IGNORECASE
        )
        clean_text = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\1\n", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"<[^>]+>", " ", clean_text)
        return clean_text.strip()
