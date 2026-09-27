from __future__ import annotations

import logging
import re

from pylatexenc.latex2text import LatexNodes2Text  # type: ignore[import-untyped]
from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexMacroNode,
    LatexMathNode,
    LatexNode,
    LatexWalker,
)

from deep_context.core.types import (
    DocumentElementType,
    DocumentNode,
    DocumentTree,
    EquationDataModel,
    FigureDataModel,
    Provenance,
    TableDataModel,
    generate_stable_doc_id,
    generate_stable_node_id,
)

logger = logging.getLogger(__name__)

MATH_ENVIRONMENTS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "eqnarray",
    "eqnarray*",
    "displaymath",
    "split",
    "math",
}

SECTION_MACROS = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}

FORMATTING_MACROS = {
    "\\frac",
    "\\left",
    "\\right",
    "\\begin",
    "\\end",
    "\\text",
    "\\mathrm",
    "\\mathbf",
    "\\mathit",
    "\\quad",
    "\\qquad",
    "\\limits",
    "\\nolimits",
    "\\displaystyle",
    "\\style",
    "\\label",
    "\\tag",
    "\\nonumber",
}


class LaTeXParser:
    """
    Native LaTeX document parser using pylatexenc.
    Preserves exact equations, symbols, formulas, section hierarchies, tables,
    and complete provenance.
    """

    @classmethod
    def _extract_line(cls, source: str, pos: int) -> int:
        if pos <= 0:
            return 1
        return source[:pos].count("\n") + 1

    @classmethod
    def _extract_symbols(cls, latex_str: str) -> list[str]:
        """Extract Greek letters, math operators, and identifier variables."""
        macros = re.findall(r"\\[a-zA-Z]+", latex_str)
        math_macros = [m for m in macros if m not in FORMATTING_MACROS]
        cleaned = re.sub(r"\\[a-zA-Z]+", " ", latex_str)
        single_chars = re.findall(r"\b[a-zA-Z]\b", cleaned)
        combined = math_macros + single_chars
        # Deduplicate preserving order
        return list(dict.fromkeys(combined))

    @classmethod
    def _parse_symbolic(cls, latex_str: str) -> str | None:
        """Attempt SymPy parsing of LaTeX math if antlr runtime is present."""
        try:
            from sympy.parsing.latex import parse_latex  # type: ignore[import-untyped]

            expr = parse_latex(latex_str)
            return str(expr)
        except Exception:
            return None

    @classmethod
    def _macro_arg_text(cls, node: LatexMacroNode) -> str:
        if not node.nodeargd or not node.nodeargd.argnlist:
            return ""
        valid_args = [a for a in node.nodeargd.argnlist if a is not None]
        if not valid_args:
            return ""
        arg = valid_args[-1]
        if hasattr(arg, "nodelist") and arg.nodelist:
            try:
                return LatexNodes2Text().nodelist_to_text(arg.nodelist).strip()
            except Exception:
                pass
        return arg.latex_verbatim().strip("{} \t\r\n")

    @classmethod
    def _parse_tabular_env(
        cls, env_node: LatexEnvironmentNode, source: str
    ) -> TableDataModel | None:
        raw_text = env_node.latex_verbatim()
        # Find inner tabular content
        body_match = re.search(
            r"\\begin\{tabular\}\s*(\[[^\]]*\])?\s*\{[^}]*\}(.*?)\\end\{tabular\}",
            raw_text,
            re.DOTALL,
        )
        if not body_match:
            return None
        body = body_match.group(2)
        converter = LatexNodes2Text()

        # Split on LaTeX newline commands \\ or \tabularnewline
        raw_rows = body.replace(r"\tabularnewline", "\n").replace("\\\\", "\n").split("\n")
        parsed_rows: list[list[str]] = []
        for row in raw_rows:
            cleaned = (
                row.replace(r"\hline", "")
                .replace(r"\toprule", "")
                .replace(r"\midrule", "")
                .replace(r"\bottomrule", "")
                .strip()
            )
            if not cleaned:
                continue
            cols = [converter.latex_to_text(col).strip() for col in cleaned.split("&")]
            parsed_rows.append(cols)

        if not parsed_rows:
            return None

        headers = parsed_rows[0]
        rows = parsed_rows[1:] if len(parsed_rows) > 1 else []

        # Markdown representation
        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for r in rows:
            # Pad or trim to header length
            row_cells = r + [""] * max(0, len(headers) - len(r))
            md_lines.append("| " + " | ".join(row_cells[: len(headers)]) + " |")
        md_table = "\n".join(md_lines)

        return TableDataModel(
            num_rows=len(rows),
            num_cols=len(headers),
            headers=headers,
            rows=rows,
            markdown=md_table,
        )

    @classmethod
    def parse_tree(
        cls,
        content: str,
        source_uri: str | None = None,
        title: str = "",
        tenant_id: str = "default",
    ) -> DocumentTree:
        """
        Builds a canonical DocumentTree from LaTeX source code.
        """
        if "\x00" in content:
            content = content.replace("\x00", "")

        walker = LatexWalker(content)
        top_nodes, _, _ = walker.get_latex_nodes()
        converter = LatexNodes2Text()

        # 1. Extract Document Title
        doc_title = title
        if not doc_title:
            for n in top_nodes:
                if isinstance(n, LatexMacroNode) and n.macroname == "title":
                    doc_title = cls._macro_arg_text(n)
                    break
        if not doc_title:
            doc_title = (source_uri.split("/")[-1] if source_uri else None) or "LaTeX Document"

        doc_id = generate_stable_doc_id(
            tenant_id=tenant_id,
            title=doc_title,
            source_uri=source_uri,
            content=content,
        )

        root_node = DocumentNode(
            id=generate_stable_node_id(doc_id, "root", 0),
            document_id=doc_id,
            parent_id=None,
            node_type=DocumentElementType.DOCUMENT,
            reading_order=0,
            text=doc_title,
            raw_text=doc_title,
            section_path=doc_title,
            provenance=Provenance(
                source_uri=source_uri,
                page_number=1,
                line_range=(1, cls._extract_line(content, len(content))),
                parser="LaTeXParser",
                parser_version="1.0.0",
                extraction_method="pylatexenc_ast",
            ),
        )

        nodes: list[DocumentNode] = [root_node]
        reading_order = 1
        equation_counter = 1

        # Section tracking stack: list of (level, title, node_id)
        current_sections: list[tuple[int, str, str]] = []

        def get_current_parent_and_path() -> tuple[str, str]:
            if not current_sections:
                return root_node.id, doc_title
            path = doc_title + " > " + " > ".join(s[1] for s in current_sections)
            return current_sections[-1][2], path

        # Flatten or unpack \begin{document} if present
        def unpack_nodes(nlist: list[LatexNode]) -> list[LatexNode]:
            unpacked: list[LatexNode] = []
            for n in nlist:
                if isinstance(n, LatexEnvironmentNode) and n.environmentname == "document":
                    unpacked.extend(n.nodelist)
                else:
                    unpacked.append(n)
            return unpacked

        flat_nodes = unpack_nodes(top_nodes)

        # Buffer for continuous paragraph text
        text_buffer: list[str] = []
        text_start_pos = 0
        text_end_pos = 0

        def flush_text_buffer() -> None:
            nonlocal reading_order, text_buffer, text_start_pos, text_end_pos
            if not text_buffer:
                return
            combined_raw = "".join(text_buffer)
            text_buffer = []
            try:
                converted = converter.latex_to_text(combined_raw).strip()
            except Exception:
                converted = combined_raw.strip()
            # Split into paragraphs if multiple blank lines
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", converted) if p.strip()]
            for p in paragraphs:
                if not p:
                    continue
                parent_id, sec_path = get_current_parent_and_path()
                s_line = cls._extract_line(content, text_start_pos)
                e_line = cls._extract_line(content, text_end_pos)
                p_node = DocumentNode(
                    id=generate_stable_node_id(doc_id, "paragraph", reading_order, p[:80]),
                    document_id=doc_id,
                    parent_id=parent_id,
                    node_type=DocumentElementType.PARAGRAPH,
                    reading_order=reading_order,
                    text=p,
                    raw_text=p,
                    section_path=sec_path,
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(s_line, e_line),
                        char_span=(text_start_pos, text_end_pos),
                        parser="LaTeXParser",
                        parser_version="1.0.0",
                        extraction_method="pylatexenc_ast",
                    ),
                )
                nodes.append(p_node)
                reading_order += 1

        for node in flat_nodes:
            # 2. Sectioning macros
            if isinstance(node, LatexMacroNode) and node.macroname in SECTION_MACROS:
                flush_text_buffer()
                sec_level = SECTION_MACROS[node.macroname]
                sec_title = cls._macro_arg_text(node) or node.macroname.capitalize()

                # Adjust section stack
                while current_sections and current_sections[-1][0] >= sec_level:
                    current_sections.pop()

                parent_id = current_sections[-1][2] if current_sections else root_node.id
                sec_start_line = cls._extract_line(content, node.pos)
                sec_end_line = cls._extract_line(content, node.pos + node.len)

                path_str = (
                    doc_title
                    + " > "
                    + " > ".join(s[1] for s in current_sections)
                    + f" > {sec_title}"
                    if current_sections
                    else f"{doc_title} > {sec_title}"
                )

                sec_node = DocumentNode(
                    id=generate_stable_node_id(doc_id, "section", reading_order, sec_title),
                    document_id=doc_id,
                    parent_id=parent_id,
                    node_type=DocumentElementType.HEADING,
                    reading_order=reading_order,
                    text=sec_title,
                    raw_text=node.latex_verbatim(),
                    section_path=path_str,
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(sec_start_line, sec_end_line),
                        char_span=(node.pos, node.pos + node.len),
                        raw_ref=sec_title,
                        parser="LaTeXParser",
                        parser_version="1.0.0",
                        extraction_method="pylatexenc_ast",
                    ),
                    metadata={"heading_level": sec_level},
                )
                nodes.append(sec_node)
                reading_order += 1
                current_sections.append((sec_level, sec_title, sec_node.id))
                continue

            # 3. Mathematical Environments (equation, align, gather, etc.)
            if isinstance(node, LatexEnvironmentNode) and node.environmentname in MATH_ENVIRONMENTS:
                flush_text_buffer()
                parent_id, sec_path = get_current_parent_and_path()
                s_line = cls._extract_line(content, node.pos)
                e_line = cls._extract_line(content, node.pos + node.len)

                # Extract label and math content
                label = None
                math_parts: list[str] = []
                for child in node.nodelist:
                    if isinstance(child, LatexMacroNode) and child.macroname == "label":
                        label = cls._macro_arg_text(child)
                    else:
                        math_parts.append(child.latex_verbatim())

                math_source = "".join(math_parts).strip()
                if not math_source:
                    math_source = node.latex_verbatim().strip()

                try:
                    math_text = converter.latex_to_text(math_source).strip()
                except Exception:
                    math_text = math_source

                eq_num_str = f"({equation_counter})"
                symbols = cls._extract_symbols(math_source)
                sym_repr = cls._parse_symbolic(math_source)

                eq_data = EquationDataModel(
                    latex=math_source,
                    normalized_latex=math_text,
                    equation_number=eq_num_str,
                    is_inline=False,
                    symbolic_repr=sym_repr,
                    variables=symbols,
                    extraction_method="latex_parser",
                    confidence=1.0,
                )

                label_part = f" [{label}]" if label else ""
                full_eq_text = f"Equation {eq_num_str}{label_part}\nLaTeX: {math_source}\nReadable: {math_text}"

                eq_node = DocumentNode(
                    id=generate_stable_node_id(doc_id, "equation", reading_order, math_source),
                    document_id=doc_id,
                    parent_id=parent_id,
                    node_type=DocumentElementType.EQUATION,
                    reading_order=reading_order,
                    text=full_eq_text,
                    raw_text=node.latex_verbatim(),
                    section_path=sec_path,
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(s_line, e_line),
                        char_span=(node.pos, node.pos + node.len),
                        raw_ref=label or eq_num_str,
                        parser="LaTeXParser",
                        parser_version="1.0.0",
                        extraction_method="pylatexenc_ast",
                    ),
                    equation_data=eq_data,
                    metadata={"equation_number": eq_num_str, "label": label},
                )
                nodes.append(eq_node)
                reading_order += 1
                equation_counter += 1
                continue

            # 4. Display Math Node ($$ or \[ \])
            if isinstance(node, LatexMathNode) and node.displaytype == "display":
                flush_text_buffer()
                parent_id, sec_path = get_current_parent_and_path()
                s_line = cls._extract_line(content, node.pos)
                e_line = cls._extract_line(content, node.pos + node.len)

                math_source = "".join(c.latex_verbatim() for c in node.nodelist).strip()
                try:
                    math_text = converter.latex_to_text(math_source).strip()
                except Exception:
                    math_text = math_source

                eq_num_str = f"({equation_counter})"
                symbols = cls._extract_symbols(math_source)
                sym_repr = cls._parse_symbolic(math_source)

                eq_data = EquationDataModel(
                    latex=math_source,
                    normalized_latex=math_text,
                    equation_number=eq_num_str,
                    is_inline=False,
                    symbolic_repr=sym_repr,
                    variables=symbols,
                    extraction_method="latex_parser",
                    confidence=1.0,
                )

                eq_node = DocumentNode(
                    id=generate_stable_node_id(doc_id, "equation", reading_order, math_source),
                    document_id=doc_id,
                    parent_id=parent_id,
                    node_type=DocumentElementType.EQUATION,
                    reading_order=reading_order,
                    text=f"Equation {eq_num_str}\nLaTeX: {math_source}\nReadable: {math_text}",
                    raw_text=node.latex_verbatim(),
                    section_path=sec_path,
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(s_line, e_line),
                        char_span=(node.pos, node.pos + node.len),
                        parser="LaTeXParser",
                        parser_version="1.0.0",
                        extraction_method="pylatexenc_ast",
                    ),
                    equation_data=eq_data,
                )
                nodes.append(eq_node)
                reading_order += 1
                equation_counter += 1
                continue

            # 5. Table & Tabular environments
            if isinstance(node, LatexEnvironmentNode) and node.environmentname in (
                "table",
                "table*",
                "tabular",
            ):
                flush_text_buffer()
                parent_id, sec_path = get_current_parent_and_path()
                s_line = cls._extract_line(content, node.pos)
                e_line = cls._extract_line(content, node.pos + node.len)

                caption = None
                label = None
                tabular_child = None

                for child in node.nodelist:
                    if isinstance(child, LatexMacroNode) and child.macroname == "caption":
                        caption = cls._macro_arg_text(child)
                    elif isinstance(child, LatexMacroNode) and child.macroname == "label":
                        label = cls._macro_arg_text(child)
                    elif (
                        isinstance(child, LatexEnvironmentNode)
                        and child.environmentname == "tabular"
                    ):
                        tabular_child = child

                target_env = tabular_child or node
                t_data = cls._parse_tabular_env(target_env, content)
                if t_data:
                    t_data.caption = caption
                    table_text = (f"Table: {caption}\n" if caption else "") + t_data.markdown
                    t_node = DocumentNode(
                        id=generate_stable_node_id(
                            doc_id, "table", reading_order, label or caption or ""
                        ),
                        document_id=doc_id,
                        parent_id=parent_id,
                        node_type=DocumentElementType.TABLE,
                        reading_order=reading_order,
                        text=table_text,
                        raw_text=node.latex_verbatim(),
                        section_path=sec_path,
                        provenance=Provenance(
                            source_uri=source_uri,
                            page_number=1,
                            line_range=(s_line, e_line),
                            char_span=(node.pos, node.pos + node.len),
                            raw_ref=label or caption,
                            parser="LaTeXParser",
                            parser_version="1.0.0",
                            extraction_method="pylatexenc_ast",
                        ),
                        table_data=t_data,
                        metadata={"label": label, "caption": caption},
                    )
                    nodes.append(t_node)
                    reading_order += 1
                continue

            # 6. Figure environments
            if isinstance(node, LatexEnvironmentNode) and node.environmentname in (
                "figure",
                "figure*",
            ):
                flush_text_buffer()
                parent_id, sec_path = get_current_parent_and_path()
                s_line = cls._extract_line(content, node.pos)
                e_line = cls._extract_line(content, node.pos + node.len)

                caption = None
                label = None
                img_path = None
                for child in node.nodelist:
                    if isinstance(child, LatexMacroNode) and child.macroname == "caption":
                        caption = cls._macro_arg_text(child)
                    elif isinstance(child, LatexMacroNode) and child.macroname == "label":
                        label = cls._macro_arg_text(child)
                    elif isinstance(child, LatexMacroNode) and child.macroname == "includegraphics":
                        img_path = cls._macro_arg_text(child)

                fig_text = f"Figure: {caption or 'Untitled'}" + (f" [{label}]" if label else "")
                fig_data = FigureDataModel(
                    caption=caption,
                    storage_uri=img_path,
                )
                fig_node = DocumentNode(
                    id=generate_stable_node_id(
                        doc_id, "figure", reading_order, label or caption or ""
                    ),
                    document_id=doc_id,
                    parent_id=parent_id,
                    node_type=DocumentElementType.FIGURE,
                    reading_order=reading_order,
                    text=fig_text,
                    raw_text=node.latex_verbatim(),
                    section_path=sec_path,
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(s_line, e_line),
                        char_span=(node.pos, node.pos + node.len),
                        raw_ref=label or caption,
                        parser="LaTeXParser",
                        parser_version="1.0.0",
                        extraction_method="pylatexenc_ast",
                    ),
                    figure_data=fig_data,
                    metadata={"label": label, "caption": caption, "graphic": img_path},
                )
                nodes.append(fig_node)
                reading_order += 1
                continue

            # 7. Abstract environment
            if isinstance(node, LatexEnvironmentNode) and node.environmentname == "abstract":
                flush_text_buffer()
                parent_id, sec_path = get_current_parent_and_path()
                s_line = cls._extract_line(content, node.pos)
                e_line = cls._extract_line(content, node.pos + node.len)
                try:
                    abs_text = converter.nodelist_to_text(node.nodelist).strip()
                except Exception:
                    abs_text = node.latex_verbatim().strip()
                abs_node = DocumentNode(
                    id=generate_stable_node_id(doc_id, "abstract", reading_order, "Abstract"),
                    document_id=doc_id,
                    parent_id=parent_id,
                    node_type=DocumentElementType.PARAGRAPH,
                    reading_order=reading_order,
                    text=f"Abstract: {abs_text}",
                    raw_text=node.latex_verbatim(),
                    section_path=f"{doc_title} > Abstract",
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(s_line, e_line),
                        char_span=(node.pos, node.pos + node.len),
                        parser="LaTeXParser",
                        parser_version="1.0.0",
                        extraction_method="pylatexenc_ast",
                    ),
                    metadata={"is_abstract": True},
                )
                nodes.append(abs_node)
                reading_order += 1
                continue

            # 8. Skip preamble macros like documentclass, usepackage, author, maketitle
            if isinstance(node, LatexMacroNode) and node.macroname in (
                "documentclass",
                "usepackage",
                "author",
                "date",
                "maketitle",
                "bibliographystyle",
                "bibliography",
            ):
                continue

            # 9. Skip comments
            if isinstance(node, LatexCommentNode):
                continue

            # 10. Buffer regular content (chars, inline math, formatting macros)
            if not text_buffer:
                text_start_pos = node.pos
            text_end_pos = node.pos + node.len
            text_buffer.append(node.latex_verbatim())

        # Final flush
        flush_text_buffer()

        # Reconstruct children_ids
        node_map = {n.id: n for n in nodes}
        for n in nodes:
            if n.parent_id and n.parent_id in node_map:
                node_map[n.parent_id].children_ids.append(n.id)

        return DocumentTree(
            document_id=doc_id,
            title=doc_title,
            source_uri=source_uri,
            doc_type="latex",
            nodes=nodes,
            root_node_id=root_node.id,
        )

    parse_string = parse_tree
