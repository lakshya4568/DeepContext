from __future__ import annotations

import ast
import logging
import re

import tree_sitter

from deep_context.core.types import (
    DocumentElementType,
    DocumentNode,
    DocumentTree,
    Provenance,
    SourceCodeDataModel,
    generate_stable_doc_id,
    generate_stable_node_id,
)

logger = logging.getLogger(__name__)

# Tree-sitter language loading with lazy cache
_TS_LANGUAGES: dict[str, tree_sitter.Language] = {}


def _get_ts_language(lang: str) -> tree_sitter.Language | None:
    if lang in _TS_LANGUAGES:
        return _TS_LANGUAGES[lang]
    try:
        if lang in ("javascript", "js", "jsx"):
            import tree_sitter_javascript as tsjs

            language = tree_sitter.Language(tsjs.language())
        elif lang in ("typescript", "ts"):
            import tree_sitter_typescript as tsts

            language = tree_sitter.Language(tsts.language_typescript())
        elif lang in ("tsx",):
            import tree_sitter_typescript as tsts

            language = tree_sitter.Language(tsts.language_tsx())
        elif lang in ("c", "h"):
            import tree_sitter_c as tsc

            language = tree_sitter.Language(tsc.language())
        elif lang in ("cpp", "c++", "cc", "hpp"):
            # Use c language parser if cpp specific is not installed
            import tree_sitter_c as tsc

            language = tree_sitter.Language(tsc.language())
        elif lang in ("python", "py"):
            import tree_sitter_python as tspy

            language = tree_sitter.Language(tspy.language())
        else:
            return None
        _TS_LANGUAGES[lang] = language
        return language
    except Exception as e:
        logger.debug("Tree-sitter language load for '%s' failed: %s", lang, e)
        return None


class CodeParser:
    """
    Syntax-aware source code parser using AST and Tree-sitter.
    Extracts structured classes, functions, methods, imports, signatures, and docstrings
    without duplicating class code into each method chunk.
    """

    @classmethod
    def _normalize_lang(cls, doc_type: str, source_uri: str | None = None) -> str:
        t = doc_type.lower()
        if source_uri:
            uri_lower = source_uri.lower()
            if uri_lower.endswith(".py"):
                return "python"
            if uri_lower.endswith(".ts"):
                return "typescript"
            if uri_lower.endswith(".tsx"):
                return "tsx"
            if uri_lower.endswith(".js") or uri_lower.endswith(".jsx"):
                return "javascript"
            if uri_lower.endswith((".c", ".h")):
                return "c"
            if uri_lower.endswith((".cpp", ".cc", ".hpp", ".cxx")):
                return "cpp"
            if uri_lower.endswith(".go"):
                return "go"
            if uri_lower.endswith(".rs"):
                return "rust"
            if uri_lower.endswith(".java"):
                return "java"

        if t in ("python", "code", "py"):
            return "python"
        if t in ("typescript", "ts"):
            return "typescript"
        if t in ("javascript", "js", "jsx"):
            return "javascript"
        if t in ("c", "h"):
            return "c"
        if t in ("cpp", "c++"):
            return "cpp"
        return t

    @classmethod
    def parse_tree(
        cls,
        content: str,
        doc_type: str = "code",
        source_uri: str | None = None,
        title: str = "",
        language: str | None = None,
        tenant_id: str = "default",
    ) -> DocumentTree:
        effective_type = language or doc_type
        lang = cls._normalize_lang(effective_type, source_uri)
        if lang == "python":
            try:
                return cls._parse_python(
                    content, source_uri=source_uri, title=title, tenant_id=tenant_id
                )
            except Exception as e:
                logger.warning(
                    "Python AST parsing failed (%s); falling back to Tree-sitter/regex", e
                )

        ts_lang = _get_ts_language(lang)
        if ts_lang is not None:
            try:
                return cls._parse_tree_sitter(
                    content,
                    lang=lang,
                    ts_language=ts_lang,
                    source_uri=source_uri,
                    title=title,
                    tenant_id=tenant_id,
                )
            except Exception as e:
                logger.warning(
                    "Tree-sitter parsing failed for %s (%s); falling back to regex blocks", lang, e
                )

        return cls._parse_regex_blocks(
            content, lang=lang, source_uri=source_uri, title=title, tenant_id=tenant_id
        )

    # -----------------------------------------------------------------------
    # Python AST Parser
    # -----------------------------------------------------------------------

    @classmethod
    def _parse_python(
        cls,
        content: str,
        source_uri: str | None = None,
        title: str = "",
        tenant_id: str = "default",
    ) -> DocumentTree:
        import textwrap

        clean_code = textwrap.dedent(content)
        doc_title = title or (source_uri.split("/")[-1] if source_uri else "Python Module")
        doc_id = generate_stable_doc_id(
            tenant_id=tenant_id,
            title=doc_title,
            source_uri=source_uri,
            content=clean_code,
        )
        lines = clean_code.splitlines()

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
                line_range=(1, len(lines)),
                parser="CodeParser",
                parser_version="1.0.0",
                extraction_method="python_ast",
            ),
        )

        nodes: list[DocumentNode] = [root_node]
        reading_order = 1

        py_ast = ast.parse(clean_code)

        # 1. Module-level docstring
        module_doc = ast.get_docstring(py_ast)
        if module_doc:
            doc_node = DocumentNode(
                id=generate_stable_node_id(doc_id, "docstring", reading_order, module_doc[:80]),
                document_id=doc_id,
                parent_id=root_node.id,
                node_type=DocumentElementType.PARAGRAPH,
                reading_order=reading_order,
                text=f"Module Docstring: {module_doc}",
                raw_text=module_doc,
                section_path=f"{doc_title} > docstring",
                provenance=Provenance(
                    source_uri=source_uri,
                    page_number=1,
                    parser="CodeParser",
                    parser_version="1.0.0",
                    extraction_method="python_ast",
                ),
            )
            nodes.append(doc_node)
            root_node.children_ids.append(doc_node.id)
            reading_order += 1

        # 2. Extract module-level imports
        imports: list[str] = []
        for item in py_ast.body:
            if isinstance(item, ast.Import):
                for alias in item.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(item, ast.ImportFrom):
                module = item.module or ""
                names = ", ".join(alias.name for alias in item.names)
                imports.append(f"from {module} import {names}")

        # 3. Classes and functions
        for item in py_ast.body:
            if isinstance(item, ast.ClassDef):
                start_line = item.lineno
                end_line = getattr(item, "end_lineno", len(lines))
                cls_doc = ast.get_docstring(item)

                # Class signature / definition excluding methods
                # Extract first lines up to docstring or first method
                first_body_lineno = item.body[0].lineno if item.body else end_line
                cls_header_lines = lines[start_line - 1 : first_body_lineno - 1]
                cls_text = "\n".join(cls_header_lines).strip()
                if cls_doc:
                    cls_text += f'\n    """{cls_doc}"""'
                if not cls_text:
                    cls_text = f"class {item.name}:"

                c_code_data = SourceCodeDataModel(
                    language="python",
                    symbol_name=item.name,
                    symbol_type="class",
                    parent_symbol=None,
                    imports=imports,
                    signature=f"class {item.name}",
                    docstring=cls_doc,
                    start_line=start_line,
                    end_line=end_line,
                    file_path=source_uri,
                )

                class_node = DocumentNode(
                    id=generate_stable_node_id(doc_id, "class", reading_order, item.name),
                    document_id=doc_id,
                    parent_id=root_node.id,
                    node_type=DocumentElementType.CODE,
                    reading_order=reading_order,
                    text=cls_text,
                    raw_text="\n".join(lines[start_line - 1 : end_line]),
                    section_path=f"{doc_title} > class {item.name}",
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(start_line, end_line),
                        raw_ref=item.name,
                        parser="CodeParser",
                        parser_version="1.0.0",
                        extraction_method="python_ast",
                    ),
                    code_data=c_code_data,
                    metadata={
                        "symbol": item.name,
                        "kind": "class",
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                )
                nodes.append(class_node)
                root_node.children_ids.append(class_node.id)
                reading_order += 1

                # Class methods
                for sub in item.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start = sub.lineno
                        m_end = getattr(sub, "end_lineno", len(lines))
                        m_doc = ast.get_docstring(sub)
                        method_code = "\n".join(lines[m_start - 1 : m_end])
                        sig = f"def {sub.name}(...)"

                        m_code_data = SourceCodeDataModel(
                            language="python",
                            symbol_name=sub.name,
                            symbol_type="method",
                            parent_symbol=item.name,
                            imports=imports,
                            signature=sig,
                            docstring=m_doc,
                            start_line=m_start,
                            end_line=m_end,
                            file_path=source_uri,
                        )

                        m_node = DocumentNode(
                            id=generate_stable_node_id(
                                doc_id, "method", reading_order, f"{item.name}.{sub.name}"
                            ),
                            document_id=doc_id,
                            parent_id=class_node.id,
                            node_type=DocumentElementType.CODE,
                            reading_order=reading_order,
                            text=method_code,
                            raw_text=method_code,
                            section_path=f"{doc_title} > class {item.name} > method {sub.name}",
                            provenance=Provenance(
                                source_uri=source_uri,
                                page_number=1,
                                line_range=(m_start, m_end),
                                raw_ref=f"{item.name}.{sub.name}",
                                parser="CodeParser",
                                parser_version="1.0.0",
                                extraction_method="python_ast",
                            ),
                            code_data=m_code_data,
                            metadata={
                                "symbol": sub.name,
                                "class_name": item.name,
                                "kind": "method",
                                "start_line": m_start,
                                "end_line": m_end,
                            },
                        )
                        nodes.append(m_node)
                        class_node.children_ids.append(m_node.id)
                        reading_order += 1

            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f_start = item.lineno
                f_end = getattr(item, "end_lineno", len(lines))
                f_doc = ast.get_docstring(item)
                func_code = "\n".join(lines[f_start - 1 : f_end])
                is_async = isinstance(item, ast.AsyncFunctionDef)
                sig = f"{'async ' if is_async else ''}def {item.name}(...)"

                f_code_data = SourceCodeDataModel(
                    language="python",
                    symbol_name=item.name,
                    symbol_type="function",
                    parent_symbol=None,
                    imports=imports,
                    signature=sig,
                    docstring=f_doc,
                    start_line=f_start,
                    end_line=f_end,
                    file_path=source_uri,
                )

                f_node = DocumentNode(
                    id=generate_stable_node_id(doc_id, "function", reading_order, item.name),
                    document_id=doc_id,
                    parent_id=root_node.id,
                    node_type=DocumentElementType.CODE,
                    reading_order=reading_order,
                    text=func_code,
                    raw_text=func_code,
                    section_path=f"{doc_title} > function {item.name}",
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(f_start, f_end),
                        raw_ref=item.name,
                        parser="CodeParser",
                        parser_version="1.0.0",
                        extraction_method="python_ast",
                    ),
                    code_data=f_code_data,
                    metadata={
                        "symbol": item.name,
                        "kind": "function",
                        "start_line": f_start,
                        "end_line": f_end,
                    },
                )
                nodes.append(f_node)
                root_node.children_ids.append(f_node.id)
                reading_order += 1

        # If no classes or functions found, include entire script
        if len(nodes) == 1:
            raw_node = DocumentNode(
                id=generate_stable_node_id(doc_id, "code_module", 1, doc_title),
                document_id=doc_id,
                parent_id=root_node.id,
                node_type=DocumentElementType.CODE,
                reading_order=1,
                text=content,
                raw_text=content,
                section_path=doc_title,
                provenance=Provenance(
                    source_uri=source_uri,
                    page_number=1,
                    line_range=(1, len(lines)),
                    parser="CodeParser",
                    parser_version="1.0.0",
                    extraction_method="python_ast",
                ),
                code_data=SourceCodeDataModel(
                    language="python", file_path=source_uri, start_line=1, end_line=len(lines)
                ),
            )
            nodes.append(raw_node)
            root_node.children_ids.append(raw_node.id)

        return DocumentTree(
            document_id=doc_id,
            title=doc_title,
            source_uri=source_uri,
            doc_type="python",
            nodes=nodes,
            root_node_id=root_node.id,
        )

    # -----------------------------------------------------------------------
    # Tree-sitter Parser (JS/TS/C/C++)
    # -----------------------------------------------------------------------

    @classmethod
    def _parse_tree_sitter(
        cls,
        content: str,
        lang: str,
        ts_language: tree_sitter.Language,
        source_uri: str | None = None,
        title: str = "",
        tenant_id: str = "default",
    ) -> DocumentTree:
        doc_title = title or (
            source_uri.split("/")[-1] if source_uri else f"{lang.capitalize()} Source"
        )
        doc_id = generate_stable_doc_id(
            tenant_id=tenant_id,
            title=doc_title,
            source_uri=source_uri,
            content=content,
        )
        lines = content.splitlines()

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
                line_range=(1, len(lines)),
                parser="CodeParser",
                parser_version="1.0.0",
                extraction_method="tree_sitter",
            ),
        )

        nodes: list[DocumentNode] = [root_node]
        reading_order = 1

        parser = tree_sitter.Parser(ts_language)
        content_bytes = content.encode("utf-8")
        ts_tree = parser.parse(content_bytes)

        # Helper to unwrap export statements
        def unwrap_node(n: tree_sitter.Node) -> tuple[tree_sitter.Node, bool]:
            if n.type in ("export_statement", "export_default_statement"):
                for child in n.children:
                    if child.type in (
                        "class_declaration",
                        "function_declaration",
                        "interface_declaration",
                        "type_alias_declaration",
                        "abstract_class_declaration",
                    ):
                        return child, True
            return n, False

        # Extract top-level imports
        imports: list[str] = []
        for child in ts_tree.root_node.children:
            if child.type in ("import_statement", "preproc_include"):
                imp_text = (
                    content_bytes[child.start_byte : child.end_byte]
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                imports.append(imp_text)

        for raw_child in ts_tree.root_node.children:
            child, is_exported = unwrap_node(raw_child)

            # 1. Class declarations
            if child.type in ("class_declaration", "abstract_class_declaration"):
                name_node = child.child_by_field_name("name")
                cls_name = (
                    content_bytes[name_node.start_byte : name_node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    if name_node
                    else "AnonymousClass"
                )
                start_line = child.start_point.row + 1
                end_line = child.end_point.row + 1

                # Class header/summary
                body_node = child.child_by_field_name("body")
                header_end_byte = body_node.start_byte if body_node else child.end_byte
                cls_header = (
                    content_bytes[child.start_byte : header_end_byte]
                    .decode("utf-8", errors="replace")
                    .strip()
                )

                c_code_data = SourceCodeDataModel(
                    language=lang,
                    symbol_name=cls_name,
                    symbol_type="class",
                    parent_symbol=None,
                    imports=imports,
                    signature=cls_header,
                    start_line=start_line,
                    end_line=end_line,
                    file_path=source_uri,
                )

                class_node = DocumentNode(
                    id=generate_stable_node_id(doc_id, "class", reading_order, cls_name),
                    document_id=doc_id,
                    parent_id=root_node.id,
                    node_type=DocumentElementType.CODE,
                    reading_order=reading_order,
                    text=cls_header,
                    raw_text=content_bytes[child.start_byte : child.end_byte].decode(
                        "utf-8", errors="replace"
                    ),
                    section_path=f"{doc_title} > class {cls_name}",
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(start_line, end_line),
                        char_span=(child.start_byte, child.end_byte),
                        raw_ref=cls_name,
                        parser="CodeParser",
                        parser_version="1.0.0",
                        extraction_method="tree_sitter",
                    ),
                    code_data=c_code_data,
                    metadata={
                        "symbol": cls_name,
                        "kind": "class",
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                )
                nodes.append(class_node)
                root_node.children_ids.append(class_node.id)
                reading_order += 1

                # Parse methods inside class_body
                if body_node:
                    for m in body_node.children:
                        if m.type in ("method_definition", "method_declaration"):
                            m_name_node = m.child_by_field_name("name")
                            m_name = (
                                content_bytes[m_name_node.start_byte : m_name_node.end_byte].decode(
                                    "utf-8", errors="replace"
                                )
                                if m_name_node
                                else "anonymousMethod"
                            )
                            m_start = m.start_point.row + 1
                            m_end = m.end_point.row + 1
                            m_text = content_bytes[m.start_byte : m.end_byte].decode(
                                "utf-8", errors="replace"
                            )

                            m_code_data = SourceCodeDataModel(
                                language=lang,
                                symbol_name=m_name,
                                symbol_type="method",
                                parent_symbol=cls_name,
                                imports=imports,
                                start_line=m_start,
                                end_line=m_end,
                                file_path=source_uri,
                            )

                            m_node = DocumentNode(
                                id=generate_stable_node_id(
                                    doc_id, "method", reading_order, f"{cls_name}.{m_name}"
                                ),
                                document_id=doc_id,
                                parent_id=class_node.id,
                                node_type=DocumentElementType.CODE,
                                reading_order=reading_order,
                                text=m_text,
                                raw_text=m_text,
                                section_path=f"{doc_title} > class {cls_name} > method {m_name}",
                                provenance=Provenance(
                                    source_uri=source_uri,
                                    page_number=1,
                                    line_range=(m_start, m_end),
                                    char_span=(m.start_byte, m.end_byte),
                                    raw_ref=f"{cls_name}.{m_name}",
                                    parser="CodeParser",
                                    parser_version="1.0.0",
                                    extraction_method="tree_sitter",
                                ),
                                code_data=m_code_data,
                                metadata={
                                    "symbol": m_name,
                                    "class_name": cls_name,
                                    "kind": "method",
                                    "start_line": m_start,
                                    "end_line": m_end,
                                },
                            )
                            nodes.append(m_node)
                            class_node.children_ids.append(m_node.id)
                            reading_order += 1

            # 2. Function declarations / definitions
            elif child.type in ("function_declaration", "function_definition"):
                name_node = child.child_by_field_name("name") or child.child_by_field_name(
                    "declarator"
                )
                func_name = ""
                if name_node:
                    # In C, declarator can be nested (e.g. function_declarator)
                    if name_node.type == "function_declarator":
                        inner_decl = name_node.child_by_field_name("declarator")
                        if inner_decl:
                            func_name = content_bytes[
                                inner_decl.start_byte : inner_decl.end_byte
                            ].decode("utf-8", errors="replace")
                    if not func_name:
                        func_name = content_bytes[name_node.start_byte : name_node.end_byte].decode(
                            "utf-8", errors="replace"
                        )
                if not func_name:
                    func_name = "anonymousFunction"

                start_line = child.start_point.row + 1
                end_line = child.end_point.row + 1
                func_text = content_bytes[child.start_byte : child.end_byte].decode(
                    "utf-8", errors="replace"
                )

                f_code_data = SourceCodeDataModel(
                    language=lang,
                    symbol_name=func_name,
                    symbol_type="function",
                    parent_symbol=None,
                    imports=imports,
                    start_line=start_line,
                    end_line=end_line,
                    file_path=source_uri,
                )

                func_node = DocumentNode(
                    id=generate_stable_node_id(doc_id, "function", reading_order, func_name),
                    document_id=doc_id,
                    parent_id=root_node.id,
                    node_type=DocumentElementType.CODE,
                    reading_order=reading_order,
                    text=func_text,
                    raw_text=func_text,
                    section_path=f"{doc_title} > function {func_name}",
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(start_line, end_line),
                        char_span=(child.start_byte, child.end_byte),
                        raw_ref=func_name,
                        parser="CodeParser",
                        parser_version="1.0.0",
                        extraction_method="tree_sitter",
                    ),
                    code_data=f_code_data,
                    metadata={
                        "symbol": func_name,
                        "kind": "function",
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                )
                nodes.append(func_node)
                root_node.children_ids.append(func_node.id)
                reading_order += 1

            # 3. Interfaces / Types (TypeScript)
            elif child.type in ("interface_declaration", "type_alias_declaration"):
                name_node = child.child_by_field_name("name")
                type_name = (
                    content_bytes[name_node.start_byte : name_node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    if name_node
                    else "AnonymousType"
                )
                start_line = child.start_point.row + 1
                end_line = child.end_point.row + 1
                type_text = content_bytes[child.start_byte : child.end_byte].decode(
                    "utf-8", errors="replace"
                )

                t_code_data = SourceCodeDataModel(
                    language=lang,
                    symbol_name=type_name,
                    symbol_type="interface",
                    parent_symbol=None,
                    imports=imports,
                    start_line=start_line,
                    end_line=end_line,
                    file_path=source_uri,
                )

                type_node = DocumentNode(
                    id=generate_stable_node_id(doc_id, "interface", reading_order, type_name),
                    document_id=doc_id,
                    parent_id=root_node.id,
                    node_type=DocumentElementType.CODE,
                    reading_order=reading_order,
                    text=type_text,
                    raw_text=type_text,
                    section_path=f"{doc_title} > interface {type_name}",
                    provenance=Provenance(
                        source_uri=source_uri,
                        page_number=1,
                        line_range=(start_line, end_line),
                        char_span=(child.start_byte, child.end_byte),
                        raw_ref=type_name,
                        parser="CodeParser",
                        parser_version="1.0.0",
                        extraction_method="tree_sitter",
                    ),
                    code_data=t_code_data,
                    metadata={
                        "symbol": type_name,
                        "kind": "interface",
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                )
                nodes.append(type_node)
                root_node.children_ids.append(type_node.id)
                reading_order += 1

        if len(nodes) == 1:
            raw_node = DocumentNode(
                id=generate_stable_node_id(doc_id, "code_module", 1, doc_title),
                document_id=doc_id,
                parent_id=root_node.id,
                node_type=DocumentElementType.CODE,
                reading_order=1,
                text=content,
                raw_text=content,
                section_path=doc_title,
                provenance=Provenance(
                    source_uri=source_uri,
                    page_number=1,
                    line_range=(1, len(lines)),
                    parser="CodeParser",
                    parser_version="1.0.0",
                    extraction_method="tree_sitter",
                ),
                code_data=SourceCodeDataModel(
                    language=lang, file_path=source_uri, start_line=1, end_line=len(lines)
                ),
            )
            nodes.append(raw_node)
            root_node.children_ids.append(raw_node.id)

        return DocumentTree(
            document_id=doc_id,
            title=doc_title,
            source_uri=source_uri,
            doc_type=lang,
            nodes=nodes,
            root_node_id=root_node.id,
        )

    # -----------------------------------------------------------------------
    # Fallback Regex Block Parser
    # -----------------------------------------------------------------------

    @classmethod
    def _parse_regex_blocks(
        cls,
        content: str,
        lang: str,
        source_uri: str | None = None,
        title: str = "",
        tenant_id: str = "default",
    ) -> DocumentTree:
        doc_title = title or (
            source_uri.split("/")[-1] if source_uri else f"{lang.capitalize()} Source"
        )
        doc_id = generate_stable_doc_id(
            tenant_id=tenant_id,
            title=doc_title,
            source_uri=source_uri,
            content=content,
        )
        lines = content.splitlines()

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
                line_range=(1, len(lines)),
                parser="CodeParser",
                parser_version="1.0.0",
                extraction_method="regex_fallback",
            ),
        )

        nodes: list[DocumentNode] = [root_node]
        reading_order = 1

        blocks = re.split(
            r"\n(?=(?:def |class |function |export |public |private |func |fn ))", content
        )
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            first_line = block.strip().splitlines()[0][:60]
            b_node = DocumentNode(
                id=generate_stable_node_id(doc_id, "block", reading_order, first_line),
                document_id=doc_id,
                parent_id=root_node.id,
                node_type=DocumentElementType.CODE,
                reading_order=reading_order,
                text=block.strip(),
                raw_text=block.strip(),
                section_path=f"{doc_title} > block_{i + 1}",
                provenance=Provenance(
                    source_uri=source_uri,
                    page_number=1,
                    parser="CodeParser",
                    parser_version="1.0.0",
                    extraction_method="regex_fallback",
                ),
                code_data=SourceCodeDataModel(
                    language=lang, signature=first_line, file_path=source_uri
                ),
                metadata={"block_index": i + 1, "first_line": first_line},
            )
            nodes.append(b_node)
            root_node.children_ids.append(b_node.id)
            reading_order += 1

        if len(nodes) == 1:
            raw_node = DocumentNode(
                id=generate_stable_node_id(doc_id, "code_module", 1, doc_title),
                document_id=doc_id,
                parent_id=root_node.id,
                node_type=DocumentElementType.CODE,
                reading_order=1,
                text=content,
                raw_text=content,
                section_path=doc_title,
                provenance=Provenance(
                    source_uri=source_uri,
                    page_number=1,
                    line_range=(1, len(lines)),
                    parser="CodeParser",
                    parser_version="1.0.0",
                    extraction_method="regex_fallback",
                ),
                code_data=SourceCodeDataModel(
                    language=lang, file_path=source_uri, start_line=1, end_line=len(lines)
                ),
            )
            nodes.append(raw_node)
            root_node.children_ids.append(raw_node.id)

        return DocumentTree(
            document_id=doc_id,
            title=doc_title,
            source_uri=source_uri,
            doc_type=lang,
            nodes=nodes,
            root_node_id=root_node.id,
        )
