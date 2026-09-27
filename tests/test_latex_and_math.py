"""Tests for LaTeX parsing, equation extraction, mathematical representations, and parent-child chunking."""

from deep_context.core.types import DocumentElementType
from deep_context.ingestion.chunker import ParentChildChunker
from deep_context.ingestion.latex_parser import LaTeXParser
from deep_context.ingestion.parser import DocumentParser

SAMPLE_LATEX = r"""
\documentclass{article}
\title{Quantum Mechanics and Wave Equations}
\author{Erwin Schrodinger}

\begin{document}
\maketitle

\begin{abstract}
We present the fundamental formulation of non-relativistic wave mechanics.
\end{abstract}

\section{Introduction}
Wave-particle duality governs microscopic physical systems.
The Planck relation connects energy and frequency:
\begin{equation}
E = h \nu
\label{eq:planck}
\end{equation}

\section{The Time-Dependent Schrödinger Equation}
For a particle of mass $m$ in a potential $V(\mathbf{r}, t)$, the state vector $\Psi(\mathbf{r}, t)$ evolves according to:
\begin{equation}
i \hbar \frac{\partial \Psi}{\partial t} = \left( -\frac{\hbar^2}{2m} \nabla^2 + V(\mathbf{r}, t) \right) \Psi
\label{eq:schrodinger}
\end{equation}
Here, $\hbar = h / (2\pi)$ is the reduced Planck constant.

\subsection{Energy Eigenvalues}
For stationary states, the time-independent equation reduces to:
\begin{align}
\hat{H} \psi &= E \psi \\
\left(-\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + V(x)\right)\psi(x) &= E \psi(x)
\end{align}

\section{Experimental Parameters}
\begin{tabular}{|l|c|r|}
\hline
Constant & Symbol & Value \\
\hline
Planck constant & $h$ & $6.626 \times 10^{-34} \text{ J s}$ \\
Reduced Planck & $\hbar$ & $1.054 \times 10^{-34} \text{ J s}$ \\
Electron mass & $m_e$ & $9.109 \times 10^{-31} \text{ kg}$ \\
\hline
\end{tabular}

\end{document}
"""


def test_latex_parser_hierarchy_and_sections():
    parser = LaTeXParser()
    tree = parser.parse_string(SAMPLE_LATEX, source_uri="physics_paper.tex")

    assert tree is not None
    assert bool(tree.document_id)
    assert len(tree.nodes) > 0

    # Title & Abstract
    assert "Quantum Mechanics" in tree.title
    title_nodes = [n for n in tree.nodes if "Quantum Mechanics" in n.text]
    assert len(title_nodes) >= 1

    # Sections (stored as HEADING nodes with section path)
    sections = [n for n in tree.nodes if n.node_type == DocumentElementType.HEADING]
    section_titles = [s.text for s in sections]
    assert any("Introduction" in t for t in section_titles)
    assert any("Schrödinger Equation" in t or "Schrodinger" in t for t in section_titles)
    assert any("Energy Eigenvalues" in t for t in section_titles)


def test_latex_parser_equations():
    parser = LaTeXParser()
    tree = parser.parse_string(SAMPLE_LATEX, source_uri="physics_paper.tex")

    eq_nodes = [n for n in tree.nodes if n.node_type == DocumentElementType.EQUATION]
    assert len(eq_nodes) >= 3

    # Check Planck equation
    planck_node = next(
        (
            n
            for n in eq_nodes
            if "E = h" in (n.equation_data.latex if n.equation_data else "") or "h" in n.text
        ),
        None,
    )
    assert planck_node is not None
    assert planck_node.equation_data is not None
    assert "E" in planck_node.equation_data.latex
    assert "nu" in planck_node.equation_data.latex or "h" in planck_node.equation_data.latex
    assert planck_node.equation_data.equation_number in ("(1)", "1", "eq:planck", None)
    assert planck_node.metadata.get("label") in ("eq:planck", "planck", None)

    # Check Schrödinger equation
    schrodinger_node = next(
        (
            n
            for n in eq_nodes
            if "\\hbar" in (n.equation_data.latex if n.equation_data else "") or "Psi" in n.text
        ),
        None,
    )
    assert schrodinger_node is not None
    assert schrodinger_node.equation_data is not None
    assert (
        "nabla" in schrodinger_node.equation_data.latex
        or "\\hbar" in schrodinger_node.equation_data.latex
    )
    assert len(schrodinger_node.equation_data.variables) >= 1


def test_latex_parser_tables():
    parser = LaTeXParser()
    tree = parser.parse_string(SAMPLE_LATEX, source_uri="physics_paper.tex")

    table_nodes = [n for n in tree.nodes if n.node_type == DocumentElementType.TABLE]
    assert len(table_nodes) >= 1
    tbl = table_nodes[0]
    assert tbl.table_data is not None
    assert tbl.table_data.num_rows >= 3
    assert tbl.table_data.num_cols >= 3
    assert "Planck" in tbl.table_data.markdown


def test_latex_routing_in_document_parser():
    doc_parser = DocumentParser()
    tree = doc_parser.parse_tree(SAMPLE_LATEX, doc_type="latex", source_uri="quantum.tex")

    assert tree.doc_type == "latex"
    assert len(tree.nodes) > 0
    assert any(n.node_type == DocumentElementType.EQUATION for n in tree.nodes)


def test_latex_parent_child_chunking_preserves_equations():
    doc_parser = DocumentParser()
    tree = doc_parser.parse_tree(SAMPLE_LATEX, doc_type="latex", source_uri="quantum.tex")

    chunker = ParentChildChunker()
    parents, children = chunker.chunk_tree(tree)

    assert len(parents) >= 1
    assert len(children) >= 1

    # Check that equation child chunk exists
    eq_children = [
        c
        for c in children
        if c.metadata.get("has_equation")
        or any("equation" in str(et).lower() for et in c.metadata.get("element_types", []))
    ]
    assert len(eq_children) >= 1
    eq_chunk = eq_children[0]
    assert eq_chunk.parent_chunk_id is not None
    assert eq_chunk.document_id == tree.document_id
    assert "$" in eq_chunk.content or "\\" in eq_chunk.content or "E =" in eq_chunk.content
