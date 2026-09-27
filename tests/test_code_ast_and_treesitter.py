"""Tests for Source Code AST and Tree-sitter parsing, symbol extraction, and chunking."""

from deep_context.ingestion.chunker import ParentChildChunker
from deep_context.ingestion.code_parser import CodeParser
from deep_context.ingestion.parser import DocumentParser

PYTHON_SAMPLE = '''
import math
from typing import Optional

class QuantumCircuit:
    """Represents a quantum computing circuit with register qubits."""

    def __init__(self, num_qubits: int, name: Optional[str] = None):
        self.num_qubits = num_qubits
        self.name = name or "circuit"
        self.gates: list[tuple[str, int]] = []

    def add_hadamard(self, qubit: int) -> None:
        """Applies a Hadamard gate to put qubit into superposition."""
        if qubit >= self.num_qubits:
            raise ValueError("Qubit index out of range")
        self.gates.append(("H", qubit))

    async def execute_simulation(self, shots: int = 1024) -> dict[str, int]:
        """Simulates measurement outcomes for the circuit."""
        return {"0" * self.num_qubits: shots}

def compute_fidelity(state_a: list[complex], state_b: list[complex]) -> float:
    """Computes state fidelity |<psi|phi>|^2 between two pure states."""
    dot = sum(a.conjugate() * b for a, b in zip(state_a, state_b))
    return abs(dot) ** 2
'''

JAVASCRIPT_SAMPLE = """
class TensorEngine {
    constructor(dimensions) {
        this.dimensions = dimensions;
        this.data = new Float32Array(dimensions.reduce((a, b) => a * b, 1));
    }

    forward(inputTensor) {
        // Forward pass implementation
        return inputTensor;
    }
}

function relu(x) {
    return Math.max(0, x);
}
"""

CPP_SAMPLE = """
#include <iostream>
#include <vector>

class HamiltonianSystem {
public:
    HamiltonianSystem(int degrees_of_freedom) : dof(degrees_of_freedom) {}

    double compute_energy(const std::vector<double>& q, const std::vector<double>& p) {
        double kinetic = 0.0;
        for (double val : p) {
            kinetic += 0.5 * val * val;
        }
        return kinetic;
    }

private:
    int dof;
};

int main() {
    HamiltonianSystem sys(3);
    return 0;
}
"""


def test_python_ast_parsing():
    parser = CodeParser()
    tree = parser.parse_tree(PYTHON_SAMPLE, language="python", source_uri="quantum_sim.py")

    assert tree is not None
    assert tree.doc_type in ("code", "python")
    assert len(tree.nodes) > 0

    # Class node
    class_nodes = [n for n in tree.nodes if n.code_data and n.code_data.symbol_type == "class"]
    assert len(class_nodes) == 1
    cls_node = class_nodes[0]
    assert cls_node.code_data is not None
    assert cls_node.code_data.symbol_name == "QuantumCircuit"
    assert "Represents a quantum computing circuit" in (cls_node.code_data.docstring or "")

    # Method nodes
    method_nodes = [
        n
        for n in tree.nodes
        if n.code_data and n.code_data.symbol_type in ("method", "async_method")
    ]
    assert len(method_nodes) >= 3
    method_names = [m.code_data.symbol_name for m in method_nodes if m.code_data]
    assert "__init__" in method_names
    assert "add_hadamard" in method_names
    assert "execute_simulation" in method_names

    # Check method parent points to class node
    h_node = next(
        m for m in method_nodes if m.code_data and m.code_data.symbol_name == "add_hadamard"
    )
    assert h_node.code_data is not None
    assert h_node.parent_id == cls_node.id
    assert "Hadamard" in (h_node.code_data.docstring or "")
    assert "def add_hadamard" in (h_node.code_data.signature or "")

    # Standalone function
    func_nodes = [n for n in tree.nodes if n.code_data and n.code_data.symbol_type == "function"]
    assert len(func_nodes) >= 1
    f_node = func_nodes[0]
    assert f_node.code_data is not None
    assert f_node.code_data.symbol_name == "compute_fidelity"
    assert "def compute_fidelity" in (f_node.code_data.signature or "")


def test_python_dedent_handling():
    # Code with indentation from multi-line triple-quoted string
    indented_code = """
        class NestedHelper:
            def run(self):
                return True
    """
    parser = CodeParser()
    tree = parser.parse_tree(indented_code, language="python", source_uri="nested.py")
    assert len(tree.nodes) >= 2
    assert any(n.code_data and n.code_data.symbol_name == "NestedHelper" for n in tree.nodes)


def test_javascript_tree_sitter_parsing():
    parser = CodeParser()
    tree = parser.parse_tree(JAVASCRIPT_SAMPLE, language="javascript", source_uri="tensor.js")

    assert tree is not None
    assert tree.doc_type in ("code", "javascript")
    assert len(tree.nodes) > 0

    symbol_nodes = [n for n in tree.nodes if n.code_data is not None]
    assert len(symbol_nodes) >= 2
    symbol_names = [n.code_data.symbol_name for n in symbol_nodes if n.code_data]
    assert "TensorEngine" in symbol_names
    assert any(name in ("relu", "forward", "constructor") for name in symbol_names)


def test_cpp_tree_sitter_parsing():
    parser = CodeParser()
    tree = parser.parse_tree(CPP_SAMPLE, language="cpp", source_uri="hamiltonian.cpp")

    assert tree is not None
    assert tree.doc_type in ("code", "cpp")
    assert len(tree.nodes) > 0

    symbol_nodes = [n for n in tree.nodes if n.code_data is not None]
    assert len(symbol_nodes) >= 1
    symbol_names = [n.code_data.symbol_name for n in symbol_nodes if n.code_data]
    assert any(
        name is not None
        and ("HamiltonianSystem" in name or "compute_energy" in name or "main" in name)
        for name in symbol_names
    )


def test_code_routing_and_chunking():
    doc_parser = DocumentParser()
    tree = doc_parser.parse_tree(PYTHON_SAMPLE, doc_type="code", source_uri="circuit.py")

    assert tree.doc_type in ("code", "python")
    chunker = ParentChildChunker()
    parents, children = chunker.chunk_tree(tree)

    assert len(parents) >= 1
    assert len(children) >= 1

    # Chunks preserve code symbols and content
    contents = " ".join(c.content for c in children)
    assert "QuantumCircuit" in contents
    assert "compute_fidelity" in contents
