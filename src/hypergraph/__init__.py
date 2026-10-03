"""Hypergraph subpackage for the HGNN-DRL project."""

from .builder import Hypergraph, HypergraphBuilder, add_shared_buffer, set_hyperedge_weights
from .incidence import IncidenceMatrix, build_incidence_matrix, validate_hypergraph
from .validation import check_empty_hyperedges, check_isolated_vertices, verify_hypergraph

__all__ = [
    "IncidenceMatrix",
    "build_incidence_matrix",
    "validate_hypergraph",
    "Hypergraph",
    "HypergraphBuilder",
    "add_shared_buffer",
    "set_hyperedge_weights",
    "check_empty_hyperedges",
    "check_isolated_vertices",
    "verify_hypergraph",
]

__version__ = "0.1.0"
