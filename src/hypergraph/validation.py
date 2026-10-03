"""Validation and lightweight helpers for hypergraph structures."""

from __future__ import annotations

from typing import Any, Dict, List

from .incidence import IncidenceMatrix, validate_hypergraph


def check_empty_hyperedges(incidence: IncidenceMatrix) -> bool:
    """True when at least one hyperedge has zero degree."""
    if incidence.n_hyperedges == 0:
        return False
    return bool((incidence.hyperedge_degrees == 0).any())


def check_isolated_vertices(incidence: IncidenceMatrix) -> bool:
    """True when at least one vertex has zero degree."""
    if incidence.n_vertices == 0:
        return False
    return bool((incidence.vertex_degrees == 0).any())


def verify_hypergraph(hg: Any) -> List[str]:
    """Validate a Hypergraph container; returns printable warnings."""
    warnings = list(hg.warnings)
    warnings.extend(validate_hypergraph(hg.incidence))
    return warnings
