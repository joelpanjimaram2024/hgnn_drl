"""Hypergraph builder and optional hyperedge weighting for HGNN-DRL.

The builder adds tasks to shared-data buffers (hyperedges) and optionally
weights a hyperedge by data-reuse potential, buffer size, transfer cost, or
dependency count.  Validation guards against empty and isolated structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy missing
    _HAS_SCIPY = False
    sp = None  # type: ignore[assignment]

from .incidence import IncidenceMatrix, build_incidence_matrix, validate_hypergraph


@dataclass
class Hypergraph:
    """A named hypergraph with incidence matrix and optional weights."""

    name: str
    incidence: IncidenceMatrix
    hyperedge_id_to_vertex_ids: Dict[str, List[str]] = field(
        default_factory=dict, repr=False
    )
    weights: Dict[str, float] = field(default_factory=dict, repr=False)
    warnings: List[str] = field(default_factory=list, repr=False)

    @property
    def n_tasks(self) -> int:
        return self.incidence.n_vertices

    @property
    def n_hyperedges(self) -> int:
        return self.incidence.n_hyperedges

    def vertex_degree(self, vertex_id: str) -> float:
        """Degree of a single vertex (sum of its incidence weights)."""
        idx = self.incidence.vertex_ids.index(vertex_id) if vertex_id in self.incidence.vertex_ids else -1
        if idx < 0:
            return 0.0
        return float(self.incidence.vertex_degrees[idx])

    def hyperedge_degree(self, hyperedge_id: str) -> float:
        """Degree of a single hyperedge (sum of its incidence weights)."""
        idx = self.hyperedge_id_to_vertex_ids.keys().__iter__().__next__() if False else None
        # Build a quick id->position map.
        pos = {he: i for i, he in enumerate(self.incidence.hyperedge_ids)}
        if hyperedge_id not in pos:
            return 0.0
        return float(self.incidence.hyperedge_degrees[pos[hyperedge_id]])

    def tasks_in_hyperedge(self, hyperedge_id: str) -> List[str]:
        return list(self.hyperedge_id_to_vertex_ids.get(hyperedge_id, []))

    def hyperedge_weights(self) -> Dict[str, float]:
        return dict(self.weights)


@dataclass
class HypergraphBuilder:
    """Constructs hypergraphs from a task registry and shared-buffer membership.

    Tasks are the vertices; shared-data buffers (e.g. camera frames, map tiles)
    are the hyperedges.  Each task may belong to multiple hyperedges when it
    consumes multiple shared inputs.
    """

    tasks: Dict[str, Any] = field(default_factory=dict, repr=False)
    task_ids: List[str] = field(default_factory=list, repr=False)

    # buffer_id -> list of task_ids that read it
    buffer_to_tasks: Dict[str, List[str]] = field(default_factory=dict, repr=False)

    # Optional per-buffer weights.
    buffer_weights: Dict[str, float] = field(default_factory=dict, repr=False)

    # Name of the scenario this hypergraph is modelling.
    name: str = "unnamed"

    def register_task(self, task_id: str, task_metadata: Optional[Any] = None) -> None:
        """Register a task (vertex) in the hypergraph."""
        if task_id in self.tasks:
            return
        self.tasks[task_id] = task_metadata
        self.task_ids.append(task_id)

    def add_shared_buffer(self, buffer_id: str, task_ids: Iterable[str]) -> None:
        """Associate tasks with a shared buffer (hyperedge)."""
        if buffer_id not in self.buffer_to_tasks:
            self.buffer_to_tasks[buffer_id] = []
        for tid in task_ids:
            if tid not in self.buffer_to_tasks[buffer_id]:
                self.buffer_to_tasks[buffer_id].append(tid)

    def set_buffer_weights(self, buffer_id: str, weight: float) -> None:
        """Set an optional weight for a shared buffer."""
        self.buffer_weights[buffer_id] = float(weight)

    def build(self) -> Hypergraph:
        """Materialise the incidence matrix and validate it."""
        vertex_ids = list(self.task_ids)
        he_to_vids = {
            buf_id: list(tids) for buf_id, tids in self.buffer_to_tasks.items()
        }

        incidence = build_incidence_matrix(
            vertex_ids=vertex_ids,
            hyperedge_id_to_vertex_ids=he_to_vids,
            weights=self.buffer_weights,
        )

        warnings = validate_hypergraph(incidence)
        return Hypergraph(
            name=self.name,
            incidence=incidence,
            hyperedge_id_to_vertex_ids=he_to_vids,
            weights=self.buffer_weights,
            warnings=warnings,
        )

    def validate(self) -> Hypergraph:
        """Convenience: build and immediately validate."""
        hg = self.build()
        return hg

    def __repr__(self) -> str:
        return f"HypergraphBuilder(name={self.name}, n_tasks={len(self.task_ids)}, n_buffers={len(self.buffer_to_tasks)})"


def add_shared_buffer(
    builder: HypergraphBuilder, buffer_id: str, task_ids: Iterable[str]
) -> None:
    """Module-level helper: add a shared buffer to a builder."""
    builder.add_shared_buffer(buffer_id, task_ids)


def set_hyperedge_weights(
    builder: HypergraphBuilder, weights: Dict[str, float]
) -> None:
    """Set per-buffer weights on the builder."""
    builder.buffer_weights.update(weights)
