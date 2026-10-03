"""Incidence matrix and degree calculations for HGNN-DRL.

Incidence matrix H (n_tasks x n_hyperedges) where H[v, e] = w_{v,e} if vertex
v belongs to hyperedge e.  Vertex degrees and hyperedge degrees are derived
from H so the hypergraph convolution can be built on top.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy missing
    _HAS_SCIPY = False
    sp = None  # type: ignore[assignment]


class IncidenceMatrix:
    """A sparse (or dense) task-hyperedge incidence matrix.

    Attributes
    ----------
    n_vertices : int
        Number of tasks.
    n_hyperedges : int
        Number of hyperedges.
    data : np.ndarray or scipy.sparse matrix
        The raw incidence entries.
    vertex_ids : list
        Ordered task identifiers.
    hyperedge_ids : list
        Ordered hyperedge identifiers.
    vertex_degrees : np.ndarray
        Row sums d(v) = sum_e H[v, e].
    hyperedge_degrees : np.ndarray
        Column sums d(e) = sum_v H[v, e].
    """

    def __init__(
        self,
        *,
        n_vertices: int,
        n_hyperedges: int,
        data: Optional[np.ndarray] = None,
        dtype: np.dtype = np.float64,
        vertex_ids: Optional[List[str]] = None,
        hyperedge_ids: Optional[List[str]] = None,
    ):
        self.n_vertices = int(n_vertices)
        self.n_hyperedges = int(n_hyperedges)
        self.dtype = np.dtype(dtype)

        if data is None:
            data = np.zeros((self.n_vertices, self.n_hyperedges), dtype=self.dtype)
        elif isinstance(data, np.ndarray):
            if data.shape != (self.n_vertices, self.n_hyperedges):
                raise ValueError(
                    f"data shape {data.shape} != ({self.n_vertices}, {self.n_hyperedges})"
                )
            data = data.astype(self.dtype)
        elif _HAS_SCIPY and sp.issparse(data):
            data = data.astype(self.dtype)
            if data.shape != (self.n_vertices, self.n_hyperedges):
                raise ValueError(
                    f"data shape {data.shape} != ({self.n_vertices}, {self.n_hyperedges})"
                )
        else:
            raise TypeError("data must be ndarray or scipy.sparse matrix")

        self.data = data
        self.vertex_ids = vertex_ids or [f"v{i}" for i in range(self.n_vertices)]
        self.hyperedge_ids = hyperedge_ids or [f"e{i}" for i in range(self.n_hyperedges)]

        # Degrees (raw, used for validation and degree matrices).
        # The HGNN's _diagonal_from handles zero-degree rows/cols gracefully
        # (inverse-sqrt set to 0 where degree == 0), so no clamping needed.
        if _HAS_SCIPY:
            self.vertex_degrees = np.asarray(self.data.sum(axis=1)).ravel().astype(self.dtype)
            self.hyperedge_degrees = np.asarray(self.data.sum(axis=0)).ravel().astype(self.dtype)
        else:
            self.vertex_degrees = self.data.sum(axis=1).astype(self.dtype)
            self.hyperedge_degrees = self.data.sum(axis=0).astype(self.dtype)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_vertices, self.n_hyperedges)

    def to_numpy(self) -> np.ndarray:
        if _HAS_SCIPY and sp.issparse(self.data):
            return self.data.toarray()
        return np.asarray(self.data)

    def to_scipy(self):
        if not _HAS_SCIPY:
            raise RuntimeError("scipy not available")
        return self.data

    def __repr__(self) -> str:
        return (
            f"IncidenceMatrix(n_vertices={self.n_vertices}, n_hyperedges={self.n_hyperedges}, "
            f"dtype={self.dtype}, nnz={self.nnz})"
        )

    @property
    def nnz(self) -> int:
        if _HAS_SCIPY:
            return int(self.data.nnz)
        return int(np.count_nonzero(self.data))


def build_incidence_matrix(
    *,
    vertex_ids: List[str],
    hyperedge_id_to_vertex_ids: Dict[str, List[str]],
    weights: Optional[Dict[str, float]] = None,
    dtype: np.dtype = np.float64,
) -> IncidenceMatrix:
    """Build H from explicit vertex membership.

    Parameters
    ----------
    vertex_ids : list of str
        All vertices (tasks). Ordering determines rows.
    hyperedge_id_to_vertex_ids : dict
        hyperedge_id -> [vertex ids]
    weights : dict, optional
        hyperedge_id -> weight.  Default weight is 1.0.

    Returns
    -------
    IncidenceMatrix
    """
    # Materialise explicit set of known vertices for validation.
    vertex_set = set(vertex_ids)
    row_index = {vid: i for i, vid in enumerate(vertex_ids)}

    n_vertices = len(vertex_ids)
    n_hyperedges = len(hyperedge_id_to_vertex_ids)

    # Build mapping hyperedge id -> col index once.
    he_to_col = {he_id: idx for idx, he_id in enumerate(hyperedge_id_to_vertex_ids)}

    # Collect (row, col, value) triplets.
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for he_id, vids in hyperedge_id_to_vertex_ids.items():
        col = he_to_col[he_id]
        for v in vids:
            if v not in row_index:
                raise ValueError(f"Unknown vertex {v} referenced by hyperedge {he_id}")
            r = row_index[v]
            rows.append(r)
            cols.append(col)
            vals.append(weights.get(he_id, 1.0) if weights else 1.0)

    if not rows:
        # Empty hypergraph -> fully zero incidence.
        data = np.zeros((n_vertices, n_hyperedges), dtype=dtype)
        return IncidenceMatrix(
            n_vertices=n_vertices,
            n_hyperedges=n_hyperedges,
            data=data,
            vertex_ids=vertex_ids,
            hyperedge_ids=list(hyperedge_id_to_vertex_ids.keys()),
        )

    data = np.array(vals, dtype=dtype)
    row_arr = np.array(rows, dtype=np.int64)
    col_arr = np.array(cols, dtype=np.int64)

    sparse = sp.csc_matrix((data, (row_arr, col_arr)), shape=(n_vertices, n_hyperedges))
    return IncidenceMatrix(
        n_vertices=n_vertices,
        n_hyperedges=n_hyperedges,
        data=sparse,
        vertex_ids=vertex_ids,
        hyperedge_ids=list(hyperedge_id_to_vertex_ids.keys()),
    )


def validate_hypergraph(incidence: IncidenceMatrix) -> List[str]:
    """Return a list of warnings for an incidence matrix.

    Checks that there are no empty hyperedges (zero-degree columns) and no
    isolated vertices (zero-degree rows), while tolerating intentional
    legitimate empties when the caller is certain.
    """
    warnings: List[str] = []
    if incidence.n_vertices == 0 or incidence.n_hyperedges == 0:
        return warnings

    zero_vertex_rows = np.where(incidence.vertex_degrees == 0)[0]
    zero_hyperedge_cols = np.where(incidence.hyperedge_degrees == 0)[0]

    for idx in zero_vertex_rows:
        warnings.append(f"isolated vertex: {incidence.vertex_ids[idx]}")
    for idx in zero_hyperedge_cols:
        warnings.append(f"empty hyperedge: {incidence.hyperedge_ids[idx]}")
    return warnings
