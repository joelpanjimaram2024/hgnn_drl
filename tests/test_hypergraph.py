"""Tests for hypergraph construction, incidence matrix, and degrees."""

import numpy as np
import pytest

from src.hypergraph import (
    HypergraphBuilder,
    build_incidence_matrix,
    validate_hypergraph,
    IncidenceMatrix,
)


def test_build_incidence_matrix_shared_buffer():
    """Tasks sharing a buffer belong to the same hyperedge."""
    vertex_ids = ["t0", "t1", "t2"]
    hyperedges = {"E_cam": ["t0", "t1", "t2"]}
    inc = build_incidence_matrix(vertex_ids=vertex_ids, hyperedge_id_to_vertex_ids=hyperedges)
    assert inc.n_vertices == 3
    assert inc.n_hyperedges == 1
    arr = np.asarray(inc.to_numpy())
    assert arr[0, 0] == 1.0 and arr[1, 0] == 1.0 and arr[2, 0] == 1.0
    assert arr.sum() == 3.0


def test_multiple_hyperedge_membership():
    """A task may belong to multiple hyperedges when it consumes multiple inputs."""
    vertex_ids = ["t0", "t1", "t2"]
    hyperedges = {
        "E_label": ["t0", "t1"],
        "E_weight": ["t0", "t2"],
    }
    inc = build_incidence_matrix(vertex_ids=vertex_ids, hyperedge_id_to_vertex_ids=hyperedges)
    assert inc.n_hyperedges == 2
    arr = np.asarray(inc.to_numpy())
    # t0 belongs to both hyperedges.
    assert arr[0].sum() == 2.0
    # t1 only in E_label.
    assert arr[1].sum() == 1.0
    # t2 only in E_weight.
    assert arr[2].sum() == 1.0


def test_weighted_hyperedge():
    """Hyperedges can carry configurable weights."""
    vertex_ids = ["t0", "t1"]
    hyperedges = {"E1": ["t0", "t1"]}
    weights = {"E1": 2.5}
    inc = build_incidence_matrix(
        vertex_ids=vertex_ids,
        hyperedge_id_to_vertex_ids=hyperedges,
        weights=weights,
    )
    arr = np.asarray(inc.to_numpy())
    assert arr[0, 0] == 2.5 and arr[1, 0] == 2.5
    # Vertex degree = sum of weights.
    assert inc.vertex_degrees[0] == 2.5


def test_validate_empty_hyperedge():
    """Empty hyperedges raise validation warnings."""
    vertex_ids = ["t0", "t1"]
    hyperedges = {"E1": ["t0"], "E2": []}
    inc = build_incidence_matrix(vertex_ids=vertex_ids, hyperedge_id_to_vertex_ids=hyperedges)
    warnings = validate_hypergraph(inc)
    assert any("E2" in w for w in warnings), f"empty hyperedge not flagged: {warnings}"


def test_validate_isolated_vertex():
    """Isolated (degree-zero) vertices are flagged by validation."""
    vertex_ids = ["t0", "t1"]
    hyperedges = {"E1": ["t0"]}
    inc = build_incidence_matrix(vertex_ids=vertex_ids, hyperedge_id_to_vertex_ids=hyperedges)
    warnings = validate_hypergraph(inc)
    assert any("t1" in w for w in warnings), f"isolated vertex not flagged: {warnings}"
    print(f"[test] isolated-vert warnings={warnings}")


def test_hypergraph_builder_shared_buffer():
    """Builder adds tasks to shared buffers and produces a valid hypergraph."""
    builder = HypergraphBuilder()
    for i in range(4):
        builder.register_task(f"t{i}", None)
    builder.add_shared_buffer("E_cam", ["t0", "t1", "t2", "t3"])
    builder.set_buffer_weights("E_cam", 1.0)
    hg = builder.build()
    assert hg.n_tasks == 4
    assert hg.n_hyperedges == 1
    assert len(hg.warnings) == 0


def test_hypergraph_builder_validation():
    """Builder reports warnings for empty hyperedges and isolated vertices."""
    builder = HypergraphBuilder()
    builder.register_task("t0", None)
    builder.register_task("t1", None)
    builder.add_shared_buffer("E1", ["t0"])
    # t1 is isolated (no buffer).
    # E1 is fine; no empty hyperedges here.
    hg = builder.build()
    assert any("isolated" in w for w in hg.warnings)


def test_incidence_field_initialisation():
    """IncidenceMatrix can be constructed directly with explicit fields."""
    inc = IncidenceMatrix(
        n_vertices=2,
        n_hyperedges=2,
        data=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert inc.n_vertices == 2
    assert inc.n_hyperedges == 2
    assert inc.vertex_degrees.shape == (2,)
    assert inc.hyperedge_degrees.shape == (2,)
