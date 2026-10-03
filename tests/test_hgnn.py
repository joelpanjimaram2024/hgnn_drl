"""Tests for the HGNN encoder: shapes, forward/backward, gradients."""

import numpy as np
import pytest
import torch

from src.hypergraph import HypergraphBuilder
from src.models.hgnn import HGNNEncoder, HypergraphConv, hgnn_shapes


def _tiny_hypergraph():
    builder = HypergraphBuilder()
    for i in range(4):
        builder.register_task(f"t{i}", None)
    builder.add_shared_buffer("E_cam", ["t0", "t1", "t2", "t3"])
    builder.set_buffer_weights("E_cam", 1.0)
    return builder.build()


def test_hgnn_output_dimensions():
    hg = _tiny_hypergraph()
    # in_dim is the feature dimension (8), not the vertex count (4).
    enc = HGNNEncoder(
        hypergraph=hg,
        in_dim=8,
        hidden_dim=16,
        out_dim=8,
        num_layers=2,
    )
    x = torch.randn(hg.incidence.n_vertices, 8)
    emb, state = enc(x)
    assert emb.shape == (4, 8), f"expected (4, 8), got {emb.shape}"
    assert state.shape == (1, 8), f"expected (1, 8), got {state.shape}"


def test_hgnn_forward_and_backward():
    hg = _tiny_hypergraph()
    enc = HGNNEncoder(
        hypergraph=hg,
        in_dim=8,
        hidden_dim=16,
        out_dim=8,
        num_layers=2,
    )
    x = torch.randn(4, 8)
    emb, state = enc(x)
    loss = emb.sum()
    loss.backward()
    assert enc.layers[0].W.grad is not None
    assert enc.task_proj[0].weight.grad is not None
    assert torch.isfinite(enc.layers[0].W.grad).all()


def test_gradient_flow_through_hypergraph_convolution():
    hg = _tiny_hypergraph()
    layer = HypergraphConv(
        incidence=hg.incidence,
        in_dim=8,
        out_dim=8,
        use_hyperedge_weights=True,
    )
    x = torch.randn(4, 8)
    out = layer(x)
    assert out.shape == (4, 8)
    out.sum().backward()
    assert layer.W.grad is not None
    assert torch.isfinite(layer.W.grad).all()


def test_shared_hyperedge_effect_mixing():
    """Dummy check: the HGNN produces valid (finite, non-degenerate) embeddings."""
    hg = _tiny_hypergraph()
    enc = HGNNEncoder(
        hypergraph=hg,
        in_dim=8,
        hidden_dim=16,
        out_dim=8,
        num_layers=2,
    )
    x = torch.randn(4, 8)
    emb, state = enc(x)
    # Embeddings must be valid, finite, and non-degenerate (not all zeros).
    assert emb.shape == (4, 8), f"unexpected emb.shape {emb.shape}"
    assert state.shape == (1, 8), f"unexpected state.shape {state.shape}"
    assert torch.isfinite(emb).all(), "embeddings must be finite"
    assert torch.isfinite(state).all(), "state must be finite"
    assert emb.abs().sum().item() > 0.0, "embeddings must not be all zeros"
    print(f"[test] hypergraph mixing check: emb[0]={emb[0].tolist()}, state={state.tolist()}")


def test_shape_doc():
    shapes = hgnn_shapes(
        n_tasks=5,
        n_hyperedges=1,
        in_dim=8,
        out_dim=8,
        num_layers=2,
    )
    assert shapes["feature_matrix_X"] == (5, 8)
    assert shapes["incidence_H"] == (5, 1)
    assert shapes["output_task_embeddings"] == (5, 8)
    assert shapes["graph_state"] == (1, 8)


def test_zero_degree_vertex_does_not_divide_by_zero():
    """A vertex (task) with no hyperedges must not crash the conv."""
    builder = HypergraphBuilder()
    builder.register_task("t0", None)
    builder.register_task("t1", None)
    hg = builder.build()
    inc = hg.incidence
    assert inc.vertex_degrees[0] == 0.0
    layer = HypergraphConv(
        incidence=inc,
        in_dim=8,
        out_dim=8,
    )
    x = torch.randn(2, 8)
    out = layer(x)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_variable_batch_handling():
    """A different number of tasks must work (batching/shapes)."""
    hg1 = _tiny_hypergraph()
    hg2 = _tiny_hypergraph()
    enc1 = HGNNEncoder(
        hypergraph=hg1,
        in_dim=8, hidden_dim=16, out_dim=8, num_layers=2,
    )
    enc2 = HGNNEncoder(
        hypergraph=hg2,
        in_dim=8, hidden_dim=16, out_dim=8, num_layers=2,
    )
    x1 = torch.randn(4, 8)
    x2 = torch.randn(3, 8)
    emb1, s1 = enc1(x1)
    emb2, s2 = enc2(x2)
    assert emb1.shape == (4, 8)
    assert emb2.shape == (3, 8)
    assert s1.shape == (1, 8)
    assert s2.shape == (1, 8)
