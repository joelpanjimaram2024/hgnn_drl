"""Pairwise Graph Convolutional Network (GCN) baseline.

The GCN+PPO baseline encodes tasks as nodes of a pairwise graph and learns
task embeddings with graph convolutions.  It is the comparison point for the
hypergraph encoder: it uses the same shared-data relations as edges (one
edge per shared buffer) but *cannot* express many-to-many hyperedges that
contain more than two vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from ..hypergraph import Hypergraph
except ImportError:
    Hypergraph = None
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class GraphConv(nn.Module):
    """One graph-convolution layer (symmetric normalisation, like GCN)."""

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        bound = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = nn.Parameter(torch.empty(in_dim, out_dim).uniform_(-bound, bound))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """x: (N, F_in); adj: (N, N) adjacency (symmetric normalised)."""
        xw = x @ self.W
        out = adj @ xw
        out = self.activation(out)
        return self.dropout(out)


class GCNEncoder(nn.Module):
    """Stack of graph convolution layers with a fixed-size pooled state."""

    def __init__(
        self,
        *,
        hypergraph: Hypergraph,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        activation_name: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hypergraph = hypergraph
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        act_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}.get(
            activation_name.lower(), nn.ReLU()
        )

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            layers.append(
                GraphConv(
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    dropout=dropout,
                    activation=nn.Identity() if i == num_layers - 1 else act_fn,
                )
            )
        self.layers = nn.ModuleList(layers)

        self.task_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
        self.graph_proj = nn.Linear(out_dim, out_dim)

    def _build_adjacency(self, hyperedge_ids: Optional[list] = None) -> Tensor:
        """Build the pairwise adjacency from the hypergraph incidence.

        Chosen order of construction: tasks that share a buffer are connected
        by a single edge whose weight is the incidence overlap count.
        """
        H_raw = self.hypergraph.incidence.to_scipy() if _HAS_SCIPY else self.hypergraph.incidence.to_numpy()
        if _HAS_SCIPY:
            import scipy.sparse as sp
            if sp.issparse(H_raw):
                H = H_raw.toarray()
            else:
                H = np.asarray(H_raw, dtype=np.float32)
        else:
            H = np.asarray(H_raw, dtype=np.float32)
        H = torch.tensor(H, dtype=torch.float32)
        N = H.shape[0]

        # Overlap adjacency: A = H @ H^T (tasks sharing buffers get high weight).
        A = H @ H.t()

        # Add self-loops for numerical stability.
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.where(deg > 0, 1.0 / torch.sqrt(torch.clamp(deg, min=1e-12)), 0.0)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        # Symmetrize (numerical guard).
        A_norm = (A_norm + A_norm.t()) / 2.0
        return A_norm

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        adj = self._build_adjacency()
        N_adj = adj.shape[0]

        # Broadcast a single-task feature vector (1 row) to all N tasks in the
        # hypergraph so that the graph convolution can use the full adjacency
        # matrix.  This is used by the schedulers that call the encoder with a
        # per-task feature vector while the encoder operates on the whole
        # hypergraph.
        if x.shape[0] == 1 and N_adj > 1:
            x = x.repeat(N_adj, 1)

        h = x
        for layer in self.layers:
            h = layer(h, adj)
        emb = self.task_proj(h)

        # If we broadcast the input, return only the embedding of the first task.
        if x.shape[0] != N_adj and x.shape[0] > 1:
            emb = emb[:N_adj]

        state = emb.mean(dim=0, keepdim=True)
        return emb, self.graph_proj(state)


# scipy availability (mirrors the hypergraph module).
try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from ..hypergraph import IncidenceMatrix
