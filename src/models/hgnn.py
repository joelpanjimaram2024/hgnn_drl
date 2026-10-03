"""Trainable Hypergraph Neural Network (HGNN) encoder.

Implements a mathematically valid hypergraph convolution:

    X_next = activation(
        D_v^(-1/2)  H  W_e  D_e^(-1)  H^T  D_v^(-1/2)  X  W
    )

where
    H  : task-hyperedge incidence matrix
    D_v: vertex-degree matrix
    D_e: hyperedge-degree matrix
    W_e: hyperedge-weight diagonal
    X  : task feature matrix
    W  : trainable weight matrix

Supporting pieces:
    - Graph (pairwise) GCN variant, used by the GCN+PPO baseline.
    - Task-level pooling to a fixed-size state vector for the DRL policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..hypergraph import IncidenceMatrix, Hypergraph

try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy missing
    _HAS_SCIPY = False
    sp = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Numerically stable block diagonal degree matrices.
# ---------------------------------------------------------------------------
def _diagonal_from(degrees: np.ndarray, *, for_inverse_sqrt: bool = False) -> Tensor:
    """Convert a degree vector to a 2-D diagonal tensor.

    ``inverse_sqrt`` only affects rows where the degree is zero (they are set
    to zero, never one).
    """
    d = np.asarray(degrees, dtype=np.float64).ravel().copy()
    if for_inverse_sqrt:
        # Numerically stable: only invert where degree > 0.
        d = np.where(d > 0.0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    else:
        d = np.where(d > 0.0, d, 0.0)
    return torch.tensor(d, dtype=torch.float32, requires_grad=False)


class _BlockDiag(nn.Module):
    """Sparse wrapper for a diagonal degree matrix."""

    def __init__(self, incidence: IncidenceMatrix, *, normalize_inverse_sqrt: bool = False):
        super().__init__()
        self.incidence = incidence
        self.is_inverse_sqrt = normalize_inverse_sqrt

    def forward(self, x: Tensor) -> Tensor:
        """Multiply x by the diagonal matrix (left or right)."""
        diag = _diagonal_from(
            self.incidence.vertex_degrees, for_inverse_sqrt=self.is_inverse_sqrt
        )
        # diag @ x  (left multiply).  For symmetric normalisation we need the
        # same scaling applied on both sides around the H W_e D_e^-1 H^T block.
        return diag * x


# ---------------------------------------------------------------------------
# Core hypergraph convolution.
# ---------------------------------------------------------------------------
class HypergraphConv(nn.Module):
    """One trainable hypergraph-aggregation layer."""

    def __init__(
        self,
        *,
        incidence: IncidenceMatrix,
        in_dim: int,
        out_dim: int,
        use_hyperedge_weights: bool = True,
        degree_normalization: str = "symmetric",
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        weight_initializer: str = "glorot",
    ):
        super().__init__()
        self.incidence = incidence
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.use_hyperedge_weights = use_hyperedge_weights
        self.degree_normalization = degree_normalization
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Trainable weight matrix W.
        if weight_initializer == "glorot":
            bound = np.sqrt(6.0 / (in_dim + out_dim))
            w_init = torch.empty(in_dim, out_dim).uniform_(-bound, bound)
        elif weight_initializer == "xavier":
            w_init = torch.empty(in_dim, out_dim)
            nn.init.xavier_uniform_(w_init)
        elif weight_initializer == "kaiming":
            w_init = torch.empty(in_dim, out_dim)
            nn.init.kaiming_uniform_(w_init, a=np.sqrt(5))
        else:
            raise ValueError(f"Unknown weight_initializer: {weight_initializer}")
        self.W = nn.Parameter(w_init)

        # Optional per-hyperedge weight vector (trainable or fixed).
        if self.use_hyperedge_weights and incidence.n_hyperedges > 0:
            if degree_normalization == "weighted":
                # Learnable hyperedge scale; initialised to 1/N so sum ~ 1.
                self.W_e = nn.Parameter(torch.ones(incidence.n_hyperedges) / max(incidence.n_hyperedges, 1))
            else:
                self.W_e = None
        else:
            self.W_e = None

        # Degree matrices (moved with the module).
        self.register_buffer("_D_v_sqrt", _diagonal_from(incidence.vertex_degrees, for_inverse_sqrt=True))
        self.register_buffer("_D_v", _diagonal_from(incidence.vertex_degrees, for_inverse_sqrt=False))
        self.register_buffer("_D_e", _diagonal_from(incidence.hyperedge_degrees, for_inverse_sqrt=False))

        self._cached_shifted: Optional[Tensor] = None

    # -- helpers -----------------------------------------------------------
    def _shifted_incidence(self, x: Optional[Tensor] = None) -> Tensor:
        """Return H under a shared-data causal shift.

        For the research claim we need H, not just the adjacency matrix.
        Caching avoids re-running sparse ops every forward pass.
        """
        if self._cached_shifted is None:
            data = self.incidence.to_scipy() if _HAS_SCIPY else self.incidence.to_numpy()
            self._cached_shifted = data
        return self._cached_shifted

    # -- main math ---------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        hyperedge_ids: Optional[List[str]] = None,
        vertex_ids: Optional[List[str]] = None,
    ) -> Tensor:
        """Run the hypergraph convolution.

        Implements the mathematically valid hypergraph convolution:

            X_next = activation(
                D_v^(-1/2)  H  W_e  D_e^(-1)  H^T  D_v^(-1/2)  X  W
            )

        where
            H  : task-hyperedge incidence matrix (N, E)
            D_v: vertex-degree matrix (N, N) diagonal
            D_e: hyperedge-degree matrix (E, E) diagonal
            W_e: hyperedge-weight diagonal (E, E)
            X  : task feature matrix (N, F_in)
            W  : trainable weight matrix (F_in, F_out)

        Parameters
        ----------
        x : (N, F) tensor of task features.
        hyperedge_ids, vertex_ids : optional, used when x is already batched.

        Returns
        -------
        y : (N, F_out) tensor of updated task embeddings.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        N, F_in = x.shape
        assert self.in_dim == F_in, (
            f"input dim mismatch: got {F_in}, expected {self.in_dim}"
        )

        _H = self.incidence.to_scipy() if _HAS_SCIPY else self.incidence.to_numpy()
        if sp is not None and sp.issparse(_H):
            H = _H.toarray()
        else:
            H = np.asarray(_H)
        H = torch.tensor(np.asarray(H, dtype=np.float32), dtype=torch.float32)

        # Slice the incidence matrix to match the batch size (variable batch
        # handling).  The adjacency matrix A is (N, N) where N = batch size.
        n_batch = H.shape[0]
        if H.shape[0] != N:
            H = H[:N]
            _D_v_sqrt = self._D_v_sqrt[:N]
            _D_v = self._D_v[:N]
            _D_e = self._D_e
        else:
            _D_v_sqrt = self._D_v_sqrt
            _D_v = self._D_v
            _D_e = self._D_e

        # D_v^(-1/2)
        D_v_inv_sqrt = _D_v_sqrt.to(H.device)  # (N,)

        # D_e^(-1)
        D_e_inv = _D_e.to(H.device).reciprocal()  # (E,)

        # ---- Build the message-passing matrix:
        #     A = D_v^(-1/2) H W_e D_e^(-1) H^T D_v^(-1/2)
        #
        # Step 1: D_v^(-1/2) H  =>  (N, E)  via row scaling
        A = H * D_v_inv_sqrt.unsqueeze(1)

        # Step 2: D_v^(-1/2) H W_e  =>  (N, E)  via column scaling by W_e
        if self.W_e is not None:
            A = A * self.W_e.to(H.device).unsqueeze(0)

        # Step 3: D_v^(-1/2) H W_e D_e^(-1)  =>  (N, E)  via column scaling by D_e^(-1)
        A = A * D_e_inv.unsqueeze(0)

        # Step 4: (D_v^(-1/2) H W_e D_e^(-1)) H^T  =>  (N, N)  via matrix multiply
        A = A @ H.t()

        # Step 5: D_v^(-1/2) A  =>  (N, N)  via row scaling by D_v^(-1/2)
        A = A * D_v_inv_sqrt.unsqueeze(1)

        # Step 6: D_v^(-1/2) A D_v^(-1/2)  =>  (N, N)  via column scaling by D_v^(-1/2)
        A = A * D_v_inv_sqrt.unsqueeze(0)

        # ---- Apply the transformation to task features
        W = self.W.to(x.device)
        xw = x @ W

        y = A @ xw
        y = self.activation(y)
        y = self.dropout(y)
        return y


class HGNNEncoder(nn.Module):
    """Stack of hypergraph convolution layers ending in a fixed-size state.

    Produces per-task embeddings (batch-first) plus an optional graph-level
    pooled representation.  Designed for variable numbers of tasks.
    """

    def __init__(
        self,
        *,
        hypergraph: Hypergraph,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        activation_name: str = "relu",
        use_attention: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.hypergraph = hypergraph
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        act_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}.get(
            activation_name.lower(), nn.ReLU()
        )
        self.activation_name = activation_name
        self.activation = act_fn

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            layers.append(
                HypergraphConv(
                    incidence=hypergraph.incidence,
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    use_hyperedge_weights=hypergraph.incidence.n_hyperedges > 0,
                    degree_normalization="symmetric",
                    activation=nn.Identity() if i == num_layers - 1 else act_fn,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Task-level projection to a fixed policy-invariant embedding.
        self.task_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

        # Optional graph-level pooling (mean over tasks).
        self.pooling = "mean"
        self.graph_proj = nn.Linear(out_dim, out_dim)

        self._task_hidden: Optional[Tensor] = None

    def forward(
        self,
        x: Tensor,
        task_ids: Optional[List[str]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through the HGNN encoder.

        Parameters
        ----------
        x : (N, F) tensor of task features.
        task_ids : optional task indentifiers used for pooling.

        Returns
        -------
        task_embeddings : (N, out_dim)
        graph_state : (1, out_dim)
        """
        h = x
        for layer in self.layers:
            h = layer(h)

        self._task_hidden = h
        task_embeddings = self.task_proj(h)

        if task_embeddings.dim() == 1:
            task_embeddings = task_embeddings.unsqueeze(0)

        if self.pooling == "mean" and task_embeddings.shape[0] > 0:
            graph_state = task_embeddings.mean(dim=0, keepdim=True)
        else:
            graph_state = task_embeddings[:1].clone()

        graph_state = self.graph_proj(graph_state)
        return task_embeddings, graph_state

    @property
    def num_tasks(self) -> int:
        return self.incidence.n_vertices

    def get_task_embeddings(self) -> Optional[Tensor]:
        return self._task_hidden

    def shared_hyperedge_effect(self, x: Tensor, scheme: str = "uniform") -> Tuple[Tensor, Tensor]:
        """Toy analysis: does the convolution mix tasks that share a hyperedge?

        Returns the raw (un-normalised) output of a single dense layer with the
        same weights as ``self.layers[0]`` applied to every pair of tasks, plus
        a similarity signal that is high when the adjacency unit of the
        hypergraph connects them.  This is used to demonstrate representational
        effect, not to replace training.
        """
        with torch.no_grad():
            if not self.layers:
                return x, x
            layer = self.layers[0]
            H = self.incidence.to_scipy() if _HAS_SCIPY else self.incidence.to_numpy()
            H = torch.tensor(H, dtype=torch.float32)
            N = x.shape[0]

            # Build the message-passing matrix A = D_v^-1/2 H W_e D_e^-1 H^T D_v^-1/2
            D_v_inv_sqrt = _diagonal_from(self.incidence.vertex_degrees, for_inverse_sqrt=True).to(x.device)  # (N,)
            D_e_inv = _diagonal_from(self.incidence.hyperedge_degrees, for_inverse_sqrt=False).to(x.device).reciprocal()  # (E,)

            # Step 1: D_v^(-1/2) H  =>  (N, E)  via row scaling
            A = H * D_v_inv_sqrt.unsqueeze(1)

            # Step 2: D_v^(-1/2) H W_e  (W_e applied as column scaling if used)
            if layer.use_hyperedge_weights and layer.W_e is not None:
                A = A * layer.W_e.to(x.device).unsqueeze(0)

            # Step 3: D_v^(-1/2) H W_e D_e^(-1)  =>  (N, E)  via column scaling
            A = A * D_e_inv.unsqueeze(0)

            # Step 4: (D_v^(-1/2) H W_e D_e^(-1)) H^T  =>  (N, N)  via matrix multiply
            A = A @ H.t()

            # Step 5: D_v^(-1/2) A  =>  (N, N)  via row scaling
            A = A * D_v_inv_sqrt.unsqueeze(1)

            # Step 6: D_v^(-1/2) A D_v^(-1/2)  =>  (N, N)  via column scaling
            A = A * D_v_inv_sqrt.unsqueeze(0)

            W = layer.W.to(x.device)
            xw = x @ W
            y = A @ xw

            # Pairwise similarity induced by shared hyperedges.
            sim = (A @ A.t()).clamp(min=0.0)
            return y, sim


# ---------------------------------------------------------------------------
# Shape documentation.
# ---------------------------------------------------------------------------
def hgnn_shapes(*, n_tasks: int, n_hyperedges: int, in_dim: int, out_dim: int, num_layers: int) -> Dict[str, Tuple[int, ...]]:
    """Return the tensor shapes produced by the HGNN for the given sizes."""
    assert n_tasks > 0, "n_tasks must be positive"
    assert n_hyperedges >= 0, "n_hyperedges must be >= 0"
    shapes = {
        "feature_matrix_X": (n_tasks, in_dim),
        "incidence_H": (n_tasks, n_hyperedges),
        "output_task_embeddings": (n_tasks, out_dim),
        "graph_state": (1, out_dim),
        "per_layer_pre_activation": (n_tasks, out_dim),
    }
    return shapes
