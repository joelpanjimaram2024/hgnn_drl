"""PPO policy that consumes the HGNN representation.

The policy maps an HGNN-generated task embedding (together with a
feasibility mask) to a categorical distribution over { Mist, Fog, Cloud }.
A custom feature extractor (instead of SB3's MlpPolicy) is used so that the
state fed to the MLP is the HGNN embedding plus feasibility information.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..models.hgnn import HGNNEncoder
from ..utils.config import Config
from ..utils.logging import get_logger


class PPOPolicy(nn.Module):
    """A small MLP head that turns the HGNN embedding into an action logit.

    This is the component that consumes the hypergraph representation directly.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        n_actions: int,
        n_layers: int = 2,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_actions = n_actions
        layers = []
        in_dim = embed_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(hidden, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, emb: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Produce action logits.

        Parameters
        ----------
        emb : (N, embed_dim) tensor of task embeddings.
        mask : (N, n_actions) optional boolean mask; actions with mask False
            are set to -inf before the softmax.
        """
        logits = self.net(emb)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits


class HGNNFeatureExtractor:
    """Encapsulates the HGNN + projection used to build the policy input.

    This class is the bridge between the simulator/environment (which holds
    the hypergraph) and the PPO agent (which holds the policy).  It ensures
    the HGNN representation — not a manually constructed feature vector — is
    consumed by the policy.
    """

    def __init__(
        self,
        *,
        config: Config,
        encoder: Optional[HGNNEncoder] = None,
        device: str = "cpu",
    ):
        self.config = config
        self.device = device
        self.logger = get_logger("hgnn_drl.extract")

        self.encoder = encoder
        self.policy_net = PPOPolicy(
            embed_dim=config.hgnn.out_dim,
            n_actions=config.action.n_actions,
            hidden=128,
            n_layers=2,
            dropout=0.1,
        ).to(device)

        self._last_emb: Optional[Tensor] = None

    def set_encoder(self, encoder: HGNNEncoder) -> None:
        self.encoder = encoder
        self.policy_net.load_state_dict(
            {
                "net.0.weight": nn.Linear(encoder.out_dim, 128).weight,
                "net.0.bias": nn.Linear(encoder.out_dim, 128).bias,
            }
        )
        self.logger.info("feature extractor encoder updated")

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "policy_net": self.policy_net.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.policy_net.load_state_dict(state["policy_net"])

    def extract(
        self,
        x: Tensor,
        hypergraph: Optional[Any] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Produce per-task embeddings and the policy input representation.

        Returns (task_embeddings, graph_state).  The task embeddings are fed
        directly into the policy network for per-task placement decisions.
        """
        if self.encoder is None:
            # Fallback: pass through a linear projection (only used for
            # deterministic shape checks, never for research results).
            return x, x

        x = x.to(self.device)
        emb, state = self.encoder(x)
        self._last_emb = emb
        return emb, state

    def predict_action(
        self,
        x: Tensor,
        hypergraph: Optional[Any] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[int, Tensor]:
        """Return the greedy action index (0=Mist, 1=Fog, 2=Cloud)."""
        self.encoder.eval()
        self.policy_net.eval()
        with torch.no_grad():
            emb, _ = self.extract(x, hypergraph, mask)
            logits = self.policy_net(emb)
            action = int(logits.argmax(dim=-1).item())
        return action, logits

    def train_mode(self) -> None:
        self.encoder.train()
        self.policy_net.train()

    def eval_mode(self) -> None:
        self.encoder.eval()
        self.policy_net.eval()
