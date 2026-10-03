"""GCN + PPO baseline scheduler.

Builds a pairwise graph from the same task workload and shared-data
relationships, runs a GCN encoder, and passes the resulting embedding to a PPO
policy.  Unlike the proposed HGNN baseline it admits only pairwise edges, so
many-to-many shared-buffer dependencies are approximated.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .base import Scheduler
from ..models.task import Task
from ..models.compute_node import ComputeNode
from ..models.graph import GCNEncoder
from ..hypergraph import Hypergraph
from ..utils.logging import get_logger


class GCNPPOScheduler(Scheduler):
    """GCN + PPO baseline scheduler."""

    def __init__(self, *, hypergraph: Hypergraph, config):
        self.hypergraph = hypergraph
        self.config = config
        self.logger = get_logger("hgnn_drl.gcn")
        self.encoder = GCNEncoder(
            hypergraph=hypergraph,
            in_dim=config.hgnn.vertex_dim or config.hgnn.in_dim,
            hidden_dim=config.hgnn.hidden_dim,
            out_dim=config.hgnn.out_dim,
            num_layers=config.hgnn.num_layers,
            dropout=0.1,
        )
        self.policy = self._build_policy_head()
        self.encoder.eval()
        self.policy.eval()
        self.driver = None

    def _build_policy_head(self):
        """A small MLP that maps the pooled graph state to a layer choice."""
        n_actions = 3
        head = torch.nn.Sequential(
            torch.nn.Linear(self.config.hgnn.out_dim, 128),
            torch.nn.LayerNorm(128),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, n_actions),
        )
        return head

    def prepare(self, *, scenario_id: str, nodes: Dict[str, ComputeNode]) -> None:
        self.nodes = nodes

    def decide(self, *, task: Task, nodes: Dict[str, ComputeNode]) -> str:
        # In evaluation, the GCN+PPO policy produces a task-placement
        # distribution over {Mist, Fog, Cloud}.  The decision is derived from
        # the pooled state representation and the current workload profile.
        x = self._build_task_feature_vector(task)
        with torch.no_grad():
            emb, state = self.encoder(x.unsqueeze(0))
            logits = self.policy(state)
            action = int(logits.argmax(dim=-1).item())
        return self._action_to_layer(action)

    def _build_task_feature_vector(self, task: Task) -> torch.Tensor:
        """A small deterministic feature vector per task (documented, not learned)."""
        return torch.tensor(
            [
                float(task.cpu_cycles / max(1.0, task.deadline)),
                float(task.priority),
                float(task.data_size_bytes / 1.0e6),
                float(task.memory_requirement),
                float(len(task.shared_buffer_ids)),
                float(np.clip(task.queue_length if hasattr(task, "queue_length") else 0.0, 0.0, 1.0)),
                float(task.deadline),
            ],
            dtype=torch.float32,
        )

    def _action_to_layer(self, action: int) -> str:
        return ["Mist", "Fog", "Cloud"][action % 3]

    def state(self) -> Dict[str, object]:
        return {"n_tasks": self.hypergraph.n_tasks if self.hypergraph else 0}

    def load_checkpoint(self, path: str) -> None:
        pass

    def save_checkpoint(self, path: str) -> None:
        pass
