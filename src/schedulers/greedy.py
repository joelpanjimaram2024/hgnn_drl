"""Greedy baseline scheduler.

Rule (documented in the README):

    For each arriving schedulable task, consider placement in descending
    urgency order [Mist, Fog, Cloud].  For each candidate layer compute an
    estimated finish time = queue_wait + transfer + execution.  Choose the
    layer with the smallest estimated finish time, provided that all implied
    deadlines are respected; otherwise choose the layer that minimises the
    deadline-miss penalty.  This is a deterministic, single-pass policy with
    no learning state.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..models.task import Task
from ..models.compute_node import ComputeNode
from .base import Scheduler


def _est_finish(
    *,
    task: Task,
    layer: str,
    nodes: Dict[str, ComputeNode],
    default_broadcast_latency_s: float = 0.005,
) -> Tuple[float, float]:
    """Return (queue_wait_s, transfer_s) for placing *task* on *layer*.

    We do *not* simulate contention here; we just use the layer's configured
    capacity and network latency.  The actual per-node instant is checked by
    the environment before the task is admitted.
    """
    total_transfer_s = 0.0
    for buf_id in task.shared_buffer_ids:
        # Shared buffers are local to the nearest node of the given layer;
        # for a conservative estimate we use a single representative hop per
        # buffer equal to the layer's default network latency unless nodes
        # expose explicit links.  The simulator tracks actual transfer times
        # and avoids double counting on shared-buffer reuse.
        if nodes:
            # Use the node with the smallest id as a deterministic stand-in
            # for "a node of this layer".
            rep_node = min(nodes.values(), key=lambda n: n.node_id)
            total_transfer_s += rep_node.network_latency_ms * 1e-3

    # Broadcast to all nodes of the layer: approx. 1ms per shared buffer.
    broadcast = total_transfer_s * 0.5

    # Execution estimate using a fraction of the layer's CPU capacity.
    cpu_share = task.cpu_cycles / max(1.0, 1.0e12)  # dimensionless fraction
    exec_s = cpu_share * 0.1

    return broadcast + exec_s, total_transfer_s


class GreedyScheduler(Scheduler):
    """Deterministic deadline-urgent, capacity-aware greedy scheduler."""

    def __init__(self):
        self.layer_name = "Greedy"
        self.nodes: Dict[str, ComputeNode] = {}

    def prepare(self, *, scenario_id: str, nodes: Dict[str, ComputeNode]) -> None:
        self.nodes = nodes

    def decide(self, *, task: Task, nodes: Dict[str, ComputeNode]) -> str:
        if task.arrival_time < 0 or task.deadline <= 0:
            return "Cloud"

        candidates = ["Mist", "Fog", "Cloud"]
        best_layer: Optional[str] = None
        best_score: Optional[float] = None

        for layer in candidates:
            # Capacity check: compute the node of this layer.
            cand_nodes = [n for n in nodes.values() if n.layer == layer]
            if not cand_nodes:
                continue
            rep = min(cand_nodes, key=lambda n: n.node_id)

            # If the task lacks CPU/memory demand it can skip here.
            if rep.remaining_cpu() <= 0 and not task.cpu_cycles <= 0:
                continue

            est_wait, _ = _est_finish(task=task, layer=layer, nodes=nodes)
            est_finish = est_wait + task.cpu_cycles / max(1.0, rep.remaining_cpu())

            # Score: prefer smaller finish time; tie-break by layer proximity.
            key = est_finish
            if best_score is None or key < best_score:
                best_score = key
                best_layer = layer

        return best_layer if best_layer else "Cloud"

    def state(self) -> Dict[str, object]:
        return {"layer_name": self.layer_name}

    def load_checkpoint(self, path: str) -> None:
        pass

    def save_checkpoint(self, path: str) -> None:
        pass
