"""Base scenario generator.

Scenarios produce a task set and node set for a Mist-Fog-Cloud scenario.
The factory, drone-fleet, and ICU telemetry generators inherit this interface
so the experiment runner can treat them uniformly.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..models.compute_node import ComputeNode
from ..models.task import Task
from ..utils.config import Config


class ScenarioGenerator:
    """Base class for scenario workload generators."""

    def __init__(self, config: Config):
        self.config = config

    def name(self) -> str:
        return "base"

    def node_count(self) -> Tuple[int, int, int]:
        """Return (mist_nodes, fog_nodes, cloud_nodes)."""
        return (0, 0, 0)

    def generate_tasks(self, *, n_tasks: int, seed: int) -> List[Task]:
        """Generate *n_tasks* for this scenario."""
        raise NotImplementedError

    def generate_nodes(self) -> Dict[str, ComputeNode]:
        """Generate the compute-node graph for this scenario."""
        raise NotImplementedError

    def scenario_config(self) -> Dict[str, object]:
        """Return scenario-specific parameters for the experiment runner."""
        return {}
