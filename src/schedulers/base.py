"""Scheduler abstraction for the Mist-Fog-Cloud environment.

The scheduler decides, for each schedulable task, which execution layer
(Mist / Fog / Cloud) gets the task.  Concrete implementations include the
proposed HGNN+PPO policy, the baselines (Cloud-only, Mist-only, Greedy),
and the GCN+PPO comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ..models.task import Task
from ..models.compute_node import ComputeNode


class Scheduler(ABC):
    """Base class for all schedulers.

    A scheduler is queried once per decision epoch (per task arrival and/or
    per simulation step) and returns a placement decision.
    """

    @abstractmethod
    def prepare(self, *, scenario_id: str, nodes: Dict[str, ComputeNode]) -> None:
        """Prepare the scheduler with the current compute graph.

        Parameters
        ----------
        scenario_id : str
            Identifier of the active scenario.
        nodes : dict
            Mapping node_id -> ComputeNode.
        """
        raise NotImplementedError

    @abstractmethod
    def decide(self, *, task: Task, nodes: Dict[str, ComputeNode]) -> str:
        """Return the layer name for *task*.

        Expected return value: "Mist", "Fog", or "Cloud".
        """
        raise NotImplementedError

    @abstractmethod
    def state(self) -> Dict[str, object]:
        """Return the scheduler's current state for logging / checkpointing."""
        return {}

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load a saved policy / scheduler state."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Persist the scheduler's state."""
        raise NotImplementedError


class BaselineScheduler(Scheduler):
    """Shared machinery for stateless baselines (Cloud-only, Mist-only)."""

    layer: str  # class attribute set by subclasses

    def prepare(self, *, scenario_id: str, nodes: Dict[str, ComputeNode]) -> None:
        pass

    def decide(self, *, task: Task, nodes: Dict[str, ComputeNode]) -> str:
        return self.layer

    def state(self) -> Dict[str, object]:
        return {"layer": self.layer}

    def load_checkpoint(self, path: str) -> None:
        pass

    def save_checkpoint(self, path: str) -> None:
        pass
