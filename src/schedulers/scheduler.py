"""Scheduler abstraction for the Mist-Fog-Cloud environment.

The scheduler decides, for each schedulable task, which execution layer
(Mist / Fog / Cloud) gets the task.  Concrete implementations include the
proposed HGNN+PPO policy, the baselines (Cloud-only, Mist-only, Greedy),
and the GCN+PPO comparison.

This module is exported so third-party code can do::

    from scheduler import Scheduler

and still get the intended behaviour (a class instance), which is the
correct fix over forcing ``Scheduler`` to be both a class and a function.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from ..models.task import Task
from ..models.compute_node import ComputeNode


class Scheduler(ABC):
    """Base class for all schedulers.

    A scheduler is queried once per decision epoch (per task arrival and/or
    per simulation step) and returns a placement decision.

    This class is the *only* definition of the "Scheduler" name.  Third-party
    modules that want to implement a custom scheduler simply subclass it and
    implement :meth:`decide`.
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

    @abstractmethod
    def decide(self, *, task: Task, nodes: Dict[str, ComputeNode]) -> str:
        """Return the layer name for *task*.

        Expected return value: "Mist", "Fog", or "Cloud".
        """

    @abstractmethod
    def state(self) -> Dict[str, object]:
        """Return the scheduler's current state for logging / checkpointing."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load a saved policy / scheduler state."""

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Persist the scheduler's state."""


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


# Keep a thin backwards-compatible alias for code that imported the class
# directly from this module before the rewrite.
SchedulerInstance = Scheduler
