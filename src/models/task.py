"""Task data model.

Each ``Task`` represents a unit of work that may be executed on one of the
Mist / Fog / Cloud computing layers.  Attributes mirror those requested in the
project plan (sections 5/9/16): arrival time, data size, CPU demand, deadline,
priority, shared buffers, predecessor constraints, energy, and deadline status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..utils.config import Config


@dataclass
class Task:
    """A single schedulable work item.

    Attributes mirror the project spec (sec 5 / 9 / 16).  All times are in
    simulated seconds, data sizes in bytes, CPU demand in CPU-cycles, memory
    in normalized units (0..1 per layer), and energy in Joules.
    """

    task_id: str
    scenario_id: str
    arrival_time: float

    # Resource & data attributes
    data_size_bytes: int = 0
    input_buffer_ids: List[str] = field(default_factory=list)
    cpu_cycles: float = 0.0
    memory_requirement: float = 0.0
    deadline: float = 0.0
    priority: float = 0.5

    # Provenance & topology
    source_node_id: Optional[str] = None
    shared_buffer_ids: List[str] = field(default_factory=list)
    predecessor_task_ids: List[str] = field(default_factory=list)
    task_type: str = "compute"
    battery_sensitive: bool = False

    # Scheduling outcomes
    assigned_execution_layer: Optional[str] = None   # "Mist" | "Fog" | "Cloud"
    execution_start_time: Optional[float] = None
    completion_time: Optional[float] = None
    energy_consumed: float = 0.0
    deadline_missed: bool = False
    queue_wait_time: float = 0.0
    transfer_time: float = 0.0
    compute_time: float = 0.0
    done: bool = False
    infeasible_at_arrival: bool = False

    # Derived (populated by the simulator)
    latency: float = 0.0
    layer: Optional[str] = None
    node_id: Optional[str] = None
    energy_compute: float = 0.0
    energy_transmit: float = 0.0
    energy_reception: float = 0.0
    released_time: float = 0.0  # first time any predecessor finished

    # Observable metadata used for dashboards / reporting
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Normalise types / defaults without changing external fields.
        self.arrival_time = float(self.arrival_time)
        self.cpu_cycles = float(self.cpu_cycles)
        self.memory_requirement = float(self.memory_requirement)
        self.deadline = float(self.deadline)
        self.priority = float(self.priority)
        self.data_size_bytes = int(self.data_size_bytes)
        self.input_buffer_ids = list(self.input_buffer_ids)
        self.shared_buffer_ids = list(self.shared_buffer_ids)
        self.predecessor_task_ids = list(self.predecessor_task_ids)

    # ------------------------------------------------------------------
    # Public helpers used by the simulator & tests.
    # ------------------------------------------------------------------
    def deadline_slack(self, current_time: float) -> float:
        """Remaining time before the task's deadline expires (s)."""
        return float(self.deadline - (current_time - self.arrival_time))

    def is_schedulable(self, *, available_cpu: float, available_memory: float) -> bool:
        """Feasibility check against a node's instantaneous capacity.

        Returns True when the task can be placed without exceeding CPU,
        memory, or deadline constraints.  Pre-emption is ignored for the
        first implementation; tasks not allowed to pre-empt are never
        migrated mid-flight.
        """
        if self.cpu_cycles <= 0 or self.memory_requirement <= 0:
            return True
        # Availability check: task fits in the remaining capacity.
        fits = available_cpu >= 1.0 and available_memory >= 1.0
        if not fits:
            return False
        if self.predecessor_task_ids:
            return False  # still waiting on predecessors
        return True

    def total_demand(self) -> float:
        """Weighted demand used for load balancing (units arbitrary)."""
        return self.cpu_cycles * self.priority

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable representation of the task."""
        return {
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "arrival_time": self.arrival_time,
            "data_size_bytes": self.data_size_bytes,
            "input_buffer_ids": list(self.input_buffer_ids),
            "cpu_cycles": self.cpu_cycles,
            "memory_requirement": self.memory_requirement,
            "deadline": self.deadline,
            "priority": self.priority,
            "source_node_id": self.source_node_id,
            "shared_buffer_ids": list(self.shared_buffer_ids),
            "predecessor_task_ids": list(self.predecessor_task_ids),
            "task_type": self.task_type,
            "battery_sensitive": self.battery_sensitive,
            "assigned_execution_layer": self.assigned_execution_layer,
            "execution_start_time": self.execution_start_time,
            "completion_time": self.completion_time,
            "energy_consumed": self.energy_consumed,
            "deadline_missed": self.deadline_missed,
            "done": self.done,
            "infeasible_at_arrival": self.infeasible_at_arrival,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Construct a Task from a serialisable dict."""
        excluded = {"meta"}
        kwargs = {k: v for k, v in data.items() if k not in excluded}
        task = cls(**kwargs)
        task.meta = data.get("meta", {})
        return task

    def __repr__(self):
        return (
            f"Task(id={self.task_id}, layer={self.assigned_execution_layer}, "
            f"deadline_missed={self.deadline_missed}, done={self.done})"
        )
