"""Compute node model.

Models heterogeneous resources across the three layers (Mist / Fog / Cloud):
CPU, memory, queue, bandwidth, network link, and battery state for the
drone scenario.  Resources are tracked per-node in a homogeneous manner so
that the simulator can deterministically enforce capacity constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ..utils.config import Config
from .network_link import NetworkLink


@dataclass
class ComputeNode:
    """A single computing resource (Mist device / Fog gateway / Cloud host)."""

    node_id: str
    layer: str  # "Mist" | "Fog" | "Cloud"
    cpu_capacity: float = 1.0
    memory_capacity: float = 1.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_length: int = 0
    available_bandwidth: float = 100.0  # Mbps
    network_latency_ms: float = 1.0
    is_battery_powered: bool = False
    battery_capacity_wh: float = 100.0
    battery_remaining_wh: float = 100.0
    battery_charging: bool = False
    network_interface: Optional[NetworkLink] = None
    max_queue_length: int = 1000
    name: str = ""

    # Internal state (simulator owns these).
    _active_tasks: Dict[str, float] = field(
        default_factory=dict, repr=False, init=False
    )  # task_id -> remaining execute time
    _arrival_times: Dict[str, float] = field(
        default_factory=dict, repr=False, init=False
    )  # task_id -> arrival
    _release_times: Dict[str, float] = field(
        default_factory=dict, repr=False, init=False
    )  # task_id -> completion

    def __post_init__(self):
        # Ensure capacity is positive to avoid division-by-zero later.
        self.cpu_capacity = float(self.cpu_capacity)
        self.memory_capacity = float(self.memory_capacity)
        self.available_bandwidth = float(self.available_bandwidth)
        self.cpu_utilization = min(1.0, self.cpu_utilization)
        self.memory_utilization = min(1.0, self.memory_utilization)
        self.queue_length = max(0, self.queue_length)
        if self.network_interface is not None:
            if self.network_interface.bandwidth <= 0:
                self.available_bandwidth = max(self.available_bandwidth, 1.0)
        if self.battery_capacity_wh <= 0:
            self.battery_remaining_wh = 0.0

    # ------------------------------------------------------------------
    # Capacity & utilisation queries.
    # ------------------------------------------------------------------
    def remaining_cpu(self) -> float:
        """Available CPU capacity (fraction) at this instant."""
        return max(0.0, self.cpu_capacity - self.cpu_utilization)

    def remaining_memory(self) -> float:
        return max(0.0, self.memory_capacity - self.memory_utilization)

    def cpu_fraction(self) -> float:
        return min(1.0, self.cpu_utilization / self.cpu_capacity) if self.cpu_capacity else 0.0

    def memory_fraction(self) -> float:
        return min(1.0, self.memory_utilization / self.memory_capacity) if self.memory_capacity else 0.0

    # ------------------------------------------------------------------
    # Battery state (drone scenario).
    # ------------------------------------------------------------------
    def battery_remaining(self) -> float:
        return float(self.battery_remaining_wh)

    def battery_capacity(self) -> float:
        return float(self.battery_capacity_wh)

    def battery_fraction(self) -> float:
        return self.battery_remaining() / self.battery_capacity() if self.battery_capacity() else 0.0

    def consume_battery(self, joules: float) -> None:
        """Drain battery by ``joules`` (J).  Units: 1 Wh = 3600 J."""
        if self.is_battery_powered and self.battery_remaining_wh > 0:
            self.battery_remaining_wh -= joules / 3600.0
            if self.battery_remaining_wh < 0:
                self.battery_remaining_wh = 0.0

    def is_battery_depleted(self) -> bool:
        return self.battery_remaining_wh <= 0

    # ------------------------------------------------------------------
    # Queueing helpers.
    # ------------------------------------------------------------------
    def enqueue_task(self, task_id: str, arrival_time: float) -> None:
        self._active_tasks[task_id] = 0.0
        self._arrival_times[task_id] = arrival_time
        self.queue_length = max(0, self.queue_length + 1)

    def dequeue_task(self, task_id: str) -> Optional[float]:
        """Remove a completed task; returns its completion time."""
        self._active_tasks.pop(task_id, None)
        self._arrival_times.pop(task_id, None)
        self._release_times.pop(task_id, None)
        self.queue_length = max(0, self.queue_length - 1)
        return self._release_times.pop(task_id, None)

    def record_execution_start(self, task_id: str, start_time: float) -> None:
        self._active_tasks[task_id] = 0.0
        self._arrival_times[task_id] = start_time

    def record_execution_finish(self, task_id: str, finish_time: float) -> None:
        self._active_tasks.pop(task_id, None)
        self._release_times[task_id] = finish_time
        self.queue_length = max(0, self.queue_length - 1)

    def add_cpu_utilization(self, delta: float) -> None:
        self.cpu_utilization = min(
            self.cpu_capacity, self.cpu_utilization + float(delta)
        )

    def add_memory_utilization(self, delta: float) -> None:
        self.memory_utilization = min(
            self.memory_capacity, self.memory_utilization + float(delta)
        )

    def reset_utilization(self) -> None:
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.queue_length = 0

    # ------------------------------------------------------------------
    # Serialisation.
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "layer": self.layer,
            "cpu_capacity": self.cpu_capacity,
            "memory_capacity": self.memory_capacity,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "queue_length": self.queue_length,
            "available_bandwidth": self.available_bandwidth,
            "network_latency_ms": self.network_latency_ms,
            "is_battery_powered": self.is_battery_powered,
            "battery_capacity_wh": self.battery_capacity_wh,
            "battery_remaining_wh": self.battery_remaining_wh,
            "battery_charging": self.battery_charging,
            "max_queue_length": self.max_queue_length,
            "active_count": len(self._active_tasks),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ComputeNode":
        return cls(**data)
