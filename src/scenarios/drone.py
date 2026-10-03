"""Case Study B: Autonomous Drone Fleet for Precision Agriculture.

Eight drones perform soil-moisture analysis, crop-disease detection,
irrigation monitoring, geospatial mapping, weather telemetry, trajectory
planning, collision avoidance, and crop-yield modelling.  Simulated
hyperedges connect shared geographical maps, synchronized weather buffers,
and fleet-state telemetry.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..models.compute_node import ComputeNode
from ..models.task import Task
from ..utils.config import Config
from .base import ScenarioGenerator


class DroneScenario(ScenarioGenerator):
    """Drone-fleet precision-agriculture generator."""

    NAME = "drone"

    # Task classes with expected layer tendencies (not hard rules).
    TASK_CLASSES = [
        "soil_moisture",
        "crop_disease",
        "irrigation_monitoring",
        "geospatial_mapping",
        "weather_telemetry",
        "trajectory_planning",
        "collision_avoidance",
        "crop_yield_modeling",
    ]

    def __init__(self, config: Config):
        super().__init__(config)
        self._node_ids = self._build_node_ids()

    def name(self) -> str:
        return self.NAME

    def _build_node_ids(self) -> Dict[str, ComputeNode]:
        res = self.config.resources
        nodes: Dict[str, ComputeNode] = {}
        for d in range(self.config.scenario.num_drones):
            # Drone host node.
            battery = self.config.scenario.drone_battery_initial_wh
            nodes[f"Drone_{d}"] = ComputeNode(
                node_id=f"Drone_{d}",
                layer="Mist",
                cpu_capacity=res.cpu_mist,
                memory_capacity=res.memory_mist,
                available_bandwidth=res.bandwidth,
                network_latency_ms=2.0,
                is_battery_powered=True,
                battery_capacity_wh=self.config.scenario.drone_battery_capacity_wh,
                battery_remaining_wh=battery,
            )
        for i in range(self.config.scenario.num_fog_nodes):
            nodes[f"Fog_{i}"] = ComputeNode(
                node_id=f"Fog_{i}",
                layer="Fog",
                cpu_capacity=res.cpu_fog,
                memory_capacity=res.memory_fog,
                available_bandwidth=res.bandwidth,
                network_latency_ms=5.0,
            )
        for i in range(self.config.scenario.num_cloud_nodes):
            nodes[f"Cloud_{i}"] = ComputeNode(
                node_id=f"Cloud_{i}",
                layer="Cloud",
                cpu_capacity=res.cpu_cloud,
                memory_capacity=res.memory_cloud,
                available_bandwidth=res.bandwidth,
                network_latency_ms=30.0,
            )
        return nodes

    def generate_nodes(self) -> Dict[str, ComputeNode]:
        return dict(self._node_ids)

    def generate_tasks(self, *, n_tasks: int, seed: int) -> List[Task]:
        rng = np.random.default_rng(seed)
        tasks: List[Task] = []
        n_drones = self.config.scenario.num_drones
        per_drone = max(1, n_tasks // n_drones)

        for d in range(n_drones):
            t0 = d * 0.2
            for ti, task_class in enumerate(self.TASK_CLASSES):
                i = ti + d * len(self.TASK_CLASSES)
                if i >= n_tasks:
                    break
                is_collision = task_class == "collision_avoidance"
                deadline = 0.3 if is_collision else float(rng.uniform(0.6, 1.5))
                cpu = float(rng.uniform(0.2, 0.7))
                mem = float(rng.uniform(0.2, 0.7))
                priority = 1.0 if is_collision else float(rng.uniform(0.0, 0.9))
                tasks.append(
                    Task(
                        task_id=f"drone_{d}_{task_class}",
                        scenario_id=self.name(),
                        arrival_time=float(t0 + i * 0.01),
                        data_size_bytes=int(rng.uniform(5.0e5, 3.0e6)),
                        cpu_cycles=cpu,
                        memory_requirement=mem,
                        deadline=deadline,
                        priority=priority,
                        input_buffer_ids=[],
                        shared_buffer_ids=[],
                        source_node_id=None,
                        predecessor_task_ids=[],
                        task_type=task_class,
                        battery_sensitive=True,
                    )
                )
        return tasks[:n_tasks]

    def scenario_config(self) -> Dict[str, object]:
        return {
            "num_drones": self.config.scenario.num_drones,
            "num_fog_nodes": self.config.scenario.num_fog_nodes,
            "num_cloud_nodes": self.config.scenario.num_cloud_nodes,
            "drone_battery_capacity_wh": self.config.scenario.drone_battery_capacity_wh,
            "drone_battery_initial_wh": self.config.scenario.drone_battery_initial_wh,
        }
