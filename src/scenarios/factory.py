"""Case Study A: Automated Smart Factory Quality Inspection.

Simulates an automotive assembly line with several high-speed 4K cameras.
Four inspection tasks (crack detection, bolt alignment, weld quality, thermal
analysis) consume the same camera frame, forming the shared-data hyperedge

    E_camera = { Crack Detection, Bolt Alignment, Weld Quality, Thermal Analysis }
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..models.compute_node import ComputeNode
from ..models.task import Task
from ..utils.config import Config
from .base import ScenarioGenerator


class FactoryScenario(ScenarioGenerator):
    """Factory quality-inspection scenario generator."""

    NAME = "factory"

    def __init__(self, config: Config):
        super().__init__(config)
        self._node_ids = self._build_node_ids()

    def name(self) -> str:
        return self.NAME

    def _build_node_ids(self):
        res = self.config.resources
        nodes = {}
        for i in range(self.config.scenario.num_mist_nodes):
            nodes[f"Mist_{i}"] = ComputeNode(
                node_id=f"Mist_{i}",
                layer="Mist",
                cpu_capacity=res.cpu_mist,
                memory_capacity=res.memory_mist,
                available_bandwidth=res.bandwidth,
                network_latency_ms=1.0,
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
        camera_count = self.config.scenario.num_cameras
        per_camera = max(1, n_tasks // camera_count)

        # Camera frame size ~ 4K (configurable).
        frame_bytes = self.config.scenario.camera_frame_size_bytes

        for cam in range(camera_count):
            t0 = cam * 0.5  # staggered camera start
            # Four tasks per camera, each reading the same camera frame.
            for ti, label in enumerate(
                ["Crack Detection", "Bolt Alignment", "Weld Quality", "Thermal Analysis"]
            ):
                i = ti + cam * len(["Crack Detection", "Bolt Alignment", "Weld Quality", "Thermal Analysis"])
                if i >= n_tasks:
                    break
                cpu = float(rng.uniform(0.3, 0.9))
                mem = float(rng.uniform(0.3, 0.8))
                deadline = float(self.config.scenario.nominal_deadline_ms / 1000.0)
                # Every task shares the same camera buffer.
                tasks.append(
                    Task(
                        task_id=f"factory_camera{cam}_{label}",
                        scenario_id=self.name(),
                        arrival_time=float(t0 + i * 0.01),
                        data_size_bytes=int(frame_bytes),
                        cpu_cycles=cpu,
                        memory_requirement=mem,
                        deadline=deadline,
                        priority=float(rng.uniform(0.5, 1.0)),
                        input_buffer_ids=[f"camera_frame_{cam}"],
                        shared_buffer_ids=[f"camera_frame_{cam}"],
                        source_node_id=None,
                        predecessor_task_ids=[],
                        task_type="compute",
                        battery_sensitive=False,
                    )
                )
        return tasks[:n_tasks]

    def scenario_config(self) -> Dict[str, object]:
        return {
            "num_cameras": self.config.scenario.num_cameras,
            "camera_frame_size_bytes": self.config.scenario.camera_frame_size_bytes,
            "nominal_deadline_ms": self.config.scenario.nominal_deadline_ms,
            "num_mist_nodes": self.config.scenario.num_mist_nodes,
            "num_fog_nodes": self.config.scenario.num_fog_nodes,
            "num_cloud_nodes": self.config.scenario.num_cloud_nodes,
        }
