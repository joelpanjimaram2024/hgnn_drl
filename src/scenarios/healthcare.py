"""Case Study C: Remote Healthcare and ICU Telemetry.

Synthetic patient-monitoring streams (multi-lead ECG, pulse oximetry, arterial
blood pressure) generate tasks for arrhythmia detection, sepsis-risk
estimation, vital-sign aggregation, alert dispatch, and historical trend
analysis.  Shared patient-vital buffers form the hyperedges.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..models.compute_node import ComputeNode
from ..models.task import Task
from ..utils.config import Config
from .base import ScenarioGenerator


class HealthcareScenario(ScenarioGenerator):
    """ICU / remote-healthcare telemetry scenario generator."""

    NAME = "healthcare"

    TASK_CLASSES = [
        "ecg_reading",
        "spo2_reading",
        "abp_reading",
        "arrhythmia_detection",
        "sepsis_risk_estimation",
        "vital_sign_aggregation",
        "alert_dispatch",
        "trend_analysis",
    ]

    PRIORITY_CLASSES = {
        "alert_dispatch": 0.95,
        "arrhythmia_detection": 0.85,
        "sepsis_risk_estimation": 0.75,
        "vital_sign_aggregation": 0.6,
        "ecg_reading": 0.5,
        "spo2_reading": 0.5,
        "abp_reading": 0.5,
        "trend_analysis": 0.2,
    }

    def __init__(self, config: Config):
        super().__init__(config)
        self._node_ids = self._build_node_ids()

    def name(self) -> str:
        return self.NAME

    def _build_node_ids(self) -> Dict[str, ComputeNode]:
        res = self.config.resources
        nodes: Dict[str, ComputeNode] = {}
        for i in range(self.config.scenario.num_health_stream_nodes):
            nodes[f"Monitor_{i}"] = ComputeNode(
                node_id=f"Monitor_{i}",
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
        n_monitors = self.config.scenario.num_health_stream_nodes
        per_monitor = max(1, n_tasks // n_monitors)

        for m in range(n_monitors):
            t0 = m * 0.15
            for ti, task_class in enumerate(self.TASK_CLASSES):
                i = ti + m * len(self.TASK_CLASSES)
                if i >= n_tasks:
                    break
                priority = self.PRIORITY_CLASSES[task_class]
                deadline = 0.3 if task_class == "alert_dispatch" else float(rng.uniform(0.5, 1.5))
                cpu = float(rng.uniform(0.1, 0.5))
                mem = float(rng.uniform(0.2, 0.6))
                tasks.append(
                    Task(
                        task_id=f"patient{m}_{task_class}",
                        scenario_id=self.name(),
                        arrival_time=float(t0 + i * 0.01),
                        data_size_bytes=int(rng.uniform(1.0e5, 8.0e5)),
                        cpu_cycles=cpu,
                        memory_requirement=mem,
                        deadline=deadline,
                        priority=priority,
                        input_buffer_ids=[],
                        shared_buffer_ids=[],
                        source_node_id=None,
                        predecessor_task_ids=[],
                        task_type=task_class,
                        battery_sensitive=False,
                    )
                )
        return tasks[:n_tasks]

    def scenario_config(self) -> Dict[str, object]:
        return {
            "num_monitors": self.config.scenario.num_health_stream_nodes,
            "num_fog_nodes": self.config.scenario.num_fog_nodes,
            "num_cloud_nodes": self.config.scenario.num_cloud_nodes,
        }
