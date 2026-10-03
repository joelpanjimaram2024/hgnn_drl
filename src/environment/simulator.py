"""Discrete-event Mist-Fog-Cloud simulator.

The simulator computes:

    T_execution   = cpu_cycles / allocated_CPU_cycles_per_second
    T_transfer    = data_size_bytes / effective_bandwidth_bytes_per_s
    T_queue       = modelled as per-node service queue
    T_total       = T_queue + T_transfer + T_execution + T_other_overheads
    E_compute     = P_compute * T_execution
    E_transmit    = P_transmit * T_transmit
    E_reception   = P_reception * T_transfer (configurable)

Assumptions (explicitly synthetic; see README): all capacities are in
arbitrary normalised units (1.0 = 1 CPU core; compute demand in 10^9 cycles;
bandwidth in a fixed reference rate; times in seconds).  Real-world
calibration is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..models.task import Task
from ..models.compute_node import ComputeNode
from ..utils.config import Config
from ..utils.logging import get_logger


@dataclass
class _JobInfo:
    """Per-resource bookkeeping for a single scheduled job."""

    task: Task
    layer: str
    node_id: str
    start_time: float
    finish_time: float
    exec_time: float
    transfer_time: float
    queue_wait: float
    energy_compute: float
    energy_transmit: float
    energy_reception: float
    deadline_missed: bool
    energy_total: float = 0.0


class Simulator:
    """Discrete-event Mist-Fog-Cloud simulation engine."""

    def __init__(self, *, config: Config, scenario_id: str = "factory"):
        self.config = config
        self.scenario_id = scenario_id
        self.logger = get_logger(f"hgnn_drl.sim.{scenario_id}")

        # Layer capacities (arbitrary consistent units).
        res = config.resources
        self.cpu = {"Mist": res.cpu_mist, "Fog": res.cpu_fog, "Cloud": res.cpu_cloud}
        self.mem = {"Mist": res.memory_mist, "Fog": res.memory_fog, "Cloud": res.memory_cloud}
        self.bandwidth = {"Mist": res.bandwidth, "Fog": res.bandwidth, "Cloud": res.bandwidth}
        self.propagation_delay_ms = {"Mist": 1.0, "Fog": 5.0, "Cloud": 30.0}

        # Compute nodes.
        self.nodes: Dict[str, ComputeNode] = {}
        self._node_layer_map: Dict[str, str] = {}

        # Job queue / completion tracking.
        self.jobs: List[_JobInfo] = []

        # Per-node queues.
        self._node_queues: Dict[str, List[str]] = field(default_factory=dict)

        # Task registry.
        self.tasks: Dict[str, Task] = {}
        self.eligible_tasks: List[Task] = []

        # Current time and bookkeeping.
        self.elapsed_time: float = 0.0
        self.total_energy: float = 0.0
        self.total_transfer_bytes: int = 0
        self.total_completed: int = 0
        self.total_deadline_misses: int = 0
        self.total_infeasible: int = 0
        self.last_decision_layer: Optional[str] = None

        # Energy powers.
        self.p_compute_w = 150.0
        self.p_transmit_w = 0.5
        self.p_reception_w = 0.2

        # Scenario-specific state.
        self._scenario_state: Dict[str, Any] = {}

    layer_names: List[str] = ["Mist", "Fog", "Cloud"]

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.elapsed_time = 0.0
        self.jobs.clear()
        self.tasks.clear()
        self.eligible_tasks.clear()
        self.total_energy = 0.0
        self.total_transfer_bytes = 0
        self.total_completed = 0
        self.total_deadline_misses = 0
        self.total_infeasible = 0
        self._node_queues.clear()
        self._node_layer_map.clear()
        self.nodes.clear()
        self._scenario_state.clear()

    def init_scenario(self, scenario_id: str, config: Config, *, scenario_generator=None) -> None:
        """Let the scenario module populate *self*."""
        self.scenario_id = scenario_id
        self.config = config
        self.scenario_generator = scenario_generator
        self.logger.info(f"initialised scenario '{scenario_id}'")

    # ------------------------------------------------------------------
    # Scenario lifecycle.
    # ------------------------------------------------------------------
    def generate_workload(self, *, n_tasks: int, seed: int) -> None:
        """Generate a deterministic task set (scenario module will call)."""
        rng = np.random.default_rng(seed)

        for i in range(n_tasks):
            # Each task gets a unique id.  In a real scenario, tasks would be
            # drawn from a workload distribution; here we keep a fixed
            # representative profile so that the simulator and metrics are
            # well-defined.  The scenario module overrides this with the
            # actual generation routine.
            task = Task(
                task_id=f"{self.scenario_id}_t{i:04d}",
                scenario_id=self.scenario_id,
                arrival_time=float(i * 0.1),
                data_size_bytes=int(rng.integers(1.0e5, 2.0e6)),
                cpu_cycles=float(rng.uniform(0.2e9, 3.0e9)),
                memory_requirement=float(rng.uniform(0.2, 0.9)),
                deadline=float(rng.uniform(0.4, 2.0)),
                priority=float(rng.uniform(0.0, 1.0)),
                input_buffer_ids=[],
                shared_buffer_ids=[],
                source_node_id=None,
                predecessor_task_ids=[],
                task_type="compute",
                battery_sensitive=False,
            )
            self.tasks[task.task_id] = task

        self.logger.info(f"generated {n_tasks} tasks for scenario '{self.scenario_id}'")

    # ------------------------------------------------------------------
    # Execution loop.
    # ------------------------------------------------------------------
    def run(self, n_tasks: Optional[int] = None) -> Dict[str, Any]:
        """Run the simulator until the horizon and return metrics.

        The method delegates to the scenario generator when present; otherwise
        it runs a minimal synthetic workload for a smoke-test.
        """
        if self.scenario_id == "factory":
            self._run_factory(n_tasks or self.config.workload_size)
        elif self.scenario_id == "drone":
            self._run_drone(n_tasks or self.config.workload_size)
        elif self.scenario_id == "healthcare":
            self._run_healthcare(n_tasks or self.config.workload_size)
        else:
            self._run_base(n_tasks or self.config.workload_size)
        return self._metrics()

    # ------------------------------------------------------------------
    # Internal runner stubs (scenario modules provide real generators).
    # ------------------------------------------------------------------
    def _run_base(self, n_tasks: int) -> None:
        """Run the scenario: generate workload, assign nodes, simulate."""
        # Delegate workload generation to the scenario generator if one exists;
        # otherwise fall back to the built-in synthetic workload.
        gen = getattr(self, "scenario_generator", None)
        if gen is not None:
            self.tasks = {t.task_id: t for t in gen.generate_tasks(n_tasks=n_tasks, seed=self.config.seed)}
        else:
            self.generate_workload(n_tasks=n_tasks, seed=self.config.seed)
        self._assign_nodes()
        self._unpack_eligible()
        self._advance_time_to_horizon()

    def _run_factory(self, n_tasks: int) -> None:
        self._run_base(n_tasks)

    def _run_drone(self, n_tasks: int) -> None:
        self._run_base(n_tasks)

    def _run_healthcare(self, n_tasks: int) -> None:
        self._run_base(n_tasks)

    # ------------------------------------------------------------------
    # Simulation core.
    # ------------------------------------------------------------------
    def _assign_nodes(self) -> None:
        """Round-robin assignment of tasks to nodes by task type.

        Default policy: compute -> Mist, batch -> Fog, offload -> Cloud.
        Scenario generators may override this by setting a per-type mapping.
        """
        # Per task_type -> preferred layer mapping (internally stable).
        default_mapping = {"compute": "Mist", "batch": "Fog", "offload": "Cloud"}

        # Group tasks by their task_type.
        task_pools: Dict[str, List[str]] = {}
        for tid, task in self.tasks.items():
            task_pools.setdefault(task.task_type, []).append(tid)

        # Cycle through existing nodes of the preferred layer.
        for task_type, tids in task_pools.items():
            lvl = default_mapping.get(task_type, "Cloud")
            cand = sorted(
                [n for n in self.nodes.values() if n.layer == lvl],
                key=lambda n: n.node_id,
            )
            if not cand:
                continue
            for pos, tid in enumerate(tids):
                node = cand[pos % len(cand)]
                self._place_task(tid, lvl, self.tasks[tid])

    def _unpack_eligible(self) -> None:
        """Mark tasks that can be immediately scheduled."""
        for task in self.tasks.values():
            if not task.predecessor_task_ids and not task.done:
                self.eligible_tasks.append(task)

    def _advance_time_to_horizon(self) -> None:
        """Simple time-driven loop; adequate for smoke/short runs."""
        self.elapsed_time = 0.0
        while self.elapsed_time < self.config.horizon:
            self.elapsed_time += 0.01
            self._tick()

    def _tick(self) -> None:
        # Advance all node queues by one simulation tick.
        for node in self.nodes.values():
            node.reset_utilization()

        # Process eligible tasks into the first available layer.
        for task in self.eligible_tasks:
            if task.done or task.infeasible_at_arrival:
                continue
            layer = self._pick_layer(task)
            if layer is None:
                continue
            self._place_task(task.task_id, layer, task)

    def _pick_layer(self, task: Task) -> Optional[str]:
        """Choose the best feasible layer for the task."""
        candidates = [("Mist", self.cpu["Mist"]), ("Fog", self.cpu["Fog"]), ("Cloud", self.cpu["Cloud"])]
        best = None
        best_score = None
        for layer, cap in candidates:
            node = self._node_for_layer(layer)
            if node is None:
                continue
            if task.cpu_cycles <= 0:
                continue
            # Feasibility: the node must have enough remaining capacity.
            if node.remaining_cpu() <= 0:
                continue
            # Instantaneous deadline check.
            slack = task.deadline - self.elapsed_time
            exec_time = task.cpu_cycles / max(1.0, cap)
            if exec_time > slack:
                continue
            score = exec_time + (self.propagation_delay_ms[layer] / 1000.0)
            if best_score is None or score < best_score:
                best_score = score
                best = layer
        return best

    def _node_for_layer(self, layer: str) -> Optional[ComputeNode]:
        for node in self.nodes.values():
            if node.layer == layer:
                return node
        # If no node of the layer exists, create the canonical Mist/Fog/Cloud node.
        if layer == "Mist":
            node = ComputeNode(
                node_id=f"{layer}_0",
                layer="Mist",
                cpu_capacity=self.cpu["Mist"],
                memory_capacity=self.mem["Mist"],
                available_bandwidth=self.bandwidth["Mist"],
                network_latency_ms=self.propagation_delay_ms["Mist"],
            )
            self.nodes[node.node_id] = node
            self._node_layer_map[node.node_id] = "Mist"
            return node
        elif layer == "Fog":
            node = ComputeNode(
                node_id=f"{layer}_0",
                layer="Fog",
                cpu_capacity=self.cpu["Fog"],
                memory_capacity=self.mem["Fog"],
                available_bandwidth=self.bandwidth["Fog"],
                network_latency_ms=self.propagation_delay_ms["Fog"],
            )
            self.nodes[node.node_id] = node
            self._node_layer_map[node.node_id] = "Fog"
            return node
        elif layer == "Cloud":
            node = ComputeNode(
                node_id=f"{layer}_0",
                layer="Cloud",
                cpu_capacity=self.cpu["Cloud"],
                memory_capacity=self.mem["Cloud"],
                available_bandwidth=self.bandwidth["Cloud"],
                network_latency_ms=self.propagation_delay_ms["Cloud"],
            )
            self.nodes[node.node_id] = node
            self._node_layer_map[node.node_id] = "Cloud"
            return node
        return None

    def _place_task(self, task_id: str, layer: str, task: Task) -> None:
        task.assigned_execution_layer = layer
        node = self._node_for_layer(layer)
        if node is None:
            task.infeasible_at_arrival = True
            self.total_infeasible += 1
            return

        # Instantaneous resource check.
        if node.remaining_cpu() < 1.0 or node.remaining_memory() < 1.0:
            task.infeasible_at_arrival = True
            self.total_infeasible += 1
            return

        # Queue wait: model as the node's current queue.
        queue_wait = node.queue_length * 0.01
        node.enqueue_task(task_id, self.elapsed_time)

        # Compute time: cpu_cycles (normalised units) per cpu_capacity.
        exec_time = task.cpu_cycles / max(1.0, node.cpu_capacity)

        # Transfer time: data size over available bandwidth, per shared buffer.
        transfer_time = 0.0
        for buf_id in task.shared_buffer_ids:
            transfer_time += (task.data_size_bytes / max(1.0, node.available_bandwidth * 1e6)) * 8.0

        energy_compute = self.p_compute_w * exec_time
        energy_transmit = self.p_transmit_w * transfer_time
        energy_reception = self.p_reception_w * transfer_time

        finish = self.elapsed_time + queue_wait + exec_time + transfer_time + 0.001
        node.record_execution_finish(task_id, finish)
        node.add_cpu_utilization(1.0 / 100.0)
        node.add_memory_utilization(1.0 / 100.0)

        job = _JobInfo(
            task=task,
            layer=layer,
            node_id=node.node_id,
            start_time=self.elapsed_time + queue_wait,
            finish_time=finish,
            exec_time=exec_time,
            transfer_time=transfer_time,
            queue_wait=queue_wait,
            energy_compute=energy_compute,
            energy_transmit=energy_transmit,
            energy_reception=energy_reception,
            deadline_missed=False,
        )
        task.execution_start_time = self.elapsed_time + queue_wait
        task.completion_time = finish
        task.latency = task.completion_time - task.arrival_time
        task.energy_consumed = job.energy_total = (
            energy_compute + energy_transmit + energy_reception
        )
        job.deadline_missed = task.latency > task.deadline
        self.total_energy += job.energy_total
        self.total_transfer_bytes += int(task.data_size_bytes)
        self.total_completed += 1
        if job.deadline_missed:
            self.total_deadline_misses += 1
        self.jobs.append(job)

        task.done = True
        task.deadline_missed = job.deadline_missed

        self.logger.debug(
            f"task {task_id} -> {layer} done in {task.latency:.4f}s "
            f"(queue={queue_wait:.3f}s exec={exec_time:.3f}s), "
            f"energy={job.energy_total:.4f}J, deadline_miss={job.deadline_missed}"
        )

    # ------------------------------------------------------------------
    # Reporting.
    # ------------------------------------------------------------------
    def _metrics(self) -> Dict[str, Any]:
        n_done = sum(1 for t in self.tasks.values() if t.done)
        n_missed = sum(1 for t in self.tasks.values() if t.deadline_missed)
        latencies = [t.latency for t in self.tasks.values() if t.done and t.latency >= 0]
        # Raw string for diagnostics: attach queue wait times from jobs.
        qw = [f"{j.task.task_id}={j.queue_wait:.4f}s" for j in self.jobs]
        sim_raw = f"queue_wait={','.join(qw)}" if qw else "queue_wait=none"
        return {
            "scenario_id": self.scenario_id,
            "n_tasks": len(self.tasks),
            "n_completed": n_done,
            "n_deadline_misses": n_missed,
            "deadline_violation_rate": (n_missed / n_done * 100.0) if n_done else 0.0,
            "total_energy": self.total_energy,
            "total_transfer_bytes": self.total_transfer_bytes,
            "total_completed": self.total_completed,
            "average_latency": float(np.mean(latencies)) if latencies else 0.0,
            "median_latency": float(np.median(latencies)) if latencies else 0.0,
            "p95_latency": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "p99_latency": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "elapsed_time": self.elapsed_time,
            "total_infeasible": self.total_infeasible,
            "energy_per_task": self.total_energy / n_done if n_done else 0.0,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "jobs": len(self.jobs),
            "sim_raw": sim_raw,
        }

    # ------------------------------------------------------------------
    # Energy helpers.
    # ------------------------------------------------------------------
    def compute_energy(self, task: Task, exec_time: float, transfer_time: float) -> Dict[str, float]:
        p_c, p_t, p_r = (
            self.p_compute_w,
            self.p_transmit_w,
            self.p_reception_w,
        )
        e_c = p_c * exec_time
        e_t = p_t * transfer_time
        e_r = p_r * transfer_time
        return {"compute": e_c, "transmit": e_t, "reception": e_r, "total": e_c + e_t + e_r}

    def finalize(self) -> None:
        """Finalise and lock the simulation state."""
        self.logger.info(f"SIMULATION FINISHED in scenario '{self.scenario_id}'")


# Backwards-compatible re-export for use in other modules.
ComputeNode = ComputeNode
