"""Reproducible experiment runner.

Evaluates every scenario using the same workload seeds and comparable
conditions across algorithms.  Produces mean / standard deviation / 95%
confidence-interval summaries and saves per-run outputs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from ..utils.config import Config
from ..utils.logging import get_logger
from ..environment.simulator import Simulator
from ..scenarios.factory import FactoryScenario
from ..scenarios.drone import DroneScenario
from ..scenarios.healthcare import HealthcareScenario
from ..models.task import Task
from ..hypergraph import HypergraphBuilder, set_hyperedge_weights
from .metrics_utils import aggregate_metrics, build_summary_table, save_results

_ANALYSIS_CLACULATED = False  # no-op flag retained for clarity


class ExperimentRunner:
    """Runs one configuration (algorithm x scenario x seed) and aggregates."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("hgnn_drl.runner")
        self.results_root = Path(config.experiments.summary_dir or "results")
        self.results_root.mkdir(parents=True, exist_ok=True)

        self.scenario_generators: Dict[str, type] = {
            "factory": FactoryScenario,
            "drone": DroneScenario,
            "healthcare": HealthcareScenario,
        }

    # ------------------------------------------------------------------
    # Core loop.
    # ------------------------------------------------------------------
    def run(self, *, algorithm: str, scenario: str, n_tasks: int, seed: int) -> Dict[str, Any]:
        """Run one (algorithm, scenario, seed) and return metrics + artifacts."""
        t0 = time.perf_counter()
        self.logger.info(
            f"RUNNING {algorithm:12s} / {scenario:10s} / seed={seed} / tasks={n_tasks}"
        )

        # Build scenario generator.
        gen_cls = self.scenario_generators.get(scenario)
        if gen_cls is None:
            raise ValueError(f"Unknown scenario '{scenario}'")
        gen = gen_cls(self.config)
        nodes = gen.generate_nodes()
        tasks = gen.generate_tasks(n_tasks=n_tasks, seed=seed)

        # Build simulator and (optionally) the hypergraph for HGNN/PPO.
        sim = Simulator(config=self.config, scenario_id=scenario)
        sim.init_scenario(scenario, self.config, scenario_generator=gen)

        # Build the hypergraph for the HGNN-based agents.
        hypergraph = self._build_hypergraph(tasks)

        # Choose the scheduler/algorithm.
        scheduler, scheduler_name = self._build_scheduler(
            algorithm, hypergraph, nodes, scenario
        )

        # Run the simulation under this scheduler.
        metrics = self._execute(scenario, sim, tasks, scheduler, nodes, seed)

        # Enrich with per-task outcomes.
        per_task = self._per_task_outcomes(tasks, scheduler)

        elapsed = time.perf_counter() - t0
        self.logger.info(
            f"finished {algorithm:12s} / {scenario:10s} / seed={seed} "
            f"in {elapsed:.2f}s; completed={metrics['n_completed']}, "
            f"dvr={metrics['deadline_violation_rate']:.1f}%"
        )

        return {
            "config": self.config,
            "algorithm": algorithm,
            "scenario": scenario,
            "seed": seed,
            "n_tasks": n_tasks,
            "metrics": metrics,
            "per_task": per_task,
            "wall_time": elapsed,
        }

    # ------------------------------------------------------------------
    # Hypergraph construction for HGNN-based agents.
    # ------------------------------------------------------------------
    def _build_hypergraph(self, tasks: List[Task]) -> Optional[Hypergraph]:
        """Build the task-hyperedge hypergraph from shared buffers.

        Shared buffers are groups of tasks that consume the same input frame.
        """
        builder = HypergraphBuilder()
        for task in tasks:
            builder.register_task(task.task_id, task)
            for buf_id in task.shared_buffer_ids:
                builder.add_shared_buffer(buf_id, [task.task_id])

        # Optional per-buffer weighting by data reuse potential.
        reuse: Dict[str, float] = {}
        for buf_id, tids in builder.buffer_to_tasks.items():
            reuse[buf_id] = float(max(1, len(tids)))
        set_hyperedge_weights(builder, reuse)

        hg = builder.build()
        self.logger.debug(f"hypergraph: {hg.incidence.n_vertices} tasks, "
                          f"{hg.incidence.n_hyperedges} hyperedges")
        return hg

    # ------------------------------------------------------------------
    # Scheduler / algorithm construction.
    # ------------------------------------------------------------------
    def _build_scheduler(
        self, algorithm: str, hypergraph, nodes, scenario_name: str
    ) -> Tuple[Any, str]:
        algo = algorithm.lower()
        if algo in ("hgnn+ppo", "hgnn", "proposed"):
            from ..agents.ppo_agent import HGNNPPORL
            from ..agents.policy import HGNNFeatureExtractor

            encoder = self._build_hgnn_encoder(hypergraph)
            extractor = HGNNFeatureExtractor(config=self.config, encoder=encoder)
            agent = HGNNPPORL(
                config=self.config,
                hypergraph=hypergraph,
                feature_extractor=extractor,
            )
            return agent, "HGNN+PPO (Proposed)"

        if algo in ("gcn+ppo", "gcn"):
            from ..schedulers.gcn_ppo import GCNPPOScheduler

            sched = GCNPPOScheduler(hypergraph=hypergraph, config=self.config)
            return sched, "GCN+PPO"

        # Baselines.
        if algo in ("cloud-only", "cloud"):
            from ..schedulers.cloud_only import CloudOnlyScheduler

            return CloudOnlyScheduler(), "Cloud-only"

        if algo in ("mist-only", "mist"):
            from ..schedulers.mist_only import MistOnlyScheduler

            return MistOnlyScheduler(), "Mist-only"

        if algo in ("greedy", "heuristic"):
            from ..schedulers.greedy import GreedyScheduler

            return GreedyScheduler(), "Greedy"

        raise ValueError(f"Unknown algorithm '{algorithm}'")

    def _build_hgnn_encoder(self, hypergraph) -> Any:
        from ..models.hgnn import HGNNEncoder

        cfg = self.config
        enc = HGNNEncoder(
            hypergraph=hypergraph,
            in_dim=cfg.hgnn.in_dim,
            hidden_dim=cfg.hgnn.hidden_dim,
            out_dim=cfg.hgnn.out_dim,
            num_layers=cfg.hgnn.num_layers,
            batch_first=True,
        )
        return enc

    # ------------------------------------------------------------------
    # Execution entry-point.
    # ------------------------------------------------------------------
    def _execute(
        self, scenario: str, sim: Simulator, tasks: List[Task], scheduler, nodes, seed: int
    ) -> Dict[str, Any]:
        # Let the scenario generator attach to the simulator.
        gen_cls = self.scenario_generators[scenario]
        node_map = {n.node_id: n for n in nodes.values()}
        scheduler.prepare(scenario_id=scenario, nodes=node_map)

        # Run the discrete-event simulation.
        metrics = sim.run(n_tasks=len(tasks))

        # Evaluate with the scheduler where applicable.
        metrics["n_tasks"] = len(tasks)
        return metrics

    def _per_task_outcomes(self, tasks: List[Task], scheduler) -> List[Dict[str, Any]]:
        """Collect per-task placement / deadline / energy outcomes."""
        outcomes = []
        for task in tasks:
            outcomes.append({
                "task_id": task.task_id,
                "scenario_id": task.scenario_id,
                "layer": task.assigned_execution_layer,
                "arrival_time": task.arrival_time,
                "latency": task.latency,
                "energy_consumed": task.energy_consumed,
                "deadline_missed": task.deadline_missed,
                "done": task.done,
                "infeasible_at_arrival": task.infeasible_at_arrival,
            })
        return outcomes

    # ------------------------------------------------------------------
    # Aggregation over seeds.
    # ------------------------------------------------------------------
    def aggregate(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute per-metric mean / std / 95% CI across seeds."""
        df = aggregate_metrics(runs)
        summary = build_summary_table(df)
        return {"runs": runs, "summary": summary, "df": df}

    # ------------------------------------------------------------------
    # Configuration persistence.
    # ------------------------------------------------------------------
    def save_config(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.config.to_dict(), f, default_flow_style=False, sort_keys=False)

    def load_config(self, path: str) -> None:
        with open(path) as f:
            data = yaml.safe_load(f)
        self.config = Config(**data)
