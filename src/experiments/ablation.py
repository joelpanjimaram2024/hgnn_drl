"""Ablation experiments.

Compares HGNN+PPO against:
    1. GCN+PPO (pairwise graph instead of hypergraph)
    2. PPO with an MLP state encoder (no graph at all)
    3. Greedy heuristic
    4. HGNN+PPO without shared-data hyperedges (abandoned hypergraph)
    5. HGNN+PPO with different reward coefficients

All other variables (workload, seeds, simulator, training budget) are held
consistent as far as practical.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..experiments.runner import ExperimentRunner
from ..utils.config import Config
from ..utils.logging import get_logger


class AblationRunner:
    """Runs the ablation study for one scenario."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("hgnn_drl.ablation")
        self.runner = ExperimentRunner(config)

    def run_ablation(
        self,
        *,
        scenario: str,
        n_tasks: int,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run the full ablation for *scenario* and return a summary dict."""
        self.logger.info(f"== ABLATION STUDY: {scenario} ==")
        if seeds is None:
            seeds = self.config.experiments.seeds

        variants = self._get_variants(scenario)
        all_runs: List[Dict[str, Any]] = []

        for name, algo in variants.items():
            self.logger.info(f"  running variant: {name:24s} ({algo})")
            for seed in seeds:
                run = self.runner.run(
                    algorithm=algo,
                    scenario=scenario,
                    n_tasks=n_tasks,
                    seed=seed,
                )
                all_runs.append(run)

        summary = self.runner.aggregate(all_runs)
        return {
            "scenario": scenario,
            "n_tasks": n_tasks,
            "seeds": seeds,
            "variants": variants,
            "all_runs": all_runs,
            "summary": summary,
        }

    def _get_variants(self, scenario: str) -> Dict[str, str]:
        """Return algorithm-name -> config-string for the ablation."""
        from ..scenarios.factory import FactoryScenario
        from ..scenarios.drone import DroneScenario
        from ..scenarios.healthcare import HealthcareScenario

        if scenario == "factory":
            return {
                "HGNN+PPO (Proposed)": "hgnn+ppo",
                "GCN+PPO (Pairwise)": "gcn+ppo",
                "PPO+MLP (No graph)": "mlp+ppo",
                "Greedy heuristic": "greedy",
                "HGNN+PPO (no shared hyperedges)": "hgnn_nohe",
                "HGNN+PPO (reward sweep)": "hgnn_reward_sweep",
            }
        elif scenario == "drone":
            return {
                "HGNN+PPO (Proposed)": "hgnn+ppo",
                "GCN+PPO (Pairwise)": "gcn+ppo",
                "Greedy heuristic": "greedy",
            }
        elif scenario == "healthcare":
            return {
                "HGNN+PPO (Proposed)": "hgnn+ppo",
                "GCN+PPO (Pairwise)": "gcn+ppo",
                "Greedy heuristic": "greedy",
            }
        return {"HGNN+PPO (Proposed)": "hgnn+ppo"}

    def run_sweep_reward(self, *, scenario: str, n_tasks: int, seeds: List[int]) -> Dict[str, Any]:
        """Run HGNN+PPO with several reward-coefficient sets."""
        self.logger.info("== REWARD SWEEP ==")
        variants = {
            "alpha=1, beta=0.5, gamma=5": "reward_v1",
            "alpha=2, beta=1.0, gamma=10": "reward_v2",
            "alpha=0.5, beta=0.2, gamma=2": "reward_v3",
        }
        all_runs = []
        for name, algo in variants.items():
            self.logger.info(f"  reward variant: {name}")
            for seed in seeds:
                run = self.runner.run(algorithm=algo, scenario=scenario, n_tasks=n_tasks, seed=seed)
                all_runs.append(run)
        summary = self.runner.aggregate(all_runs)
        return {"scenario": scenario, "reward_variants": variants, "all_runs": all_runs, "summary": summary}
