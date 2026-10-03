"""Reward function for the HGNN-DRL environment.

Reward (configurable, centralised) -- the task-level reward is the primary
signal; the episode-level return is the sum of task-level rewards.

        R = -alpha * L_norm
          - beta  * E_norm
          - gamma * V_deadline
          - delta * C_norm
          - eta   * O_resource

where
    L_norm  : normalised latency (per-task, normalising by scenario deadline)
    E_norm  : normalised energy consumption (per-task, by scenario peak)
    V_deadline: 1 if the task missed its deadline, else 0
    C_norm  : normalised communication cost (per-task transfer bytes)
    O_resource: resource overload / infeasibility penalty
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..models.task import Task
from ..utils.config import Config
from ..utils.logging import get_logger

# Numeric tolerances.
EPS = 1e-12


def compute_reward(
    *,
    task: Task,
    scenario_metric_reference: Dict[str, float],
    config: Config,
    layer: str,
) -> float:
    """Return the scalar reward for one completed/scheduled task.

    Parameters
    ----------
    task : Task
        The task being scheduled / completed.
    scenario_metric_reference : dict
        Scenario-specific reference scales (e.g. peak latency, peak energy)
        so that metrics are normalised to a comparable range.
    config : Config
        Global configuration (reward coefficients).
    layer : str
        The chosen execution layer.

    Returns
    -------
    float
        Scalar reward.  Negative values encourage the agent to minimise the
        weighted metrics; large positive penalty when a deadline is missed or
        the task is infeasible.
    """
    alpha = config.reward.alpha
    beta = config.reward.beta
    gamma = config.reward.gamma
    delta = config.reward.delta
    eta = config.reward.eta
    deadline_mult = config.reward.deadline_penalty_multiplier

    # --- raw terms --------------------------------------------------------
    latency_raw = task.latency if task.latency > 0 else 0.0
    energy_raw = task.energy_consumed if task.energy_consumed > 0 else 0.0

    # Communication cost: bytes transferred (per the spec, C_norm is
    # a communication metric).  We use bytes as the raw communication cost;
    # normalising by a reference peak in bytes.
    comm_cost_raw = float(task.data_size_bytes)

    # Resource overload: infeasibility or a large queue at the assigned layer.
    overloaded = 0.0
    if task.infeasible_at_arrival:
        overloaded = 1.0
    elif task.assigned_execution_layer is not None:
        # Queue length signal (task-level): use 0..1 scaled by scenario config.
        overloaded = min(1.0, task.queue_wait_time * 10.0)

    # --- normalise --------------------------------------------------------
    L_norm = _normalise(latency_raw, scenario_metric_reference.get("peak_latency", 1.0))
    E_norm = _normalise(energy_raw, scenario_metric_reference.get("peak_energy", 1.0))
    C_norm = _normalise(comm_cost_raw, scenario_metric_reference.get("peak_comms_bytes", 1.0))

    # --- combine ----------------------------------------------------------
    reward = (
        -alpha * L_norm
        - beta * E_norm
        - delta * C_norm
        - eta * overloaded
    )

    # Deadline violation: strong penalty (configurable multiplier).  We also
    # add a fixed gamma to make the missed-deadline component separate and
    # explicitly tunable.
    if task.deadline_missed or task.infeasible_at_arrival:
        reward -= gamma * deadline_mult

    # Keep the reward within a sane, bounded range for PPO stability.
    reward = float(np.clip(reward, -20.0, 20.0))
    return reward


def _normalise(value: float, peak: float) -> float:
    """Scale *value* by *peak* (clipping at 1.0 for stability)."""
    if peak <= 0 or not np.isfinite(peak):
        return 0.0
    return float(np.clip(value / max(peak, EPS), 0.0, 1.0))


def reward_statistics(rewards: list[float]) -> Dict[str, float]:
    """Return summary statistics of a reward list."""
    if not rewards:
        return {"sum": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "sum": float(np.sum(rewards)),
        "mean": float(np.mean(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
    }
