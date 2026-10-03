"""End-to-end experiment runner test (single seed, small workload)."""

import sys
from pathlib import Path

# Ensure project root is on sys.path.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from src.experiments.runner import ExperimentRunner
from src.utils.config import Config


def test_experiment_runner_runs_single_seed():
    cfg = Config(seed=42, workload_size=20)
    runner = ExperimentRunner(cfg)
    run = runner.run(
        algorithm="greedy",
        scenario="factory",
        n_tasks=20,
        seed=42,
    )
    assert run["algorithm"] == "greedy"
    assert run["scenario"] == "factory"
    assert run["seed"] == 42
    m = run["metrics"]
    assert m["n_completed"] >= 0
    print(f"[test] experiment runner OK: completed={m['n_completed']}")
