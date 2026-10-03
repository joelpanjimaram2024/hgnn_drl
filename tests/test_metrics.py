"""Tests for the central metrics formulas."""

import numpy as np
import pytest

from src.experiments.metrics_utils import (
    _safe_pct_change,
    _nanmean,
    _nanstd,
    _nanpercentile,
    compute_metrics,
)


def test_safe_pct_change_zero_baseline():
    """Avoid division by zero when baseline is zero."""
    assert _safe_pct_change(0.0, 5.0) == 0.0
    assert _safe_pct_change(0.0, 0.0) == 0.0
    assert _safe_pct_change(10.0, 5.0) == 100.0 * (10.0 - 5.0) / 10.0
    print(f"[test] pct_change check: {100.0 * (10.0 - 5.0)/10.0}")


def test_nan_mean():
    assert _nanmean([1.0, 2.0, 3.0]) == 2.0
    assert _nanmean([]) == 0.0
    assert _nanmean([np.nan, 2.0, 3.0]) == 2.5


def test_nan_std():
    assert _nanstd([1.0, 2.0, 3.0]) > 0.0
    assert _nanstd([]) == 0.0


def test_nan_percentile():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _nanpercentile(vals, 50) == 3.0
    assert _nanpercentile(vals, 95) > 4.0
    assert _nanpercentile([], 50) == 0.0


def test_compute_metrics_construction():
    per_task = [
        {"task_id": "t0", "done": True, "latency": 0.5, "energy_consumed": 1.0, "deadline_missed": False},
        {"task_id": "t1", "done": False, "latency": 0.0, "energy_consumed": 0.0, "deadline_missed": False},
    ]
    sim_metrics = {
        "n_tasks": 2,
        "total_transfer_bytes": 1000,
        "elapsed_time": 5.0,
        "nodes": {},
    }
    m = compute_metrics(per_task=per_task, sim_metrics=sim_metrics,
                        algorithm="HGNN+PPO", scenario="factory")
    assert m["average_latency_s"] > 0.0
    assert m["deadline_violation_rate_pct"] >= 0.0
    assert m["n_completed"] == 1
    assert m["throughput_tasks_per_s"] > 0.0
    print(f"[test] metrics: DVR={m['deadline_violation_rate_pct']:.1f}%, "
          f"avg_lat={m['average_latency_s']:.4f}s")
