"""Central metrics utilities and experiment aggregation.

Implements the metrics from the project plan (section 16):
    * Average latency
    * Tail latency (P95 / P99)
    * Deadline violation rate
    * Energy (total, per-task, compute vs transmit)
    * Communication overhead
    * Throughput
    * Resource utilisation
    * Training performance
    * Percentage change vs baseline
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# Numeric tolerance used to avoid division by zero in percentage change.
_EPS = 1e-12


def _safe_pct_change(baseline: float, proposed: float) -> float:
    """100 * (baseline - proposed) / baseline; returns 0 if baseline ~ 0."""
    if abs(baseline) < _EPS:
        return 0.0
    return 100.0 * (baseline - proposed) / baseline


def _nanmean(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def _nanstd(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _nanpercentile(values: List[float], q: float) -> float:
    vals = sorted([float(v) for v in values if v is not None and np.isfinite(v)])
    return float(np.percentile(vals, q)) if vals else 0.0


def compute_metrics(
    *,
    per_task: List[Dict[str, Any]],
    sim_metrics: Dict[str, Any],
    algorithm: str,
    scenario: str,
) -> Dict[str, Any]:
    """Compute the project metrics for a single run.

    Parameters
    ----------
    per_task : list of dict
        One entry per task (from the experiment runner).
    sim_metrics : dict
        Raw simulator metrics returned by the discrete-event engine.
    algorithm : str
        Name of the algorithm (for labelling).
    scenario : str
        Scenario name.
    """
    # Latency
    latencies = [t["latency"] for t in per_task if t.get("done") and t["latency"] >= 0]
    avg_lat = _nanmean(latencies)
    med_lat = _nanmean(latencies) and _nanpercentile(latencies, 50)
    p95_lat = _nanpercentile(latencies, 95)
    p99_lat = _nanpercentile(latencies, 99)

    # Deadline
    n_done = sum(1 for t in per_task if t.get("done"))
    n_missed = sum(1 for t in per_task if t.get("deadline_missed"))
    # Denominator: tasks that reached a scheduling decision (either completed
    # or recorded as infeasible).  Per the spec, dropped/infeasible tasks are
    # excluded unless explicitly stated.
    n_scheduled = n_done + sum(1 for t in per_task if t.get("infeasible_at_arrival"))
    dvr = (n_missed / max(n_scheduled, 1)) * 100.0 if n_scheduled else 0.0

    # Energy
    energy_vals = [t["energy_consumed"] for t in per_task if t.get("done")]
    total_energy = float(np.sum(energy_vals)) if energy_vals else 0.0
    energy_per_task = (total_energy / max(n_done, 1)) if n_done else 0.0

    # Communication overhead
    comm_bytes = sim_metrics.get("total_transfer_bytes", 0)
    n_transfers = sim_metrics.get("total_transfers", 0)

    # Throughput (completed tasks per simulated second)
    elapsed = sim_metrics.get("elapsed_time", 0.0) or 1e-12
    throughput = n_done / elapsed if elapsed > 0 else 0.0

    # Resource utilisation (from simulator node state)
    util_rows = []
    for node_id, node in sim_metrics.get("nodes", {}).items():
        util_rows.append({
            "node": node_id,
            "layer": node.get("layer"),
            "cpu_util": node.get("cpu_utilization", 0.0),
            "mem_util": node.get("memory_utilization", 0.0),
            "queue_len": node.get("queue_length", 0),
        })
    util_df = pd.DataFrame(util_rows) if util_rows else pd.DataFrame()

    return {
        "algorithm": algorithm,
        "scenario": scenario,
        "n_tasks": sim_metrics.get("n_tasks", 0),
        "n_completed": n_done,
        "n_deadline_misses": n_missed,
        "n_infeasible": sim_metrics.get("total_infeasible", 0),
        "deadline_violation_rate_pct": dvr,
        "average_latency_s": avg_lat,
        "median_latency_s": med_lat,
        "p95_latency_s": p95_lat,
        "p99_latency_s": p99_lat,
        "total_energy_J": total_energy,
        "energy_per_task_J": energy_per_task,
        "communication_bytes": comm_bytes,
        "communication_transfers": n_transfers,
        "throughput_tasks_per_s": throughput,
        "resource_util_df": util_df,
        "sim_raw": sim_metrics,
    }


def aggregate_metrics(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten per-run metrics into a tidy dataframe (one row per run)."""
    rows = []
    for run in runs:
        m = run.get("metrics", {})
        # The simulator reports metrics under *its* key names; map them to
        # the tidy column names used by aggregation and the summary table.
        def _get(key: str, default: Any = None) -> Any:
            return m.get(key, default)

        rows.append({
            "algorithm": run.get("algorithm"),
            "scenario": run.get("scenario"),
            "seed": run.get("seed"),
            "n_tasks": run.get("n_tasks"),
            "n_completed": m.get("n_completed"),
            "n_deadline_misses": m.get("n_deadline_misses"),
            "deadline_violation_rate_pct": m.get("deadline_violation_rate"),
            "average_latency_s": m.get("average_latency"),
            "median_latency_s": m.get("median_latency"),
            "p95_latency_s": m.get("p95_latency"),
            "p99_latency_s": m.get("p99_latency"),
            "total_energy_J": m.get("total_energy"),
            "energy_per_task_J": m.get("energy_per_task"),
            "communication_bytes": m.get("total_transfer_bytes", 0),
            "communication_transfers": m.get("total_transfers", m.get("total_transfer_bytes", 0)),
            "throughput_tasks_per_s": _nanmean(
                [m.get("n_completed", 0) / (m.get("elapsed_time", 1e-12) or 1e-12)]
            ),
            "wall_time_s": run.get("wall_time", 0.0),
            "n_infeasible": m.get("total_infeasible", 0),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_summary_table(df: pd.DataFrame) -> Dict[str, Any]:
    """Build per-metric mean / std / 95% CI and percentage change vs baselines."""
    if df.empty:
        return {"means": {}, "stds": {}, "cis": {}, "pct_change": {}}

    summary: Dict[str, Any] = {"means": {}, "stds": {}, "cis": {}, "pct_change": {}}

    metric_cols = [
        "deadline_violation_rate_pct",
        "average_latency_s",
        "median_latency_s",
        "p95_latency_s",
        "p99_latency_s",
        "total_energy_J",
        "energy_per_task_J",
        "throughput_tasks_per_s",
    ]

    for metric in metric_cols:
        if metric not in df.columns:
            continue
        series = df[metric].dropna()
        if series.empty:
            continue
        mean = float(series.mean())
        std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        n = len(series)
        ci = 1.96 * std / float(np.sqrt(n)) if n > 1 else 0.0
        summary["means"][metric] = mean
        summary["stds"][metric] = std
        summary["cis"][metric] = ci

        # Percentage change relative to the Cloud-only baseline.
        cloud = df.loc[df["algorithm"].str.lower() == "cloud-only", metric]
        if len(cloud) > 0 and cloud.notna().any() and abs(cloud.mean()) > _EPS:
            pct = 100.0 * (cloud.mean() - mean) / cloud.mean()
            summary["pct_change"][metric] = pct
        else:
            summary["pct_change"][metric] = 0.0

    # Counts
    summary["counts"] = df.value_counts(subset=["algorithm", "scenario"]).to_dict()

    return summary


def _json_safe(obj: Any) -> Any:
    """Recursively normalise a summary dict for YAML/JSON serialisation.

    Converts non-string keys (e.g. tuple keys from pandas ``value_counts``),
    numpy scalars, numpy arrays and any object that exposes ``to_dict`` into
    plain JSON/YAML-safe values.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _json_safe(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return _json_safe(obj.to_dict())
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def save_results(
    summary: Dict[str, Any],
    *,
    out_dir: str = "results",
    scenario: str = "",
    algorithm: str = "",
) -> None:
    """Persist experiment summaries (CSV + YAML/JSON) under *out_dir*."""
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)

    # Summary table CSV.
    # ``aggregate()`` nests the summary under ``summary["summary"]``; fall
    # back to top-level keys when the caller passes a pre-built summary.
    means = summary.get("means", None)
    stds = summary.get("stds", None)
    cis = summary.get("cis", None)
    if means is None:
        means = summary.get("summary", {}).get("means", {}) or {}
        stds = summary.get("summary", {}).get("stds", {}) or {}
        cis = summary.get("summary", {}).get("cis", {}) or {}
    rows = [
        {"metric": k, "mean": means[k], "std": stds.get(k, 0.0), "ci95": cis.get(k, 0.0)}
        for k in means
    ]
    csv_path = os.path.join(out_dir, f"summary_{scenario or 'all'}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Human-readable summary.
    txt_path = os.path.join(out_dir, f"summary_{scenario or 'all'}.txt")
    with open(txt_path, "w") as f:
        f.write("Experiment summary\n")
        f.write("=" * 60 + "\n")
        for metric, mean in means.items():
            std = stds.get(metric, 0.0)
            ci = cis.get(metric, 0.0)
            pct = summary.get("pct_change", {}).get(metric, 0.0)
            f.write(f"{metric:35s} mean={mean:12.4f} std={std:10.4f} ci95={ci:10.4f} pct_vs_cloud={pct:8.2f}%\n")

    # JSON of full summary (recursion-safe for non-string keys).
    json_path = os.path.join(out_dir, f"summary_{scenario or 'all'}.json")
    with open(json_path, "w") as f:
        json.dump(_json_safe(summary), f, indent=2)

    # Raw per-run metrics table.
    runs_df = aggregate_metrics([r for r in summary.get("runs", [])])
    raw_csv = os.path.join(out_dir, f"runs_{scenario or 'all'}.csv")
    runs_df.to_csv(raw_csv, index=False)
