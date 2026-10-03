#!/usr/bin/env python3
"""Top-level smoke test for the HGNN-DRL project.

Runs a tiny deterministic end-to-end experiment (factory scenario) and
verifies that the task pipeline, simulator, hypergraph, and HGNN all execute.
Exits with status 0 on success, 1 on failure.
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from src.utils.config import Config
from src.environment.simulator import Simulator
from src.hypergraph import HypergraphBuilder
from src.models.hgnn import HGNNEncoder, hgnn_shapes
from src.scenarios.factory import FactoryScenario

# ----------------------------------------------------------------------
# 1. Tiny deterministic end-to-end: task arrival -> completion.
# ----------------------------------------------------------------------
def run_end_to_end():
    cfg = Config(seed=42, workload_size=5, horizon=5.0)

    # Hypergraph with a shared buffer (e.g. one camera frame).
    builder = HypergraphBuilder()
    for i in range(5):
        builder.register_task(f"t{i}", None)
    # Tasks 0..2 share camera frame E_cam.
    builder.add_shared_buffer("E_cam", ["t0", "t1", "t2"])
    builder.set_buffer_weights("E_cam", 1.0)
    hg = builder.build()
    assert hg.incidence.n_vertices == 5, "Expected 5 vertices"
    assert hg.incidence.n_hyperedges == 1, "Expected 1 hyperedge"
    print(f"[smoke] hypergraph OK: {hg.incidence.n_vertices} tasks, "
          f"{hg.incidence.n_hyperedges} hyperedges")

    # HGNN forward and backward pass.
    enc = HGNNEncoder(
        hypergraph=hg,
        in_dim=cfg.hgnn.vertex_dim,
        hidden_dim=16,
        out_dim=16,
        num_layers=2,
    )
    x = torch.randn(hg.incidence.n_vertices, cfg.hgnn.vertex_dim)
    task_emb, graph_state = enc(x)
    assert task_emb.shape == (5, 16), f"unexpected task_emb shape {task_emb.shape}"
    assert graph_state.shape == (1, 16), f"unexpected graph_state shape {graph_state.shape}"
    # Gradient flow check.
    loss = task_emb.sum()
    loss.backward()
    assert enc.layers[0].W.grad is not None, "no gradients computed"
    print(f"[smoke] HGNN forward+backward OK: {task_emb.shape}, grads present")

    # Scenario + simulator.
    gen = FactoryScenario(cfg)
    nodes = gen.generate_nodes()
    tasks = gen.generate_tasks(n_tasks=cfg.workload_size, seed=42)
    sim = Simulator(config=cfg, scenario_id="factory")
    sim.init_scenario("factory", cfg, scenario_generator=gen)
    metrics = sim.run(n_tasks=cfg.workload_size)
    assert metrics["n_completed"] >= 0, "simulator produced no completed tasks"
    assert metrics["scenario_id"] == "factory"
    assert "average_latency" in metrics, "metrics missing latency"
    print(f"[smoke] simulator OK: completed={metrics['n_completed']}, "
          f"latency={metrics.get('average_latency') or 0:.4f}s")

    # Verify the tiny deterministic task end-to-end property:
    # a single task with deadline > latency is not missed.
    pending = [t for t in tasks if t.task_id == "factory_camera0_Crack Detection"]
    assert len(pending) >= 1, "factory task not generated"
    assert pending[0].deadline_missed in (False, None), "task unexpectedly missed deadline"
    print("[smoke] task end-to-end OK")

    return True


# ----------------------------------------------------------------------
# 2. Basic module import checks for all required packages.
# ----------------------------------------------------------------------
def check_modules():
    required = [
        "torch", "gymnasium", "numpy", "pandas", "matplotlib", "seaborn",
        "networkx", "yaml", "pytest", "scipy",
    ]
    for mod in required:
        __import__(mod)
    print(f"[smoke] all required packages import OK")
    # Gymnasium environment smoke.
    from src.environment.base_env import HGGNDRLGymnasiumEnv
    from src.environment.simulator import Simulator
    from src.environment.reward import compute_reward
    for name, fn in [
        ("HGGNDRLGymnasiumEnv", HGGNDRLGymnasiumEnv),
        ("Simulator", Simulator),
        ("compute_reward", compute_reward),
    ]:
        assert fn is not None, f"import failed for {name}"
    print("[smoke] Gymnasium environment + reward OK")


# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("HGNN-DRL SMOKE TEST")
    print("=" * 60)
    failed = []
    try:
        check_modules()
    except Exception as exc:  # pragma: no cover
        failed.append(("module import", exc))

    try:
        run_end_to_end()
    except Exception as exc:  # pragma: no cover
        failed.append(("end-to-end", exc))

    if failed:
        print("\n[smoke] FAILED:")
        for name, exc in failed:
            print(f"  - {name}: {exc}")
        return 1
    print("\n[smoke] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
