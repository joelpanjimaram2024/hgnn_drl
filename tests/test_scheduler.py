"""Tests for the scheduler and baseline algorithms."""

import numpy as np
import pytest

from src.models.task import Task
from src.models.compute_node import ComputeNode
from src.schedulers.cloud_only import CloudOnlyScheduler
from src.schedulers.mist_only import MistOnlyScheduler
from src.schedulers.greedy import GreedyScheduler
from src.schedulers.gcn_ppo import GCNPPOScheduler


def _make_nodes():
    return {
        "Mist_0": ComputeNode(node_id="Mist_0", layer="Mist", cpu_capacity=1.0, memory_capacity=1.0),
        "Fog_0": ComputeNode(node_id="Fog_0", layer="Fog", cpu_capacity=4.0, memory_capacity=8.0),
        "Cloud_0": ComputeNode(node_id="Cloud_0", layer="Cloud", cpu_capacity=16.0, memory_capacity=32.0),
    }


def test_cloud_only_scheduler():
    sched = CloudOnlyScheduler()
    sched.prepare(scenario_id="factory", nodes=_make_nodes())
    task = Task(task_id="t0", scenario_id="factory", arrival_time=0.0,
                data_size_bytes=1000, cpu_cycles=0.5, memory_requirement=0.5,
                deadline=1.0, priority=0.5)
    layer = sched.decide(task=task, nodes=_make_nodes())
    assert layer == "Cloud"
    print(f"[test] cloud-only -> {layer}")


def test_mist_only_scheduler():
    sched = MistOnlyScheduler()
    sched.prepare(scenario_id="factory", nodes=_make_nodes())
    task = Task(task_id="t0", scenario_id="factory", arrival_time=0.0,
                data_size_bytes=1000, cpu_cycles=0.5, memory_requirement=0.5,
                deadline=1.0, priority=0.5)
    layer = sched.decide(task=task, nodes=_make_nodes())
    assert layer == "Mist"
    print(f"[test] mist-only -> {layer}")


def test_greedy_scheduler_decides():
    sched = GreedyScheduler()
    sched.prepare(scenario_id="factory", nodes=_make_nodes())
    task = Task(task_id="t0", scenario_id="factory", arrival_time=0.0,
                data_size_bytes=1000, cpu_cycles=0.5, memory_requirement=0.5,
                deadline=1.0, priority=0.5)
    layer = sched.decide(task=task, nodes=_make_nodes())
    assert layer in ("Mist", "Fog", "Cloud")
    print(f"[test] greedy -> {layer}")


def test_gcn_ppo_scheduler_construction():
    """GCN+PPO baseline is constructible (no training required in tests)."""
    from src.hypergraph import HypergraphBuilder

    builder = HypergraphBuilder()
    for i in range(4):
        builder.register_task(f"t{i}", None)
    builder.add_shared_buffer("E_cam", ["t0", "t1", "t2", "t3"])
    hg = builder.build()

    from src.utils.config import Config
    cfg = Config(seed=42, workload_size=10)
    # GCN encoder input dim must match the feature vector size (7 features).
    cfg.hgnn.vertex_dim = 7
    cfg.hgnn.in_dim = 7
    sched = GCNPPOScheduler(hypergraph=hg, config=cfg)
    assert sched.encoder is not None
    task = Task(task_id="t0", scenario_id="factory", arrival_time=0.0,
                data_size_bytes=1000, cpu_cycles=0.5, memory_requirement=0.5,
                deadline=1.0, priority=0.5)
    layer = sched.decide(task=task, nodes=_make_nodes())
    assert layer in ("Mist", "Fog", "Cloud")
    print(f"[test] gcn-ppo -> {layer}")
