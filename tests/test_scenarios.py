"""Tests for the three scenario generators."""

import numpy as np
import pytest

from src.scenarios.factory import FactoryScenario
from src.scenarios.drone import DroneScenario
from src.scenarios.healthcare import HealthcareScenario
from src.utils.config import Config


def _base_config():
    return Config(seed=42, workload_size=20)


def test_factory_scenario_nodes_and_tasks():
    cfg = _base_config()
    sc = FactoryScenario(cfg)
    nodes = sc.generate_nodes()
    tasks = sc.generate_tasks(n_tasks=20, seed=42)
    assert len(nodes) >= 6, "factory needs at least 6 mist nodes"
    assert len(tasks) == 20, f"expected 20 tasks, got {len(tasks)}"
    assert all(t.scenario_id == "factory" for t in tasks)
    print(f"[test] factory: {len(nodes)} nodes, {len(tasks)} tasks")


def test_drone_scenario_nodes_and_tasks():
    cfg = _base_config()
    sc = DroneScenario(cfg)
    nodes = sc.generate_nodes()
    tasks = sc.generate_tasks(n_tasks=20, seed=42)
    assert len(nodes) >= 8, "drone needs at least 8 drone nodes"
    assert len(tasks) == 20, f"expected 20 tasks, got {len(tasks)}"
    print(f"[test] drone: {len(nodes)} nodes, {len(tasks)} tasks")


def test_healthcare_scenario_nodes_and_tasks():
    cfg = _base_config()
    sc = HealthcareScenario(cfg)
    nodes = sc.generate_nodes()
    tasks = sc.generate_tasks(n_tasks=20, seed=42)
    assert len(nodes) >= 4, "healthcare needs at least 4 monitor nodes"
    assert len(tasks) == 20, f"expected 20 tasks, got {len(tasks)}"
    print(f"[test] healthcare: {len(nodes)} nodes, {len(tasks)} tasks")


def test_scenario_config_serializable():
    cfg = _base_config()
    for cls in (FactoryScenario, DroneScenario, HealthcareScenario):
        sc = cls(cfg)
        config = sc.scenario_config()
        assert isinstance(config, dict)
        assert len(config) > 0
