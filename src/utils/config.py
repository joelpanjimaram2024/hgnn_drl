"""Central configuration system.

All tunable parameters live here or in YAML files under ``configs/``.
Simulation assumptions (units, network models, baseline weights) are
explicitly separated from measured hardware parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


@dataclass
class HypergraphConfig:
    """Hypergraph construction parameters."""

    vertex_dim: int = 8
    hyperedge_dim: int = 4
    hidden_dim: int = 32
    use_hyperedge_weights: bool = True
    degree_normalization: str = "symmetric"  # symmetric / left / right
    shared_buffer_weight_factor: float = 1.0
    cache_size: int = 10000


@dataclass
class HGNNConfig:
    """Hypergraph Neural Network encoder parameters."""

    in_dim: int = 8
    hidden_dim: int = 64
    out_dim: int = 64
    num_layers: int = 3
    activation: str = "relu"
    use_attention: bool = False
    dropout: float = 0.0
    batch_first: bool = True
    vertex_dim: int = 8


@dataclass
class DiscreteActionConfig:
    """Discrete action space for scheduler."""

    n_actions: int = 3  # Mist, Fog, Cloud
    action_names: List[str] = field(
        default_factory=lambda: ["Mist", "Fog", "Cloud"]
    )


@dataclass
class PPOConfig:
    """Proximal Policy Optimization training configuration."""

    algo: str = "PPO"
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    device: str = "auto"
    n_envs: int = 1
    total_timesteps: int = 50000
    log_interval: int = 10
    render_mode: Optional[str] = None
    verbose: int = 1
    tensorboard_log: Optional[str] = None


@dataclass
class ReplayBufferConfig:
    """Stored experiences for off-policy baselines where applicable."""

    capacity: int = 100000
    state_dim: int = 64
    action_dim: int = 3
    hidden_dim: int = 64


@dataclass
class ResourceConfig:
    """Compute resource capacities."""

    cpu_mist: float = 1.0
    memory_mist: float = 1.0
    cpu_fog: float = 4.0
    memory_fog: float = 8.0
    cpu_cloud: float = 16.0
    memory_cloud: float = 32.0
    bandwidth: float = 100.0  # Mbps
    propagation_delay_ms: float = 1.0


@dataclass
class TaskConfig:
    """Task workload parameters."""

    default_cpu_cycles: float = 1.0e9
    default_data_size_bytes: int = 1.0e6
    default_memory_requirement: float = 0.5
    default_deadline: float = 1.0
    default_priority: float = 0.5
    arrival_rate: float = 1.0
    max_shared_buffers: int = 4
    task_types: tuple = ("compute", "offload", "fusion")
    battery_sensitive: bool = False
    preemption_allowed: bool = False


@dataclass
class RewardConfig:
    """Reward function coefficients."""

    alpha: float = 1.0        # latency
    beta: float = 0.5         # energy
    gamma: float = 5.0        # deadline violation
    delta: float = 0.3        # communication cost
    eta: float = 1.0          # resource overload
    deadline_penalty_multiplier: float = 10.0
    normalize: bool = True


@dataclass
class BurstConfig:
    """Bursty workload parameters."""

    burst_probability: float = 0.0
    burst_scale: float = 5.0
    burst_cooldown: int = 5


@dataclass
class ScenarioConfig:
    """Scenario-specific parameters."""

    num_mist_nodes: int = 6
    num_fog_nodes: int = 2
    num_cloud_nodes: int = 1
    num_cameras: int = 6
    camera_frame_size_bytes: int = 8.0e6  # 4K 30fps approx
    frame_arrival_rate_hz: float = 2.0
    nominal_deadline_ms: float = 50.0
    drone_battery_capacity_wh: float = 100.0
    drone_battery_initial_wh: float = 100.0
    num_drones: int = 8
    num_health_stream_nodes: int = 4
    num_patients: int = 3
    healthcare_emergency_penalty: float = 10.0
    vehicle_mode: str = "computational"


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    scenario: str = "factory"
    agent: str = "HGNN+PPO"
    horizon: str = "long"
    train_episodes: int = 50
    eval_episodes: int = 10
    seed: int = 42
    device: str = "cpu"
    record_video: bool = False
    checkpoint_freq: int = 5
    checkpoint_dir: str = "checkpoints"
    summary_dir: str = "results"


@dataclass
class ExperimentConfig:
    """Experiment runner configuration."""

    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_reps: int = 5
    workload_size: int = 50
    train_episodes: int = 20
    eval_episodes: int = 5
    reward_debug: bool = False
    save_checkpoints: bool = True
    save_plots: bool = True
    save_results: bool = True

    # Experiment runner defaults (mirrored at Config top-level).
    summary_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    log_file: Optional[str] = None


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configurations."""

    # System
    name: str = "HGNN-DRL"
    version: str = "0.1.0"
    seed: int = 42
    device: str = "cpu"

    # Hypergraph
    hypergraph: HypergraphConfig = field(default_factory=HypergraphConfig)

    # HGNN
    hgnn: HGNNConfig = field(default_factory=HGNNConfig)

    # Actions
    action: DiscreteActionConfig = field(default_factory=DiscreteActionConfig)

    # PPO
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Replay
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)

    # Resources
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    # Tasks
    tasks: TaskConfig = field(default_factory=TaskConfig)

    # Reward
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Burst
    burst: BurstConfig = field(default_factory=BurstConfig)

    # Scenarios
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiments
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Extra runtime knobs
    workload_size: int = 50
    # Horizon in seconds (numeric); small default for tests/simulations.
    horizon: float = 3600.0  # 1 hour in seconds
    reward_debug: bool = False
    save_checkpoints: bool = True
    save_plots: bool = True
    save_results: bool = True

    # Experiment runner defaults.
    summary_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    log_file: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def save(self, path: str) -> None:
        """Write the full configuration to a YAML file."""
        data = self.to_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Recursive conversion of dataclasses to plain dicts."""
        result: Dict[str, Any] = {}
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                result[key] = val
            elif isinstance(val, (str, int, float, bool, list)):
                result[key] = val
            elif isinstance(val, (np.integer,)):
                result[key] = int(val)
            elif isinstance(val, (np.floating,)):
                result[key] = float(val)
            elif isinstance(val, np.ndarray):
                result[key] = val.tolist()
            elif hasattr(val, "to_dict"):
                result[key] = val.to_dict()
            else:
                result[key] = str(val)
        return result

    def to_tuple(self) -> tuple:
        """Render the config as a hashable tuple for seeding."""
        return tuple(sorted(self.to_dict().items()))

    def to_device(self, device: str) -> None:
        """No-Op for config object; device routing lives in model code."""
        self.device = device

    def ensure_device(self) -> str:
        """Resolve a usable torch device from the configured device string."""
        return self.device

    def __post_init__(self):
        # Coerce numpy integers/floats to Python natives
        for key, val in self.__dict__.items():
            if isinstance(val, np.integer):
                setattr(self, key, int(val))
            elif isinstance(val, np.floating):
                setattr(self, key, float(val))
            elif isinstance(val, np.ndarray):
                setattr(self, key, val.tolist())
        # Build deterministic seed sequence per sub-run
        self._rng = np.random.default_rng(self.seed)

    def sub_seed(self, offset: int = 0) -> int:
        """Derive a reproducible sub-seed from the base seed."""
        return int(self.seed) + offset
