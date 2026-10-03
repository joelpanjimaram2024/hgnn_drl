"""Core simulator + Gymnasium environment wrapper.

The simulator implements a discrete-event Mist-Fog-Cloud model:

    T_execution   = cpu_cycles / allocated_CPU_cycles_per_second
    T_transfer    = data_size_bytes / effective_bandwidth_bytes_per_s
    T_total       = T_queue + T_transfer + T_execution + T_overhead
    E_compute     = P_compute * T_execution
    E_transmit    = P_transmit * T_transmit
    E_reception   = P_reception * T_transfer (when configured)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional gymnasium; we use a lightweight custom wrapper if missing.
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except Exception:  # pragma: no cover
    _GYM_AVAILABLE = False
    gym = None  # type: ignore
    spaces = None  # type: ignore


from .simulator import Simulator
from .reward import compute_reward
from .shared_buffer_manager import SharedBufferManager
from ..models.task import Task
from ..utils.config import Config
from ..utils.logging import get_logger

if _GYM_AVAILABLE:
    from gymnasium import spaces


class HGGNDRLGymnasiumEnv:
    """Gymnasium-compatible environment: Mist-Fog-Cloud scheduler.

    State observation is a per-step vector of resource utilizations, queue
    lengths, task backlog, and an aggregated HGNN embedding through the
    policy pipeline.  The environment is *not* a full MDP-reset-per-episode
    in the tightest sense: it advances a discrete-event clock and is designed
    to be driven by a PPO agent via a single decision every N simulation steps.
    """

    MATCHES_GYM = False

    def __init__(
        self,
        *,
        config: Config,
        scenario_id: str,
        nodes: Optional[Dict[str, "ComputeNode"]] = None,
        tasks: Optional[List[Task]] = None,
        device: str = "cpu",
    ):
        self.config = config
        self.scenario_id = scenario_id
        self.device = device
        self.logger = get_logger(f"hgnn_drl.env.{scenario_id}")

        self.simulator = Simulator(config=config, scenario_id=scenario_id)
        self.shared_buffer_manager = SharedBufferManager(config=config)

        # Descriptive properties populated by the scenario generator.
        self.node_count: int = 0
        self.task_count: int = 0
        self.layer_names: List[str] = ["Mist", "Fog", "Cloud"]

        # Observation / action spaces.
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

        # Current state.
        self.observation: Optional[np.ndarray] = None
        self.state: Dict[str, Any] = {}
        self.internal_step: int = 0
        self.episode_return: float = 0.0
        self.episode_length: int = 0
        self.current_task_id: Optional[str] = None

        # Placeholder for scenario tasks (set by scenario module).
        self.tasks: Dict[str, Task] = {}

        self._reset_simulator_attributes()

    # ------------------------------------------------------------------
    # Gym compatibility.
    # ------------------------------------------------------------------
    def _build_observation_space(self):
        if not _GYM_AVAILABLE:
            return None
        return spaces.Box(low=0.0, high=1.0, shape=(64,), dtype=np.float32)

    def _build_action_space(self):
        if not _GYM_AVAILABLE:
            return None
        return spaces.Discrete(3)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one internal decision tick."""
        start = time.perf_counter()

        if not _GYM_AVAILABLE:
            return (
                self.observation,
                0.0,
                False,
                False,
                {"error": "gymnasium not installed"},
            )

        # Map discrete action to layer name.
        layer_map = {0: "Mist", 1: "Fog", 2: "Cloud"}
        layer = layer_map.get(action, "Cloud")

        # Advance the discrete-event simulator by one step.
        self.internal_step += 1
        self.episode_length += 1

        # Execute the scheduler decision in the simulator.
        self._execute_decision(layer)

        # Gather a simple observation vector.
        self.observation = self._build_observation()

        # Reward is computed after each task completion in the simulator.
        self.episode_return += self._reward_for_last_step()

        terminated = self.simulator.elapsed_time >= self.config.horizon
        truncated = False

        self.logger.debug(
            f"step={self.internal_step} layer={layer} obs_dim={len(self.observation) if self.observation is not None else 0}"
        )

        info = self._info_dict()

        elapsed = time.perf_counter() - start
        # Keep per-step overhead bounded (a few ms on CPU).
        self.logger.debug(f"step wall_time={elapsed:.4f}s")

        return self.observation, float(self.episode_return), terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self.internal_step = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.current_task_id = None

        self._reset_simulator_attributes()

        # Re-initialise scenario-specific computation.
        self.simulator.init_scenario(self.scenario_id, self.config)

        if self.simulator.nodes:
            self.node_count = len(self.simulator.nodes)
        else:
            self.node_count = 0

        # Run a small workload so that the state is non-empty.
        self.simulator.run(self.config.workload_size)
        self.tasks = self.simulator.tasks

        self.observation = self._build_observation()
        self.state = self._state_dict()
        self.logger.info(
            f"reset completed: {self.node_count} nodes, {len(self.tasks)} tasks, "
            f"elapsed={self.simulator.elapsed_time:.3f}s"
        )
        return self.observation, self.state

    # ------------------------------------------------------------------
    # Observation / state helpers.
    # ------------------------------------------------------------------
    def _reset_simulator_attributes(self):
        self.simulator.reset()
        self.episode_return = 0.0
        self.episode_length = 0

    def _state_dict(self) -> Dict[str, Any]:
        return {
            "internal_step": self.internal_step,
            "elapsed_time": self.simulator.elapsed_time,
            "node_count": self.node_count,
            "task_count": len(self.tasks),
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
        }

    def _build_observation(self) -> Optional[np.ndarray]:
        if not _GYM_AVAILABLE:
            return None
        if self.simulator is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        # Simple resource utilization vector.
        dims = []
        for layer in self.layer_names:
            layer_nodes = [n for n in self.simulator.nodes.values() if n.layer == layer]
            if not layer_nodes:
                dims.extend([0.0, 0.0, 0.0])
                continue
            cpu = sum(n.cpu_utilization for n in layer_nodes) / len(layer_nodes)
            mem = sum(n.memory_utilization for n in layer_nodes) / len(layer_nodes)
            q = sum(n.queue_length for n in layer_nodes) / max(1, len(layer_nodes))
            dims.extend([float(cpu), float(mem), float(q)])
        # Pad to a fixed size.
        obs = np.array(dims, dtype=np.float32)
        if obs.shape[0] < self.observation_space.shape[0]:
            obs = np.pad(obs, (0, self.observation_space.shape[0] - obs.shape[0]))
        return obs

    def _reward_for_last_step(self):
        # Reward is computed centrally in the simulator after each job outcome.
        return 0.0

    def _info_dict(self) -> Dict[str, Any]:
        return {
            "internal_step": self.internal_step,
            "elapsed_time": self.simulator.elapsed_time,
            "layer_names": self.layer_names,
            "task_count": len(self.tasks),
        }

    def render(self, mode: str = "human"):
        if mode == "human":
            print(
                f"[HGNN-DRL env] step={self.internal_step} "
                f"elapsed={self.simulator.elapsed_time:.3f}s "
                f"tasks={len(self.tasks)}"
            )

    def close(self):
        pass


# Re-export the simulator class for later usage.
from .simulator import Simulator  # noqa: E402
from .simulator import ComputeNode  # noqa: E402

__all__ = [
    "HGGNDRLGymnasiumEnv",
    "Simulator",
    "ComputeNode",
]
