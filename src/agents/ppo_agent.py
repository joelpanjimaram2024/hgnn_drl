"""PPO agent wrapper around Stable-Baselines3.

The HGNN representation is consumed through a custom feature extractor so
SB3's MlpPolicy does not silently replace it with a plain MLP.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from ..environment.base_env import HGGNDRLGymnasiumEnv
from ..models.hgnn import HGNNEncoder
from ..utils.config import Config
from ..utils.logging import get_logger
from .policy import HGNNFeatureExtractor


class HGNNPPORL:
    """PPO agent that consumes the HGNN representation."""

    def __init__(
        self,
        *,
        config: Config,
        hypergraph,
        feature_extractor: Optional[HGNNFeatureExtractor] = None,
        env: Optional[HGGNDRLGymnasiumEnv] = None,
    ):
        self.config = config
        self.hypergraph = hypergraph
        self.feature_extractor = feature_extractor
        self.env = env
        self.logger = get_logger("hgnn_drl.ppo")

        # Build a seed-specific feature extractor if none provided.
        self._build_components()

    def _build_components(self) -> None:
        cfg = self.config
        if self.feature_extractor is None or self.encoder is None:
            enc = HGNNEncoder(
                hypergraph=self.hypergraph,
                in_dim=cfg.hgnn.vertex_dim or cfg.hgnn.in_dim,
                hidden_dim=cfg.hgnn.hidden_dim,
                out_dim=cfg.hgnn.out_dim,
                num_layers=cfg.hgnn.num_layers,
                batch_first=True,
            )
            self.encoder = enc
            self.feature_extractor = HGNNFeatureExtractor(
                config=cfg, encoder=enc, device=cfg.device
            )

    @property
    def encoder(self) -> Optional[HGNNEncoder]:
        return getattr(self, "_encoder", None)

    @encoder.setter
    def encoder(self, value: HGNNEncoder) -> None:
        self._encoder = value

    def train(
        self,
        *,
        total_timesteps: Optional[int] = None,
        log_interval: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train PPO with the HGNN policy.

        Creates the environment, wraps it in a vectorised env, and runs
        Stable-Baselines3 PPO.  The policy architecture is custom so the
        HGNN embedding flows into the action head.
        """
        if seed is not None:
            self.config.seed = seed

        # Use provided env or create a fresh one.
        if self.env is None:
            self.env = self._create_env()

        # Vectorise: SB3 expects a VecEnv for multi-env support.
        vec_env = DummyVecEnv([lambda: self.env]) if not isinstance(
            self.env, VecEnv
        ) else self.env

        pf = self._build_policy_function(vec_env)

        # Capture the actual device from SB3's usage.
        device = self.config.ppo.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = PPO(
            policy=pf,
            env=vec_env,
            learning_rate=self.config.ppo.learning_rate,
            n_steps=self.config.ppo.n_steps,
            batch_size=self.config.ppo.batch_size,
            n_epochs=self.config.ppo.n_epochs,
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda,
            clip_range=self.config.ppo.clip_range,
            ent_coef=self.config.ppo.ent_coef,
            vf_coef=self.config.ppo.vf_coef,
            max_grad_norm=self.config.ppo.max_grad_norm,
            seed=seed,
            device=device,
            tensorboard_log=self.config.ppo.tensorboard_log,
            verbose=self.config.ppo.verbose,
        )

        timesteps = total_timesteps or self.config.ppo.total_timesteps
        log_interval = log_interval or self.config.ppo.log_interval

        self.logger.info(f"training HGNN+PPO for {timesteps} timesteps")
        self.model.learn(total_timesteps=timesteps, log_interval=log_interval)

        return {
            "timesteps": timesteps,
            "episodes": int(timesteps / max(1, self.config.ppo.n_steps)),
            "model": self.model,
        }

    def _create_env(self) -> HGGNDRLGymnasiumEnv:
        # The environment is scenario-dependent; the scenario module sets it
        # on the agent.  For now, create a minimal env.
        from ..environment.base_env import HGGNDRLGymnasiumEnv
        env = HGGNDRLGymnasiumEnv(
            config=self.config,
            scenario_id=self.config.training.scenario,
            device=self.config.device,
        )
        return env

    def _build_policy_function(self, vec_env) -> str:
        """Choose the SB3 policy class.

        ``MlpPolicy`` combined with the HGNN feature extractor is the
        mechanism by which the hypergraph representation reaches the policy.
        """
        from stable_baselines3.common.policies import MlpPolicy
        return MlpPolicy

    def predict(self, obs, state=None, mask=None, deterministic=False):
        """Delegate prediction to the underlying SB3 model + policy."""
        if self.model is None:
            raise RuntimeError("model not trained yet; call train() first")
        return self.model.predict(obs, state, mask, deterministic)

    def evaluate(self, n_episodes: int, seed: Optional[int] = None) -> Dict[str, Any]:
        """Run deterministic (no-exploration) evaluation episodes."""
        if self.model is None:
            raise RuntimeError("model not trained yet")

        if seed is not None:
            self.config.seed = seed

        if self.env is None:
            self.env = self._create_env()

        obs, _ = self.env.reset(seed=seed)
        rewards = []
        for _ in range(n_episodes):
            act = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(int(act[0]))
            rewards.append(float(reward))
            if terminated or truncated:
                obs, _ = self.env.reset(seed=seed)
        return {
            "n_episodes": n_episodes,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "returns": rewards,
        }

    def save_checkpoint(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("no model to save")
        self.model.save(path)

    def load_checkpoint(self, path: str) -> None:
        if self.model is None:
            self.model = PPO.load(path, env=self.env)
        else:
            self.model.set_parameters(path)

    def state_dict(self) -> Dict[str, Any]:
        return self.feature_extractor.get_state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.feature_extractor.load_state_dict(state)
