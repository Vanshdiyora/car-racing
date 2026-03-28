"""AI controller — wraps a trained PPO agent for race inference."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from src.agent.ppo_agent import PPOAgent
from src.env.car_env import make_env


class AIController:
    """Load a PPO checkpoint and expose a simple ``get_action(obs)`` API."""

    def __init__(
        self,
        checkpoint: str | pathlib.Path,
        train_cfg: dict[str, Any],
        env_cfg: dict[str, Any],
    ) -> None:
        # Discover n_actions
        tmp = make_env(env_cfg, seed=0)
        n_actions = tmp.action_space.n  # type: ignore[union-attr]
        tmp.close()

        self.agent = PPOAgent(train_cfg, env_cfg, n_actions)
        self.agent.load(checkpoint)
        self.agent.policy.eval()

    def get_action(self, obs: np.ndarray) -> int:
        """Return a deterministic action for the given observation."""
        return self.agent.predict(obs, deterministic=True)
