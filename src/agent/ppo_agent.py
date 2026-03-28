"""PPO Agent — full training & inference interface.

Implements:
- On-policy rollout storage with GAE
- Clipped surrogate objective with entropy bonus
- Checkpoint save / load
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.agent.policy import ActorCriticPolicy
from src.agent.utils import get_device, obs_to_tensor

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Rollout buffer
# -----------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-length rollout storage for on-policy data collection."""

    def __init__(self, n_steps: int, n_envs: int, obs_shape: tuple[int, ...]) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
    ) -> None:
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = done
        self.log_probs[self._ptr] = log_prob
        self.values[self._ptr] = value
        self._ptr += 1

    def reset(self) -> None:
        self._ptr = 0

    @property
    def full(self) -> bool:
        return self._ptr >= self.n_steps

    def compute_gae(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1.0 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def flatten(self) -> dict[str, np.ndarray]:
        b = self.n_steps * self.n_envs
        return {
            "obs": self.obs.reshape(b, *self.obs.shape[2:]),
            "actions": self.actions.reshape(b),
            "log_probs": self.log_probs.reshape(b),
            "advantages": self.advantages.reshape(b),
            "returns": self.returns.reshape(b),
            "values": self.values.reshape(b),
        }

    def get_batches(self, batch_size: int, rng: np.random.Generator):
        total = self.n_steps * self.n_envs
        indices = rng.permutation(total)
        for start in range(0, total, batch_size):
            yield indices[start : start + batch_size]


# -----------------------------------------------------------------------
# PPO Agent
# -----------------------------------------------------------------------

class PPOAgent:
    """Production-ready PPO agent for discrete CarRacing."""

    def __init__(self, train_cfg: dict[str, Any], env_cfg: dict[str, Any], n_actions: int) -> None:
        t = train_cfg.get("training", {})
        p = env_cfg.get("preprocessing", {})

        self.gamma = t.get("gamma", 0.99)
        self.gae_lambda = t.get("gae_lambda", 0.95)
        self.clip_range = t.get("clip_range", 0.2)
        self.vf_coef = t.get("vf_coef", 0.5)
        self.ent_coef = t.get("ent_coef", 0.01)
        self.max_grad_norm = t.get("max_grad_norm", 0.5)
        self.n_epochs = t.get("n_epochs", 10)
        self.batch_size = t.get("batch_size", 64)
        self.n_steps = t.get("n_steps", 2048)
        self.normalize_advantage = t.get("normalize_advantage", True)

        self.device = get_device(t.get("device", "auto"))
        self.n_actions = n_actions

        in_channels = p.get("frame_stack", 4)
        self.policy = ActorCriticPolicy(in_channels, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=t.get("learning_rate", 3e-4), eps=1e-5
        )

        resize = tuple(p.get("resize", [84, 84]))
        obs_shape = (*resize, in_channels)
        n_envs = 1  # overridden by trainer for vec envs
        self.buffer = RolloutBuffer(self.n_steps, n_envs, obs_shape)
        self.rng = np.random.default_rng(t.get("seed", 42))
        self._update_count = 0

    def set_n_envs(self, n: int) -> None:
        """Re-create the rollout buffer for *n* parallel environments."""
        obs_shape = self.buffer.obs.shape[2:]
        self.buffer = RolloutBuffer(self.n_steps, n, obs_shape)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, *, deterministic: bool = False) -> int:
        """Return a single action (for evaluation / race mode)."""
        with torch.no_grad():
            obs_t = obs_to_tensor(obs, self.device)
            logits, _ = self.policy(obs_t)
            if deterministic:
                return int(logits.argmax(dim=-1).item())
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().item())

    def predict_batch(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (actions, log_probs, values) for a batch."""
        with torch.no_grad():
            obs_t = obs_to_tensor(obs, self.device)
            actions, log_probs, _, values = self.policy.get_action_and_value(obs_t)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = obs_to_tensor(obs, self.device)
            return self.policy.get_value(obs_t).cpu().numpy()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        flat = self.buffer.flatten()
        obs_all = torch.as_tensor(flat["obs"], device=self.device).float().permute(0, 3, 1, 2)
        actions_all = torch.as_tensor(flat["actions"], device=self.device).long()
        old_log_probs = torch.as_tensor(flat["log_probs"], device=self.device).float()
        advantages = torch.as_tensor(flat["advantages"], device=self.device).float()
        returns = torch.as_tensor(flat["returns"], device=self.device).float()

        if self.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {"loss": 0.0, "pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            for idx in self.buffer.get_batches(self.batch_size, self.rng):
                b = torch.as_tensor(idx, device=self.device).long()
                _, new_lp, entropy, new_val = self.policy.get_action_and_value(
                    obs_all[b], actions_all[b]
                )

                ratio = torch.exp(new_lp - old_log_probs[b])
                pg1 = ratio * advantages[b]
                pg2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages[b]
                pg_loss = -torch.min(pg1, pg2).mean()
                vf_loss = torch.nn.functional.mse_loss(new_val, returns[b])
                ent_loss = -entropy.mean()
                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["loss"] += loss.item()
                stats["pg_loss"] += pg_loss.item()
                stats["vf_loss"] += vf_loss.item()
                stats["entropy"] += entropy.mean().item()
                n_updates += 1

        self._update_count += 1
        self.buffer.reset()
        return {k: v / max(n_updates, 1) for k, v in stats.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_count": self._update_count,
            },
            path,
        )
        logger.info("Saved PPO checkpoint → %s", path)

    def load(self, path: str | pathlib.Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(data["policy"])
        self.optimizer.load_state_dict(data["optimizer"])
        self._update_count = data.get("update_count", 0)
        logger.info("Loaded PPO checkpoint ← %s (updates=%d)", path, self._update_count)
