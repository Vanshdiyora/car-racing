"""Model evaluation — run episodes and collect statistics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.env.car_env import make_env

logger = logging.getLogger(__name__)


def evaluate_agent(
    agent: Any,
    env_cfg: dict[str, Any],
    *,
    n_episodes: int = 10,
    deterministic: bool = True,
    seed: int = 0,
    render: bool = False,
) -> float:
    """Run *n_episodes* and return the mean total reward.

    Works with any agent that exposes ``predict(obs, deterministic=...)``.
    """
    env = make_env(env_cfg, seed=seed, render=render)
    rewards: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        rewards.append(total_reward)

    env.close()
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    logger.info("Eval %d episodes: %.1f ± %.1f", n_episodes, mean, std)
    return mean


def evaluate_with_details(
    agent: Any,
    env_cfg: dict[str, Any],
    *,
    n_episodes: int = 10,
    deterministic: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Detailed evaluation returning per-episode stats."""
    env = make_env(env_cfg, seed=seed)
    episodes: list[dict[str, Any]] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        episodes.append({"episode": ep, "reward": total_reward, "steps": steps})

    env.close()
    rewards = [e["reward"] for e in episodes]
    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
    }
