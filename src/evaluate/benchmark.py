"""Benchmark: compare PPO agent performance against human baseline stats."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import numpy as np

from src.agent.ppo_agent import PPOAgent
from src.evaluate.evaluate import evaluate_with_details
from src.utils import load_env_config, load_train_config

logger = logging.getLogger(__name__)


def benchmark_agent(
    checkpoint: str,
    env_cfg: dict[str, Any] | None = None,
    train_cfg: dict[str, Any] | None = None,
    n_episodes: int = 50,
    seed: int = 0,
    output_path: str = "outputs/reports/benchmark.json",
) -> dict[str, Any]:
    """Run a comprehensive benchmark and save results as JSON."""
    env_cfg = env_cfg or load_env_config()
    train_cfg = train_cfg or load_train_config()

    from src.env.car_env import make_env

    tmp = make_env(env_cfg, seed=seed)
    n_actions = tmp.action_space.n  # type: ignore[union-attr]
    tmp.close()

    agent = PPOAgent(train_cfg, env_cfg, n_actions)
    agent.load(checkpoint)

    results = evaluate_with_details(
        agent, env_cfg, n_episodes=n_episodes, deterministic=True, seed=seed
    )

    # Reference human performance (typical for CarRacing-v3 discrete)
    results["human_baseline"] = {
        "note": "Approximate human scores on CarRacing-v3",
        "mean_reward": 800.0,
        "std_reward": 100.0,
    }

    results["comparison"] = {
        "ai_vs_human_mean": results["mean_reward"] - 800.0,
        "ai_superhuman": results["mean_reward"] > 800.0,
    }

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark saved → %s", out)
    logger.info(
        "AI mean=%.1f | Human baseline=800 | Diff=%+.1f",
        results["mean_reward"],
        results["comparison"]["ai_vs_human_mean"],
    )
    return results
