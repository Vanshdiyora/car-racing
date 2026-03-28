"""Top-level training script — can be invoked directly or via main.py."""

from __future__ import annotations

import argparse
from typing import Any

from src.agent.ppo_agent import PPOAgent
from src.env.car_env import make_env
from src.train.trainer import Trainer
from src.utils import load_env_config, load_train_config, set_global_seed, setup_logger


def run_training(
    train_cfg: dict[str, Any] | None = None,
    env_cfg: dict[str, Any] | None = None,
    n_envs: int = 4,
) -> PPOAgent:
    """Entry point for programmatic training."""
    train_cfg = train_cfg or load_train_config()
    env_cfg = env_cfg or load_env_config()

    logger = setup_logger("car_racing")
    seed = train_cfg.get("training", {}).get("seed", 42)
    set_global_seed(seed)

    # Discover action count from a temporary env
    tmp_env = make_env(env_cfg, seed=seed)
    n_actions = tmp_env.action_space.n  # type: ignore[union-attr]
    tmp_env.close()

    agent = PPOAgent(train_cfg, env_cfg, n_actions)
    trainer = Trainer(agent, train_cfg, env_cfg)

    if n_envs > 1:
        return trainer.train_vec_env(n_envs=n_envs)
    return trainer.train_single_env()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CarRacing")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    run_training(
        train_cfg=load_train_config(args.train_config),
        env_cfg=load_env_config(args.env_config),
        n_envs=args.n_envs,
    )


if __name__ == "__main__":
    main()
