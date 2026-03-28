"""Trainer: orchestrates the PPO training loop with callbacks."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from tqdm import trange

from src.agent.ppo_agent import PPOAgent
from src.env.car_env import make_env, make_vec_env
from src.train.callbacks import CheckpointCallback, CSVCallback, TensorBoardCallback

logger = logging.getLogger(__name__)


class Trainer:
    """High-level trainer that manages environment interaction, updates, and logging."""

    def __init__(
        self,
        agent: PPOAgent,
        train_cfg: dict[str, Any],
        env_cfg: dict[str, Any],
    ) -> None:
        self.agent = agent
        self.train_cfg = train_cfg
        self.env_cfg = env_cfg

        t = train_cfg.get("training", {})
        ckpt = train_cfg.get("checkpoints", {})
        log_cfg = train_cfg.get("logging", {})
        ev = train_cfg.get("evaluation", {})

        self.total_timesteps = t.get("total_timesteps", 2_000_000)
        self.n_steps = t.get("n_steps", 2048)
        self.seed = t.get("seed", 42)

        self.eval_freq = ev.get("eval_freq", 20_000)
        self.n_eval_episodes = ev.get("n_eval_episodes", 10)
        self.eval_deterministic = ev.get("deterministic", True)

        # Callbacks
        self.ckpt_cb = CheckpointCallback(
            save_dir=ckpt.get("save_dir", "models/ppo_baseline"),
            best_dir=ckpt.get("best_model_dir", "models/ppo_best"),
            save_freq=ckpt.get("save_freq", 50_000),
        )
        self.tb_cb = TensorBoardCallback(log_cfg.get("tensorboard_dir", "logs/tensorboard"))
        self.csv_cb = CSVCallback(
            log_cfg.get("csv_path", "logs/training_logs.csv"),
            fields=["step", "loss", "pg_loss", "vf_loss", "entropy", "mean_reward", "time"],
        )

    # ------------------------------------------------------------------
    # Single-env training (simpler, good for debugging)
    # ------------------------------------------------------------------

    def train_single_env(self) -> PPOAgent:
        """Train in a single environment (agent.n_envs = 1)."""
        env = make_env(self.env_cfg, seed=self.seed)
        self.agent.set_n_envs(1)

        logger.info("Starting PPO training (single env) — %d timesteps", self.total_timesteps)
        logger.info("Device: %s", self.agent.device)

        obs, _ = env.reset(seed=self.seed)
        global_step = 0
        start = time.time()

        n_rollouts = self.total_timesteps // self.n_steps

        for rollout in trange(1, n_rollouts + 1, desc="PPO"):
            self.agent.buffer.reset()
            for _ in range(self.n_steps):
                action = self.agent.predict(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store as arrays for buffer compatibility
                self.agent.buffer.add(
                    obs[np.newaxis],
                    np.array([action]),
                    np.array([reward], dtype=np.float32),
                    np.array([float(done)], dtype=np.float32),
                    np.zeros(1, dtype=np.float32),  # log_prob placeholder
                    np.zeros(1, dtype=np.float32),   # value placeholder
                )

                obs = next_obs
                global_step += 1
                if done:
                    obs, _ = env.reset()

            last_val = self.agent.get_value(obs[np.newaxis])
            self.agent.buffer.compute_gae(last_val, self.agent.gamma, self.agent.gae_lambda)
            metrics = self.agent.update()

            self.tb_cb.log_scalars("train", metrics, global_step)
            self.ckpt_cb.on_step(self.agent, global_step)

            if global_step % self.eval_freq < self.n_steps:
                mean_r = self._evaluate()
                self.tb_cb.log_scalars("eval", {"mean_reward": mean_r}, global_step)
                self.ckpt_cb.on_eval(self.agent, mean_r, global_step)
                self.csv_cb.log({
                    "step": global_step, **metrics,
                    "mean_reward": mean_r, "time": time.time() - start,
                })
                logger.info("Step %d | eval=%.1f | loss=%.4f", global_step, mean_r, metrics["loss"])

        self.agent.save("models/ppo_baseline/final_model.pt")
        self.tb_cb.close()
        self.csv_cb.close()
        env.close()
        return self.agent

    # ------------------------------------------------------------------
    # Vectorised training (faster)
    # ------------------------------------------------------------------

    def train_vec_env(self, n_envs: int = 4) -> PPOAgent:
        """Train with *n_envs* parallel environments for faster collection."""
        vec_env = make_vec_env(self.env_cfg, n_envs=n_envs, seed=self.seed)
        self.agent.set_n_envs(n_envs)

        logger.info("Starting PPO training (%d envs) — %d timesteps", n_envs, self.total_timesteps)
        logger.info("Device: %s", self.agent.device)

        obs, _ = vec_env.reset(seed=self.seed)
        global_step = 0
        start = time.time()

        n_rollouts = self.total_timesteps // (self.n_steps * n_envs)

        for rollout in trange(1, n_rollouts + 1, desc="PPO"):
            self.agent.buffer.reset()
            for _ in range(self.n_steps):
                actions, log_probs, values = self.agent.predict_batch(obs)
                next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
                dones = np.logical_or(terminated, truncated).astype(np.float32)

                self.agent.buffer.add(obs, actions, rewards, dones, log_probs, values)
                obs = next_obs
                global_step += n_envs

            last_val = self.agent.get_value(obs)
            self.agent.buffer.compute_gae(last_val, self.agent.gamma, self.agent.gae_lambda)
            metrics = self.agent.update()

            self.tb_cb.log_scalars("train", metrics, global_step)
            self.ckpt_cb.on_step(self.agent, global_step)

            if global_step % self.eval_freq < (self.n_steps * n_envs):
                mean_r = self._evaluate()
                self.tb_cb.log_scalars("eval", {"mean_reward": mean_r}, global_step)
                self.ckpt_cb.on_eval(self.agent, mean_r, global_step)
                self.csv_cb.log({
                    "step": global_step, **metrics,
                    "mean_reward": mean_r, "time": time.time() - start,
                })
                logger.info("Step %d | eval=%.1f | loss=%.4f", global_step, mean_r, metrics["loss"])

        self.agent.save("models/ppo_baseline/final_model.pt")
        self.tb_cb.close()
        self.csv_cb.close()
        vec_env.close()
        return self.agent

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> float:
        from src.evaluate.evaluate import evaluate_agent

        return evaluate_agent(
            self.agent,
            self.env_cfg,
            n_episodes=self.n_eval_episodes,
            deterministic=self.eval_deterministic,
            seed=self.seed + 10_000,
        )
