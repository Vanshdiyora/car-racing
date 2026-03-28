"""Training callbacks: TensorBoard logging, CSV logging, checkpointing."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

from src.utils.logger import CSVLogger

logger = logging.getLogger(__name__)


class CheckpointCallback:
    """Save agent checkpoints at fixed intervals and track the best model."""

    def __init__(
        self,
        save_dir: str,
        best_dir: str,
        save_freq: int = 50_000,
    ) -> None:
        self._save_dir = pathlib.Path(save_dir)
        self._best_dir = pathlib.Path(best_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._best_dir.mkdir(parents=True, exist_ok=True)
        self._save_freq = save_freq
        self._best_reward = -float("inf")

    def on_step(self, agent: Any, global_step: int) -> None:
        if global_step > 0 and global_step % self._save_freq == 0:
            path = self._save_dir / f"checkpoint_{global_step}.pt"
            agent.save(path)

    def on_eval(self, agent: Any, mean_reward: float, global_step: int) -> None:
        if mean_reward > self._best_reward:
            self._best_reward = mean_reward
            agent.save(self._best_dir / "best_model.pt")
            logger.info(
                "New best model! reward=%.1f at step %d", mean_reward, global_step
            )


class TensorBoardCallback:
    """Write scalars to TensorBoard."""

    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, tag_prefix: str, metrics: dict[str, float], step: int) -> None:
        for key, val in metrics.items():
            self.writer.add_scalar(f"{tag_prefix}/{key}", val, step)

    def close(self) -> None:
        self.writer.close()


class CSVCallback:
    """Append training metrics to a CSV file."""

    def __init__(self, path: str, fields: list[str]) -> None:
        self._logger = CSVLogger(path, fields)

    def log(self, row: dict[str, Any]) -> None:
        self._logger.log(row)

    def close(self) -> None:
        self._logger.close()
