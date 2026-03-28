"""Race metrics: lap time, speed, penalties, track completion."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RaceMetrics:
    """Accumulates real-time metrics for one racer (human or AI)."""

    label: str = "player"
    total_reward: float = 0.0
    steps: int = 0
    tiles_visited: int = 0
    grass_steps: int = 0
    penalties: float = 0.0
    lap_times: list[float] = field(default_factory=list)

    # Internal state
    _lap_start: float = field(default=0.0, repr=False)
    _started: bool = field(default=False, repr=False)

    def start(self) -> None:
        self._lap_start = time.time()
        self._started = True

    def record_step(self, reward: float, on_grass: bool = False) -> None:
        self.total_reward += reward
        self.steps += 1
        if on_grass:
            self.grass_steps += 1
            self.penalties += 0.1

    def finish_lap(self) -> float:
        if self._started:
            lap = time.time() - self._lap_start
            self.lap_times.append(lap)
            self._lap_start = time.time()
            return lap
        return 0.0

    @property
    def avg_lap_time(self) -> float:
        return sum(self.lap_times) / len(self.lap_times) if self.lap_times else 0.0

    @property
    def total_time(self) -> float:
        return sum(self.lap_times)

    def summary(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "total_reward": round(self.total_reward, 1),
            "steps": self.steps,
            "tiles_visited": self.tiles_visited,
            "grass_steps": self.grass_steps,
            "penalties": round(self.penalties, 2),
            "laps_completed": len(self.lap_times),
            "avg_lap_time": round(self.avg_lap_time, 2),
            "total_time": round(self.total_time, 2),
        }


def compute_score(
    metrics: RaceMetrics,
    track_weight: float = 1.0,
    time_weight: float = 0.5,
    penalty_weight: float = 0.3,
) -> float:
    """Compute a composite race score (higher is better)."""
    track_score = metrics.total_reward * track_weight
    time_bonus = max(0.0, 300.0 - metrics.total_time) * time_weight if metrics.total_time > 0 else 0.0
    penalty_deduction = metrics.penalties * penalty_weight
    return track_score + time_bonus - penalty_deduction
