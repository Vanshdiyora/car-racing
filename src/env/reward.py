"""Custom reward shaping for the CarRacing environment.

Wraps the base environment to add bonuses/penalties that guide the agent
toward track-following, speed maintenance, and penalise off-track behaviour.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class RewardShaper(gym.Wrapper):
    """Augment the default reward signal with domain-specific shaping."""

    def __init__(
        self,
        env: gym.Env,
        *,
        speed_reward_weight: float = 0.1,
        grass_penalty: float = -0.5,
        backward_penalty: float = -1.0,
        standing_still_penalty: float = -0.1,
        tile_visit_bonus: float = 1.0,
    ) -> None:
        super().__init__(env)
        self._speed_w = speed_reward_weight
        self._grass_pen = grass_penalty
        self._backward_pen = backward_penalty
        self._still_pen = standing_still_penalty
        self._tile_bonus = tile_visit_bonus
        self._prev_tiles: int = 0
        self._step_count: int = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_tiles = 0
        self._step_count = 0
        return obs, info

    def step(self, action: Any) -> tuple[NDArray, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        shaped_reward = float(base_reward)

        # Access the underlying CarRacing env to read car state
        car_env = self._unwrap_car_env()
        if car_env is not None:
            car = getattr(car_env, "car", None)
            if car is not None:
                # Speed bonus: encourage the agent to maintain speed
                speed = np.sqrt(car.hull.linearVelocity[0] ** 2 + car.hull.linearVelocity[1] ** 2)
                shaped_reward += self._speed_w * min(speed, 50.0) / 50.0

                # Standing still penalty
                if speed < 1.0 and self._step_count > 10:
                    shaped_reward += self._still_pen

                # Grass detection via wheel contact
                on_grass = any(
                    getattr(w, "is_off_track", False) or (hasattr(w, "grass") and w.grass)
                    for w in car.wheels
                )
                if on_grass:
                    shaped_reward += self._grass_pen

            # Tile visit bonus
            tile_count = getattr(car_env, "tile_visited_count", 0)
            new_tiles = tile_count - self._prev_tiles
            if new_tiles > 0:
                shaped_reward += self._tile_bonus * new_tiles
                self._prev_tiles = tile_count

        info["shaped_reward"] = shaped_reward
        info["base_reward"] = float(base_reward)
        return obs, shaped_reward, terminated, truncated, info

    def _unwrap_car_env(self):
        """Walk the wrapper chain to find the raw CarRacing environment."""
        env = self.env
        while hasattr(env, "env"):
            if type(env).__name__ == "CarRacing":
                return env
            env = env.env
        if type(env).__name__ == "CarRacing":
            return env
        # Check unwrapped
        return getattr(self, "unwrapped", None)
