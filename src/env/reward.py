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
        spin_penalty: float = -0.5,
        drift_penalty: float = -0.4,
        highspeed_turn_penalty: float = -0.3,
    ) -> None:
        super().__init__(env)
        self._speed_w = speed_reward_weight
        self._grass_pen = grass_penalty
        self._backward_pen = backward_penalty
        self._still_pen = standing_still_penalty
        self._tile_bonus = tile_visit_bonus
        self._spin_pen = spin_penalty
        self._drift_pen = drift_penalty
        self._hs_turn_pen = highspeed_turn_penalty
        self._prev_tiles: int = 0
        self._step_count: int = 0
        self._car_env_cache: Any = None  # cached unwrapped env reference

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_tiles = 0
        self._step_count = 0
        self._car_env_cache = self._unwrap_car_env()  # refresh on reset
        return obs, info

    def step(self, action: Any) -> tuple[NDArray, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        shaped_reward = float(base_reward)

        # Use cached reference instead of walking wrapper chain every step
        car_env = self._car_env_cache
        if car_env is not None:
            car = getattr(car_env, "car", None)
            if car is not None:
                vx, vy = car.hull.linearVelocity
                speed = np.sqrt(vx ** 2 + vy ** 2)

                # Car heading vector (forward direction)
                angle = car.hull.angle
                forward_x = -np.sin(angle)
                forward_y = np.cos(angle)

                # Dot product: positive = moving forward, negative = backward
                forward_speed = vx * forward_x + vy * forward_y

                # Speed bonus: only reward *forward* speed
                if forward_speed > 0:
                    shaped_reward += self._speed_w * min(forward_speed, 50.0) / 50.0

                # Backward penalty: penalise when car is moving backward
                if forward_speed < -1.0:
                    shaped_reward += self._backward_pen

                # Standing still penalty
                if speed < 1.0 and self._step_count > 10:
                    shaped_reward += self._still_pen

                # Spin penalty: penalise high angular velocity (U-turns)
                angular_vel = abs(car.hull.angularVelocity)
                if angular_vel > 0.5:
                    shaped_reward += self._spin_pen * min(angular_vel / 3.0, 1.0)

                # Drift penalty: lateral slip at speed
                lateral_speed = abs(vx * forward_y - vy * forward_x)
                if speed > 10.0 and lateral_speed > 2.0:
                    drift_ratio = min(lateral_speed / speed, 1.0)
                    shaped_reward += self._drift_pen * drift_ratio

                # High-speed turn penalty
                if speed > 30.0 and angular_vel > 0.3:
                    severity = min(angular_vel / 2.0, 1.0) * min(speed / 60.0, 1.0)
                    shaped_reward += self._hs_turn_pen * severity

                # Grass detection — wheel.tiles is empty when on grass
                on_grass = any(
                    len(getattr(w, "tiles", {1})) == 0
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
