"""Frame preprocessing: grayscale, resize, normalisation, frame stacking."""

from __future__ import annotations

import collections
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert RGB observations to single-channel grayscale."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        h, w = self.observation_space.shape[:2]  # type: ignore[index]
        self.observation_space = gym.spaces.Box(0, 255, shape=(h, w, 1), dtype=np.uint8)

    def observation(self, obs: NDArray) -> NDArray:
        grey = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return grey[..., np.newaxis]


# Try to use OpenCV for fast resizing; fall back to numpy if unavailable
try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class ResizeWrapper(gym.ObservationWrapper):
    """Resize observations to target (H, W) using OpenCV (fast) or numpy."""

    def __init__(self, env: gym.Env, size: tuple[int, int]) -> None:
        super().__init__(env)
        self._size = size  # (H, W)
        channels = self.observation_space.shape[2] if len(self.observation_space.shape) == 3 else 1  # type: ignore[union-attr]
        self.observation_space = gym.spaces.Box(0, 255, shape=(*size, channels), dtype=np.uint8)

    def observation(self, obs: NDArray) -> NDArray:
        squeeze = obs.ndim == 3 and obs.shape[-1] == 1
        src = obs.squeeze(-1) if squeeze else obs
        if _HAS_CV2:
            resized = _cv2.resize(src, (self._size[1], self._size[0]), interpolation=_cv2.INTER_AREA)
        else:
            # Fast numpy nearest-neighbour resize (no PIL dependency)
            h, w = self._size
            oh, ow = src.shape[:2]
            row_idx = (np.arange(h) * oh // h).astype(int)
            col_idx = (np.arange(w) * ow // w).astype(int)
            resized = src[np.ix_(row_idx, col_idx)]
        resized = np.asarray(resized, dtype=np.uint8)
        if resized.ndim == 2:
            resized = resized[..., np.newaxis]
        return resized


class FrameSkipWrapper(gym.Wrapper):
    """Repeat each action for *skip* frames, summing rewards."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = max(1, skip)

    def step(self, action: Any) -> tuple[NDArray, SupportsFloat, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class FrameStackWrapper(gym.Wrapper):
    """Stack the last *n* observations along the channel axis."""

    def __init__(self, env: gym.Env, n: int = 4) -> None:
        super().__init__(env)
        self._n = n
        low = np.repeat(env.observation_space.low, n, axis=-1)  # type: ignore[union-attr]
        high = np.repeat(env.observation_space.high, n, axis=-1)  # type: ignore[union-attr]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)
        self._frames: collections.deque[NDArray] = collections.deque(maxlen=n)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self._n):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> NDArray:
        return np.concatenate(list(self._frames), axis=-1)


class NormalizeWrapper(gym.ObservationWrapper):
    """Scale pixel observations from [0, 255] to [0.0, 1.0]."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32  # type: ignore[union-attr]
        )

    def observation(self, obs: NDArray) -> NDArray:
        return obs.astype(np.float32) / 255.0
