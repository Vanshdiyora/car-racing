"""CarRacing environment factory — assembles wrappers from config."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from src.env.preprocess import (
    FrameSkipWrapper,
    FrameStackWrapper,
    GrayscaleWrapper,
    NormalizeWrapper,
    ResizeWrapper,
)
from src.env.reward import RewardShaper


def make_env(
    env_cfg: dict[str, Any],
    seed: int = 0,
    render: bool = False,
) -> gym.Env:
    """Build a fully-wrapped CarRacing environment.

    Parameters
    ----------
    env_cfg : dict
        Loaded from ``configs/env_config.yaml``.
    seed : int
        Random seed for the environment.
    render : bool
        If *True*, force ``render_mode="human"``.
    """
    e = env_cfg.get("environment", {})
    p = env_cfg.get("preprocessing", {})
    r = env_cfg.get("reward_shaping", {})

    render_mode = "human" if render else e.get("render_mode")
    env = gym.make(
        e.get("name", "CarRacing-v3"),
        continuous=e.get("continuous", False),
        render_mode=render_mode,
        max_episode_steps=e.get("max_episode_steps", 1000000),
    )

    # Frame skip
    env = FrameSkipWrapper(env, skip=p.get("frame_skip", 4))

    # Reward shaping (before pixel transforms so it can read car state)
    if r.get("enabled", False):
        env = RewardShaper(
            env,
            speed_reward_weight=r.get("speed_reward_weight", 0.1),
            grass_penalty=r.get("grass_penalty", -0.5),
            backward_penalty=r.get("backward_penalty", -1.0),
            standing_still_penalty=r.get("standing_still_penalty", -0.1),
            tile_visit_bonus=r.get("tile_visit_bonus", 1.0),
        )

    # Pixel preprocessing
    if p.get("grayscale", True):
        env = GrayscaleWrapper(env)

    resize = p.get("resize", [84, 84])
    env = ResizeWrapper(env, size=tuple(resize))

    env = FrameStackWrapper(env, n=p.get("frame_stack", 4))

    if p.get("normalize", True):
        env = NormalizeWrapper(env)

    env.reset(seed=seed)
    return env


def make_race_env(
    seed: int = 42,
    continuous: bool = True,
    max_episode_steps: int = 10_000,
) -> gym.Env:
    """Create a raw CarRacing env for race mode — no wrappers, no frame skip.

    Human plays on a continuous-action env for natural driving feel.
    Returns RGB frames directly via ``env.render()``.
    """
    env = gym.make(
        "CarRacing-v3",
        continuous=continuous,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )
    env.reset(seed=seed)
    return env


def make_vec_env(
    env_cfg: dict[str, Any],
    n_envs: int,
    seed: int = 0,
) -> gym.vector.VectorEnv:
    """Create a vectorised environment for parallel data collection."""

    def _thunk(i: int):
        def _init():
            return make_env(env_cfg, seed=seed + i)
        return _init

    return gym.vector.AsyncVectorEnv([_thunk(i) for i in range(n_envs)])
