"""Tests for environment wrappers and factory."""

import numpy as np
import gymnasium as gym

from src.env.preprocess import (
    GrayscaleWrapper,
    ResizeWrapper,
    FrameSkipWrapper,
    FrameStackWrapper,
    NormalizeWrapper,
)


def test_grayscale_wrapper():
    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
    env = GrayscaleWrapper(env)
    obs, _ = env.reset(seed=0)
    assert obs.shape[2] == 1
    assert obs.dtype == np.uint8
    env.close()


def test_resize_wrapper():
    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, size=(84, 84))
    obs, _ = env.reset(seed=0)
    assert obs.shape[:2] == (84, 84)
    env.close()


def test_frame_stack_wrapper():
    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, size=(84, 84))
    env = FrameStackWrapper(env, n=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 4)
    env.close()


def test_normalize_wrapper():
    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, size=(84, 84))
    env = FrameStackWrapper(env, n=4)
    env = NormalizeWrapper(env)
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.max() <= 1.0
    env.close()
