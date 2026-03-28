"""Tests for the PPO agent and policy network."""

import numpy as np
import torch

from src.agent.policy import ActorCriticPolicy, NatureCNN
from src.agent.utils import obs_to_tensor


def test_nature_cnn_output_shape():
    net = NatureCNN(in_channels=4, feature_dim=512)
    x = torch.randn(2, 4, 84, 84)
    out = net(x)
    assert out.shape == (2, 512)


def test_actor_critic_forward():
    net = ActorCriticPolicy(in_channels=4, n_actions=5)
    x = torch.randn(2, 4, 84, 84)
    logits, value = net(x)
    assert logits.shape == (2, 5)
    assert value.shape == (2, 1)


def test_get_action_and_value():
    net = ActorCriticPolicy(in_channels=4, n_actions=5)
    x = torch.randn(3, 4, 84, 84)
    action, log_prob, entropy, value = net.get_action_and_value(x)
    assert action.shape == (3,)
    assert log_prob.shape == (3,)
    assert entropy.shape == (3,)
    assert value.shape == (3,)


def test_obs_to_tensor_single():
    obs = np.random.rand(84, 84, 4).astype(np.float32)
    t = obs_to_tensor(obs, torch.device("cpu"))
    assert t.shape == (1, 4, 84, 84)


def test_obs_to_tensor_batch():
    obs = np.random.rand(8, 84, 84, 4).astype(np.float32)
    t = obs_to_tensor(obs, torch.device("cpu"))
    assert t.shape == (8, 4, 84, 84)
