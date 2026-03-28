"""Custom CNN policy network for PPO.

Uses a Nature-DQN style convolutional backbone with separate actor and
critic heads.  Compatible with both standalone PPO and stable-baselines3.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NatureCNN(nn.Module):
    """Nature-DQN convolutional feature extractor.

    Input : (B, C, H, W) float32 in [0, 1]
    Output: (B, feature_dim)
    """

    def __init__(self, in_channels: int, feature_dim: int = 512) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            flat_size = self.conv(dummy).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.conv(x))


class ActorCriticPolicy(nn.Module):
    """Shared-backbone actor-critic network for discrete PPO."""

    def __init__(self, in_channels: int, n_actions: int, feature_dim: int = 512) -> None:
        super().__init__()
        self.features = NatureCNN(in_channels, feature_dim)
        self.actor_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (action_logits, state_value)."""
        feat = self.features(x)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample or evaluate actions.

        Returns
        -------
        action, log_prob, entropy, value
        """
        logits, value = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def get_value(self, x: Tensor) -> Tensor:
        feat = self.features(x)
        return self.critic_head(feat).squeeze(-1)
