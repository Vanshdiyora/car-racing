"""Agent-level utilities: tensor conversion, device helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def get_device(preference: str = "auto") -> torch.device:
    """Resolve device string to a ``torch.device``."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> Tensor:
    """Convert HWC numpy observation(s) to BCHW float tensor on *device*.

    Handles both single (H, W, C) and batched (B, H, W, C) observations.
    """
    t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if t.ndim == 3:
        t = t.permute(2, 0, 1).unsqueeze(0)
    elif t.ndim == 4:
        t = t.permute(0, 3, 1, 2)
    return t
