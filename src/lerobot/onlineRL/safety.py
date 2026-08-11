"""Action validation used before any Piper command is emitted."""

from __future__ import annotations

import torch


def safe_action(action: torch.Tensor, action_dim: int, limit: float) -> torch.Tensor:
    action = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
    if action.numel() != action_dim:
        raise ValueError(f"Expected {action_dim} action values, got {action.numel()}.")
    if not torch.isfinite(action).all():
        raise ValueError("Refusing non-finite robot action.")
    return action.clamp(-limit, limit)
