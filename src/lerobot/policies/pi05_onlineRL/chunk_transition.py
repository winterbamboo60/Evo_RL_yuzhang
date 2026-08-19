"""Sliding-window conversion for fixed-horizon PI05 online-RL transitions."""

from __future__ import annotations

from typing import Any

import torch

from lerobot.utils.constants import ACTION


def _vector(value: Any, *, like: torch.Tensor) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=like.device)
    return tensor.to(device=like.device, dtype=like.dtype).reshape(-1)


def _intervention(transition: dict) -> bool:
    value = (transition.get("complementary_info") or {}).get("is_intervention", False)
    if isinstance(value, torch.Tensor):
        return bool(value.detach().float().max().item() > 0.5)
    return bool(value)


def build_sliding_window_transitions(
    episode: list[dict], horizon: int = 50
) -> list[dict]:
    """Return one padded macro transition for every primitive step in an episode."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not episode:
        return []

    result = []
    episode_length = len(episode)
    first_action = episode[0][ACTION].reshape(-1)
    action_dim = first_action.numel()

    for start in range(episode_length):
        valid_length = min(horizon, episode_length - start)
        action_chunk = first_action.new_zeros((horizon, action_dim))
        rewards = first_action.new_zeros(horizon)
        interventions = torch.zeros(horizon, dtype=torch.bool, device=first_action.device)
        mask = first_action.new_zeros(horizon)

        for offset in range(valid_length):
            primitive = episode[start + offset]
            action_chunk[offset] = primitive[ACTION].reshape(-1)
            rewards[offset] = _vector(primitive["reward"], like=first_action)[0]
            interventions[offset] = _intervention(primitive)
            mask[offset] = 1

        end = start + valid_length - 1
        next_start = start + horizon
        next_mask = first_action.new_zeros(horizon)
        if next_start < episode_length:
            next_mask[: min(horizon, episode_length - next_start)] = 1

        result.append(
            {
                "state": episode[start]["state"],
                ACTION: episode[start][ACTION],
                "target_action_chunk": action_chunk,
                "reward": rewards,
                "intervene_flags": interventions,
                "valid_action_mask": mask,
                "next_valid_action_mask": next_mask,
                "next_state": episode[end]["next_state"],
                "done": bool(episode[end]["done"]),
                "truncated": bool(episode[end]["truncated"]),
                "complementary_info": episode[start].get("complementary_info"),
            }
        )

    return result


def split_episodes(transitions: list[dict]) -> list[list[dict]]:
    episodes, current = [], []
    for transition in transitions:
        current.append(transition)
        if bool(transition["done"]) or bool(transition["truncated"]):
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes


def sliding_windows(transitions: list[dict], horizon: int = 50) -> list[dict]:
    return [
        macro
        for episode in split_episodes(transitions)
        for macro in build_sliding_window_transitions(episode, horizon)
    ]
