"""Stable messages exchanged between execution actor and training learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ActionSource = Literal["vla", "actor", "human"]


@dataclass
class Transition:
    z_rl: torch.Tensor
    proprio: torch.Tensor
    ref_chunk: torch.Tensor
    action: torch.Tensor
    next_z_rl: torch.Tensor
    next_proprio: torch.Tensor
    next_ref_chunk: torch.Tensor
    reward: float
    terminal: bool
    truncated: bool
    action_source: ActionSource
    intervention: bool

    def cpu(self) -> "Transition":
        values = {}
        for name, value in self.__dict__.items():
            values[name] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
        return Transition(**values)


@dataclass
class Episode:
    episode_id: str
    transitions: list[Transition]
    outcome: Literal["success", "failure", "timeout"]
    actor_version: int


@dataclass
class ActorWeights:
    version: int
    state_dict: dict[str, torch.Tensor]
