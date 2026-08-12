"""Small RLT Stage2 actor and twin-Q model with the RLinf input contract."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .config import ModelConfig


class RLTActorCritic(nn.Module):
    """Actor and twin critic over ``z_rl``, proprio and action chunks."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.chunk_dim = cfg.num_action_chunks * cfg.action_dim
        self.ref_dim = cfg.ref_num_action_chunks * cfg.action_dim
        actor_in = cfg.z_dim + cfg.proprio_dim + self.ref_dim
        critic_in = cfg.z_dim + cfg.proprio_dim + self.ref_dim + self.chunk_dim
        self.actor = self._mlp(actor_in, self.chunk_dim, cfg.hidden_dim)
        self.q1 = self._mlp(critic_in, 1, cfg.hidden_dim)
        self.q2 = self._mlp(critic_in, 1, cfg.hidden_dim)

    @staticmethod
    def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def _actor_input(self, z_rl: torch.Tensor, proprio: torch.Tensor, ref_chunk: torch.Tensor) -> torch.Tensor:
        return torch.cat((z_rl, proprio, ref_chunk.flatten(start_dim=1)), dim=-1)

    def act(self, z_rl: torch.Tensor, proprio: torch.Tensor, ref_chunk: torch.Tensor) -> torch.Tensor:
        return self.actor(self._actor_input(z_rl, proprio, ref_chunk)).view(-1, self.cfg.num_action_chunks, self.cfg.action_dim)

    def q_values(self, z_rl: torch.Tensor, proprio: torch.Tensor, ref_chunk: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((self._actor_input(z_rl, proprio, ref_chunk), action.flatten(start_dim=1)), dim=-1)
        return self.q1(features), self.q2(features)

    def actor_state_dict(self) -> dict[str, torch.Tensor]:
        return copy.deepcopy(self.actor.state_dict())

    def load_actor_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state_dict)


def soft_update(target: RLTActorCritic, source: RLTActorCritic, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)
