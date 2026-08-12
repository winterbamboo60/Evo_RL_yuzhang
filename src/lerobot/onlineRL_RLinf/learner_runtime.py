"""Episode-based RLT learner. It owns all critic and optimizer state."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .config import OnlineRLConfig
from .protocol import ActorWeights, Episode, Transition
from .replay import EpisodeReplayBuffer
from .rlt_model import RLTActorCritic, soft_update


class Learner:
    def __init__(self, cfg: OnlineRLConfig):
        self.cfg = cfg
        self.device = torch.device("cpu" if cfg.runtime.dry_run or not torch.cuda.is_available() else cfg.runtime.learner_device)
        self.model = RLTActorCritic(cfg.model).to(self.device)
        self.target = copy.deepcopy(self.model).to(self.device).eval()
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(list(self.model.q1.parameters()) + list(self.model.q2.parameters()), lr=3e-4)
        self.replay = EpisodeReplayBuffer(cfg.replay.max_cached_episodes, cfg.replay.sample_window_episodes)
        self.update_step = 0
        self.actor_version = 0

    def ingest(self, episode: Episode) -> ActorWeights | None:
        self.replay.add(episode)
        if len(self.replay) < self.cfg.replay.min_buffer_episodes:
            return None
        for _ in range(self.cfg.replay.update_epoch):
            self._update_once(update_actor=len(self.replay) >= self.cfg.replay.train_actor_episodes)
        return ActorWeights(version=self.actor_version, state_dict={key: value.detach().cpu() for key, value in self.model.actor_state_dict().items()})

    def _batch(self) -> list[Transition]:
        return self.replay.sample(self.cfg.replay.batch_size)

    def _stack(self, transitions: list[Transition], name: str) -> torch.Tensor:
        return torch.cat([getattr(item, name).to(self.device) for item in transitions], dim=0)

    def _update_once(self, update_actor: bool) -> None:
        batch = self._batch()
        z, proprio, ref, action = (self._stack(batch, name) for name in ("z_rl", "proprio", "ref_chunk", "action"))
        next_z, next_proprio, next_ref = (self._stack(batch, name) for name in ("next_z_rl", "next_proprio", "next_ref_chunk"))
        reward = torch.tensor([[item.reward] for item in batch], dtype=torch.float32, device=self.device)
        mask = torch.tensor([[0.0 if item.terminal else 1.0] for item in batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_action = self.model.act(next_z, next_proprio, next_ref)
            tq1, tq2 = self.target.q_values(next_z, next_proprio, next_ref, next_action)
            target = reward + mask * (self.cfg.replay.gamma ** self.cfg.model.num_action_chunks) * torch.minimum(tq1, tq2)
        q1, q2 = self.model.q_values(z, proprio, ref, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        if update_actor and self.update_step % self.cfg.replay.critic_actor_ratio == 0:
            pi = self.model.act(z, proprio, ref)
            q_pi, _ = self.model.q_values(z, proprio, ref, pi)
            targets = ref[:, : self.cfg.model.num_action_chunks]
            human_mask = torch.tensor([item.intervention for item in batch], device=self.device).view(-1, 1, 1)
            bc_target = torch.where(human_mask, action, targets)
            actor_loss = -self.cfg.replay.q_weight * q_pi.mean() + self.cfg.replay.bc_weight * F.mse_loss(pi, bc_target)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_version += 1
        soft_update(self.target, self.model, self.cfg.replay.tau)
        self.update_step += 1
