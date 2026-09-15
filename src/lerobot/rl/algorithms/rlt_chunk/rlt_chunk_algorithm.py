"""Fixed-horizon actor/critic trained on frozen PI05-RLT features."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from torch.optim import Optimizer

from lerobot.lerobot_types import BatchType
from lerobot.onlineRL_evoRL.chunk_transition import (
    build_sliding_window_transitions,
)
from lerobot.onlineRL_evoRL.compact_transition import (
    SCHEMA_NAME,
    SLIDING_WINDOW_TRANSITIONS,
    compact_episode_to_bytes,
    is_compact_episode,
    make_compact_episode,
    validate_compact_episode,
)
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.rl.algorithms.base import RLAlgorithm
from lerobot.rl.algorithms.configs import TrainingStats
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.transition import move_state_dict_to_device

from .configuration_rlt_chunk import RLTChunkAlgorithmConfig

RLT_ACTOR_ARCHITECTURE = "rlinf_tanh_v1"
RLT_ACTOR_INPUT_ORDER = ("ref_action", "z_rl", "proprio")
RLT_ACTOR_NOISE_MODE = "pre_tanh_fixed_gaussian"


class RLTChunkActor(nn.Module):
    def __init__(self, z_dim, proprio_dim, horizon, action_dim, hidden_dims, fixed_std, dropout):
        super().__init__()
        dims = [z_dim + proprio_dim + horizon * action_dim, *hidden_dims, horizon * action_dim]
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(dims, dims[1:], strict=False)):
            linear = nn.Linear(left, right)
            gain = 0.01 * math.sqrt(2) if index == len(dims) - 2 else math.sqrt(2)
            nn.init.orthogonal_(linear.weight, gain)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index != len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.horizon = horizon
        self.action_dim = action_dim
        self.fixed_std = fixed_std
        self.dropout = dropout

    def raw_mean(self, z_rl, proprio, ref_action, *, apply_reference_dropout=False):
        if apply_reference_dropout and self.dropout:
            keep = torch.rand((ref_action.shape[0], 1, 1), device=ref_action.device) >= self.dropout
            ref_action = ref_action * keep
        flat = torch.cat((ref_action.flatten(1), z_rl.flatten(1), proprio.flatten(1)), dim=-1)
        return self.net(flat).view(-1, self.horizon, self.action_dim)

    def mean(self, z_rl, proprio, ref_action, *, apply_reference_dropout=False):
        raw_mean = self.raw_mean(
            z_rl,
            proprio,
            ref_action,
            apply_reference_dropout=apply_reference_dropout,
        )
        return torch.tanh(raw_mean)

    def forward(self, z_rl, proprio, ref_action, deterministic=False, *, apply_reference_dropout=False):
        raw_mean = self.raw_mean(z_rl, proprio, ref_action, apply_reference_dropout=apply_reference_dropout)
        if not deterministic:
            raw_mean = raw_mean + torch.randn_like(raw_mean) * self.fixed_std
        return torch.tanh(raw_mean)


class RLTChunkCriticEnsemble(nn.Module):
    def __init__(self, z_dim, proprio_dim, horizon, action_dim, hidden_dims, count):
        super().__init__()
        in_dim = z_dim + proprio_dim + horizon * action_dim

        def make_net():
            dims = [in_dim, *hidden_dims, 1]
            layers: list[nn.Module] = []
            for index, (left, right) in enumerate(zip(dims, dims[1:], strict=False)):
                linear = nn.Linear(left, right)
                if index == len(dims) - 2:
                    nn.init.normal_(linear.weight, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(linear.bias)
                layers.append(linear)
                if index != len(dims) - 2:
                    layers.extend((nn.LayerNorm(right), nn.Tanh()))
            return nn.Sequential(*layers)

        self.critics = nn.ModuleList(make_net() for _ in range(count))

    def forward(self, z_rl, proprio, action_chunk, valid_action_mask):
        masked_action = action_chunk * valid_action_mask.unsqueeze(-1)
        x = torch.cat((z_rl.flatten(1), proprio.flatten(1), masked_action.flatten(1)), dim=-1)
        return torch.stack([critic(x).squeeze(-1) for critic in self.critics])


def compute_chunk_td_target(reward, valid_action_mask, next_q, done, truncated, discount):
    horizon = reward.shape[-1]
    powers = discount ** torch.arange(horizon, device=reward.device, dtype=reward.dtype)
    chunk_return = (reward * valid_action_mask * powers).sum(dim=-1)
    bootstrap = valid_action_mask.sum(dim=-1).eq(horizon) & ~done.bool() & ~truncated.bool()
    return chunk_return + bootstrap * (discount**horizon) * next_q


def _split_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}


class RLTChunkAlgorithm(RLAlgorithm):
    config_class = RLTChunkAlgorithmConfig
    name = "rlt_chunk"
    uses_preprocessed_replay = True

    def __init__(self, policy: PI05RLTPolicy, config: RLTChunkAlgorithmConfig):
        if not isinstance(policy, PI05RLTPolicy):
            raise TypeError(f"rlt_chunk requires PI05RLTPolicy, got {type(policy).__name__}")
        self.policy = policy
        self.config = config
        self.policy_config = policy.config
        self._device = torch.device(policy.config.device)
        self._optimization_step = 0
        self.optimizers: dict[str, Optimizer] = {}
        self._action_queue: deque[torch.Tensor] = deque()

        action_dim = policy.config.output_features[ACTION].shape[0]
        state_feature = policy.config.input_features.get(OBS_STATE)
        if state_feature is None:
            raise ValueError("pi05_rlt requires observation.state")
        proprio_dim = state_feature.shape[0]
        z_dim = policy.config.rlt_embed_dim * policy.config.rlt_num_rl_tokens
        horizon = policy.config.chunk_size
        self.actor = RLTChunkActor(
            z_dim,
            proprio_dim,
            horizon,
            action_dim,
            config.actor_hidden_dims,
            config.fixed_std,
            config.reference_dropout_prob,
        ).to(self._device)
        self.critic_ensemble = RLTChunkCriticEnsemble(
            z_dim,
            proprio_dim,
            horizon,
            action_dim,
            config.critic_hidden_dims,
            config.num_critics,
        ).to(self._device)
        self.critic_target = deepcopy(self.critic_ensemble).requires_grad_(False).to(self._device)
        self.policy.requires_grad_(False).eval()

    def make_optimizers_and_scheduler(self) -> dict[str, Optimizer]:
        self.optimizers = {
            "actor": torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critic_ensemble.parameters(), lr=self.config.critic_lr),
        }
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self._action_queue:
            features = self.policy.extract_rlt_features(batch)
            chunk = self.actor(
                features["z_rl"],
                features["proprio"],
                features["ref_action"],
                deterministic=self.config.actor_rollout_deterministic,
                apply_reference_dropout=False,
            )
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def reset(self) -> None:
        self._action_queue.clear()
        self.policy.reset()

    def prepare_observation(self, preprocessor, observation: dict[str, Any], *, task: str):
        """Build the native PI0.5 batch, including task tokenization."""
        return preprocessor({**observation, "task": task})

    def _batch(self, batch: BatchType) -> dict[str, Any]:
        complementary = batch.get("complementary_info") or {}
        required = (
            "target_action_chunk",
            "reward_chunk",
            "intervene_flags",
            "valid_action_mask",
            "next_valid_action_mask",
        )
        missing = [key for key in required if key not in complementary]
        if missing:
            raise ValueError(f"rlt_chunk replay batch is missing compact fields: {missing}")
        result = {
            key: value.to(self._device) for key, value in batch.items() if isinstance(value, torch.Tensor)
        }
        result["state"] = {key: value.to(self._device) for key, value in batch["state"].items()}
        result["next_state"] = {key: value.to(self._device) for key, value in batch["next_state"].items()}
        result.update({key: complementary[key].to(self._device) for key in required})
        return result

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        if not self.optimizers:
            raise RuntimeError("make_optimizers_and_scheduler() must be called before update()")
        batch = self._batch(next(batch_iterator))
        mask = batch["valid_action_mask"].float()
        next_mask = batch["next_valid_action_mask"].float()
        action = batch["target_action_chunk"]
        state, next_state = batch["state"], batch["next_state"]

        q_values = self.critic_ensemble(state["z_rl"], state["proprio"], action, mask)
        with torch.no_grad():
            next_action = self.actor(next_state["z_rl"], next_state["proprio"], next_state["ref_action"])
            next_q = (
                self.critic_target(next_state["z_rl"], next_state["proprio"], next_action, next_mask)
                .min(dim=0)
                .values
            )
            target_q = compute_chunk_td_target(
                batch["reward_chunk"], mask, next_q, batch["done"], batch["truncated"], self.config.discount
            )
        critic_loss = functional.mse_loss(q_values, target_q.unsqueeze(0).expand_as(q_values))
        self.optimizers["critic"].zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.critic_ensemble.parameters(), self.config.grad_clip_norm
        ).item()
        self.optimizers["critic"].step()

        losses = {"loss_critic": critic_loss.item()}
        grad_norms = {"critic": critic_grad}
        if self._optimization_step % self.config.actor_update_interval == 0:
            for parameter in self.critic_ensemble.parameters():
                parameter.requires_grad_(False)
            predicted = self.actor(
                state["z_rl"],
                state["proprio"],
                state["ref_action"],
                apply_reference_dropout=True,
            )
            actor_q = self.critic_ensemble(state["z_rl"], state["proprio"], predicted, mask)[0].mean()
            intervention = batch["intervene_flags"].bool().unsqueeze(-1)
            bc_target = torch.where(intervention, action, state["ref_action"])
            bc = (((predicted - bc_target) ** 2) * mask.unsqueeze(-1)).sum()
            bc = bc / (mask.sum() * predicted.shape[-1]).clamp_min(1)
            actor_loss = -self.config.q_weight * actor_q + self.config.bc_weight * bc
            self.optimizers["actor"].zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.grad_clip_norm
            ).item()
            self.optimizers["actor"].step()
            for parameter in self.critic_ensemble.parameters():
                parameter.requires_grad_(True)
            losses.update(loss_actor=actor_loss.item(), bc_loss=bc.item())
            grad_norms["actor"] = actor_grad

        with torch.no_grad():
            tau = self.config.critic_target_update_weight
            for target, source in zip(
                self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True
            ):
                target.mul_(1 - tau).add_(source, alpha=tau)
        self._optimization_step += 1
        return TrainingStats(
            losses=losses,
            grad_norms=grad_norms,
            extra={"q": q_values.mean().item(), "target_q": target_q.mean().item()},
        )

    def get_weights(self) -> dict[str, Any]:
        return {"policy": move_state_dict_to_device(self.actor.state_dict(), device="cpu")}

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        actor_state = weights.get("policy", weights.get("actor"))
        if not isinstance(actor_state, dict):
            raise ValueError("rlt_chunk weight bundle is missing 'policy' actor state")
        self.actor.load_state_dict(move_state_dict_to_device(actor_state, device=device), strict=True)
        self.reset()

    def state_dict(self) -> dict[str, torch.Tensor]:
        result = {}
        for prefix, module in (
            ("actor.", self.actor),
            ("critic_ensemble.", self.critic_ensemble),
            ("critic_target.", self.critic_target),
        ):
            result.update({prefix + key: value for key, value in module.state_dict().items()})
        return result

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], device: str | torch.device = "cpu"
    ) -> None:
        del device
        self.actor.load_state_dict(_split_prefix(state_dict, "actor."), strict=True)
        self.critic_ensemble.load_state_dict(_split_prefix(state_dict, "critic_ensemble."), strict=True)
        self.critic_target.load_state_dict(_split_prefix(state_dict, "critic_target."), strict=True)
        self.reset()

    def load_legacy_actor_critic(self, checkpoint: str | Path) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.actor.load_state_dict(state["actor"], strict=True)
        critic_state = state.get("critic", state.get("critic_ensemble"))
        if not isinstance(critic_state, dict):
            raise ValueError(f"Legacy checkpoint {checkpoint} is missing critic weights")
        self.critic_ensemble.load_state_dict(critic_state, strict=True)
        self.critic_target.load_state_dict(state.get("critic_target", critic_state), strict=True)
        self.reset()

    def configure_data_iterator(self, data_mixer, batch_size, *, async_prefetch=True, queue_size=2):
        if getattr(data_mixer, "offline_buffer", None) is not None:
            raise ValueError(
                "rlt_chunk currently requires precomputed compact online replay; set dataset=null"
            )
        return data_mixer.get_iterator(batch_size, async_prefetch=async_prefetch, queue_size=queue_size)

    def ingest_transition_payload(self, payload: Any, replay_buffer) -> bool:
        if not is_compact_episode(payload):
            return False
        validate_compact_episode(payload)
        for transition in payload["transitions"]:
            complementary = dict(transition.get("complementary_info") or {})
            complementary.update(
                target_action_chunk=transition["target_action_chunk"],
                reward_chunk=transition["reward"],
                intervene_flags=transition["intervene_flags"],
                valid_action_mask=transition["valid_action_mask"],
                next_valid_action_mask=transition["next_valid_action_mask"],
            )
            replay_buffer.add(
                state=transition["state"],
                action=transition[ACTION],
                reward=float(torch.as_tensor(transition["reward"]).sum().item()),
                next_state=transition["next_state"],
                done=transition["done"],
                truncated=transition["truncated"],
                complementary_info=complementary,
            )
        return True

    @torch.no_grad()
    def serialize_episode_for_transport(
        self, transitions, preprocessor, *, task: str, policy_path: str | None
    ) -> bytes:
        episode_length = len(transitions)
        horizon = self.policy.config.chunk_size
        observations = [transition["state"] for transition in transitions] + [transitions[-1]["next_state"]]
        actions = [transition[ACTION] for transition in transitions] + [transitions[-1][ACTION]]
        features = []
        normalized_actions = []
        self.policy.reset()
        preprocessor.reset()
        try:
            for index, (observation, action) in enumerate(zip(observations, actions, strict=True)):
                raw = {
                    key: value.squeeze(0)
                    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 1
                    else value
                    for key, value in observation.items()
                }
                raw[ACTION] = action.squeeze(0) if action.ndim > 1 and action.shape[0] == 1 else action
                batch = preprocessor({**raw, "task": task})
                features.append(
                    {
                        key: value.detach().cpu()
                        for key, value in self.policy.extract_rlt_features(batch).items()
                    }
                )
                if index < episode_length:
                    normalized_actions.append(batch[ACTION].detach().cpu())
        finally:
            self.policy.reset()
            preprocessor.reset()
        primitive = []
        for index, transition in enumerate(transitions):
            complementary = transition.get("complementary_info") or {}
            intervention = complementary.get(
                "intervention",
                complementary.get("is_intervention", complementary.get("acp_indicator", False)),
            )
            if isinstance(intervention, torch.Tensor):
                intervention = bool(intervention.detach().float().max().item() > 0.5)
            primitive.append(
                {
                    "state": features[index],
                    "next_state": features[index + 1],
                    ACTION: normalized_actions[index],
                    "reward": transition["reward"],
                    "done": bool(transition["done"]),
                    "truncated": bool(transition["truncated"]),
                    "complementary_info": {"intervention": bool(intervention)},
                }
            )
        compact = make_compact_episode(
            transitions=build_sliding_window_transitions(primitive, horizon=horizon),
            metadata={"task": task, "transition_schema": SCHEMA_NAME},
            feature_model={
                "resolved_path": str(Path(policy_path).expanduser().resolve()) if policy_path else "runtime"
            },
            transition_layout=SLIDING_WINDOW_TRANSITIONS,
            primitive_steps=episode_length,
        )
        return compact_episode_to_bytes(compact)
