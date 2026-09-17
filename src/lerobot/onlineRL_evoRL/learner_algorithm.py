"""Head-only PI05-RLT learner used by the EvoRL online runtime."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

import torch

from lerobot.rl.algorithms.rlt_chunk.rlt_chunk_algorithm import (
    RLTChunkActor,
    RLTChunkAlgorithm,
    RLTChunkCriticEnsemble,
)
from lerobot.utils.constants import ACTION, OBS_STATE


class HeadOnlyRLTChunkAlgorithm(RLTChunkAlgorithm):
    """Train RLT heads from compact replay without constructing the 5B policy."""

    def __init__(self, policy_config: Any, config: Any, *, device: str | torch.device = "cpu") -> None:
        """Build only actor/critic heads from PI0.5 configuration metadata."""
        self.policy = None
        self.config = config
        self.policy_config = policy_config
        self._device = torch.device(device)
        self._optimization_step = 0
        self.optimizers = {}
        self._action_queue = deque()

        action_feature = policy_config.output_features.get(ACTION)
        state_feature = policy_config.input_features.get(OBS_STATE)
        if action_feature is None or state_feature is None:
            raise ValueError("head-only PI05-RLT requires action and observation.state features")
        action_dim = action_feature.shape[0]
        proprio_dim = state_feature.shape[0]
        z_dim = policy_config.rlt_embed_dim * policy_config.rlt_num_rl_tokens
        horizon = policy_config.chunk_size
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

    def reset(self) -> None:
        """Clear only the small actor action queue; no VLA exists on the learner."""
        self._action_queue.clear()

    def to_device(self, device: str | torch.device) -> None:
        """Move all learner-owned modules and optimizer state to one device."""
        target = torch.device(device)
        self.actor.to(target)
        self.critic_ensemble.to(target)
        self.critic_target.to(target)
        for optimizer in self.optimizers.values():
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(target)
        self._device = target

    def prepare_observation(self, *_args, **_kwargs):
        """Reject raw observation preprocessing on the compact learner."""
        raise RuntimeError("head-only learner cannot preprocess raw observations")

    def select_action(self, *_args, **_kwargs):
        """Reject PI0.5 inference on the compact learner."""
        raise RuntimeError("head-only learner cannot run PI0.5 inference")

    def serialize_episode_for_transport(self, *_args, **_kwargs):
        """Reject actor-side serialization on the compact learner."""
        raise RuntimeError("head-only learner accepts precomputed compact episodes only")
