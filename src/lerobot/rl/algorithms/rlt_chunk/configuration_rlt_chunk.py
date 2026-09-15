"""Configuration for the PI05-RLT fixed-horizon actor/critic algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (
    ActorLearnerConfig,
    ConcurrencyConfig,
)
from lerobot.rl.algorithms.configs import RLAlgorithmConfig


@RLAlgorithmConfig.register_subclass("rlt_chunk")
@dataclass
class RLTChunkAlgorithmConfig(RLAlgorithmConfig):
    actor_hidden_dims: tuple[int, ...] = (256, 256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256, 256)
    num_critics: int = 2
    fixed_std: float = 0.002
    reference_dropout_prob: float = 0.5
    q_weight: float = 0.1
    bc_weight: float = 5.0
    discount: float = 0.96
    critic_target_update_weight: float = 0.005
    actor_update_interval: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    grad_clip_norm: float = 10.0
    online_steps: int = 1_000_000
    online_step_before_learning: int = 100
    online_buffer_capacity: int = 100_000
    offline_buffer_capacity: int = 100_000
    storage_device: str = "cpu"
    actor_rollout_deterministic: bool = False
    state_keys: tuple[str, ...] = ("z_rl", "proprio", "ref_action")
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    policy_config: PreTrainedConfig | None = None

    def __post_init__(self) -> None:
        if self.num_critics < 2:
            raise ValueError("num_critics must be at least 2")
        if self.fixed_std <= 0:
            raise ValueError("fixed_std must be positive")
        if not 0.0 <= self.reference_dropout_prob <= 1.0:
            raise ValueError("reference_dropout_prob must be in [0, 1]")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if not 0.0 < self.critic_target_update_weight <= 1.0:
            raise ValueError("critic_target_update_weight must be in (0, 1]")
        if self.actor_update_interval <= 0:
            raise ValueError("actor_update_interval must be positive")
        if min(self.online_steps, self.online_step_before_learning, self.online_buffer_capacity) <= 0:
            raise ValueError("online runtime sizes must be positive")

    @classmethod
    def from_policy_config(cls, policy_cfg: Any) -> RLTChunkAlgorithmConfig:
        return cls(policy_config=policy_cfg)
