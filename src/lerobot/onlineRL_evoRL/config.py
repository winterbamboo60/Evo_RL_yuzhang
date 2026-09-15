"""Configuration owned by the EvoRL actor compatibility layer.

These options used to live in ``lerobot.configs.train``.  Keeping them local
prevents EvoRL collection concerns from changing the canonical LeRobot training
configuration and checkpoint format.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rl.train_rl import TrainRLServerPipelineConfig


@dataclass
class ActorVLAPolicyConfig:
    enabled: bool = False
    policy_path: str | None = None
    policy_poll_s: float = 5.0
    reload_on_episode_boundary: bool = True


@dataclass
class ActorOnlyConfig:
    enabled: bool = False
    episode_output_dir: str | None = None
    save_format: str = "lerobot"
    save_episode_images: bool = True
    save_episode_viewer: bool = True

    def __post_init__(self) -> None:
        if self.save_format not in {"transition", "lerobot"}:
            raise ValueError(
                "actor_only.save_format must be 'transition' or 'lerobot', "
                f"got {self.save_format!r}"
            )


@dataclass
class OnlineTransitionConfig:
    enabled: bool = False
    save_local_copy: bool = True
    episode_output_dir: str | None = None
    feature_batch_size: int = 8
    sliding_window_stride: int = 1

    def __post_init__(self) -> None:
        if self.feature_batch_size <= 0:
            raise ValueError("online_transition.feature_batch_size must be positive")
        if self.sliding_window_stride <= 0:
            raise ValueError("online_transition.sliding_window_stride must be positive")


@dataclass
class CANLeaderControlConfig:
    """Whether the configured leader arm(s) participate in online collection."""

    control: bool = False


@dataclass(kw_only=True)
class ActorPipelineConfig(TrainRLServerPipelineConfig):
    """Online actor configuration layered on canonical LeRobot RL config."""

    actor_mode: str = "online_actor"
    save_format: str = "transition"
    actor_checkpoint_path: str | None = None
    task_hotkeys_path: str | None = None
    actor_vla_policy: ActorVLAPolicyConfig = field(default_factory=ActorVLAPolicyConfig)
    actor_only: ActorOnlyConfig = field(default_factory=ActorOnlyConfig)
    online_transition: OnlineTransitionConfig = field(default_factory=OnlineTransitionConfig)
    can0: CANLeaderControlConfig = field(default_factory=CANLeaderControlConfig)
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(enabled=False, mode="guided", execution_horizon=25)
    )
    rtc_action_queue_threshold: int = 32

    def validate(self) -> None:
        super().validate()
        if self.actor_mode not in {"vla_only", "online_actor"}:
            raise ValueError("actor_mode must be vla_only or online_actor")
        if self.save_format not in {"lerobot", "transition"}:
            raise ValueError("save_format must be lerobot or transition")
        if self.actor_mode == "online_actor" and self.save_format != "transition":
            raise ValueError("online_actor mode only supports save_format=transition")
        if self.actor_only.enabled and self.online_transition.enabled:
            raise ValueError("actor_only.enabled and online_transition.enabled are mutually exclusive")
        needs_compact = self.online_transition.enabled or (
            self.actor_only.enabled and self.actor_only.save_format == "transition"
        )
        if needs_compact and not self.actor_vla_policy.enabled:
            raise ValueError("compact transition mode requires actor_vla_policy.enabled=true")
        if needs_compact and not self.actor_vla_policy.policy_path:
            raise ValueError("compact transition mode requires actor_vla_policy.policy_path")
        if self.rtc.enabled and (self.env is None or not self.env.fps or self.env.fps <= 0):
            raise ValueError("rtc.enabled=true requires env.fps > 0")
        if self.rtc.enabled and self.rtc.execution_horizon <= 0:
            raise ValueError("rtc.execution_horizon must be > 0")
        if self.rtc_action_queue_threshold < 0:
            raise ValueError("rtc_action_queue_threshold must be >= 0")
