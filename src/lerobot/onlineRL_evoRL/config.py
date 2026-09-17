"""Configuration owned by the EvoRL online actor/learner runtime.

The generic distributed-RL schema remains provided by lerobot.rl, while the
supported PI05-RLT online profile and all robot-facing options are enforced
here. Both executable entry points parse one of the classes in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rl.train_rl import TrainRLServerPipelineConfig


@dataclass
class LearnerGPUHandoffConfig:
    """Learner-owned quota and CUDA handoff settings."""

    enabled: bool = False
    update_quota_threshold: int = 200
    updates_per_episode: int = 4
    actor_release_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        """Validate learner quota arithmetic and timeout."""
        if self.update_quota_threshold <= 0:
            raise ValueError("learner gpu_handoff.update_quota_threshold must be positive")
        if self.updates_per_episode <= 0:
            raise ValueError("learner gpu_handoff.updates_per_episode must be positive")
        if self.update_quota_threshold % self.updates_per_episode:
            raise ValueError(
                "learner gpu_handoff.update_quota_threshold must be divisible by updates_per_episode"
            )
        if self.actor_release_timeout_s <= 0:
            raise ValueError("learner gpu_handoff.actor_release_timeout_s must be positive")


@dataclass
class ActorGPUHandoffConfig:
    """Actor-owned episode boundary and learner-wait settings."""

    enabled: bool = False
    update_quota_threshold: int = 200
    updates_per_episode: int = 4
    learner_release_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        """Validate actor quota arithmetic and timeout."""
        if self.update_quota_threshold <= 0:
            raise ValueError("actor gpu_handoff.update_quota_threshold must be positive")
        if self.updates_per_episode <= 0:
            raise ValueError("actor gpu_handoff.updates_per_episode must be positive")
        if self.update_quota_threshold % self.updates_per_episode:
            raise ValueError(
                "actor gpu_handoff.update_quota_threshold must be divisible by updates_per_episode"
            )
        if self.learner_release_timeout_s <= 0:
            raise ValueError("actor gpu_handoff.learner_release_timeout_s must be positive")


@dataclass
class OfflinePretrainingConfig:
    """Optional one-shot RLT Actor/Critic pretraining from compact episodes."""

    enabled: bool = False
    steps: int = 0
    feature_batch_size: int = 8
    sliding_window_stride: int = 1
    compact_dataset_path: str | None = None

    def __post_init__(self) -> None:
        """Validate the expensive feature-extraction and training phase."""
        if self.enabled and self.steps <= 0:
            raise ValueError("offline_pretraining.steps must be positive when enabled")
        if self.feature_batch_size <= 0:
            raise ValueError("offline_pretraining.feature_batch_size must be positive")
        if self.sliding_window_stride <= 0:
            raise ValueError("offline_pretraining.sliding_window_stride must be positive")


def _validate_pi05_rlt_profile(cfg: TrainRLServerPipelineConfig) -> None:
    """Reject combinations that the EvoRL compact-transition path cannot run."""
    if cfg.policy is None or cfg.policy.type != "pi05_rlt":
        policy_type = None if cfg.policy is None else cfg.policy.type
        raise ValueError(f"onlineRL_evoRL requires policy.type=pi05_rlt, got {policy_type!r}")
    if cfg.algorithm is None or cfg.algorithm.type != "rlt_chunk":
        algorithm_type = None if cfg.algorithm is None else cfg.algorithm.type
        raise ValueError(f"onlineRL_evoRL requires algorithm.type=rlt_chunk, got {algorithm_type!r}")
    if cfg.env is None:
        raise ValueError("onlineRL_evoRL requires an env configuration")
    if cfg.online_ratio != 1.0:
        raise ValueError("rlt_chunk uses compact online replay and requires online_ratio=1.0")


@dataclass(kw_only=True)
class OnlineRLPipelineConfig(TrainRLServerPipelineConfig):
    """Shared schema for this package's PI05-RLT actor and learner."""

    def validate(self) -> None:
        """Validate the shared schema and the supported online profile."""
        super().validate()
        _validate_pi05_rlt_profile(self)


@dataclass(kw_only=True)
class LearnerPipelineConfig(OnlineRLPipelineConfig):
    """Learner with optional one-shot offline initialization and online replay."""

    gpu_handoff: LearnerGPUHandoffConfig = field(default_factory=LearnerGPUHandoffConfig)
    offline_pretraining: OfflinePretrainingConfig = field(default_factory=OfflinePretrainingConfig)

    def validate(self) -> None:
        """Keep raw datasets outside learner and require compact data when enabled."""
        super().validate()
        if self.dataset is not None:
            raise ValueError(
                "learner dataset must be null; use the standalone extractor and compact_dataset_path"
            )
        if (
            self.offline_pretraining.enabled
            and not self.offline_pretraining.compact_dataset_path
        ):
            raise ValueError(
                "offline_pretraining.enabled=true requires compact_dataset_path"
            )


@dataclass
class ActorVLAPolicyConfig:
    """VLA checkpoint reload options used by the online actor."""

    enabled: bool = False
    policy_path: str | None = None
    policy_poll_s: float = 5.0
    reload_on_episode_boundary: bool = True


@dataclass
class ActorOnlyConfig:
    """Legacy actor-only collection options kept for config compatibility."""

    enabled: bool = False
    episode_output_dir: str | None = None
    save_format: str = "lerobot"
    save_episode_images: bool = True
    save_episode_viewer: bool = True

    def __post_init__(self) -> None:
        """Validate the selected actor-only serialization format."""
        if self.save_format not in {"transition", "lerobot"}:
            raise ValueError(
                f"actor_only.save_format must be 'transition' or 'lerobot', got {self.save_format!r}"
            )


@dataclass
class OnlineTransitionConfig:
    """Compact transport persistence and optional LeRobot dataset mirroring."""

    enabled: bool = False
    # Kept for config compatibility; online actors require this to remain true.
    save_local_copy: bool = True
    episode_output_dir: str | None = None
    save_lerobot_copy: bool = False
    lerobot_output_dir: str | None = None
    feature_batch_size: int = 8
    sliding_window_stride: int = 1

    def __post_init__(self) -> None:
        """Validate compact feature batching and window stride."""
        if self.feature_batch_size <= 0:
            raise ValueError("online_transition.feature_batch_size must be positive")
        if self.sliding_window_stride <= 0:
            raise ValueError("online_transition.sliding_window_stride must be positive")


@dataclass
class CANLeaderControlConfig:
    """Whether the configured leader arm(s) participate in online collection."""

    control: bool = False


@dataclass(kw_only=True)
class ActorPipelineConfig(OnlineRLPipelineConfig):
    """Robot-facing online actor configuration."""

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
    gpu_handoff: ActorGPUHandoffConfig = field(default_factory=ActorGPUHandoffConfig)

    def validate(self) -> None:
        """Validate robot-facing online actor settings."""
        super().validate()
        if self.actor_mode not in {"vla_only", "online_actor"}:
            raise ValueError("actor_mode must be vla_only or online_actor")
        if self.save_format not in {"lerobot", "transition"}:
            raise ValueError("save_format must be lerobot or transition")
        if self.actor_mode == "online_actor" and self.save_format != "transition":
            raise ValueError("online_actor mode only supports save_format=transition")
        if self.actor_mode == "online_actor" and not self.online_transition.save_local_copy:
            raise ValueError(
                "online_actor requires online_transition.save_local_copy=true so every sent "
                "compact transition also has a local copy"
            )
        if self.actor_only.enabled and self.online_transition.enabled:
            raise ValueError("actor_only.enabled and online_transition.enabled are mutually exclusive")
        needs_compact = self.online_transition.enabled or (
            self.actor_only.enabled and self.actor_only.save_format == "transition"
        )
        if needs_compact and not self.actor_vla_policy.enabled:
            raise ValueError("compact transition mode requires actor_vla_policy.enabled=true")
        if needs_compact and not self.actor_vla_policy.policy_path:
            raise ValueError("compact transition mode requires actor_vla_policy.policy_path")
        if self.online_transition.save_lerobot_copy:
            compact_dir = self.online_transition.episode_output_dir
            lerobot_dir = self.online_transition.lerobot_output_dir
            if compact_dir and lerobot_dir and compact_dir == lerobot_dir:
                raise ValueError("online_transition.episode_output_dir and lerobot_output_dir must differ")
        if self.rtc.enabled and (self.env is None or not self.env.fps or self.env.fps <= 0):
            raise ValueError("rtc.enabled=true requires env.fps > 0")
        if self.rtc.enabled and self.rtc.execution_horizon <= 0:
            raise ValueError("rtc.execution_horizon must be > 0")
        if self.rtc_action_queue_threshold < 0:
            raise ValueError("rtc_action_queue_threshold must be >= 0")
