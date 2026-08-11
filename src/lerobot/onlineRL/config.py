"""Configuration for the standalone Piper online RLT runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    z_dim: int = 2048
    proprio_dim: int = 7
    action_dim: int = 7
    ref_num_action_chunks: int = 50
    num_action_chunks: int = 50
    hidden_dim: int = 1024


@dataclass
class ReplayConfig:
    max_cached_episodes: int = 200
    sample_window_episodes: int = 200
    min_buffer_episodes: int = 2
    train_actor_episodes: int = 2
    batch_size: int = 32
    update_epoch: int = 8
    critic_actor_ratio: int = 4
    gamma: float = 0.96
    tau: float = 0.005
    q_weight: float = 0.1
    bc_weight: float = 5.0
    reference_dropout_prob: float = 0.5
    actor_sync_interval_updates: int = 1


@dataclass
class CameraConfig:
    """One OpenCV camera used by the Stage1 visual policy."""

    index_or_path: int | str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class HardwareConfig:
    follower_port: str = "can1"
    leader_port: str = "can0"
    follower_id: str = "my_piper_follower"
    leader_id: str = "my_piper_leader"
    follower_speed_ratio: int = 50
    leader_speed_ratio: int = 50
    reset_duration_s: float = 3.0
    control_hz: float = 10.0
    action_limit: float = 180.0
    top_camera: CameraConfig = field(default_factory=CameraConfig)
    wrist_camera: CameraConfig = field(default_factory=CameraConfig)


@dataclass
class RuntimeConfig:
    mode: str = "local"  # local | actor | learner
    actor_device: str = "cuda:0"
    learner_device: str = "cuda:1"
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    dry_run: bool = True
    max_episode_steps: int = 1800
    outbox_dir: str = "./online_rl_outbox"
    checkpoint_dir: str = "./online_rl_checkpoints"
    feature_extractor_factory: str = ""
    feature_extractor_kwargs: dict[str, Any] = field(default_factory=dict)
    display_data: bool = False
    debug_every_steps: int = 10


@dataclass
class OnlineRLConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if self.model.action_dim != 7:
            raise ValueError("Piper onlineRL currently requires action_dim=7.")
        if self.model.ref_num_action_chunks < self.model.num_action_chunks:
            raise ValueError("ref_num_action_chunks must be >= num_action_chunks.")
        if self.replay.max_cached_episodes < self.replay.min_buffer_episodes:
            raise ValueError("max_cached_episodes must be >= min_buffer_episodes.")
        if self.replay.critic_actor_ratio < 1 or self.replay.update_epoch < 1:
            raise ValueError("critic_actor_ratio and update_epoch must be >= 1.")
        if self.runtime.max_episode_steps < 1 or self.hardware.control_hz <= 0:
            raise ValueError("max_episode_steps and control_hz must be positive.")
        if self.runtime.debug_every_steps < 1:
            raise ValueError("runtime.debug_every_steps must be positive.")
        for name, camera in (("top_camera", self.hardware.top_camera), ("wrist_camera", self.hardware.wrist_camera)):
            if camera.width < 1 or camera.height < 1 or camera.fps < 1:
                raise ValueError(f"hardware.{name} width, height and fps must be positive.")
            if not self.runtime.dry_run and camera.index_or_path is None:
                raise ValueError(f"hardware.{name}.index_or_path is required when dry_run=false.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    allowed = {item.name for item in fields(instance)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields for {type(instance).__name__}: {sorted(unknown)}")
    for name, value in values.items():
        setattr(instance, name, value)


def load_config(path: str | Path | None) -> OnlineRLConfig:
    cfg = OnlineRLConfig()
    if path is None:
        cfg.validate()
        return cfg
    with Path(path).open() as handle:
        payload = json.load(handle)
    for section in ("model", "replay", "hardware", "runtime"):
        if section in payload:
            values = dict(payload[section])
            if section == "hardware":
                for camera_name in ("top_camera", "wrist_camera"):
                    if camera_name in values:
                        _update_dataclass(getattr(cfg.hardware, camera_name), values.pop(camera_name))
            _update_dataclass(getattr(cfg, section), values)
    cfg.validate()
    return cfg
