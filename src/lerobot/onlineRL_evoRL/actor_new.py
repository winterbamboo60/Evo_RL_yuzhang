"""EvoRL online actor on top of the current LeRobot runtime.

Hardware control, human intervention, RTC inference and episode assembly live
in this module. Learner communication reuses the current
``lerobot.rl.actor`` streaming helpers so the gRPC byte protocol remains
unchanged.
"""

# Canonical actor helpers below are intentional compatibility re-exports.
# ruff: noqa: F401

import contextlib
import gc
import inspect
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue as ThreadQueue
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from torch.multiprocessing import Queue

from lerobot.common.control_utils import predict_action
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import make_default_processors
from lerobot.rl.actor import (
    establish_learner_connection,
    get_frequency_stats,
    interactions_stream,
    learner_service_client,
    log_policy_frequency_issue,
    push_transitions_to_transport_queue,
    receive_policy,
    send_interactions,
    send_transitions,
    transitions_stream,
    update_policy_parameters,
    use_threads,
)
from lerobot.rl.algorithms.factory import make_algorithm
from lerobot.transport.utils import bytes_to_state_dict, python_object_to_bytes
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.process import ProcessSignalHandler, ensure_multiprocessing_start_method
from lerobot.utils.random_utils import set_seed
from lerobot.utils.recording_annotations import (
    EVORL_COLLECTOR_POLICY_ID_FIELD,
    EVORL_INTERVENTION_FIELD,
    EVORL_POLICY_ACTION_FIELD,
    EVORL_STATE_ACTIVE,
    EVORL_STATE_FIELD,
    EVORL_STATE_POLICY,
    EVORL_STATE_RELEASE,
    build_evorl_dataset_features,
    resolve_episode_success_from_mapping,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import Transition, move_transition_to_device
from lerobot.utils.utils import TimerManager, init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun

from .chunk_transition import build_sliding_window_transitions, sliding_window_observation_indices
from .compact_transition import (
    SCHEMA_NAME,
    SLIDING_WINDOW_TRANSITIONS,
    compact_episode_to_bytes,
    make_compact_episode,
    save_compact_episode,
)
from .config import ActorPipelineConfig
from .gym_manipulator import make_robot_env
from .keyboard_control import (
    EPISODE_FAILURE,
    EPISODE_SUCCESS,
    INTERVENTION_TOGGLE_COOLDOWN_S,
    KeyboardController,
    KeyboardState,
    start_keyboard_listener,
    stop_keyboard_listener,
)
from .piper_episode_control import (
    PoseHoldController,
    default_home_action,
    follow_policy_action,
    hold_arms_current_pose,
    home_arms_to_default,
    set_leader_manual_control,
)
from .rtc_action_runner import RTCActionChunkRunner, RTCChunkPrediction
from .wire import (
    ACTOR_GPU_RELEASED,
    WEIGHT_HANDOFF_ID_FIELD,
    make_control_message,
)


@dataclass
class TaskHotkeys:
    tasks: dict[str, str]
    default_key: str


@dataclass
class ActorControl:
    cfg: ActorPipelineConfig
    task_hotkeys: TaskHotkeys | None = None
    dataset_meta: LeRobotDatasetMetadata | None = None
    policy: Any = None
    runtime: Any = None
    smoother: Any = None
    actor_available: bool = False
    use_actor: bool = False


def _log_unhandled_exception(context, exc_type, exc_value, exc_traceback) -> None:
    logging.critical("[ACTOR] Unhandled exception in %s", context)
    logging.critical("%s", "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)).rstrip())
    for handler in logging.getLogger().handlers:
        handler.flush()


def load_task_hotkeys(path: str | Path) -> TaskHotkeys:
    data = json.loads(Path(path).expanduser().read_text())
    tasks = data.get("tasks")
    default_key = str(data.get("default_key", "")).lower()
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("task hotkey config requires a non-empty tasks object")
    normalized = {str(key).lower(): str(task).strip() for key, task in tasks.items()}
    reserved = {"a", "b", "c", "f", "r", "v"}
    invalid = [key for key, task in normalized.items() if len(key) != 1 or key in reserved or not task]
    if invalid:
        raise ValueError(f"Invalid or reserved task hotkeys: {invalid}")
    if default_key not in normalized:
        raise ValueError("task hotkey config default_key must exist in tasks")
    return TaskHotkeys(tasks=normalized, default_key=default_key)


def _find_actor_checkpoint(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate if candidate.name in {"actor_critic.pt", "model.safetensors"} else None
    for item in (
        candidate / "actor_critic.pt",
        candidate / "checkpoints/last/actor_critic.pt",
        candidate / "algorithm/model.safetensors",
        candidate / "checkpoints/last/algorithm/model.safetensors",
    ):
        if item.is_file():
            return item
    return None


def _normalize_actor_config(cfg: ActorPipelineConfig) -> None:
    cfg.actor_only.enabled = cfg.actor_mode == "vla_only"
    cfg.actor_only.save_format = cfg.save_format
    cfg.online_transition.enabled = cfg.actor_mode == "online_actor"
    cfg.validate()
    if cfg.actor_mode == "online_actor" and _find_actor_checkpoint(cfg.actor_checkpoint_path) is None:
        logging.warning(
            "No local Actor checkpoint found at %s; actor_new will remain on VLA. "
            "Press V after learner weights arrive to select the online Actor.",
            cfg.actor_checkpoint_path,
        )


def _build_actor_control(cfg: ActorPipelineConfig) -> ActorControl:
    dataset_meta = None
    if cfg.dataset is not None and (cfg.actor_mode == "online_actor" or cfg.task_hotkeys_path):
        dataset_meta = LeRobotDatasetMetadata(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
        )
    task_hotkeys = load_task_hotkeys(cfg.task_hotkeys_path) if cfg.task_hotkeys_path else None
    if task_hotkeys is not None:
        if cfg.env is None:
            raise ValueError("task hotkeys require an environment")
        if dataset_meta is not None:
            valid_tasks = set(dataset_meta.tasks.index)
            unknown = sorted(set(task_hotkeys.tasks.values()) - valid_tasks)
            if unknown:
                raise ValueError(f"Task hotkeys missing from dataset metadata: {unknown}")
        cfg.env.task = task_hotkeys.tasks[task_hotkeys.default_key]
    return ActorControl(
        cfg=cfg,
        task_hotkeys=task_hotkeys,
        dataset_meta=dataset_meta,
        use_actor=False,
    )


class ActorKeyboardController(KeyboardController):
    """Historical mode/task hotkeys layered on the shared episode hotkeys."""

    def __init__(self, control: ActorControl):
        super().__init__()
        self.control = control
        self._actor_events: ThreadQueue[str] = ThreadQueue()
        self._last_actor2_event_t = 0.0

    def push(self, key: str) -> None:
        normalized = key.lower() if len(key) == 1 else key
        tasks = self.control.task_hotkeys.tasks if self.control.task_hotkeys else {}
        if normalized == "v" or normalized in tasks:
            self._actor_events.put(normalized)
        else:
            super().push(key)

    def poll(self, state):
        state = super().poll(state)
        while True:
            try:
                key = self._actor_events.get_nowait()
            except Empty:
                break
            now = time.monotonic()
            if now - self._last_actor2_event_t < INTERVENTION_TOGGLE_COOLDOWN_S:
                continue
            self._last_actor2_event_t = now
            if key == "v":
                if not self.control.actor_available:
                    logging.error("[ACTOR] V ignored: online Actor weights are unavailable.")
                    continue
                if self.control.runtime is not None:
                    self.control.runtime.reset_action_state()
                self.control.use_actor = not self.control.use_actor
                if self.control.smoother is not None:
                    self.control.smoother.reset()
                logging.info(
                    "[ACTOR] V selected %s output",
                    "online Actor" if self.control.use_actor else "VLA",
                )
                continue
            task = self.control.task_hotkeys.tasks[key]
            if self.control.runtime is not None:
                self.control.runtime.reset_action_state()
            self.control.cfg.env.task = task
            if self.control.smoother is not None:
                self.control.smoother.reset()
            state.reset_episode = state.rerecord_episode = state.exit_episode = True
        return state


def _select_actor_action(policy_action, teleop_action, last_teleop_action, is_intervention: bool):
    if teleop_action is not None:
        last_teleop_action = teleop_action
    if is_intervention:
        if teleop_action is not None:
            return teleop_action, False, last_teleop_action
        if last_teleop_action is not None:
            return last_teleop_action, False, last_teleop_action
        if policy_action is not None:
            return policy_action, True, last_teleop_action
        return None, False, last_teleop_action
    return (
        policy_action if policy_action is not None else teleop_action,
        policy_action is not None,
        last_teleop_action,
    )


@dataclass(frozen=True)
class CheckpointFingerprint:
    resolved_path: str
    files: dict[str, tuple[int, int]]


def _fingerprint_policy_path(policy_path: str | Path) -> CheckpointFingerprint:
    path = Path(policy_path).expanduser().resolve()
    candidates = (
        [path]
        if path.is_file()
        else [
            candidate
            for pattern in (
                "config.json",
                "*.safetensors",
                "*.bin",
                "*preprocessor*.json",
                "*postprocessor*.json",
            )
            for candidate in path.glob(pattern)
            if candidate.is_file()
        ]
    )
    files = {}
    for candidate in sorted(set(candidates)):
        stat = candidate.stat()
        key = str(candidate.relative_to(path) if path.is_dir() else candidate.name)
        files[key] = (int(stat.st_mtime_ns), int(stat.st_size))
    return CheckpointFingerprint(str(path), files)


def _as_jsonable(value: Any, *, max_items: int = 32) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.flatten()[:max_items].tolist()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v, max_items=max_items) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v, max_items=max_items) for v in list(value)[:max_items]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitize_path_component(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "value"


class ActorEpisodeWriter:
    """Writes either the stable compact payload or the current LeRobot dataset format."""

    def __init__(
        self,
        cfg: Any,
        *,
        save_format: str | None = None,
        output_dir: str | None = None,
        save_images: bool | None = None,
        save_viewer: bool | None = None,
    ) -> None:
        self.cfg = cfg
        actor_only = cfg.actor_only
        self.save_format = save_format or actor_only.save_format
        self.output_dir = Path(
            output_dir or actor_only.episode_output_dir or Path(cfg.output_dir) / "actor_episodes"
        ).expanduser()
        self.save_images = actor_only.save_episode_images if save_images is None else save_images
        self.save_viewer = actor_only.save_episode_viewer if save_viewer is None else save_viewer
        self.dataset: LeRobotDataset | None = None
        self.action_names: list[str] = []
        self.observation_features: dict[str, dict[str, Any]] = {}
        self.robot_type: str | None = None
        if self.save_format == "transition":
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.episode_index = self._next_episode_index()
        else:
            if self.output_dir.exists():
                raise FileExistsError(f"LeRobotDataset output directory already exists: {self.output_dir}")
            self.episode_index = 0

    def configure_robot(self, robot: Any) -> None:
        if robot is None:
            return
        self.action_names = list(robot.action_features)
        self.observation_features = hw_to_dataset_features(
            getattr(robot, "observation_features", {}), prefix=OBS_STR, use_video=True
        )
        self.robot_type = getattr(robot, "robot_type", getattr(robot, "name", type(robot).__name__))

    def _next_episode_index(self) -> int:
        indices = []
        for child in self.output_dir.glob("episode_*"):
            with contextlib.suppress(ValueError):
                indices.append(int(child.name.rsplit("_", 1)[-1]))
        return max(indices, default=-1) + 1

    @staticmethod
    def _unbatch(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().cpu()
        return value.squeeze(0) if value.ndim and value.shape[0] == 1 else value

    def _ensure_lerobot_dataset(self, transition: Transition) -> LeRobotDataset:
        if self.dataset is not None:
            return self.dataset
        features = {key: dict(feature) for key, feature in self.observation_features.items()}
        if not features:
            for key, tensor in transition["state"].items():
                if isinstance(tensor, torch.Tensor):
                    value = self._unbatch(tensor)
                    features[key] = {"dtype": "float32", "shape": tuple(value.shape), "names": None}
        action = self._unbatch(transition["action"])
        action_feature = {
            "dtype": "float32",
            "shape": tuple(action.shape),
            "names": self.action_names or None,
        }
        features[ACTION] = action_feature
        features.update(build_evorl_dataset_features(action_feature))
        repo_id = self.cfg.dataset.repo_id if self.cfg.dataset is not None else "actor_only"
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=self.cfg.env.fps,
            root=self.output_dir,
            robot_type=self.robot_type,
            features=features,
            use_videos=True,
            image_writer_threads=4 * len([key for key in features if "image" in key]),
        )
        return self.dataset

    def save_episode(
        self,
        *,
        transitions: list[Transition],
        metadata: dict[str, Any],
        compact_episode: dict[str, Any] | None = None,
    ) -> Path:
        if not transitions:
            raise ValueError("Cannot save an empty actor episode")
        if self.save_format == "lerobot":
            return self._save_lerobot_episode(transitions, metadata)
        if compact_episode is None:
            raise ValueError("transition save format requires a compact episode payload")
        episode_dir = self.output_dir / f"episode_{self.episode_index:06d}"
        self.episode_index += 1
        episode_dir.mkdir(parents=True, exist_ok=False)
        save_compact_episode(compact_episode, episode_dir / "compact_episode.pt")
        (episode_dir / "metadata.json").write_text(
            json.dumps({**metadata, "transition_schema": SCHEMA_NAME}, indent=2, ensure_ascii=False)
        )
        return episode_dir

    def _save_lerobot_episode(self, transitions: list[Transition], metadata: dict[str, Any]) -> Path:
        cpu_transitions = [move_transition_to_device(transition=tr, device="cpu") for tr in transitions]
        dataset = self._ensure_lerobot_dataset(cpu_transitions[0])
        outcome = str(metadata.get("episode_outcome") or "none")
        episode_success = resolve_episode_success_from_mapping(
            {**metadata, "episode_outcome": outcome}, default_label="failure"
        )
        task = str(metadata.get("task") or "")
        policy_source = str(metadata.get("actor_policy_path") or "policy")
        policy_id = Path(policy_source).name
        episode_metadata = {"episode_success": episode_success}
        for transition in cpu_transitions:
            complementary = transition.get("complementary_info") or {}
            intervention = complementary.get(
                "intervention",
                complementary.get("is_intervention", complementary.get("acp_indicator", False)),
            )
            if isinstance(intervention, torch.Tensor):
                intervention = bool(intervention.detach().float().max().item() > 0.5)
            policy_action = complementary.get("policy_action")
            if policy_action is None:
                policy_action = (
                    torch.zeros_like(transition["action"]) if intervention else transition["action"]
                )
            state = complementary.get(
                "state",
                complementary.get(
                    "intervention_state", EVORL_STATE_ACTIVE if intervention else EVORL_STATE_POLICY
                ),
            )
            if isinstance(state, torch.Tensor):
                state = state.detach().reshape(-1)[0].item()
            elif isinstance(state, np.ndarray):
                state = state.reshape(-1)[0].item()
            frame = {
                key: self._unbatch(value)
                for key, value in transition["state"].items()
                if isinstance(value, torch.Tensor) and key in dataset.features
            }
            frame.update(
                {
                    ACTION: self._unbatch(transition["action"]).float(),
                    EVORL_POLICY_ACTION_FIELD: self._unbatch(policy_action).float(),
                    EVORL_INTERVENTION_FIELD: np.array([float(bool(intervention))], dtype=np.float32),
                    EVORL_STATE_FIELD: np.array([float(state)], dtype=np.float32),
                    EVORL_COLLECTOR_POLICY_ID_FIELD: "human" if intervention else policy_id,
                    "task": task,
                }
            )
            dataset.add_frame(frame)

        parameters = inspect.signature(dataset.save_episode).parameters
        if "episode_metadata" in parameters:
            dataset.save_episode(episode_metadata=episode_metadata)
        elif "extra_episode_metadata" in parameters:
            dataset.save_episode(extra_episode_metadata=episode_metadata)
        else:
            dataset.save_episode()
        return dataset.root

    def finalize(self) -> None:
        if self.dataset is not None:
            self.dataset.finalize()


class ActorEpisodeWriters:
    """Fan one accepted episode out to all configured local formats."""

    def __init__(self, writers: list[ActorEpisodeWriter]) -> None:
        """Create a non-empty local writer fan-out."""
        if not writers:
            raise ValueError("At least one actor episode writer is required")
        self.writers = writers

    def configure_robot(self, robot: Any) -> None:
        """Provide the same hardware feature contract to every writer."""
        for writer in self.writers:
            writer.configure_robot(robot)

    def save_episode(
        self,
        *,
        transitions: list[Transition],
        metadata: dict[str, Any],
        compact_episode: dict[str, Any],
    ) -> dict[str, Path]:
        """Persist the same accepted episode in every configured format."""
        paths = {}
        for writer in self.writers:
            paths[writer.save_format] = writer.save_episode(
                transitions=transitions,
                metadata=metadata,
                compact_episode=compact_episode,
            )
        return paths

    def finalize(self) -> None:
        """Finalize every writer, including video encoding and metadata."""
        for writer in self.writers:
            writer.finalize()


class ActorVLARuntime:
    """Current LeRobot policy/processor adapter with the legacy EvoRL API."""

    def __init__(self, cfg: ActorPipelineConfig):
        if not cfg.actor_vla_policy.policy_path:
            raise ValueError("actor_vla_policy.enabled=true requires actor_vla_policy.policy_path")
        self.cfg = cfg
        self.policy_path = Path(cfg.actor_vla_policy.policy_path).expanduser()
        self.last_check_t = 0.0
        self.fingerprint: CheckpointFingerprint | None = None
        self.policy_cfg = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self._rtc_runner: RTCActionChunkRunner | None = None
        self.reload()

    def reload(self) -> None:
        self.close()
        if not self.policy_path.exists():
            raise FileNotFoundError(f"VLA policy path does not exist: {self.policy_path}")
        # 0901 checkpoints identify themselves as ``pi05`` while containing
        # the former embedded RLT module. The migrated actor JSON owns the
        # current ``pi05_rlt`` config, so use it as the authoritative schema
        # and load only weights/processors from the legacy checkpoint path.
        configured_policy = self.cfg.policy
        if configured_policy is not None and getattr(configured_policy, "type", None) == "pi05_rlt":
            policy_cfg = deepcopy(configured_policy)
        else:
            policy_cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        if getattr(policy_cfg, "type", None) != "pi05_rlt":
            raise ValueError(
                "online actor requires policy.type=pi05_rlt; legacy pi05+RLT checkpoints "
                "must be paired with a migrated online RL JSON"
            )
        policy_cfg.pretrained_path = self.policy_path
        if self.cfg.policy is not None and getattr(self.cfg.policy, "device", None):
            policy_cfg.device = self.cfg.policy.device
        policy = make_policy(policy_cfg, env_cfg=self.cfg.env).eval()
        self._configure_rtc_policy(policy)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(self.policy_path),
            preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
        )
        self.policy_cfg = policy_cfg
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.fingerprint = _fingerprint_policy_path(self.policy_path)
        self.reset_action_state()

    def _configure_rtc_policy(self, policy: Any) -> None:
        init_rtc = getattr(policy, "init_rtc_processor", None)
        if not self.cfg.rtc.enabled:
            if hasattr(policy.config, "rtc_config"):
                policy.config.rtc_config = None
                if callable(init_rtc):
                    init_rtc()
            return
        if not callable(init_rtc) or not callable(getattr(policy, "predict_action_chunk", None)):
            raise ValueError(f"Policy {type(policy).__name__} does not support RTC")
        supports_rtc = getattr(policy, "supports_rtc", None)
        if callable(supports_rtc) and not supports_rtc():
            raise ValueError(f"Policy {type(policy).__name__} reports that RTC is unsupported")
        policy.config.rtc_config = self.cfg.rtc
        init_rtc()
        logging.info(
            "[ACTOR][RTC] mode=%s horizon=%d queue_threshold=%d",
            self.cfg.rtc.mode,
            self.cfg.rtc.execution_horizon,
            self.cfg.rtc_action_queue_threshold,
        )

    def _predict_rtc_chunk(
        self,
        observation_frame: dict[str, Any],
        inference_delay: int,
        previous_actions: torch.Tensor | None,
        task: str | None,
        robot_type: str | None,
    ) -> RTCChunkPrediction:
        observation = prepare_observation_for_inference(
            copy(observation_frame),
            get_safe_torch_device(self.policy_cfg.device),
            task,
            robot_type,
        )
        batch = self.preprocessor(observation)
        actions = self.policy.predict_action_chunk(
            batch,
            inference_delay=inference_delay,
            prev_chunk_left_over=previous_actions,
        )
        return RTCChunkPrediction(actions=actions)

    def _rtc(self, robot_type: str | None) -> RTCActionChunkRunner:
        if self._rtc_runner is None:
            self._rtc_runner = RTCActionChunkRunner(
                policy=self.policy,
                postprocessor=self.postprocessor,
                rtc=self.cfg.rtc,
                fps=self.cfg.env.fps,
                queue_threshold=self.cfg.rtc_action_queue_threshold,
                task=getattr(self.cfg.env, "task", None),
                robot_type=robot_type,
                chunk_predictor=self._predict_rtc_chunk,
            )
        else:
            self._rtc_runner.task = getattr(self.cfg.env, "task", None)
            self._rtc_runner.robot_type = robot_type
        return self._rtc_runner

    def reset_action_state(self) -> None:
        if self._rtc_runner is not None:
            self._rtc_runner.reset()
        for component in (self.policy, self.preprocessor, self.postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()
        if self._rtc_runner is not None:
            self._rtc_runner.resume()

    def invalidate_action_state(self) -> None:
        if self._rtc_runner is not None:
            self._rtc_runner.pause()
            self._rtc_runner.invalidate_pending_actions()

    def quiesce_action_state(self) -> None:
        if self._rtc_runner is not None:
            self._rtc_runner.pause()
            self._rtc_runner.invalidate_pending_actions()
            self._rtc_runner.wait_for_idle()

    def close(self) -> None:
        if self._rtc_runner is not None:
            self._rtc_runner.close()
            self._rtc_runner = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.policy_cfg = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def reload_if_changed(self) -> None:
        now = time.monotonic()
        if now - self.last_check_t < self.cfg.actor_vla_policy.policy_poll_s:
            return
        self.last_check_t = now
        current = _fingerprint_policy_path(self.policy_path)
        if self.fingerprint is not None and current != self.fingerprint:
            self.reload()

    def select_action(
        self, observation_frame: dict[str, Any], robot_type: str | None, device: torch.device
    ) -> torch.Tensor:
        if self.policy is None or self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("VLA runtime is not loaded")
        if self.cfg.rtc.enabled:
            action = self._rtc(robot_type).get_action(observation_frame)
            return action.to(device)
        numpy_observation = {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for key, value in observation_frame.items()
        }
        inference_device = get_safe_torch_device(self.policy_cfg.device)
        action = predict_action(
            observation=numpy_observation,
            policy=self.policy,
            device=inference_device,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=getattr(self.policy_cfg, "use_amp", False),
            task=getattr(self.cfg.env, "task", None),
            robot_type=robot_type,
        )
        return action.to(device) if isinstance(action, torch.Tensor) else action

    @torch.no_grad()
    def build_compact_episode(
        self,
        transitions: list[Transition],
        metadata: dict[str, Any],
        batch_size: int,
        robot_type: str | None = None,
    ) -> dict[str, Any]:
        if not transitions:
            raise ValueError("Cannot build a compact episode from no transitions")
        if self.policy is None or self.preprocessor is None or self.fingerprint is None:
            raise RuntimeError("VLA runtime is not loaded")
        predict_with_rlt = getattr(self.policy, "predict_action_chunk_with_rlt", None)
        if not callable(predict_with_rlt):
            raise TypeError(
                f"{type(self.policy).__name__} does not implement predict_action_chunk_with_rlt(); "
                "use a pi05_rlt checkpoint for compact EvoRL collection"
            )
        episode_length = len(transitions)
        horizon = int(self.policy_cfg.chunk_size)
        stride = self.cfg.online_transition.sliding_window_stride
        observations = [transition["state"] for transition in transitions] + [transitions[-1]["next_state"]]
        actions = [transition[ACTION] for transition in transitions] + [transitions[-1][ACTION]]
        feature_indices = sliding_window_observation_indices(episode_length, horizon, stride)
        feature_index_set = set(feature_indices)
        features_by_observation: dict[int, dict[str, torch.Tensor]] = {}
        normalized_actions = []
        feature_device = get_safe_torch_device(self.policy_cfg.device)
        for start in range(0, len(observations), batch_size):
            items = []
            for observation, action in zip(
                observations[start : start + batch_size], actions[start : start + batch_size], strict=True
            ):
                raw_observation = {
                    key: value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else np.asarray(value)
                    for key, value in observation.items()
                }
                model_input = prepare_observation_for_inference(
                    raw_observation,
                    feature_device,
                    task=metadata["task"],
                    robot_type=robot_type,
                )
                model_input[ACTION] = torch.as_tensor(action).detach().reshape(-1).to(feature_device)
                items.append(self.preprocessor(model_input))
            batch = {
                key: torch.cat([item[key] for item in items], dim=0)
                for key, value in items[0].items()
                if isinstance(value, torch.Tensor)
            }
            for offset, part in enumerate(batch[ACTION].split(1)):
                if start + offset < episode_length:
                    normalized_actions.append(part.detach().cpu())
            selected_offsets = [offset for offset in range(len(items)) if start + offset in feature_index_set]
            if not selected_offsets:
                continue
            feature_batch = {key: value[selected_offsets] for key, value in batch.items()}
            outputs = predict_with_rlt(feature_batch)
            proprio_dim = getattr(self.policy_cfg, "proprio_dim", feature_batch[OBS_STATE].shape[-1])
            batch_features = {
                "z_rl": outputs["z_rl"],
                "proprio": feature_batch[OBS_STATE][..., :proprio_dim],
                "ref_action": outputs["actions"],
            }
            for feature_index, parts in zip(
                [start + offset for offset in selected_offsets],
                zip(*(value.split(1) for value in batch_features.values()), strict=True),
                strict=True,
            ):
                features_by_observation[feature_index] = {
                    key: part.detach().cpu() for key, part in zip(batch_features, parts, strict=True)
                }

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
                    "state": features_by_observation.get(index, {}),
                    "next_state": features_by_observation.get(index + 1, {}),
                    ACTION: normalized_actions[index],
                    "reward": transition["reward"],
                    "done": bool(transition["done"]),
                    "truncated": bool(transition["truncated"]),
                    "complementary_info": {"intervention": bool(intervention)},
                }
            )
        compact_transitions = build_sliding_window_transitions(primitive, horizon=horizon, stride=stride)
        return make_compact_episode(
            transitions=compact_transitions,
            metadata={
                **metadata,
                "transition_schema": SCHEMA_NAME,
                "transition_layout": SLIDING_WINDOW_TRANSITIONS,
                "sliding_window_stride": stride,
            },
            feature_model=_as_jsonable(self.fingerprint.__dict__),
            transition_layout=SLIDING_WINDOW_TRANSITIONS,
            sliding_window_stride=stride,
            primitive_steps=episode_length,
        )


class OnlineActorRuntime(ActorVLARuntime):
    """PI05-RLT plus an independent online algorithm actor."""

    def __init__(self, control: ActorControl):
        super().__init__(control.cfg)
        self.control = control
        self.algorithm = None
        self._actor_checkpoint = _find_actor_checkpoint(self.cfg.actor_checkpoint_path)
        self._build_algorithm()
        control.policy = self.policy
        control.runtime = self
        if self._actor_checkpoint is not None:
            self._load_algorithm_actor()

    def _build_algorithm(self) -> None:
        algorithm_cfg = self.cfg.algorithm
        if algorithm_cfg is None or getattr(algorithm_cfg, "type", None) != "rlt_chunk":
            raise ValueError("online_actor requires algorithm.type=rlt_chunk")
        algorithm_cfg.policy_config = self.policy_cfg
        self.algorithm = make_algorithm(algorithm_cfg, self.policy)

    def _load_algorithm_actor(self) -> None:
        checkpoint = self._actor_checkpoint
        if checkpoint is None:
            return
        if checkpoint.suffix == ".safetensors":
            state = load_safetensors(str(checkpoint), device="cpu")
            if any(key.startswith("critic_ensemble.") for key in state):
                self.algorithm.load_state_dict(state, device=self.policy_cfg.device)
            else:
                actor_state = {
                    key.removeprefix("actor."): value
                    for key, value in state.items()
                    if key.startswith("actor.")
                }
                self.algorithm.load_weights({"policy": actor_state or state}, device=self.policy_cfg.device)
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError(f"Missing algorithm state dict in {checkpoint}")
            if "actor" in state and ("critic" in state or "critic_ensemble" in state):
                self.algorithm.load_legacy_actor_critic(checkpoint)
            else:
                self.algorithm.load_weights(
                    state if "policy" in state else {"policy": state},
                    device=self.policy_cfg.device,
                )
        self.control.actor_available = True

    def release_gpu(self) -> None:
        """Drop every CUDA-owning inference object while preserving robot transport state."""
        self.quiesce_action_state()
        self.algorithm = None
        self.control.policy = None
        super().close()
        logging.info("[ACTOR] released PI0.5 and online Actor CUDA allocations")

    def acquire_gpu(self, weights: dict[str, Any]) -> None:
        """Reload PI0.5 and apply the learner weights for the completed handoff."""
        super().reload()
        self._build_algorithm()
        self.algorithm.load_weights(weights, device=self.policy_cfg.device)
        self.control.policy = self.policy
        self.control.actor_available = True
        self.reset_action_state()
        logging.info("[ACTOR] reacquired GPU and loaded learner Actor weights")

    def reset_action_state(self) -> None:
        algorithm = getattr(self, "algorithm", None)
        if algorithm is not None:
            algorithm.reset()
        super().reset_action_state()

    def _predict_rtc_chunk(
        self,
        observation_frame: dict[str, Any],
        inference_delay: int,
        previous_actions: torch.Tensor | None,
        task: str | None,
        robot_type: str | None,
    ) -> RTCChunkPrediction:
        if not self.control.use_actor:
            return super()._predict_rtc_chunk(
                observation_frame, inference_delay, previous_actions, task, robot_type
            )
        if not self.control.actor_available:
            raise RuntimeError("online Actor weights are unavailable")
        observation = prepare_observation_for_inference(
            copy(observation_frame),
            get_safe_torch_device(self.policy_cfg.device),
            task,
            robot_type,
        )
        batch = self.preprocessor(observation)
        outputs = self.policy.predict_action_chunk_with_rlt(
            batch,
            inference_delay=inference_delay,
            prev_chunk_left_over=previous_actions,
        )
        proprio = batch[OBS_STATE]
        if proprio.ndim == 3:
            proprio = proprio[:, -1]
        chunk = self.algorithm.actor(
            outputs["z_rl"],
            proprio,
            outputs["actions"],
            deterministic=self.algorithm.config.actor_rollout_deterministic,
            apply_reference_dropout=False,
        )
        return RTCChunkPrediction(actions=chunk, apply_prefix_fusion=True)

    def select_action(self, observation_frame, robot_type, device):
        if self.cfg.rtc.enabled:
            return super().select_action(observation_frame, robot_type, device)
        if not self.control.use_actor:
            return super().select_action(observation_frame, robot_type, device)
        if not self.control.actor_available:
            raise RuntimeError("online_actor mode selected without loaded Actor weights")
        numpy_observation = {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for key, value in observation_frame.items()
        }
        batch = self.algorithm.prepare_observation(
            self.preprocessor,
            numpy_observation,
            task=getattr(self.cfg.env, "task", ""),
        )
        with torch.inference_mode():
            action = self.algorithm.select_action(batch)
        return self.postprocessor(action).to(device)


def _runtime_config(cfg: ActorPipelineConfig):
    return cfg.algorithm if hasattr(cfg.algorithm, "actor_learner_config") else cfg.policy


def _freeze_observation(frame: dict[str, Any]) -> dict[str, torch.Tensor]:
    frozen = {}
    for key, value in frame.items():
        if isinstance(value, torch.Tensor):
            frozen[key] = value.detach().cpu().clone()
        else:
            frozen[key] = torch.from_numpy(np.asarray(value).copy())
    return frozen


def _retry_io(name: str, fn, *, timeout_s: float = 2.0, interval_s: float = 0.1):
    deadline = time.perf_counter() + timeout_s
    attempts = 0
    while True:
        attempts += 1
        try:
            result = fn()
            if attempts > 1:
                logging.warning("%s recovered after %d retries", name, attempts - 1)
            return result
        except ConnectionError:
            if time.perf_counter() >= deadline:
                raise
            if attempts == 1:
                logging.warning("%s hit a transient CAN error; retrying for %.1fs", name, timeout_s)
            time.sleep(interval_s)


def _queue_has_items(queue: Queue) -> bool:
    try:
        return queue.qsize() > 0
    except (AttributeError, NotImplementedError):
        return not queue.empty()


@parser.wrap()
def actor_cli(cfg: ActorPipelineConfig):
    """Run actor_new as the sole EvoRL online hardware actor."""
    _normalize_actor_config(cfg)
    if cfg.actor_mode == "vla_only":
        raise ValueError(
            "vla_only collection is now run by lerobot-rollout; use the EvoRL rollout strategy. "
            "actor_new is retained only for online Actor collection."
        )
    control = _build_actor_control(cfg)
    display_pid = not use_threads(cfg)
    if display_pid:
        ensure_multiprocessing_start_method(_runtime_config(cfg).concurrency.multiprocessing_context)
    log_dir = Path(cfg.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"actor_{cfg.job_name}.log"
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info("[ACTOR] actor_new online runtime; log=%s", log_file)
    logging.info(
        "[ACTOR] can0.control=%s rtc=%s rtc_mode=%s",
        cfg.can0.control,
        cfg.rtc.enabled,
        cfg.rtc.mode,
    )
    sys.excepthook = lambda typ, value, tb: _log_unhandled_exception("main thread", typ, value, tb)
    threading.excepthook = lambda args: _log_unhandled_exception(
        "thread " + getattr(args.thread, "name", "unknown"),
        args.exc_type,
        args.exc_value,
        args.exc_traceback,
    )
    shutdown_event = ProcessSignalHandler(use_threads(cfg), display_pid=display_pid).shutdown_event
    return run_actor_online(cfg, shutdown_event=shutdown_event, actor_control=control)


def run_actor_online(
    cfg: ActorPipelineConfig,
    shutdown_event: Any | None = None,
    actor_control: ActorControl | None = None,
):
    """Connect to the configured EvoRL learner and run the hardware loop.

    ``actor_cli`` normalizes and validates the configuration before creating
    runtime directories. Revalidating here would reject the ``output_dir``
    immediately after ``actor_cli`` creates its ``logs`` subdirectory.
    """
    actor_control = actor_control or _build_actor_control(cfg)
    shutdown_event = shutdown_event or ProcessSignalHandler(use_threads(cfg)).shutdown_event
    learner_cfg = _runtime_config(cfg).actor_learner_config
    learner_client, grpc_channel = learner_service_client(
        host=learner_cfg.learner_host,
        port=learner_cfg.learner_port,
    )
    logging.info("[ACTOR] Establishing connection with learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        raise ConnectionError("Failed to establish connection with learner")
    if not use_threads(cfg):
        grpc_channel.close()
        grpc_channel = None

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    if use_threads(cfg):
        from threading import Thread as Worker
    else:
        from multiprocessing import Process as Worker

    workers = [
        Worker(
            target=receive_policy,
            args=(cfg, parameters_queue, shutdown_event, grpc_channel),
            daemon=True,
        ),
        Worker(
            target=send_transitions,
            args=(cfg, transitions_queue, shutdown_event, grpc_channel),
            daemon=True,
        ),
        Worker(
            target=send_interactions,
            args=(cfg, interactions_queue, shutdown_event, grpc_channel),
            daemon=True,
        ),
    ]
    compact_output_dir = cfg.online_transition.episode_output_dir or str(
        Path(cfg.output_dir) / "online_transitions"
    )
    writers = [
        ActorEpisodeWriter(
            cfg,
            save_format="transition",
            output_dir=compact_output_dir,
            save_images=False,
            save_viewer=False,
        )
    ]
    if cfg.online_transition.save_lerobot_copy:
        lerobot_output_dir = cfg.online_transition.lerobot_output_dir or str(
            Path(cfg.output_dir) / "lerobot_dataset"
        )
        writers.append(
            ActorEpisodeWriter(
                cfg,
                save_format="lerobot",
                output_dir=lerobot_output_dir,
                save_images=True,
                save_viewer=True,
            )
        )
    writer = ActorEpisodeWriters(writers)
    for worker in workers:
        worker.start()

    try:
        act_with_policy(
            cfg=cfg,
            shutdown_event=shutdown_event,
            parameters_queue=parameters_queue,
            transitions_queue=transitions_queue,
            interactions_queue=interactions_queue,
            actor_episode_writer=writer,
            actor_control=actor_control,
        )
    except Exception:
        logging.exception("[ACTOR] Unhandled exception in actor_new control loop")
        raise
    finally:
        shutdown_event.set()
        try:
            writer.finalize()
        finally:
            for worker in workers:
                worker.join(timeout=max(float(learner_cfg.queue_get_timeout) + 1.0, 3.0))
                if worker.is_alive():
                    logging.warning("[ACTOR] background worker did not exit before timeout: %s", worker)
            for queue in (transitions_queue, interactions_queue, parameters_queue):
                queue.close()
                queue.cancel_join_thread()
            if grpc_channel is not None:
                grpc_channel.close()


def act_with_policy(
    *,
    cfg: ActorPipelineConfig,
    shutdown_event: Any,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
    actor_episode_writer: ActorEpisodeWriter | ActorEpisodeWriters | None = None,
    actor_control: ActorControl | None = None,
) -> None:
    """Collect online episodes with policy-follow, immediate C takeover and RTC."""
    if actor_control is None:
        actor_control = _build_actor_control(cfg)
    if cfg.env is None:
        raise ValueError("online actor requires env")

    display_data = bool(
        cfg.env.processor.observation is not None and cfg.env.processor.observation.display_cameras
    )
    rerun_started = False
    online_env = None
    teleop = None
    runtime = None
    pose_holder = None
    try:
        if display_data:
            init_rerun(session_name="evorl_actor_new")
            rerun_started = True
        online_env, teleop = make_robot_env(cfg.env, connect_teleop=cfg.can0.control)
        robot = online_env.robot
        if actor_episode_writer is not None:
            actor_episode_writer.configure_robot(robot)
        teleop_processor, robot_action_processor, observation_processor = make_default_processors()
        action_names = list(robot.action_features)
        observation_features = hw_to_dataset_features(
            robot.observation_features, prefix=OBS_STR, use_video=False
        )
        device = get_safe_torch_device(cfg.policy.device, log=True)
        set_seed(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        runtime = OnlineActorRuntime(actor_control)
        actor_control.runtime = runtime
        if not actor_control.actor_available:
            actor_control.use_actor = False
            logging.warning(
                "[ACTOR] No local Actor weights yet; remaining on VLA. "
                "Press V after learner weights arrive to select the online Actor."
            )
        keyboard = ActorKeyboardController(actor_control)
        keyboard_state = KeyboardState()
        listener = start_keyboard_listener(keyboard)
        policy_timer = TimerManager("Policy inference", log=False)
        intervention_state = EVORL_STATE_POLICY
        transitions: list[Transition] = []
        pending: dict[str, Any] | None = None
        episode_reward = 0.0
        episode_steps = 0
        intervention_steps = 0
        episode_index = 0
        actor_session_id = uuid4().hex
        episode_limit = getattr(cfg.env, "episode_length", None)
        if episode_limit is None:
            reset_cfg = getattr(cfg.env.processor, "reset", None)
            control_time_s = getattr(reset_cfg, "control_time_s", None)
            if control_time_s and cfg.env.fps:
                episode_limit = max(int(round(control_time_s * cfg.env.fps)), 1)
        online_env.reset()
        set_leader_manual_control(teleop, False)
        home_target = default_home_action(
            robot,
            max_gripper_pos=cfg.env.processor.max_gripper_pos,
        )
        home_arms_to_default(
            robot,
            teleop,
            target_action=home_target,
            fps=cfg.env.fps or 30,
        )
        pose_holder = PoseHoldController(robot, teleop, fps=cfg.env.fps or 30)
        episodes_since_handoff = 0
        handoff_id = 0
        episodes_per_handoff = cfg.gpu_handoff.update_quota_threshold // cfg.gpu_handoff.updates_per_episode

        def action_to_dict(action: Any) -> dict[str, float]:
            if isinstance(action, dict):
                missing = [key for key in action_names if key not in action]
                if missing:
                    raise ValueError(f"Action is missing robot keys: {missing}")
                return {key: float(action[key]) for key in action_names}
            tensor = torch.as_tensor(action).detach().cpu().reshape(-1)
            if tensor.numel() != len(action_names):
                raise ValueError(
                    f"Policy returned {tensor.numel()} values for {len(action_names)} robot actions"
                )
            return {key: float(tensor[index]) for index, key in enumerate(action_names)}

        def action_to_tensor(action: dict[str, float]) -> torch.Tensor:
            return torch.tensor([action[key] for key in action_names], dtype=torch.float32)

        def reset_episode() -> None:
            nonlocal keyboard_state, intervention_state, transitions, pending
            nonlocal episode_reward, episode_steps, intervention_steps
            keyboard_state = KeyboardState()
            intervention_state = EVORL_STATE_POLICY
            transitions = []
            pending = None
            episode_reward = 0.0
            episode_steps = 0
            intervention_steps = 0
            set_leader_manual_control(teleop, False)
            runtime.reset_action_state()
            teleop_processor.reset()
            robot_action_processor.reset()
            observation_processor.reset()
            online_env.reset()

        def perform_gpu_handoff() -> None:
            nonlocal handoff_id
            while True:
                try:
                    parameters_queue.get_nowait()
                except Empty:
                    break
            handoff_id += 1
            runtime.release_gpu()
            interactions_queue.put(
                python_object_to_bytes(
                    make_control_message(
                        ACTOR_GPU_RELEASED,
                        handoff_id=handoff_id,
                        accepted_episodes=episodes_per_handoff,
                    )
                )
            )
            logging.info("[ACTOR] waiting for learner handoff=%d", handoff_id)
            deadline = time.monotonic() + cfg.gpu_handoff.learner_release_timeout_s
            weights = None
            while not shutdown_event.is_set() and time.monotonic() < deadline:
                try:
                    candidate = bytes_to_state_dict(
                        parameters_queue.get(timeout=min(0.2, max(deadline - time.monotonic(), 0.01)))
                    )
                except Empty:
                    continue
                response_id = candidate.get(WEIGHT_HANDOFF_ID_FIELD)
                if isinstance(response_id, torch.Tensor):
                    response_id = int(response_id.item())
                if response_id != handoff_id:
                    logging.warning(
                        "[ACTOR] ignored stale learner weights handoff=%s while waiting for %d",
                        response_id,
                        handoff_id,
                    )
                    continue
                weights = candidate
                break
            if weights is None:
                raise TimeoutError(f"Learner did not release GPU for handoff={handoff_id}")
            runtime.acquire_gpu(weights)

        def finish_episode(*, outcome: str | None, discard: bool, reset_to_home: bool, timeout: bool) -> None:
            nonlocal episode_index, episodes_since_handoff
            runtime.quiesce_action_state()
            pose_holder.start()
            metadata = {
                "Episodic reward": episode_reward,
                "episodic_reward": episode_reward,
                "Interaction step": interaction_step,
                "Episode intervention": int(intervention_steps > 0),
                "Intervention rate": intervention_steps / max(episode_steps, 1),
                "intervention_rate": intervention_steps / max(episode_steps, 1),
                "Episode outcome": outcome or ("timeout" if timeout else "none"),
                "episode_outcome": outcome or ("timeout" if timeout else "none"),
                "timeout": timeout,
                "task": cfg.env.task,
                "actor_policy_type": "online_actor" if actor_control.use_actor else "vla",
                "actor_policy_path": str(runtime.policy_path),
                "episode_index": episode_index,
                "episode_id": f"{cfg.job_name}:{actor_session_id}:{episode_index}",
                **get_frequency_stats(policy_timer),
            }
            accepted_episode = False
            if transitions and not discard:
                compact = runtime.build_compact_episode(
                    transitions,
                    metadata,
                    cfg.online_transition.feature_batch_size,
                    robot_type=getattr(robot, "robot_type", getattr(robot, "name", None)),
                )
                payload = compact_episode_to_bytes(compact)
                if actor_episode_writer is not None:
                    saved_paths = actor_episode_writer.save_episode(
                        transitions=transitions,
                        metadata=metadata,
                        compact_episode=compact,
                    )
                    logging.info("[ACTOR] episode=%d saved locally: %s", episode_index, saved_paths)
                transitions_queue.put(payload)
                interactions_queue.put(python_object_to_bytes(metadata))
                accepted_episode = True
                logging.info(
                    "[ACTOR] episode=%d queued steps=%d outcome=%s bytes=%d",
                    episode_index,
                    len(transitions),
                    metadata["episode_outcome"],
                    len(payload),
                )
            else:
                logging.info("[ACTOR] episode=%d discarded; no payload sent", episode_index)

            did_handoff = False
            if accepted_episode:
                episodes_since_handoff += 1
            if cfg.gpu_handoff.enabled and episodes_since_handoff >= episodes_per_handoff:
                perform_gpu_handoff()
                episodes_since_handoff = 0
                did_handoff = True
            elif not cfg.gpu_handoff.enabled:
                had_parameters = _queue_has_items(parameters_queue)
                update_policy_parameters(runtime.algorithm, parameters_queue, device)
                if had_parameters and not actor_control.actor_available:
                    actor_control.actor_available = True
                    logging.info(
                        "[ACTOR] First learner Actor weights loaded; output remains VLA until V is pressed"
                    )

            reset_cfg = getattr(cfg.env.processor, "reset", None)
            reset_time_s = float(getattr(reset_cfg, "reset_time_s", 0.0) or 0.0)
            if did_handoff or reset_to_home:
                pose_holder.stop()
                home_arms_to_default(
                    robot,
                    teleop,
                    target_action=home_target,
                    fps=cfg.env.fps or 30,
                )
            else:
                pose_holder.wait(reset_time_s)
                pose_holder.stop()
            episode_index += 1
            policy_timer.reset()
            reset_episode()

        for interaction_step in range(_runtime_config(cfg).online_steps):
            tick_start = time.perf_counter()
            if shutdown_event.is_set():
                break
            keyboard.poll(keyboard_state)
            if keyboard_state.stop:
                logging.info("[ACTOR] ESC requested shutdown")
                hold_arms_current_pose(robot, teleop)
                break
            if keyboard_state.toggle_intervention:
                if teleop is None:
                    logging.warning("[ACTOR] C ignored because can0.control=false")
                elif intervention_state == EVORL_STATE_ACTIVE:
                    set_leader_manual_control(teleop, False)
                    runtime.reset_action_state()
                    intervention_state = EVORL_STATE_RELEASE
                    logging.info("[ACTOR] Human intervention released; policy resumes")
                else:
                    runtime.invalidate_action_state()
                    set_leader_manual_control(teleop, True)
                    intervention_state = EVORL_STATE_ACTIVE
                    logging.info("[ACTOR] C takeover active immediately; pending RTC actions invalidated")

            raw_observation = _retry_io("robot.get_observation", online_env.read_raw_observation)
            processed_observation = observation_processor(raw_observation)
            observation_frame = build_dataset_frame(
                observation_features, processed_observation, prefix=OBS_STR
            )
            state = _freeze_observation(observation_frame)
            outcome = keyboard_state.episode_outcome
            if pending is not None:
                reward = 1.0 if outcome == EPISODE_SUCCESS else 0.0
                terminal = outcome in {EPISODE_SUCCESS, EPISODE_FAILURE}
                pending["complementary_info"]["success"] = torch.tensor([float(outcome == EPISODE_SUCCESS)])
                pending["complementary_info"]["failure"] = torch.tensor([float(outcome == EPISODE_FAILURE)])
                transitions.append(
                    Transition(
                        state=pending["state"],
                        action=pending["action"],
                        reward=reward,
                        next_state=state,
                        done=terminal,
                        truncated=False,
                        complementary_info=pending["complementary_info"],
                    )
                )
                episode_reward += reward
                episode_steps += 1
                intervention_steps += int(pending["intervention"])
                pending = None

            timeout = bool(episode_limit and episode_steps >= episode_limit)
            if timeout and transitions and outcome is None:
                transitions[-1]["truncated"] = True
            discard = keyboard_state.rerecord_episode or keyboard_state.reset_episode
            if keyboard_state.exit_episode or outcome is not None or timeout:
                finish_episode(
                    outcome=outcome,
                    discard=discard,
                    reset_to_home=keyboard_state.reset_episode,
                    timeout=timeout and outcome is None,
                )
                continue

            is_intervention = intervention_state == EVORL_STATE_ACTIVE
            if is_intervention:
                raw_teleop_action = _retry_io("teleop.get_action", teleop.get_action)
                selected_action = teleop_processor((raw_teleop_action, raw_observation))
                policy_action = dict.fromkeys(action_names, 0.0)
            else:
                with policy_timer:
                    predicted = runtime.select_action(
                        observation_frame,
                        getattr(robot, "robot_type", getattr(robot, "name", None)),
                        device,
                    )
                log_policy_frequency_issue(policy_timer.fps_last, cfg, interaction_step)
                selected_action = action_to_dict(predicted)
                policy_action = dict(selected_action)

            selected_action = robot_action_processor((selected_action, raw_observation))
            sent_action = _retry_io(
                "robot.send_action",
                lambda selected_action=selected_action: online_env.send_action(selected_action),
            )
            actual_action = action_to_dict(sent_action or selected_action)
            if not is_intervention:
                _retry_io(
                    "teleop.send_feedback",
                    lambda actual_action=actual_action: follow_policy_action(teleop, actual_action),
                )
            action_tensor = action_to_tensor(actual_action)
            pending = {
                "state": state,
                "action": action_tensor,
                "intervention": is_intervention,
                "complementary_info": {
                    "is_intervention": torch.tensor([float(is_intervention)]),
                    "intervention_state": torch.tensor([float(intervention_state)]),
                    "policy_action": action_to_tensor(policy_action),
                    "success": torch.tensor([0.0]),
                    "failure": torch.tensor([0.0]),
                },
            }
            if display_data:
                log_rerun_data(raw_observation, actual_action, compress_images=False)
            if intervention_state == EVORL_STATE_RELEASE:
                intervention_state = EVORL_STATE_POLICY
            if cfg.env.fps:
                precise_sleep(max(1.0 / cfg.env.fps - (time.perf_counter() - tick_start), 0.0))
    finally:
        if pose_holder is not None and pose_holder.active:
            with contextlib.suppress(Exception):
                pose_holder.stop()
        if "listener" in locals():
            stop_keyboard_listener(listener)
        if runtime is not None:
            runtime.release_gpu()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if online_env is not None:
            online_env.close()
        if rerun_started:
            shutdown_rerun()


def run_actor_only(*_args, **_kwargs):
    raise ValueError("Actor-only VLA collection moved to lerobot-rollout with the EvoRL strategy")


if __name__ == "__main__":
    actor_cli()
