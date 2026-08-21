#!/usr/bin/env python

"""Actor v2: reuse the existing actor with policy-driven loading and runtime controls."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.onlineRL_evoRL import actor as base_actor
from lerobot.onlineRL_evoRL.keyboard_control import INTERVENTION_TOGGLE_COOLDOWN_S
from lerobot.onlineRL_evoRL.process import ProcessSignalHandler
from lerobot.policies.factory import make_policy as factory_make_policy
from lerobot.utils.control_utils import prepare_observation_for_inference
from lerobot.utils.utils import get_safe_torch_device, init_logging


@dataclass(kw_only=True)
class Actor2PipelineConfig(TrainRLServerPipelineConfig):
    task_hotkeys_path: str
    actor_checkpoint_path: str | None = None


@dataclass
class TaskHotkeys:
    tasks: dict[str, str]
    default_key: str


@dataclass
class Actor2Control:
    cfg: Actor2PipelineConfig
    task_hotkeys: TaskHotkeys
    dataset_meta: LeRobotDatasetMetadata
    policy: Any = None
    runtime: Any = None
    smoother: Any = None
    actor_available: bool = False
    use_actor: bool = False


def _log_unhandled_exception(context, exc_type, exc_value, exc_traceback) -> None:
    logging.critical(
        "[ACTOR_2] Unhandled exception in %s\n%s",
        context,
        "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)).rstrip(),
    )
    for handler in logging.getLogger().handlers:
        handler.flush()


def load_task_hotkeys(path: str | Path) -> TaskHotkeys:
    config_path = Path(path).expanduser()
    data = json.loads(config_path.read_text())
    tasks = data.get("tasks")
    default_key = str(data.get("default_key", ""))
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("task hotkey config requires a non-empty 'tasks' object")

    normalized = {str(key).lower(): str(task).strip() for key, task in tasks.items()}
    reserved = {"b", "f", "i", "r", "s"}
    invalid = [key for key, task in normalized.items() if len(key) != 1 or key in reserved or not task]
    if invalid:
        raise ValueError(f"Invalid or reserved task hotkeys: {invalid}")
    if default_key.lower() not in normalized:
        raise ValueError("task hotkey config default_key must exist in tasks")
    return TaskHotkeys(tasks=normalized, default_key=default_key.lower())


def _find_actor_checkpoint(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate if candidate.name in {"actor_critic.pt", "model.safetensors"} else None
    for item in (
        candidate / "actor_critic.pt",
        candidate / "checkpoints/last/actor_critic.pt",
        candidate / "pretrained_model/model.safetensors",
        candidate / "checkpoints/last/pretrained_model/model.safetensors",
    ):
        if item.is_file():
            return item
    return None


class Actor2PolicyRuntime(base_actor.ActorVLARuntime):
    """Use the policy already created by the base actor and optionally load its RL actor head."""

    def __init__(self, control: Actor2Control):
        self.control = control
        self.cfg = control.cfg
        self.policy = control.policy
        self.policy_cfg = self.cfg.policy
        if self.policy is None or self.policy_cfg is None or not self.policy_cfg.pretrained_path:
            raise RuntimeError("actor_2 policy was not initialized from policy.pretrained_path")

        self.policy_path = Path(self.policy_cfg.pretrained_path).expanduser()
        self.last_check_t = 0.0
        self.fingerprint = base_actor._fingerprint_policy_path(self.policy_path)
        self.preprocessor, self.postprocessor = base_actor.make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=str(self.policy_path),
            dataset_stats=control.dataset_meta.stats,
        )
        self._actor_actions: deque[torch.Tensor] = deque()
        self._actor_checkpoint = _find_actor_checkpoint(self.cfg.actor_checkpoint_path)
        self._actor_checkpoint_stamp: tuple[int, int] | None = None
        self._load_actor_head()
        control.runtime = self

    def _load_actor_head(self) -> None:
        if self._actor_checkpoint is None:
            logging.warning(
                "[ACTOR_2] No Actor weights found at %s; pure VLA mode remains active.",
                self.cfg.actor_checkpoint_path,
            )
            self.control.actor_available = False
            return
        if not hasattr(self.policy, "actor") or not hasattr(self.policy, "extract_rlt_features"):
            logging.warning(
                "[ACTOR_2] policy.type=%s has no compatible RLT actor interface; pure VLA only.",
                self.policy_cfg.type,
            )
            self.control.actor_available = False
            return

        if self._actor_checkpoint.name == "actor_critic.pt":
            metadata_path = self._actor_checkpoint.parent / "manifest.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing Actor checkpoint manifest: {metadata_path}")
            metadata = json.loads(metadata_path.read_text())
            expected = {
                "base_policy_path": str(self.policy_path.resolve()),
                "chunk_size": self.policy_cfg.chunk_size,
                "z_dim": self.policy_cfg.z_dim,
                "proprio_dim": self.policy_cfg.proprio_dim,
            }
            mismatch = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatch:
                raise ValueError(f"Actor checkpoint manifest mismatch: {mismatch}")
        else:
            metadata_path = self._actor_checkpoint.parent / "config.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            source_base = metadata.get("pretrained_path") or metadata.get("base_policy_path")
            if source_base and Path(source_base).expanduser().resolve() != self.policy_path.resolve():
                raise ValueError(
                    f"Actor checkpoint base {source_base} does not match VLA {self.policy_path}"
                )

        if self._actor_checkpoint.name == "model.safetensors":
            from safetensors import safe_open

            actor_state = {}
            with safe_open(self._actor_checkpoint, framework="pt", device="cpu") as checkpoint:
                for key in checkpoint.keys():
                    if key.startswith("actor."):
                        actor_state[key.removeprefix("actor.")] = checkpoint.get_tensor(key)
        else:
            state = torch.load(
                self._actor_checkpoint,
                map_location=self.policy_cfg.device,
                weights_only=True,
            )
            actor_state = state.get("actor") if isinstance(state, dict) else None
        if not isinstance(actor_state, dict):
            raise ValueError(f"Missing 'actor' state dict in {self._actor_checkpoint}")
        self.policy.actor.load_state_dict(actor_state, strict=True)
        self.policy.actor.eval()
        stat = self._actor_checkpoint.stat()
        self._actor_checkpoint_stamp = (stat.st_mtime_ns, stat.st_size)
        self.control.actor_available = True
        logging.info("[ACTOR_2] Loaded Actor head from %s", self._actor_checkpoint)

    def reload_if_changed(self) -> None:
        now = time.monotonic()
        if now - self.last_check_t < self.cfg.actor_vla_policy.policy_poll_s:
            return
        self.last_check_t = now
        checkpoint = _find_actor_checkpoint(self.cfg.actor_checkpoint_path)
        if checkpoint is None:
            self._actor_checkpoint = None
            self.control.actor_available = False
            self.control.use_actor = False
            return
        stat = checkpoint.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if checkpoint != self._actor_checkpoint or stamp != self._actor_checkpoint_stamp:
            self._actor_checkpoint = checkpoint
            self._load_actor_head()
            self.reset_action_state()

    def reset_action_state(self) -> None:
        self._actor_actions.clear()
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def select_action(
        self,
        observation_frame: dict[str, Any],
        robot_type: str | None,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.control.use_actor:
            return super().select_action(observation_frame, robot_type, device)
        if not self.control.actor_available:
            raise RuntimeError("VLA+Actor mode selected without a loaded Actor head")

        if not self._actor_actions:
            inference_device = get_safe_torch_device(self.policy_cfg.device)
            observation = prepare_observation_for_inference(
                copy(observation_frame),
                inference_device,
                getattr(self.cfg.env, "task", None),
                robot_type,
            )
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda")
                if inference_device.type == "cuda" and getattr(self.policy_cfg, "use_amp", False)
                else nullcontext(),
            ):
                batch = self.preprocessor(observation)
                features = self.policy.extract_rlt_features(batch)
                chunk = self.policy.actor.mean(
                    features["z_rl"], features["proprio"], features["ref_action"]
                )
                self._actor_actions.extend(chunk.transpose(0, 1))

        action = self.postprocessor(self._actor_actions.popleft())
        return action.to(device) if isinstance(action, torch.Tensor) else action


class Actor2KeyboardController(base_actor.KeyboardController):
    def __init__(self, control: Actor2Control):
        super().__init__()
        self.control = control
        self._actor2_events: Queue[str] = Queue()
        self._last_actor2_event_t = 0.0

    def push(self, key: str) -> None:
        normalized = key.lower() if len(key) == 1 else key
        if normalized == "b" or normalized in self.control.task_hotkeys.tasks:
            self._actor2_events.put(normalized)
        else:
            super().push(key)

    def poll(self, state):
        state = super().poll(state)
        while True:
            try:
                key = self._actor2_events.get_nowait()
            except Empty:
                break
            now = time.monotonic()
            if now - self._last_actor2_event_t < INTERVENTION_TOGGLE_COOLDOWN_S:
                continue
            self._last_actor2_event_t = now

            if key == "b":
                if not self.control.actor_available:
                    logging.error(
                        "[ACTOR_2] B ignored: Actor weights are unavailable; continuing pure VLA control."
                    )
                    continue
                self.control.use_actor = not self.control.use_actor
                if self.control.runtime is not None:
                    self.control.runtime.reset_action_state()
                if self.control.smoother is not None:
                    self.control.smoother.reset()
                logging.info(
                    "[ACTOR_2] Action mode switched to %s; cached actions cleared.",
                    "VLA+Actor" if self.control.use_actor else "VLA",
                )
                continue

            task = self.control.task_hotkeys.tasks[key]
            self.control.cfg.env.task = task
            if self.control.runtime is not None:
                self.control.runtime.reset_action_state()
            if self.control.smoother is not None:
                self.control.smoother.reset()
            state.reset_episode = True
            state.rerecord_episode = True
            state.exit_episode = True
            logging.info(
                "[ACTOR_2] Task key %s selected %r; discarding episode and returning home.",
                key,
                task,
            )
        return state


@contextmanager
def _install_actor2_runtime(control: Actor2Control):
    original_make_policy = base_actor.make_policy
    original_runtime = base_actor.ActorVLARuntime
    original_keyboard = base_actor.KeyboardController
    original_smoother = base_actor._OnlinePolicyActionSmoother

    def make_policy_from_metadata(*args, **kwargs):
        policy_cfg = kwargs.get("cfg", args[0] if args else None)
        if policy_cfg is control.cfg.policy:
            policy = factory_make_policy(cfg=policy_cfg, ds_meta=control.dataset_meta, env_cfg=None)
            for name in ("critic_ensemble", "critic_target"):
                critic = getattr(policy, name, None)
                if critic is not None:
                    critic.to("cpu")
            control.policy = policy.eval()
            return control.policy
        return original_make_policy(*args, **kwargs)

    def make_runtime(_cfg):
        return Actor2PolicyRuntime(control)

    def make_keyboard():
        return Actor2KeyboardController(control)

    def make_smoother(*args, **kwargs):
        control.smoother = original_smoother(*args, **kwargs)
        return control.smoother

    base_actor.make_policy = make_policy_from_metadata
    base_actor.ActorVLARuntime = make_runtime
    base_actor.KeyboardController = make_keyboard
    base_actor._OnlinePolicyActionSmoother = make_smoother
    try:
        yield
    finally:
        base_actor.make_policy = original_make_policy
        base_actor.ActorVLARuntime = original_runtime
        base_actor.KeyboardController = original_keyboard
        base_actor._OnlinePolicyActionSmoother = original_smoother


@parser.wrap()
def actor_cli(cfg: Actor2PipelineConfig):
    display_pid = not base_actor.use_threads(cfg)
    init_logging(display_pid=display_pid)
    sys.excepthook = lambda exc_type, exc_value, exc_traceback: _log_unhandled_exception(
        "main thread", exc_type, exc_value, exc_traceback
    )
    threading.excepthook = lambda args: _log_unhandled_exception(
        f"thread {args.thread.name if args.thread else '<unknown>'}",
        args.exc_type,
        args.exc_value,
        args.exc_traceback,
    )

    task_hotkeys = load_task_hotkeys(cfg.task_hotkeys_path)
    if cfg.dataset is None:
        raise ValueError("actor_2 requires dataset metadata for policy loading and task validation")
    dataset_meta = LeRobotDatasetMetadata(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    valid_tasks = set(dataset_meta.tasks.index)
    unknown = sorted(set(task_hotkeys.tasks.values()) - valid_tasks)
    if unknown:
        raise ValueError(f"Task hotkey config contains tasks missing from meta/tasks.parquet: {unknown}")
    if cfg.env is None:
        raise ValueError("actor_2 requires an environment")
    cfg.env.task = task_hotkeys.tasks[task_hotkeys.default_key]
    cfg.validate()
    if not cfg.actor_vla_policy.enabled:
        raise ValueError("actor_2 requires actor_vla_policy.enabled=true")

    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_2_{cfg.job_name}.log")
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info("Actor 2 logging initialized, writing to %s", log_file)

    if display_pid:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
    logging.info(
        "[ACTOR_2] Task hotkeys: %s; default=%s (%s); B toggles VLA/VLA+Actor.",
        task_hotkeys.tasks,
        task_hotkeys.default_key,
        cfg.env.task,
    )

    shutdown_event = ProcessSignalHandler(
        base_actor.use_threads(cfg), display_pid=display_pid
    ).shutdown_event
    control = Actor2Control(cfg, task_hotkeys, dataset_meta)
    with _install_actor2_runtime(control):
        if cfg.actor_only.enabled:
            base_actor.run_actor_only(cfg=cfg, shutdown_event=shutdown_event)
        else:
            base_actor.run_actor_online(cfg=cfg, shutdown_event=shutdown_event)


if __name__ == "__main__":
    actor_cli()
