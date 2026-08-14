#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Actor server runner for distributed HILSerl robot policy training.

This script implements the actor component of the distributed HILSerl architecture.
It executes the policy in the robot environment, collects experience,
and sends transitions to the learner server for policy updates.

Examples of usage:

- Start an actor server for real robot training with human-in-the-loop intervention:
```bash
python -m lerobot.onlineRL_evoRL.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: The actor server requires a running learner server to connect to. Ensure the learner
server is started before launching the actor.

**NOTE**: Human intervention is key to HILSerl training. Press the upper right trigger button on the
gamepad to take control of the robot during training. Initially intervene frequently, then gradually
reduce interventions as the policy improves.

**WORKFLOW**:
1. Determine robot workspace bounds using `lerobot-find-joint-limits`
2. Record demonstrations with `gym_manipulator.py` in record mode
3. Process the dataset and determine camera crops with `crop_dataset_roi.py`
4. Start the learner server with the training configuration
5. Start this actor server with the same configuration
6. Use human interventions to guide policy learning

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from queue import Empty
from typing import Any

import grpc
import torch
from torch import nn
from torch.multiprocessing import Event, Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.processor import TransitionKey
from lerobot.onlineRL_evoRL.process import ProcessSignalHandler
from lerobot.onlineRL_evoRL.queue import get_last_item_from_queue
from lerobot.robots import piper_follower, so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, piper_leader, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.control_utils import predict_action
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from .keyboard_control import (
    EPISODE_FAILURE,
    EPISODE_SUCCESS,
    KeyboardController,
    KeyboardState,
    start_keyboard_listener,
    stop_keyboard_listener,
)
from .piper_episode_control import home_arms_to_default, hold_arms_current_pose

INTERVENTION_STATE_POLICY = 0.0
INTERVENTION_STATE_ACTIVE = 1.0
INTERVENTION_STATE_RELEASE = 2.0


@dataclass
class CheckpointFingerprint:
    resolved_path: str
    files: dict[str, tuple[int, int]]


def _fingerprint_policy_path(policy_path: str | Path) -> CheckpointFingerprint:
    path = Path(policy_path).expanduser()
    resolved = path.resolve()
    files: dict[str, tuple[int, int]] = {}
    candidates: list[Path]
    if resolved.is_file():
        candidates = [resolved]
    else:
        patterns = [
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "policy.safetensors",
            "*.safetensors",
            "*.bin",
            "*preprocessor*.json",
            "*postprocessor*.json",
        ]
        seen: set[Path] = set()
        candidates = []
        for pattern in patterns:
            for candidate in resolved.glob(pattern):
                if candidate.is_file() and candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
    for candidate in sorted(candidates):
        stat = candidate.stat()
        key = str(candidate.relative_to(resolved) if resolved.is_dir() else candidate.name)
        files[key] = (int(stat.st_mtime_ns), int(stat.st_size))
    return CheckpointFingerprint(resolved_path=str(resolved), files=files)


def _as_jsonable(value: Any, *, max_items: int = 32) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return [float(x) for x in value.flatten()[:max_items].tolist()]
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v, max_items=max_items) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v, max_items=max_items) for v in list(value)[:max_items]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitize_path_component(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "value"


def _tensor_to_pil_image(value: Any):
    from PIL import Image
    import numpy as np

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        while value.ndim > 3 and value.shape[0] == 1:
            value = value.squeeze(0)
        array = value.numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        if array.size and float(array.max()) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        return Image.fromarray(array[..., 0])
    if array.ndim == 3 and array.shape[-1] >= 3:
        return Image.fromarray(array[..., :3])
    raise ValueError(f"Unsupported image shape: {array.shape}")


class ActorEpisodeWriter:
    def __init__(self, cfg: TrainRLServerPipelineConfig):
        output_dir = cfg.actor_only.episode_output_dir
        if output_dir is None:
            output_dir = str(Path(cfg.output_dir) / "actor_episodes")
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = cfg.actor_only.save_episode_images
        self.save_viewer = cfg.actor_only.save_episode_viewer
        self.episode_index = self._next_episode_index()

    def _next_episode_index(self) -> int:
        max_index = -1
        for child in self.output_dir.glob("episode_*"):
            if child.is_dir():
                try:
                    max_index = max(max_index, int(child.name.split("_")[-1]))
                except ValueError:
                    continue
        return max_index + 1

    def save_episode(self, *, transitions: list[Transition], metadata: dict[str, Any]) -> Path:
        episode_dir = self.output_dir / f"episode_{self.episode_index:06d}"
        self.episode_index += 1
        episode_dir.mkdir(parents=True, exist_ok=False)
        cpu_transitions = [move_transition_to_device(transition=tr, device="cpu") for tr in transitions]
        torch.save(cpu_transitions, episode_dir / "transitions.pt")
        frames = self._write_frames(episode_dir, cpu_transitions)
        payload = {**metadata, "frames": len(frames)}
        (episode_dir / "metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        (episode_dir / "frames.json").write_text(json.dumps(frames, indent=2, ensure_ascii=False))
        if self.save_viewer:
            self._write_viewer(episode_dir, payload, frames)
        logging.info("[ACTOR] Saved actor-only episode to %s", episode_dir)
        return episode_dir

    def _write_frames(self, episode_dir: Path, transitions: list[Transition]) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for step, transition in enumerate(transitions):
            state = transition.get("state", {})
            image_paths: dict[str, str] = {}
            if self.save_images:
                for key, value in state.items():
                    if "image" not in str(key) and "pixels" not in str(key):
                        continue
                    try:
                        image = _tensor_to_pil_image(value)
                    except Exception as exc:
                        logging.debug("Skip non-image observation %s: %s", key, exc)
                        continue
                    key_dir = Path("images") / _sanitize_path_component(str(key))
                    target_dir = episode_dir / key_dir
                    target_dir.mkdir(parents=True, exist_ok=True)
                    rel_path = key_dir / f"{step:06d}.jpg"
                    image.save(episode_dir / rel_path, quality=90)
                    image_paths[str(key)] = str(rel_path)
            frames.append(
                {
                    "step": step,
                    "reward": _as_jsonable(transition.get("reward")),
                    "done": bool(transition.get("done", False)),
                    "truncated": bool(transition.get("truncated", False)),
                    "action": _as_jsonable(transition.get("action")),
                    "state": _as_jsonable({k: v for k, v in state.items() if "image" not in str(k) and "pixels" not in str(k)}),
                    "complementary_info": _as_jsonable(transition.get("complementary_info", {})),
                    "images": image_paths,
                }
            )
        return frames

    def _write_viewer(self, episode_dir: Path, metadata: dict[str, Any], frames: list[dict[str, Any]]) -> None:
        data = json.dumps({"metadata": metadata, "frames": frames}, ensure_ascii=False)
        template = """<!doctype html>
<html>
<head>
<meta charset=\"utf-8\">
<title>Actor Episode Viewer</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 0; background: #111; color: #eee; }
header { padding: 16px 20px; border-bottom: 1px solid #333; }
main { display: grid; grid-template-columns: 1fr 360px; gap: 16px; padding: 16px; }
.images { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
figure { margin: 0; background: #1b1b1b; border: 1px solid #333; }
img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #050505; }
figcaption { padding: 8px 10px; color: #bbb; font-size: 13px; }
.panel { background: #1b1b1b; border: 1px solid #333; padding: 12px; }
input[type=range] { width: 100%; }
pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; color: #ddd; }
.meta { color: #bbb; font-size: 13px; }
</style>
</head>
<body>
<header><div class=\"meta\" id=\"meta\"></div></header>
<main>
  <section><div class=\"images\" id=\"images\"></div></section>
  <aside class=\"panel\"><input id=\"slider\" type=\"range\" min=\"0\" max=\"0\" value=\"0\"><pre id=\"details\"></pre></aside>
</main>
<script>
const DATA = __DATA__;
const slider = document.getElementById('slider');
const details = document.getElementById('details');
const images = document.getElementById('images');
const meta = document.getElementById('meta');
meta.textContent = `task: ${DATA.metadata.task || ''} | outcome: ${DATA.metadata.episode_outcome || 'none'} | reward: ${DATA.metadata.episodic_reward} | steps: ${DATA.metadata.frames}`;
slider.max = Math.max(DATA.frames.length - 1, 0);
function render() {
  const frame = DATA.frames[Number(slider.value)] || {images: {}};
  images.innerHTML = '';
  const entries = Object.entries(frame.images || {});
  if (entries.length === 0) { images.textContent = 'No saved camera frames.'; }
  for (const [key, src] of entries) {
    const fig = document.createElement('figure');
    fig.innerHTML = `<img src=\"${src}\"><figcaption>${key}</figcaption>`;
    images.appendChild(fig);
  }
  details.textContent = JSON.stringify(frame, null, 2);
}
slider.addEventListener('input', render);
render();
</script>
</body>
</html>
"""
        (episode_dir / "viewer.html").write_text(template.replace("__DATA__", data))


class ActorVLARuntime:
    def __init__(self, cfg: TrainRLServerPipelineConfig):
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
        self.ds_meta = None
        self.reload()

    def reload(self) -> None:
        if not self.policy_path.exists():
            raise FileNotFoundError(f"VLA policy path does not exist: {self.policy_path}")
        policy_cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        policy_cfg.pretrained_path = self.policy_path
        if self.cfg.policy is not None and getattr(self.cfg.policy, "device", None):
            policy_cfg.device = self.cfg.policy.device
        self.ds_meta = self._make_dataset_meta()
        policy = make_policy(policy_cfg, ds_meta=self.ds_meta).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(policy_cfg.pretrained_path),
            dataset_stats=self.ds_meta.stats,
            preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
        )
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()
        self.policy_cfg = policy_cfg
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.fingerprint = _fingerprint_policy_path(self.policy_path)
        logging.info("[ACTOR] Loaded VLA policy from %s", self.policy_path)

    def _make_dataset_meta(self) -> LeRobotDatasetMetadata:
        if self.cfg.dataset is None or self.cfg.dataset.repo_id is None:
            raise ValueError("actor_vla_policy requires cfg.dataset.repo_id/root to build policy metadata")
        return LeRobotDatasetMetadata(repo_id=self.cfg.dataset.repo_id, root=self.cfg.dataset.root)

    def reload_if_changed(self) -> None:
        now = time.monotonic()
        if now - self.last_check_t < self.cfg.actor_vla_policy.policy_poll_s:
            return
        self.last_check_t = now
        current = _fingerprint_policy_path(self.policy_path)
        if self.fingerprint is not None and current == self.fingerprint:
            return
        logging.info("[ACTOR] VLA policy checkpoint changed; reloading before next episode.")
        self.reload()

    def select_action(self, env, device: torch.device) -> torch.Tensor:
        if self.policy is None or self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("VLA runtime is not loaded")
        robot = getattr(env, "robot", None)
        if robot is None:
            raise ValueError("VLA actor policy requires an environment with a robot")
        raw_observation = robot.get_observation()
        action = predict_action(
            observation=raw_observation,
            policy=self.policy,
            device=get_safe_torch_device(self.policy_cfg.device),
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=getattr(self.policy_cfg, "use_amp", False),
            task=getattr(self.cfg.env, "task", None),
            robot_type=getattr(robot, "robot_type", None),
        )
        if isinstance(action, torch.Tensor):
            action = action.to(device)
        return action


# Main entry point


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info("Actor logging initialized, writing to %s", log_file)

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    if cfg.actor_only.enabled:
        run_actor_only(cfg=cfg, shutdown_event=shutdown_event)
    else:
        run_actor_online(cfg=cfg, shutdown_event=shutdown_event)


def _get_concurrency_entity(cfg: TrainRLServerPipelineConfig):
    if use_threads(cfg):
        from threading import Thread

        return Thread
    from multiprocessing import Process

    return Process


def run_actor_only(cfg: TrainRLServerPipelineConfig, shutdown_event: Event):  # type: ignore
    logging.info("[ACTOR] Starting actor-only mode; learner connection is disabled.")
    if cfg.actor_vla_policy.enabled:
        logging.info("[ACTOR] actor_vla_policy.enabled=true: actions will be generated by VLA.")
    writer = ActorEpisodeWriter(cfg)
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    try:
        act_with_policy(
            cfg=cfg,
            shutdown_event=shutdown_event,
            parameters_queue=parameters_queue,
            transitions_queue=transitions_queue,
            interactions_queue=interactions_queue,
            actor_episode_writer=writer,
        )
    finally:
        transitions_queue.close()
        interactions_queue.close()
        parameters_queue.close()
        transitions_queue.cancel_join_thread()
        interactions_queue.cancel_join_thread()
        parameters_queue.cancel_join_thread()


def run_actor_online(cfg: TrainRLServerPipelineConfig, shutdown_event: Event):  # type: ignore
    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")
    if cfg.actor_vla_policy.enabled:
        logging.info(
            "[ACTOR] actor_vla_policy.enabled=true: actions are generated by VLA; "
            "received SAC actor updates are not used for action selection."
        )

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = _get_concurrency_entity(cfg)
    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )
    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )
    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue,
    )
    logging.info("[ACTOR] Policy process joined")

    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")


# Core algorithm functions


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
    actor_episode_writer: ActorEpisodeWriter | None = None,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.
    此函数在环境中部署策略，收集交互数据并将其推送到队列中，以便流式传输给learner。
    一个episode结束后，从队列中检索从learner收到的更新网络参数，并将其加载到网络中。

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        parameters_queue: Queue to receive updated network parameters from the learner.
        transitions_queue: Queue to send transitions to the learner.
        interactions_queue: Queue to send interactions to the learner.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    logging.info("make_env online")

    # 创建在线环境。可能是真实机器人，也可能是 gym_hil 仿真。
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    # 创建 observation processor 和 action processor。包裹各种控制器，如按键控制、超时控制等
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a SACPolicy object through the port, we create a policy instance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    # 本地创建 SACPolicy。actor 不直接接收整个 policy 对象，而是本地建模型，之后只加载 learner 推来的 state dict，Learner 每隔 n 步发送更新后的参数，以更新 Actor 的参数。
    # TODO 我们训练时使用pi05或smolvla时可以考虑type来适配，目前可能不适配pi05
    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    # VLA推理,将预训练的VLA模型加载进来，越过了make_policy的推理
    # TODO 后续需要将模型与make_policy统一起来处理，代码中有针对不同模型的处理方式
    vla_runtime = ActorVLARuntime(cfg) if cfg.actor_vla_policy.enabled else None

    # 获取初始观测
    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation， 把初始 observation 包成 transition。
    transition = create_transition(observation=obs, info=info)
    # 经过 env_processor，变成 policy 可用的 tensor/batch/device 格式。
    transition = env_processor(transition)

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    # 用于暂存当前 episode 的所有 transitions
    list_transition_to_send_to_learner = []
    episode_intervention = False
    # Add counters for intervention rate calculation
    episode_intervention_steps = 0
    episode_total_steps = 0

    policy_timer = TimerManager("Policy inference", log=False)
    keyboard_controller = KeyboardController()
    keyboard_state = KeyboardState()
    keyboard_listener = start_keyboard_listener(keyboard_controller)
    intervention_state = INTERVENTION_STATE_POLICY

    def set_teleop_manual_control(enabled: bool) -> None:
        if teleop_device is None or not hasattr(teleop_device, "set_manual_control"):
            return
        try:
            teleop_device.set_manual_control(enabled)
        except Exception:
            logging.exception("Failed to switch teleop manual-control mode to %s", enabled)

    def reset_policy_runtime() -> None:
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()

    def action_to_robot_action_dict(action) -> dict[str, float] | None:
        robot = getattr(online_env, "robot", None)
        bus = getattr(robot, "bus", None)
        motors = getattr(bus, "motors", None)
        if motors is None:
            return None
        if isinstance(action, dict):
            return {str(k): float(v) for k, v in action.items()}
        if isinstance(action, torch.Tensor):
            values = action.detach().squeeze().cpu().tolist()
        else:
            values = action
        if not isinstance(values, list):
            return None
        return {f"{name}.pos": float(values[i]) for i, name in enumerate(motors.keys()) if i < len(values)}

    def sync_policy_action_to_teleop(action) -> None:
        if teleop_device is None or not hasattr(teleop_device, "send_feedback"):
            return
        feedback = action_to_robot_action_dict(action)
        if feedback is None:
            logging.debug("Skip policy-to-teleop sync: action cannot be converted to robot dict.")
            return
        try:
            teleop_device.send_feedback(feedback)
        except Exception:
            logging.exception("Failed to sync policy action to teleop leader.")

    def reset_episode_state() -> None:
        nonlocal sum_reward_episode, list_transition_to_send_to_learner
        nonlocal episode_intervention, episode_intervention_steps, episode_total_steps
        nonlocal transition, keyboard_state, intervention_state
        sum_reward_episode = 0.0
        list_transition_to_send_to_learner = []
        episode_intervention = False
        episode_intervention_steps = 0
        episode_total_steps = 0
        keyboard_state = KeyboardState()
        intervention_state = INTERVENTION_STATE_POLICY
        set_teleop_manual_control(False)
        obs, info = online_env.reset()
        env_processor.reset()
        action_processor.reset()
        transition = create_transition(observation=obs, info=info)
        transition = env_processor(transition)

    def finish_episode(
        *,
        interaction_step: int,
        should_send: bool,
        reset_to_home: bool,
        hold_current_pose: bool,
        episode_outcome: str | None,
        timeout: bool,
    ) -> None:
        nonlocal list_transition_to_send_to_learner
        logging.info(
            "[ACTOR] Global step %s: Episode reward=%s outcome=%s send=%s timeout=%s",
            interaction_step,
            sum_reward_episode,
            episode_outcome,
            should_send,
            timeout,
        )
        if vla_runtime is None:
            update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)
        elif cfg.actor_vla_policy.reload_on_episode_boundary:
            vla_runtime.reload_if_changed()

        stats = get_frequency_stats(policy_timer)
        intervention_rate = 0.0
        if episode_total_steps > 0:
            intervention_rate = episode_intervention_steps / episode_total_steps
        episode_metadata = {
            "Episodic reward": sum_reward_episode,
            "episodic_reward": sum_reward_episode,
            "Interaction step": interaction_step,
            "Episode intervention": int(episode_intervention),
            "Intervention rate": intervention_rate,
            "intervention_rate": intervention_rate,
            "Episode outcome": episode_outcome or "timeout" if timeout else episode_outcome or "none",
            "episode_outcome": episode_outcome or "timeout" if timeout else episode_outcome or "none",
            "timeout": timeout,
            "task": getattr(cfg.env, "task", None),
            "actor_policy_type": "vla" if vla_runtime is not None else "sac",
            "actor_policy_path": str(vla_runtime.policy_path) if vla_runtime is not None else None,
            "actor_policy_fingerprint": _as_jsonable(vla_runtime.fingerprint.__dict__) if vla_runtime is not None and vla_runtime.fingerprint is not None else None,
            **stats,
        }

        if should_send and len(list_transition_to_send_to_learner) > 0:
            if actor_episode_writer is not None:
                actor_episode_writer.save_episode(
                    transitions=list_transition_to_send_to_learner,
                    metadata=episode_metadata,
                )
            else:
                push_transitions_to_transport_queue(
                    transitions=list_transition_to_send_to_learner,
                    transitions_queue=transitions_queue,
                )
                interactions_queue.put(python_object_to_bytes(episode_metadata))
        else:
            logging.info("[ACTOR] Discard current episode transitions; nothing sent or saved.")

        policy_timer.reset()
        if reset_to_home:
            home_arms_to_default(getattr(online_env, "robot", None), teleop_device)
        elif hold_current_pose:
            hold_arms_current_pose(getattr(online_env, "robot", None), teleop_device)
        reset_episode_state()

    # 初始时默认由VLA控制
    set_teleop_manual_control(False)
    try:
        for interaction_step in range(cfg.policy.online_steps):
            start_time = time.perf_counter()
            if shutdown_event.is_set():
                logging.info("[ACTOR] Shutting down act_with_policy")
                return

            keyboard_controller.poll(keyboard_state)
            if keyboard_state.stop:
                logging.info("[ACTOR] Stop requested from keyboard.")
                return

            if keyboard_state.toggle_intervention:
                if intervention_state == INTERVENTION_STATE_POLICY:
                    intervention_state = INTERVENTION_STATE_ACTIVE
                    set_teleop_manual_control(True)
                    logging.info("[ACTOR] Intervention enabled: teleop actions override policy execution.")
                else:
                    intervention_state = INTERVENTION_STATE_RELEASE
                    set_teleop_manual_control(False)
                    reset_policy_runtime()
                    logging.info("[ACTOR] Intervention release requested: returning control to policy.")

            observation = {
                k: v
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in cfg.policy.input_features
            }

            with policy_timer:
                if vla_runtime is not None:
                    # TODO 这里后续需要统一用make_policy的模型获取动作
                    action = vla_runtime.select_action(env=online_env, device=device)
                else:
                    action = policy.select_action(batch=observation)
            policy_fps = policy_timer.fps_last
            log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            next_observation = {
                k: v
                for k, v in new_transition[TransitionKey.OBSERVATION].items()
                if k in cfg.policy.input_features
            }

            is_intervention = intervention_state == INTERVENTION_STATE_ACTIVE
            executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
            if not is_intervention:
                sync_policy_action_to_teleop(executed_action)

            reward = float(new_transition[TransitionKey.REWARD])
            done = bool(new_transition.get(TransitionKey.DONE, False))
            truncated = bool(new_transition.get(TransitionKey.TRUNCATED, False))
            episode_outcome = keyboard_state.episode_outcome
            if episode_outcome == EPISODE_SUCCESS:
                reward = 1.0
                done = True
                truncated = False
            elif episode_outcome == EPISODE_FAILURE:
                reward = 0.0
                done = True
                truncated = False

            sum_reward_episode += float(reward)
            episode_total_steps += 1
            if is_intervention:
                episode_intervention = True
                episode_intervention_steps += 1

            complementary_info = {
                "discrete_penalty": torch.tensor(
                    [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
                ),
                "is_intervention": torch.tensor([float(is_intervention)]),
                "intervention_state": torch.tensor([float(intervention_state)]),
                "success": torch.tensor([float(episode_outcome == EPISODE_SUCCESS)]),
                "failure": torch.tensor([float(episode_outcome == EPISODE_FAILURE)]),
                "actor_policy_is_vla": torch.tensor([float(vla_runtime is not None)]),
            }
            state_to_store = transition[TransitionKey.OBSERVATION] if actor_episode_writer is not None else observation
            next_state_to_store = (
                new_transition[TransitionKey.OBSERVATION] if actor_episode_writer is not None else next_observation
            )
            list_transition_to_send_to_learner.append(
                Transition(
                    state=state_to_store,
                    action=executed_action,
                    reward=reward,
                    next_state=next_state_to_store,
                    done=done,
                    truncated=truncated,
                    complementary_info=complementary_info,
                )
            )

            transition = new_transition

            discard_episode = keyboard_state.rerecord_episode or keyboard_state.reset_episode
            keyboard_episode_end = keyboard_state.exit_episode or episode_outcome is not None
            if done or truncated or keyboard_episode_end:
                should_send = not discard_episode
                reset_to_home = keyboard_state.reset_episode
                hold_current_pose = episode_outcome in {EPISODE_SUCCESS, EPISODE_FAILURE} or keyboard_state.rerecord_episode
                finish_episode(
                    interaction_step=interaction_step,
                    should_send=should_send,
                    reset_to_home=reset_to_home,
                    hold_current_pose=hold_current_pose,
                    episode_outcome=episode_outcome,
                    timeout=bool(truncated and episode_outcome is None and not discard_episode),
                )

            if intervention_state == INTERVENTION_STATE_RELEASE:
                intervention_state = INTERVENTION_STATE_POLICY

            if cfg.env.fps is not None:
                dt_time = time.perf_counter() - start_time
                precise_sleep(max(1 / cfg.env.fps - dt_time, 0.0))
    finally:
        stop_keyboard_listener(keyboard_listener)


#  Communication Functions - Group all gRPC/messaging functions


def establish_learner_connection(
    stub: services_pb2_grpc.LearnerServiceStub,
    shutdown_event: Event,  # type: ignore
    attempts: int = 30,
):
    """Establish a connection with the learner.

    Args:
        stub (services_pb2_grpc.LearnerServiceStub): The stub to use for the connection.
        shutdown_event (Event): The event to check if the connection should be established.
        attempts (int): The number of attempts to establish the connection.
    Returns:
        bool: True if the connection is established, False otherwise.
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # Force a connection attempt and check state
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    """
    Returns a client for the learner service.

    GRPC uses HTTP/2, which is a binary protocol and multiplexes requests over a single connection.
    So we need to create only one client and reuse it.
    """

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel

# Actor 接收 learner 参数的后台逻辑
def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
):
    """Receive parameters from the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        parameters_queue (Queue): The queue to receive the parameters.
        shutdown_event (Event): The event to check if the process should shutdown.
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: Queue,
    shutdown_event: any,  # Event,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends transitions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Transition Data:
        - A batch of transitions (observation, action, reward, next observation) is collected.
        - Transitions are moved to the CPU and serialized using PyTorch.
        - The serialized data is wrapped in a `services_pb2.Transition` message and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor transitions process logging initialized")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends interactions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Interaction Messages:
        - Contains useful statistics about episodic rewards and policy timings.
        - The message is serialized using `pickle` and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor interactions process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


# 后台发送流
def transitions_stream(shutdown_event: Event, transitions_queue: Queue, timeout: float) -> services_pb2.Empty:  # type: ignore
    while not shutdown_event.is_set():
        try:
            # 从 transitions_queue 取消息
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        # 通过 send_bytes_in_chunks 发给 learner 的 SendTransitions gRPC 接口
        yield from send_bytes_in_chunks(
            message, services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()


def interactions_stream(
    shutdown_event: Event,
    interactions_queue: Queue,
    timeout: float,  # type: ignore
) -> services_pb2.Empty:
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


#  Policy functions

# 更新 actor 的 policy 参数
def update_policy_parameters(policy: SACPolicy, parameters_queue: Queue, device):
    # 从 parameters_queue 取最新参数
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # TODO: check encoder parameter synchronization possible issues:
        # 1. When shared_encoder=True, we're loading stale encoder params from actor's state_dict
        #    instead of the updated encoder params from critic (which is optimized separately)
        # 2. When freeze_vision_encoder=True, we waste bandwidth sending/loading frozen params
        # 3. Need to handle encoder params correctly for both actor and discrete_critic
        # Potential fixes:
        # - Send critic's encoder state when shared_encoder=True
        # - Skip encoder params entirely when freeze_vision_encoder=True
        # - Ensure discrete_critic gets correct encoder state (currently uses encoder_critic)

        # Load actor state dict 加载 actor state dict
        actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
        policy.actor.load_state_dict(actor_state_dict)

        # Load discrete critic if present 如果有 discrete critic，也加载
        if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
            discrete_critic_state_dict = move_state_dict_to_device(
                state_dicts["discrete_critic"], device=device
            )
            policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
            logging.info("[ACTOR] Loaded discrete critic parameters from Learner.")


#  Utilities functions

# transition发送函数
def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """Send transitions to learner in smaller chunks to avoid network issues.

    Args:
        transitions: List of transitions to send
        message_queue: Queue to send messages to learner
        chunk_size: Size of each chunk to send
    """
    transition_to_send_to_learner = []
    for transition in transitions:
        # 把transition移到cpu上，避免显存占用过大
        tr = move_transition_to_device(transition=transition, device="cpu")
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"Found NaN values in transition {key}")

        transition_to_send_to_learner.append(tr)
    # 序列化后放入 transitions_queue
    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))


def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """Get the frequency statistics of the policy.

    Args:
        timer (TimerManager): The timer with collected metrics.

    Returns:
        dict[str, float]: The frequency statistics of the policy.
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    # 根据配置决定 actor 里的并发方式是用 线程 还是用 进程。
    return cfg.policy.concurrency.actor == "threads"


if __name__ == "__main__":
    actor_cli()
