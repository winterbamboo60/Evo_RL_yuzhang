# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Core recording loop used by `lerobot_record.py`."""

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch

from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.robots import Robot
from lerobot.scripts.recording_hil import (
    INTERVENTION_STATE_ACTIVE,
    INTERVENTION_STATE_POLICY,
    INTERVENTION_STATE_RELEASE,
    ACPInferenceConfig,
    PolicySyncDualArmExecutor,
    _capture_policy_runtime_state,
    _predict_policy_action_with_acp_inference,
)
from lerobot.teleoperators import Teleoperator, koch_leader, omx_leader, so_leader
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.recording_annotations import resolve_collector_policy_id
from lerobot.utils.recording_events import EventConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import log_rerun_data

T = TypeVar("T")


@dataclass(frozen=True)
class _RTCChunkPrediction:
    """One normalized action chunk produced for the RTC execution queue."""

    actions: torch.Tensor
    apply_prefix_fusion: bool = False


RTCChunkPredictor = Callable[
    [dict[str, np.ndarray], int, torch.Tensor | None, str | None, str | None],
    torch.Tensor | _RTCChunkPrediction,
]


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


class _OnlinePolicyActionSmoother:
    """逐拍在线滑窗均值平滑：维护最近 window 拍动作的滚动缓冲，输出窗口算术平均。

    对齐 inference 端的 ArmSmoother（窗口=9 的滑动均值，且会平滑包括夹爪在内的全部动作
    维度）。record_loop 每拍只拿到一个动作，无法对整段 chunk 卷积，这里改成等价的因果滑动
    均值（最近 window 拍的均值）；window 取奇数与 inference 一致，policy.reset 时清空缓冲。
    """

    def __init__(self, action_keys, window: int = 9):
        self.action_keys = list(action_keys)
        self.window = window if window % 2 == 1 else window + 1
        self._buffers: dict[str, deque] = {k: deque(maxlen=self.window) for k in self.action_keys}

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf.clear()

    def __call__(self, action: dict) -> dict:
        out = dict(action)  # 不改原 dict，保证原始策略动作可单独入库
        for k in self.action_keys:
            if k in action:
                buf = self._buffers[k]
                buf.append(float(action[k]))
                out[k] = sum(buf) / len(buf)
        return out


def _attenuate_gripper_value(v: float) -> float:
    """夹爪开合非线性衰减，与 inference / piper_follower 硬件公式逐字一致。

    gripper.pos 以 unit 为量纲，硬件按 milli(×1000，int(round())) 处理并 clip 到 101100；分段：
    ratio>=0.4 线性、0.04~0.4 三次曲线、<0.04 归零。返回值仍为 unit 量纲。

    注意：衰减逻辑已从 piper_follower.send_action 上移到此处（避免二次衰减），send_action 只对
    实际下发动作做 clip，不再做非线性衰减。
    """
    raw = int(round(float(v) * 1000.0))  # 等价 unit_to_milli
    ratio = float(np.clip(raw, 0, 101100)) / 101100
    if ratio >= 0.4:
        attenuated_raw = int(ratio * 101100)
    elif ratio >= 0.04:
        attenuated_raw = int(((ratio - 0.04) / 0.36) ** 3 * 0.4 * 101100)
    else:
        attenuated_raw = 0
    return attenuated_raw / 1000.0  # 等价 milli_to_unit


def _postprocess_policy_action(
    action: dict,
    smoother: "_OnlinePolicyActionSmoother",
    joint_limits: dict | None = None,
    gripper_key: str = "gripper.pos",
    attenuate_gripper: bool = True,
) -> dict:
    """对策略生成的动作做质量后处理：滑窗平滑（含夹爪）+ 可选关节限位 + 夹爪非线性衰减。

    与 inference 一致的顺序：先对全部维度滑窗平滑（含夹爪），再对夹爪做非线性衰减。
    """
    action = smoother(action)  # ① 滑窗平滑（与 inference ArmSmoother 对齐，平滑全部动作维度含夹爪）
    if joint_limits:  # ⑤ 可选关节限位
        action = dict(action)
        for k, (lo, hi) in joint_limits.items():
            if k in action:
                action[k] = float(np.clip(action[k], lo, hi))
    if attenuate_gripper and gripper_key in action:  # ③ 夹爪非线性衰减（与 inference 逐字一致）
        action = dict(action)
        action[gripper_key] = _attenuate_gripper_value(action[gripper_key])
    return action


class _RTCActionChunkRunner:
    """Generate RTC chunks on one worker while the main loop consumes the current chunk."""

    def __init__(
        self,
        *,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        rtc: RTCConfig,
        fps: int,
        queue_threshold: int,
        task: str | None,
        robot_type: str | None,
        chunk_predictor: RTCChunkPredictor | None = None,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.rtc = rtc
        self.fps = fps
        self.queue_threshold = queue_threshold
        self.task = task
        self.robot_type = robot_type
        self.chunk_predictor = chunk_predictor
        self.device = get_safe_torch_device(policy.config.device)
        self.action_queue = ActionQueue(rtc)
        self.latency_tracker = LatencyTracker()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-inference")
        self.future: Future[
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool, float]
        ] | None = None
        self.future_generation = 0
        self.action_index_before_inference = 0
        self.generation = 0
        self.closed = False

    def _infer_chunk(
        self,
        observation_frame: dict[str, np.ndarray],
        inference_delay: int,
        previous_actions: torch.Tensor | None,
        task: str | None,
        robot_type: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool, float]:
        started_at = time.perf_counter()
        amp_context = (
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.policy.config.use_amp
            else nullcontext()
        )
        with torch.no_grad(), amp_context:
            if self.chunk_predictor is None:
                observation = prepare_observation_for_inference(
                    observation_frame, self.device, task, robot_type
                )
                observation = self.preprocessor(observation)
                prediction = _RTCChunkPrediction(
                    actions=self.policy.predict_action_chunk(
                        observation,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=previous_actions,
                    )
                )
            else:
                prediction = self.chunk_predictor(
                    observation_frame,
                    inference_delay,
                    previous_actions,
                    task,
                    robot_type,
                )
                if isinstance(prediction, torch.Tensor):
                    prediction = _RTCChunkPrediction(actions=prediction)

            actions = prediction.actions
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            if actions.ndim != 3 or actions.shape[0] != 1:
                raise ValueError(
                    "RTC chunk predictor must return actions shaped [1, time, action] or [time, action], "
                    f"got {tuple(actions.shape)}"
                )
            original_actions = actions.squeeze(0).detach().clone()
            processed_actions = None
            if not prediction.apply_prefix_fusion:
                processed_actions = self.postprocessor(actions).squeeze(0).detach().clone()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        previous_snapshot = previous_actions.detach().clone() if previous_actions is not None else None
        return (
            original_actions,
            processed_actions,
            previous_snapshot,
            prediction.apply_prefix_fusion,
            time.perf_counter() - started_at,
        )

    def _start_inference(self, observation_frame: dict[str, np.ndarray]) -> None:
        previous_actions = self.action_queue.get_left_over()
        if previous_actions is not None:
            previous_actions = previous_actions.detach().clone()
        inference_delay = math.ceil((self.latency_tracker.max() or 0.0) * self.fps)
        self.future_generation = self.generation
        self.action_index_before_inference = self.action_queue.get_action_index()
        self.future = self.executor.submit(
            self._infer_chunk,
            deepcopy(observation_frame),
            inference_delay,
            previous_actions,
            self.task,
            self.robot_type,
        )

    def _collect_inference(self, *, wait: bool = False) -> None:
        future = self.future
        if future is None or (not wait and not future.done()):
            return

        generation = self.future_generation
        action_index_before_inference = self.action_index_before_inference
        try:
            (
                original_actions,
                processed_actions,
                previous_actions,
                apply_prefix_fusion,
                latency,
            ) = future.result()
        except Exception:
            self.future = None
            if generation != self.generation:
                logging.warning("Discarding a failed stale RTC inference.", exc_info=True)
                return
            raise
        self.future = None

        if generation != self.generation:
            return

        real_delay = self.action_queue.get_action_index() - action_index_before_inference
        if apply_prefix_fusion:
            original_actions = self._fuse_action_prefix(
                original_actions,
                previous_actions,
                real_delay,
            )
            processed_actions = (
                self.postprocessor(original_actions.unsqueeze(0)).squeeze(0).detach().clone()
            )
        if processed_actions is None:
            raise RuntimeError("RTC chunk postprocessing did not produce an action chunk")
        self.action_queue.merge(
            original_actions,
            processed_actions,
            real_delay,
            action_index_before_inference,
        )
        self.latency_tracker.add(latency)

    def _fuse_action_prefix(
        self,
        actions: torch.Tensor,
        previous_actions: torch.Tensor | None,
        real_delay: int,
    ) -> torch.Tensor:
        """Blend a non-denoising policy chunk into the unconsumed previous chunk."""
        if previous_actions is None:
            return actions
        if actions.ndim != 2 or previous_actions.ndim != 2:
            raise ValueError("RTC prefix fusion expects [time, action] tensors")
        if actions.shape[1] != previous_actions.shape[1]:
            raise ValueError(
                "RTC prefix fusion action dimensions differ: "
                f"{actions.shape[1]} != {previous_actions.shape[1]}"
            )

        horizon = min(
            self.rtc.execution_horizon,
            actions.shape[0],
            previous_actions.shape[0],
        )
        if horizon <= 0:
            return actions

        from lerobot.policies.rtc.modeling_rtc import RTCProcessor

        start = min(max(real_delay, 0), horizon)
        weights = RTCProcessor(self.rtc).get_prefix_weights(start, horizon, actions.shape[0])
        weights = weights[:horizon].to(device=actions.device, dtype=actions.dtype).unsqueeze(-1)
        previous_prefix = previous_actions[:horizon].to(device=actions.device, dtype=actions.dtype)
        fused = actions.clone()
        fused[:horizon] = weights * previous_prefix + (1.0 - weights) * fused[:horizon]
        return fused

    def get_action(self, observation_frame: dict[str, np.ndarray]) -> torch.Tensor:
        self._collect_inference()
        if self.action_queue.qsize() <= self.queue_threshold and self.future is None:
            self._start_inference(observation_frame)

        action = self.action_queue.get()
        if action is None:
            self._collect_inference(wait=True)
            action = self.action_queue.get()
        if action is None:
            raise RuntimeError(
                "RTC action queue is empty after inference; increase the queue threshold or chunk size."
            )
        return action.unsqueeze(0)

    def invalidate(self) -> None:
        """Drop queued/stale results without touching a possibly running policy inference."""
        self.generation += 1
        self.action_queue = ActionQueue(self.rtc)
        self.latency_tracker.reset()

    def wait_for_idle(self) -> None:
        self._collect_inference(wait=True)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.invalidate()
        self.wait_for_idle()
        self.executor.shutdown(wait=True)


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    policy_sync_executor: PolicySyncDualArmExecutor | None = None,
    intervention_state_machine_enabled: bool = True,
    collector_policy_id_policy: str = "policy",
    collector_policy_id_human: str = "human",
    acp_inference: ACPInferenceConfig | None = None,
    rtc: RTCConfig | None = None,
    rtc_action_queue_threshold: int = 32,
    communication_retry_timeout_s: float = 2.0,
    communication_retry_interval_s: float = 0.1,
    event_config: EventConfig | None = None,
    episode_events: list[dict] | None = None,
    dataset_features: dict[str, dict] | None = None,
) -> RobotAction | None:
    if acp_inference is None:
        acp_inference = ACPInferenceConfig()
    rtc_enabled = rtc is not None and rtc.enabled
    if rtc_enabled:
        if policy is None or preprocessor is None or postprocessor is None:
            raise ValueError("`rtc.enabled=true` requires a policy and its processors.")
        if acp_inference.enable:
            raise ValueError("RTC and ACP inference cannot be enabled at the same time.")
        if rtc.execution_horizon <= 0:
            raise ValueError("`rtc.execution_horizon` must be > 0.")
        if rtc_action_queue_threshold < 0:
            raise ValueError("`rtc_action_queue_threshold` must be >= 0.")

    # Per-episode quality events, each shaped {"step": <frame_index>, "event": <event_name>}.
    # The set of events and their hotkeys is described by `event_config` (loaded from a JSON file).
    # The caller may pass an empty list (default) or a pre-seeded one; record_loop appends events
    # detected from the configured hotkeys in-place so the caller can persist them after the episode.
    if episode_events is None:
        episode_events = []
    # Which event features the dataset actually stores per frame (e.g. complementary_info.bad_depth).
    event_features_present = (
        event_config is not None
        and dataset is not None
        and any(feature in dataset.features for feature in event_config.feature_keys.values())
    )
    # Group any pre-seeded events by step so they get re-applied to the matching frames.
    preseeded_events_by_step = (
        event_config.index_events_by_step(episode_events) if event_config is not None else {}
    )
    # Track already-recorded (step, event) pairs to avoid duplicating pre-seeded entries.
    recorded_event_pairs = {(int(e["step"]), e["event"]) for e in episode_events}

    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    feature_mapping = dataset.features if dataset is not None else dataset_features
    if feature_mapping is None and policy is not None:
        raise ValueError("Policy execution requires a dataset or `dataset_features` for feature mapping.")

    action_feature_names = feature_mapping[ACTION]["names"] if feature_mapping is not None else None
    if action_feature_names is None:
        if hasattr(robot.action_features, "keys"):
            action_feature_names = list(robot.action_features.keys())
        else:
            action_feature_names = list(robot.action_features)
    zero_policy_action = dict.fromkeys(action_feature_names, 0.0)
    # 策略动作质量后处理：对策略自身生成的全部动作维度（含夹爪）做在线滑窗平滑，与 inference
    # 端 ArmSmoother(窗口=9，平滑全部 7 维) 对齐；接管/teleop 动作不处理。
    policy_action_smoother = _OnlinePolicyActionSmoother(action_feature_names, window=9)
    has_teleop = isinstance(teleop, (Teleoperator, list))
    intervention_enabled = intervention_state_machine_enabled and policy is not None and has_teleop
    intervention_state = INTERVENTION_STATE_POLICY
    last_teleop_action: RobotAction | None = None
    last_robot_action_to_send: RobotAction | None = None
    teleop_fallback_warned = False

    teleop_arm_for_mode_switch: Any | None = None
    if isinstance(teleop, Teleoperator):
        teleop_arm_for_mode_switch = teleop
    elif isinstance(teleop, list):
        teleop_arm_for_mode_switch = teleop_arm

    def set_teleop_manual_control(enabled: bool) -> None:
        if teleop_arm_for_mode_switch is None:
            return
        if not hasattr(teleop_arm_for_mode_switch, "set_manual_control"):
            return
        try:
            teleop_arm_for_mode_switch.set_manual_control(enabled)
        except Exception:
            logging.exception("Failed to switch teleop manual-control mode to %s", enabled)

    if policy is None:
        # During reset/teleop-only loops keep leader backdrivable for manual dragging.
        # 若无策略模型，主臂进入重力补偿/可拖动模式，由人工操作主臂
        # 但固件里
        set_teleop_manual_control(True)

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        policy_action_smoother.reset()

    rtc_runner = (
        _RTCActionChunkRunner(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            rtc=rtc,
            fps=fps,
            queue_threshold=rtc_action_queue_threshold,
            task=single_task,
            robot_type=robot.robot_type,
        )
        if rtc_enabled
        else None
    )

    cond_policy_runtime_state: dict[str, Any] | None = None
    uncond_policy_runtime_state: dict[str, Any] | None = None
    if policy is not None and acp_inference.enable and acp_inference.use_cfg:
        cond_policy_runtime_state = _capture_policy_runtime_state(policy)
        uncond_policy_runtime_state = _capture_policy_runtime_state(policy)

    if intervention_enabled:
        # Start in S0: policy drives both arms, teleop arm should accept feedback commands.
        set_teleop_manual_control(False)

    def run_with_connection_retry(action_name: str, fn: Callable[[], T]) -> T:
        timeout_s = max(communication_retry_timeout_s, 0.0)
        interval_s = max(communication_retry_interval_s, 0.0)
        deadline_t = time.perf_counter() + timeout_s
        attempts = 0
        first_error: ConnectionError | None = None

        while True:
            attempts += 1
            try:
                result = fn()
                if attempts > 1:
                    elapsed_s = timeout_s - max(deadline_t - time.perf_counter(), 0.0)
                    logging.warning(
                        "%s recovered after %d retries in %.2fs.",
                        action_name,
                        attempts - 1,
                        elapsed_s,
                    )
                return result
            except ConnectionError as error:
                if first_error is None:
                    first_error = error
                    logging.warning(
                        "%s failed with transient communication error; retrying for up to %.2fs (%s)",
                        action_name,
                        timeout_s,
                        error,
                    )

                if timeout_s <= 0.0:
                    raise

                remaining_s = deadline_t - time.perf_counter()
                if remaining_s <= 0.0:
                    raise

                sleep_s = interval_s if interval_s > 0.0 else remaining_s
                time.sleep(min(sleep_s, remaining_s))

    timespent = 0
    start_episode_t = time.perf_counter()
    # Tracks whether the episode ended via a keyboard event (right/left/esc/s/f) rather than by
    # reaching the maximum recording time, so the caller can distinguish a timeout.
    ended_by_event = False
    # 录制时长由调用方传入的 control_time_s（即 --dataset.episode_time_s）决定；为 None 时不限时长，
    # 仅靠键盘事件结束。达到 control_time_s 后正常退出循环 -> 下方标记为 episode_timeout。
    try:
        while control_time_s is None or timespent < control_time_s:
            start_loop_t = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                ended_by_event = True
                break

            if events.get("toggle_intervention", False):
                events["toggle_intervention"] = False
                if intervention_enabled:
                    if intervention_state == INTERVENTION_STATE_POLICY:     # 开始接管
                        intervention_state = INTERVENTION_STATE_ACTIVE
                        if rtc_runner is not None:
                            rtc_runner.invalidate()
                        set_teleop_manual_control(True)
                        logging.info("Intervention enabled (S1): teleop actions now override policy execution.")
                    else:                                                   # 结束接管
                        if last_robot_action_to_send is None:
                            logging.warning(
                                "Cannot release intervention before an action has been sent to the robot."
                            )
                            continue
                        run_with_connection_retry(
                            "teleop.send_feedback",
                            lambda action=last_robot_action_to_send: teleop_arm_for_mode_switch.send_feedback(
                                action
                            ),
                        )
                        intervention_state = INTERVENTION_STATE_RELEASE
                        if rtc_runner is not None:
                            rtc_runner.wait_for_idle()
                        if policy is not None and preprocessor is not None and postprocessor is not None:
                            policy.reset()
                            preprocessor.reset()
                            postprocessor.reset()
                            policy_action_smoother.reset()
                            if acp_inference.enable and acp_inference.use_cfg:
                                cond_policy_runtime_state = _capture_policy_runtime_state(policy)
                                uncond_policy_runtime_state = _capture_policy_runtime_state(policy)
                        if rtc_runner is not None:
                            rtc_runner.invalidate()
                        if policy is not None and preprocessor is not None and postprocessor is not None:
                            logging.info("Policy cache reset on release: next policy action is recomputed.")
                        logging.info("Intervention release requested (S2): returning control to policy.")
                else:
                    logging.info("Intervention toggle ignored because policy+teleop are not both active.")

            # Get robot observation
            obs = robot.get_observation()

            # Applies a pipeline to the raw robot observation, default is IdentityProcessor
            obs_processed = robot_observation_processor(obs)

            observation_frame = (
                build_dataset_frame(feature_mapping, obs_processed, prefix=OBS_STR)
                if feature_mapping is not None
                else {}
            )

            # Get action from policy and/or teleop
            act_processed_policy: RobotAction | None = None
            act_processed_teleop: RobotAction | None = None
            # logging.info(f"Loop start: intervention_state={intervention_state}, policy={'on' if policy else 'off'}, teleop={'on' if teleop else 'off'}")
            if (
                policy is not None
                and preprocessor is not None
                and postprocessor is not None
                and not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE)  # 没接管
            ):
                if rtc_runner is None:
                    policy_action = _predict_policy_action_with_acp_inference(
                        observation_frame=observation_frame,
                        policy=policy,
                        device=get_safe_torch_device(policy.config.device),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=single_task,
                        robot_type=robot.robot_type,
                        acp_inference=acp_inference,
                        cond_runtime_state=cond_policy_runtime_state,
                        uncond_runtime_state=uncond_policy_runtime_state,
                    )
                else:
                    policy_action = rtc_runner.get_action(observation_frame)
                act_processed_policy = make_robot_action(policy_action, feature_mapping)
                # logging.info("policy_action: %s \n act_processed_policy: %s", policy_action, act_processed_policy)



            if isinstance(teleop, Teleoperator):
                act = run_with_connection_retry("teleop.get_action", teleop.get_action)

                # Applies a pipeline to the raw teleop action, default is IdentityProcessor
                act_processed_teleop = teleop_action_processor((act, obs))

            elif isinstance(teleop, list):
                arm_action = run_with_connection_retry("teleop_arm.get_action", teleop_arm.get_action)
                arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                keyboard_action = teleop_keyboard.get_action()
                base_action = robot._from_keyboard_to_base_action(keyboard_action)
                act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
                act_processed_teleop = teleop_action_processor((act, obs))

            if act_processed_policy is None and act_processed_teleop is None:
                logging.info(
                    "No policy or teleoperator provided, skipping action generation."
                    "This is likely to happen when resetting the environment without a teleop device."
                    "The robot won't be at its rest position at the start of the next episode."
                )
                continue

            if act_processed_teleop is not None:
                last_teleop_action = act_processed_teleop
                teleop_fallback_warned = False

            policy_action_for_storage = (
                act_processed_policy if act_processed_policy is not None else zero_policy_action
            )

            is_intervention = 0.0
            if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:  # 接管状态用主臂动作
                is_intervention = 1.0
                if act_processed_teleop is not None:
                    action_values = act_processed_teleop
                elif last_teleop_action is not None:
                    action_values = last_teleop_action
                    if not teleop_fallback_warned:
                        logging.warning(
                            "Intervention is active but no fresh teleop action is available; reusing last teleop action."
                        )
                        teleop_fallback_warned = True
                elif act_processed_policy is not None:
                    action_values = act_processed_policy
                    if not teleop_fallback_warned:
                        logging.warning(
                            "Intervention is active but teleop action is unavailable; falling back to policy action."
                        )
                        teleop_fallback_warned = True
                else:
                    action_values = zero_policy_action
                    if not teleop_fallback_warned:
                        logging.warning(
                            "Intervention is active but no teleop/policy action is available; sending zero action."
                        )
                        teleop_fallback_warned = True
            else:  # 未接管用推理动作
                action_values = act_processed_policy if act_processed_policy is not None else act_processed_teleop

            # Applies a pipeline to the action, default is IdentityProcessor
            # robot_action_to_send = robot_action_processor((action_values, obs))
            # logging.info(f"robot_action_to_send: {robot_action_to_send}")

            # Send action to robot
            # Action can eventually be clipped using `max_relative_target`,
            # so action actually sent is saved in the dataset. action = postprocessor.process(action)
            # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
            selected_from_policy = act_processed_policy is not None and action_values is act_processed_policy
            if policy_sync_executor is not None and selected_from_policy:  # 推理动作发给主臂和从臂
                # 对 VLA 生成的动作做质量后处理（滑窗平滑 + 可选限位 + 夹爪非线性衰减），提高数据质量。
                # 覆盖 action_values：既影响实际下发，也让数据集 action_frame 记录后处理后的动作。
                # 夹爪非线性衰减在此处理（与 inference 一致）；send_action 不再二次衰减。
                action_values = _postprocess_policy_action(
                    action_values,
                    policy_action_smoother,
                )

                # 按需求：VLA 生成动作时不再单独保留原始动作，complementary_info.policy_action
                # 也记录平滑后的动作（训练直接用平滑后的动作作为目标）。
                policy_action_for_storage = action_values

                # Applies a pipeline to the action, default is IdentityProcessor
                robot_action_to_send = robot_action_processor((action_values, obs))
                # 需要在send_action中添加偏置【已完成】
                # logging.info(f"将动作发给主臂+从臂")
                _sent_action = run_with_connection_retry(
                    "policy_sync_executor.send_action",
                    lambda robot_action_to_send=robot_action_to_send: policy_sync_executor.send_action(
                        robot_action_to_send, add_offset=not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE)  # 只有在非接管状态才添加偏置,即模型自行生成的动作才添加偏置，接管状态下主臂由teleop控制，不添加偏置
                    ),
                )
            else:  # 接管动作只发给从臂，一般由接管状态下主臂由teleop控制，不添加偏置
                # Applies a pipeline to the action, default is IdentityProcessor
                robot_action_to_send = robot_action_processor((action_values, obs))

                # logging.info(f"将动作发给从臂")
                _sent_action = run_with_connection_retry(
                    "robot.send_action",
                    lambda robot_action_to_send=robot_action_to_send: robot.send_action(
                        robot_action_to_send, add_offset=not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE)  # 只有在非接管状态才添加偏置,即模型自行生成的动作才添加偏置，接管状态下主臂由teleop控制，不添加偏置
                    ),
                )

            # Keep the exact common-coordinate action that can1 accepted. It can be reused to move
            # can0 out of manual/MIT mode without depending on potentially stale leader feedback.
            last_robot_action_to_send = dict(robot_action_to_send)

            # Write to dataset
            if dataset is not None:
                # Resolve quality events for the frame we are about to add. Its step equals the
                # frame_index add_frame() will assign (the current episode buffer size).
                active_events: set[str] = set()
                if event_config is not None:
                    current_step = dataset.episode_buffer["size"] if dataset.episode_buffer is not None else 0
                    active_events = set(preseeded_events_by_step.get(current_step, set()))
                    # Consume the momentary hotkey markers and attribute them to this frame.
                    for event_name, marker_key in event_config.marker_keys.items():
                        if events.get(marker_key, False):
                            events[marker_key] = False
                            active_events.add(event_name)
                    for event_name in active_events:
                        pair = (current_step, event_name)
                        if pair not in recorded_event_pairs:
                            recorded_event_pairs.add(pair)
                            episode_events.append({"step": current_step, "event": event_name})

                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                policy_action_frame = build_dataset_frame(
                    dataset.features, policy_action_for_storage, prefix="complementary_info.policy_action"
                )
                frame = {**observation_frame, **action_frame, **policy_action_frame, "task": single_task}

                if "complementary_info.is_intervention" in dataset.features:
                    frame["complementary_info.is_intervention"] = np.array([is_intervention], dtype=np.float32)
                if "complementary_info.state" in dataset.features:
                    frame["complementary_info.state"] = np.array([intervention_state], dtype=np.float32)
                if "complementary_info.collector_policy_id" in dataset.features:
                    frame["complementary_info.collector_policy_id"] = resolve_collector_policy_id(
                        intervention_enabled=intervention_enabled,
                        is_intervention=bool(is_intervention),
                        selected_from_policy=selected_from_policy,
                        policy_id=collector_policy_id_policy,
                        human_id=collector_policy_id_human,
                    )
                if event_features_present:
                    # Per-frame quality-event flags (0.0/1.0), always present so the frame stays
                    # schema-consistent for add_frame validation.
                    for feature, flag in event_config.flags(active_events).items():
                        if feature in dataset.features:
                            frame[feature] = flag
                dataset.add_frame(frame)

            if display_data:
                log_rerun_data(
                    observation=obs_processed, action=action_values, compress_images=display_compressed_images
                )

            if intervention_state == INTERVENTION_STATE_RELEASE:
                intervention_state = INTERVENTION_STATE_POLICY

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1 / fps - dt_s, 0.0))
            # precise_sleep(max(1 / 20 - dt_s, 0.0))  # MM

            timespent = time.perf_counter() - start_episode_t

    finally:
        if rtc_runner is not None:
            rtc_runner.close()

    # The loop only breaks on a keyboard event; otherwise it ended by reaching the max recording
    # time. Signal this so the caller can mark the episode as failed and reset the arms.
    events["episode_timeout"] = not ended_by_event
    return last_robot_action_to_send
