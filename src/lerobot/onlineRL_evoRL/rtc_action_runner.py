"""Callback-driven RTC runner used by the EvoRL online actor.

The rollout RTC engine owns its complete policy stack. Online EvoRL needs the
same queue/invalidation behavior but supplies either a VLA chunk or an
Actor-head chunk dynamically, so this adapter accepts a chunk callback.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.device_utils import get_safe_torch_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RTCChunkPrediction:
    actions: torch.Tensor
    apply_prefix_fusion: bool = False


RTCChunkPredictor = Callable[
    [dict[str, np.ndarray], int, torch.Tensor | None, str | None, str | None],
    torch.Tensor | RTCChunkPrediction,
]


class RTCActionChunkRunner:
    """Produce chunks asynchronously while exposing one action per control tick.

    Every in-flight result carries a generation number. Intervention increments
    it and clears the queue immediately, so a stale model result cannot be
    merged after the human has taken control.
    """

    def __init__(
        self,
        *,
        policy: Any,
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        rtc: RTCConfig,
        fps: float,
        queue_threshold: int,
        task: str | None,
        robot_type: str | None,
        chunk_predictor: RTCChunkPredictor,
    ) -> None:
        self.policy = policy
        self.postprocessor = postprocessor
        self.rtc = rtc
        self.fps = float(fps)
        self.queue_threshold = queue_threshold
        self.task = task
        self.robot_type = robot_type
        self.chunk_predictor = chunk_predictor
        self.device = get_safe_torch_device(policy.config.device)
        self.action_queue = ActionQueue(rtc)
        self.latency_tracker = LatencyTracker()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evorl-rtc")
        self.future: Future | None = None
        self.future_generation = 0
        self.action_index_before_inference = 0
        self.generation = 0
        self.paused = False
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
        amp = (
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and getattr(self.policy.config, "use_amp", False)
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            prediction = self.chunk_predictor(
                observation_frame, inference_delay, previous_actions, task, robot_type
            )
            if isinstance(prediction, torch.Tensor):
                prediction = RTCChunkPrediction(actions=prediction)
            actions = prediction.actions
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            if actions.ndim != 3 or actions.shape[0] != 1:
                raise ValueError(
                    "RTC chunk callback must return [1,T,A] or [T,A], "
                    f"got {tuple(actions.shape)}"
                )
            original = actions.squeeze(0).detach().clone()
            processed = None
            if not prediction.apply_prefix_fusion:
                processed = self.postprocessor(actions).squeeze(0).detach().clone()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        previous = previous_actions.detach().clone() if previous_actions is not None else None
        return original, processed, previous, prediction.apply_prefix_fusion, time.perf_counter() - started_at

    def _start_inference(self, observation_frame: dict[str, np.ndarray]) -> None:
        previous = self.action_queue.get_left_over()
        if previous is not None:
            previous = previous.detach().clone()
        delay = math.ceil((self.latency_tracker.max() or 0.0) * self.fps)
        self.future_generation = self.generation
        self.action_index_before_inference = self.action_queue.get_action_index()
        self.future = self.executor.submit(
            self._infer_chunk,
            deepcopy(observation_frame),
            delay,
            previous,
            self.task,
            self.robot_type,
        )

    def _collect_inference(self, *, wait: bool = False) -> None:
        future = self.future
        if future is None or (not wait and not future.done()):
            return
        generation = self.future_generation
        index_before = self.action_index_before_inference
        try:
            original, processed, previous, needs_fusion, latency = future.result()
        except Exception:
            self.future = None
            if generation != self.generation:
                logger.warning("Discarding a failed stale online RTC inference", exc_info=True)
                return
            raise
        self.future = None
        if generation != self.generation:
            logger.debug("Discarded stale online RTC generation %d", generation)
            return
        real_delay = self.action_queue.get_action_index() - index_before
        if needs_fusion:
            original = self._fuse_action_prefix(original, previous, real_delay)
            processed = self.postprocessor(original.unsqueeze(0)).squeeze(0).detach().clone()
        if processed is None:
            raise RuntimeError("RTC chunk postprocessing produced no actions")
        self.action_queue.merge(original, processed, real_delay, index_before, task=self.task)
        self.latency_tracker.add(latency)

    def _fuse_action_prefix(
        self, actions: torch.Tensor, previous_actions: torch.Tensor | None, real_delay: int
    ) -> torch.Tensor:
        if previous_actions is None:
            return actions
        if actions.ndim != 2 or previous_actions.ndim != 2:
            raise ValueError("RTC prefix fusion expects [T,A] tensors")
        if actions.shape[1] != previous_actions.shape[1]:
            raise ValueError("RTC prefix fusion action dimensions differ")
        horizon = min(self.rtc.execution_horizon, actions.shape[0], previous_actions.shape[0])
        if horizon <= 0:
            return actions
        from lerobot.policies.rtc.modeling_rtc import RTCProcessor

        start = min(max(real_delay, 0), horizon)
        weights = RTCProcessor(self.rtc).get_prefix_weights(start, horizon, actions.shape[0])
        weights = weights[:horizon].to(device=actions.device, dtype=actions.dtype).unsqueeze(-1)
        fused = actions.clone()
        previous = previous_actions[:horizon].to(device=actions.device, dtype=actions.dtype)
        fused[:horizon] = weights * previous + (1.0 - weights) * fused[:horizon]
        return fused

    def get_action(self, observation_frame: dict[str, np.ndarray]) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("RTC runner is closed")
        if self.paused:
            raise RuntimeError("RTC action requested while intervention is active")
        self._collect_inference()
        if self.action_queue.qsize() <= self.queue_threshold and self.future is None:
            self._start_inference(observation_frame)
        action = self.action_queue.get()
        if action is None:
            self._collect_inference(wait=True)
            action = self.action_queue.get()
        if action is None:
            raise RuntimeError("RTC action queue is empty after inference")
        return action.unsqueeze(0)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def invalidate_pending_actions(self) -> None:
        self.generation += 1
        self.action_queue.clear()
        self.latency_tracker.reset()

    def wait_for_idle(self) -> None:
        self._collect_inference(wait=True)

    def reset(self) -> None:
        self.invalidate_pending_actions()
        self.wait_for_idle()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.invalidate_pending_actions()
        self.wait_for_idle()
        self.executor.shutdown(wait=True)
