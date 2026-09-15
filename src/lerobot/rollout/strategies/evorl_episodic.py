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

"""Episodic rollout strategy: mirrors the behavior of ``lerobot-record``.

- Policy drives the robot during each recording episode.
- An optional teleoperator can drive the robot during reset phases so the
  operator can bring the environment back to its starting configuration.
  If no teleop is connected the robot stays in its current position.
- Keyboard controls:

      Right arrow  — end the current episode or reset phase early
      A            — discard the current episode and re-record it
      B / F        — mark the episode successful / failed
      C            — toggle immediate human intervention
      R            — return home, discard, and re-record
      Escape       — stop the recording session

Dataset naming follows the rollout convention: repo names must start with ``rollout_``.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)
from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.recording_annotations import (
    EVORL_COLLECTOR_POLICY_ID_FIELD,
    EVORL_INTERVENTION_FIELD,
    EVORL_POLICY_ACTION_FIELD,
    EVORL_STATE_ACTIVE,
    EVORL_STATE_FIELD,
    EVORL_STATE_POLICY,
    EVORL_STATE_RELEASE,
    infer_collector_policy_version,
    normalize_episode_success_label,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import log_visualization_data

from ..configs import EvoRLEpisodicStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)


class EvoRLEpisodicStrategy(RolloutStrategy):
    """Policy-driven multi-episode recording, mirrors the behavior of ``lerobot-record``.

    Each recording episode runs the policy for maximum ``dataset.episode_time_s``
    seconds, recording one frame per policy action (``1/fps`` cadence — with
    ``interpolation_multiplier > 1`` the interpolated ticks only send commands
    to the robot).  A reset phase of ``dataset.reset_time_s``
    follows every episode (except the last) so the operator can manually
    reset the environment.  During the reset phase, an optional teleoperator
    drives the robot; if none is present the robot returns to its initial joint positions captured at startup.

    The policy state (hidden state, RTC queue, interpolator) is reset at
    the start of each recording episode.

    Keyboard events:
        right arrow  → end current episode or reset phase early
        A            → discard & re-record current episode
        B / F        → mark the episode successful / failed
        R            → return home, discard & re-record
        ESC          → stop the session
        C            → immediately toggle human intervention when enabled
    """

    config: EvoRLEpisodicStrategyConfig

    def __init__(self, config: EvoRLEpisodicStrategyConfig) -> None:
        super().__init__(config)
        self._listener = None
        self._events: dict | None = None
        self._last_action: dict | None = None

    def setup(self, ctx: RolloutContext) -> None:
        """Start the inference engine and attach the keyboard listener."""
        self._init_engine(ctx)
        if self.config.enable_intervention and ctx.hardware.teleop is None:
            raise ValueError("EvoRL intervention is enabled but no teleoperator is connected.")
        self._listener, self._events = init_keyboard_listener(
            intervention_toggle_key=(
                self.config.intervention_key if self.config.enable_intervention else None
            ),
            episode_success_key=self.config.success_key,
            episode_failure_key=self.config.failure_key,
            rerecord_episode_key=self.config.rerecord_key,
            reset_episode_key=self.config.reset_key,
        )
        if self.config.enable_intervention:
            logger.info("EvoRL episodic strategy ready; intervention key=%s", self.config.intervention_key)
        else:
            logger.info("EvoRL inference-only mode; leader teleoperator and human intervention key are disabled")

    def run(self, ctx: RolloutContext) -> None:
        """Main multi-episode recording loop."""
        cfg = ctx.runtime.cfg
        dataset_cfg = cfg.dataset
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        features = ctx.data.dataset_features

        fps = cfg.fps
        episode_time_s = dataset_cfg.episode_time_s
        reset_time_s = dataset_cfg.reset_time_s
        num_episodes = dataset_cfg.num_episodes
        single_task = dataset_cfg.single_task or cfg.task
        play_sounds = cfg.play_sounds

        display_compressed = (
            True
            if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
            else cfg.display_compressed_images
        )

        # One timer for the whole session: episodes get their own cadence line, and
        # the run summary averages across them without the untimed reset phases.
        timer = CycleTimer(fps, self._interpolator.multiplier)

        with VideoEncodingManager(dataset):
            try:
                recorded_episodes = 0
                while recorded_episodes < num_episodes and not events["stop_recording"]:
                    if ctx.runtime.shutdown_event.is_set():
                        break

                    # Reset policy state at episode start (discard leftover hidden state / queue)
                    self._engine.reset()
                    self._interpolator.reset()
                    # A reset interpolator re-primes over two consecutive inference
                    # ticks, exactly like loop start-up, so exempt the group that
                    # spans them instead of reporting a healthy episode as slow.
                    timer.restart()
                    self._engine.resume()

                    events["episode_outcome"] = None
                    events["reset_episode"] = False
                    log_say(f"Recording episode {dataset.num_episodes}", play_sounds)
                    timed_out = self._policy_loop(
                        ctx=ctx,
                        robot=robot,
                        events=events,
                        features=features,
                        timer=timer,
                        control_time_s=episode_time_s,
                        dataset=dataset,
                        single_task=single_task,
                    )

                    if events["stop_recording"] or ctx.runtime.shutdown_event.is_set():
                        dataset.clear_episode_buffer()
                        break

                    outcome = events.get("episode_outcome")
                    if timed_out and outcome is None:
                        outcome = self.config.default_timeout_outcome
                        logger.info("Episode timed out; using outcome=%s", outcome)
                    outcome = normalize_episode_success_label(outcome)

                    # Reset phase, skip after the last episode (but run when re-recording)
                    if not events["stop_recording"] and (
                        recorded_episodes < num_episodes - 1 or events["rerecord_episode"]
                    ):
                        log_say("Reset the environment", play_sounds)

                        if events.get("reset_episode"):
                            self._reset_to_home(ctx)
                        else:
                            # Outside policy execution and explicit human intervention,
                            # keep both arms enabled at the follower's measured terminal
                            # pose until the next episode begins.
                            self._hold_last_follower_position(
                                ctx=ctx,
                                robot=robot,
                                teleop=teleop,
                                events=events,
                                fps=fps,
                                control_time_s=reset_time_s,
                                display_data=cfg.display_data,
                                display_mode=cfg.display_mode,
                                display_compressed=display_compressed,
                            )

                    if events["rerecord_episode"]:
                        log_say("Re-record episode", play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        dataset.clear_episode_buffer()
                        timer.log_episode_summary("discarded episode")

                        # returns to its initial joint positions captured at startup
                        if (
                            not teleop
                            and self.config.reset_to_initial_position
                            and not events.get("reset_episode")
                        ):
                            self.return_to_initial_position(hw=ctx.hardware, duration_s=1)

                        continue

                    if self.config.require_outcome and outcome is None:
                        logger.warning(
                            "Episode ended without %s/%s outcome; discarding and re-recording.",
                            self.config.success_key.upper(),
                            self.config.failure_key.upper(),
                        )
                        dataset.clear_episode_buffer()
                        timer.log_episode_summary("discarded unlabeled episode")
                        continue

                    episode_index = dataset.num_episodes
                    episode_frames = (
                        dataset.writer.episode_buffer["size"]
                        if dataset.writer.episode_buffer is not None
                        else 0
                    )
                    dataset.save_episode(
                        episode_metadata={"episode_success": outcome} if outcome is not None else None
                    )
                    logger.info(
                        "Saved episode %d (outcome=%s, frames=%d)",
                        episode_index,
                        outcome or "unlabeled",
                        episode_frames,
                    )
                    recorded_episodes += 1
                    timer.log_episode_summary(f"episode {episode_index}")
            finally:
                logger.info("EvoRL episodic loop ended; discarding any unlabeled partial episode")
                timer.log_run_summary()
                dataset.clear_episode_buffer()

    def _log_evorl_telemetry(
        self,
        observation: dict,
        action: dict,
        runtime,
        intervening: bool,
    ) -> None:
        """Log policy/human source together with current observation and action."""
        self._log_telemetry(
            {**observation, "intervention": intervening},
            action,
            runtime,
        )

    def _policy_loop(
        self,
        ctx: RolloutContext,
        robot,
        events: dict,
        features: dict,
        timer: CycleTimer,
        control_time_s: float,
        dataset,
        single_task: str,
    ) -> bool:
        """Record one mixed policy/human episode using the current rollout engine."""
        interpolator = self._interpolator
        engine = self._engine
        teleop = ctx.hardware.teleop
        actuated_teleop = (
            self.config.enable_intervention and teleop is not None and teleop_supports_feedback(teleop)
        )
        intervening = False
        intervention_state = EVORL_STATE_POLICY
        policy_action_names = features[EVORL_POLICY_ACTION_FIELD].get("names")
        if policy_action_names is None:
            policy_action_names = list(ctx.data.ordered_action_keys)
        policy_id = infer_collector_policy_version(ctx.runtime.cfg.policy)

        intervention_tick = 0
        last_action: dict | None = None
        timestamp = 0.0
        start_t = time.perf_counter()

        while timestamp < control_time_s:
            timer.tick(new_cycle=interpolator.needs_new_action() if not intervening else True)

            if events.get("toggle_intervention"):
                events["toggle_intervention"] = False
                if not self.config.enable_intervention or teleop is None:
                    logger.warning("Ignoring intervention request: human intervention is disabled")
                elif not intervening:
                    engine.pause()
                    # Match the 0901 RTC handover semantics: once C is pressed, neither
                    # queued actions nor an inference already in flight may be dispatched
                    # after human control begins.  Sync engines implement this as a no-op.
                    engine.invalidate_pending_actions()
                    if actuated_teleop:
                        # The leader already tracks every dispatched policy action,
                        # so it is aligned with the follower. Release it immediately
                        # and consume its live pose below on this same control tick.
                        teleop.disable_torque()
                    intervening = True
                    intervention_tick = 0
                    intervention_state = EVORL_STATE_ACTIVE
                    logger.info("Human intervention started")
                else:
                    if actuated_teleop:
                        teleop.enable_torque()
                    intervening = False
                    self.reset_control_state()
                    intervention_state = EVORL_STATE_RELEASE
                    timer.restart()
                    engine.resume()
                    last_action = None
                    logger.info("Human intervention ended; inference state reset")

            if events["exit_early"]:
                events["exit_early"] = False
                break
            if ctx.runtime.shutdown_event.is_set() or events["stop_recording"]:
                break

            with timer.section("observe"):
                obs = robot.get_observation()

            if intervening:
                with timer.section("process_obs"):
                    obs_processed = ctx.processors.robot_observation_processor(obs)
                with timer.section("teleop"):
                    teleop_action = teleop.get_action()
                    action_dict = ctx.processors.teleop_action_processor((teleop_action, obs))
                    robot_action = ctx.processors.robot_action_processor((action_dict, obs))
                with timer.section("send"):
                    robot.send_action(robot_action)
                last_action = action_dict
                should_record = intervention_tick % interpolator.multiplier == 0
                intervention_tick += 1
            else:
                with timer.section("process_obs"):
                    obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                if self._handle_warmup(ctx.runtime.cfg.use_torch_compile, timer):
                    continue
                action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timer)
                should_record = action_dict is not None and interpolator.emitted_policy_action
                if action_dict is not None:
                    last_action = action_dict
                    if actuated_teleop:
                        with timer.section("leader_follow"):
                            teleop.send_feedback(action_dict)

            if action_dict is not None:
                with timer.section("telemetry"):
                    self._log_evorl_telemetry(obs_processed, action_dict, ctx.runtime, intervening)
                if should_record:
                    with timer.section("record"):
                        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                        policy_action_values = (
                            dict.fromkeys(policy_action_names, 0.0) if intervening else action_dict
                        )
                        policy_action_frame = build_dataset_frame(
                            features,
                            policy_action_values,
                            prefix=EVORL_POLICY_ACTION_FIELD,
                        )
                        dataset.add_frame(
                            {
                                **obs_frame,
                                **action_frame,
                                **policy_action_frame,
                                "task": single_task,
                                EVORL_INTERVENTION_FIELD: np.array([float(intervening)], dtype=np.float32),
                                EVORL_STATE_FIELD: np.array([intervention_state], dtype=np.float32),
                                EVORL_COLLECTOR_POLICY_ID_FIELD: ("human" if intervening else policy_id),
                            }
                        )
                        if intervention_state == EVORL_STATE_RELEASE:
                            intervention_state = EVORL_STATE_POLICY

            engine.pump_query(obs_processed)
            timer.wait()
            timestamp = time.perf_counter() - start_t

        self._last_action = last_action
        if intervening and actuated_teleop:
            teleop.enable_torque()
        engine.pause()
        return timestamp >= control_time_s

    def _reset_to_home(self, ctx: RolloutContext, duration_s: float = 3.0) -> None:
        """Return the follower, and the leader when present, to action-space zero."""
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        obs = robot.get_observation()
        if teleop is None:
            action_current = {key: value for key, value in obs.items() if key.endswith(".pos")}
        else:
            raw_current = teleop.get_action()
            action_current = ctx.processors.teleop_action_processor((raw_current, obs))
        robot_current = ctx.processors.robot_action_processor((action_current, obs))
        action_target = dict.fromkeys(action_current, 0.0)
        robot_target = ctx.processors.robot_action_processor((action_target, obs))
        follower_smooth_move_to(robot, robot_current, robot_target, duration_s=duration_s)
        if teleop is not None and teleop_supports_feedback(teleop):
            teleop_smooth_move_to(teleop, action_target, duration_s=duration_s)
        self.reset_control_state()

    def _hold_last_follower_position(
        self,
        ctx: RolloutContext,
        robot,
        teleop,
        events: dict,
        fps: float,
        control_time_s: float,
        display_data: bool,
        display_mode: str,
        display_compressed: bool,
    ) -> None:
        """Hold the follower's measured terminal pose between episodes.

        This is an idle state: neither policy inference nor human intervention
        controls the robot. The fixed follower pose is refreshed on both the
        follower and any actuated leader so neither arm drops into a free/manual
        mode before the next episode starts.
        """
        processors = ctx.processors
        control_interval = 1.0 / fps
        obs = robot.get_observation()
        fallback_action = self._last_action or {}
        hold_action = {
            key: obs[key] if key in obs else fallback_action[key]
            for key in ctx.data.ordered_action_keys
            if key in obs or key in fallback_action
        }
        missing_keys = [key for key in ctx.data.ordered_action_keys if key not in hold_action]
        if missing_keys:
            raise ValueError(
                "Cannot hold the follower's terminal pose because position values are missing for "
                f"action keys: {missing_keys}."
            )

        robot_hold_action = processors.robot_action_processor((hold_action, obs))
        actuated_teleop = teleop is not None and teleop_supports_feedback(teleop)

        logger.info("Holding follower terminal pose between episodes: %s", hold_action)
        robot.send_action(robot_hold_action)
        if actuated_teleop:
            teleop.send_feedback(hold_action)

        timestamp = 0.0
        start_t = time.perf_counter()

        while timestamp < control_time_s:
            loop_start = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            if ctx.runtime.shutdown_event.is_set():
                break

            obs = robot.get_observation()
            robot.send_action(robot_hold_action)
            if actuated_teleop:
                teleop.send_feedback(hold_action)

            if display_data:
                obs_processed = processors.robot_observation_processor(obs)
                log_visualization_data(
                    display_mode,
                    observation=obs_processed,
                    action=hold_action,
                    compress_images=display_compressed,
                )

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            precise_sleep(max(sleep_t, 0.0))
            timestamp = time.perf_counter() - start_t

    def teardown(self, ctx: RolloutContext) -> None:
        """Finalize data and release every resource, even if one cleanup step fails."""
        cfg = ctx.runtime.cfg
        play_sounds = cfg.play_sounds
        cleanup_error = None

        def run_cleanup(description, operation):
            nonlocal cleanup_error
            try:
                return operation()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                logger.exception("%s failed", description)
                return None

        run_cleanup(
            "Stop-recording announcement",
            lambda: log_say("Stop recording", play_sounds, blocking=True),
        )

        if self._listener is not None:
            run_cleanup("Keyboard listener shutdown", self._listener.stop)

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            run_cleanup("Dataset finalization", ctx.data.dataset.finalize)

        if cfg.dataset is not None and cfg.dataset.push_to_hub and ctx.data.dataset is not None:
            uploaded = run_cleanup(
                "Dataset upload",
                lambda: safe_push_to_hub(
                    ctx.data.dataset,
                    tags=cfg.dataset.tags,
                    private=cfg.dataset.private,
                ),
            )
            if uploaded:
                logger.info("Dataset uploaded to hub")
                run_cleanup(
                    "Dataset-upload announcement",
                    lambda: log_say("Dataset uploaded to hub", play_sounds),
                )

        run_cleanup(
            "Hardware teardown",
            lambda: self._teardown_hardware(
                ctx.hardware,
                return_to_initial_position=cfg.return_to_initial_position,
            ),
        )
        run_cleanup("Exit announcement", lambda: log_say("Exiting", play_sounds))

        if cleanup_error is not None:
            logger.error("EvoRL episodic strategy teardown completed with errors")
            raise cleanup_error
        logger.info("EvoRL episodic strategy teardown complete")
