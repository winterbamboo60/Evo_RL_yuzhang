"""Episode-boundary controls shared by single- and dual-Piper actors."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any


def _pose_action(robot: Any) -> dict[str, float]:
    observation = robot.get_observation()
    action_keys = list(robot.action_features)
    missing = [key for key in action_keys if key not in observation]
    if missing:
        raise KeyError(f"Cannot hold pose; observation is missing {missing}")
    return {key: float(observation[key]) for key in action_keys}


def set_leader_manual_control(teleop: Any | None, enabled: bool) -> None:
    """Switch any connected leader between manual and actuated modes."""
    if teleop is None:
        return
    setter = getattr(teleop, "set_manual_control", None)
    if callable(setter):
        setter(enabled)
        return
    torque = getattr(teleop, "disable_torque" if enabled else "enable_torque", None)
    if callable(torque):
        torque()


def follow_policy_action(teleop: Any | None, action: dict[str, float]) -> None:
    """Mirror the executed follower target to one or two leader arms."""
    if teleop is None:
        return
    feedback = getattr(teleop, "send_feedback", None)
    if callable(feedback):
        feedback(action)


def hold_arms_current_pose(robot: Any, teleop: Any | None = None) -> dict[str, float]:
    """Keep followers enabled at their last measured pose and align leaders."""
    action = _pose_action(robot)
    robot.send_action(action)
    set_leader_manual_control(teleop, False)
    follow_policy_action(teleop, action)
    logging.info("Episode ended: follower pose is held; motor enable state is preserved.")
    return action


class PoseHoldController:
    """Refresh a fixed terminal pose while episode data or handoff work runs."""

    def __init__(self, robot: Any, teleop: Any | None = None, *, fps: float = 30.0) -> None:
        """Create an idle pose controller for all configured action keys."""
        self.robot = robot
        self.teleop = teleop
        self.interval_s = 1.0 / max(float(fps), 1.0)
        self.action: dict[str, float] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def active(self) -> bool:
        """Return whether the refresh worker is running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> dict[str, float]:
        """Capture the measured pose and continuously resend it until stopped."""
        self.stop()
        self.action = _pose_action(self.robot)
        set_leader_manual_control(self.teleop, False)
        self.robot.send_action(self.action)
        follow_policy_action(self.teleop, self.action)
        self._error = None
        self._stop_event.clear()

        def hold_loop() -> None:
            try:
                while not self._stop_event.wait(self.interval_s):
                    self.robot.send_action(self.action)
                    follow_policy_action(self.teleop, self.action)
            except BaseException as error:
                self._error = error
                self._stop_event.set()

        self._thread = threading.Thread(target=hold_loop, name="evorl-pose-hold", daemon=True)
        self._thread.start()
        logging.info("Episode ended: continuously holding follower pose %s", self.action)
        return dict(self.action)

    def wait(self, duration_s: float) -> None:
        """Keep holding for a bounded reset interval and surface worker failures."""
        deadline = time.monotonic() + max(float(duration_s), 0.0)
        while self.active and time.monotonic() < deadline:
            if self._error is not None:
                raise RuntimeError("Pose hold worker failed") from self._error
            time.sleep(min(self.interval_s, max(deadline - time.monotonic(), 0.0)))
        if self._error is not None:
            raise RuntimeError("Pose hold worker failed") from self._error

    def stop(self) -> None:
        """Stop refreshing only after the next controller has taken ownership."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(self.interval_s * 4, 1.0))
        if self._thread.is_alive():
            raise RuntimeError("Pose hold worker did not stop")
        self._thread = None
        if self._error is not None:
            error = self._error
            self._error = None
            raise RuntimeError("Pose hold worker failed") from error


def default_home_action(robot: Any, *, max_gripper_pos: float) -> dict[str, float]:
    """Return all-zero joints with every configured gripper fully open."""
    return {
        key: float(max_gripper_pos) if "gripper" in key.lower() else 0.0
        for key in robot.action_features
    }


def home_arms_to_default(
    robot: Any,
    teleop: Any | None = None,
    *,
    target_action: dict[str, float] | None = None,
    duration_s: float = 4.0,
    fps: float = 30.0,
) -> dict[str, float]:
    """Move single/dual followers and leaders smoothly to the configured home pose."""
    start = _pose_action(robot)
    target = dict(target_action or dict.fromkeys(start, 0.0))
    missing = [key for key in start if key not in target]
    if missing:
        raise ValueError(f"Reset target is missing action keys: {missing}")
    set_leader_manual_control(teleop, False)
    steps = max(int(max(duration_s, 0.0) * max(fps, 1.0)), 1)
    for step in range(1, steps + 1):
        ratio = step / steps
        action = {key: start[key] + (float(target[key]) - start[key]) * ratio for key in start}
        robot.send_action(action)
        follow_policy_action(teleop, action)
        time.sleep(1.0 / max(fps, 1.0))
    logging.info("Leader and follower arm(s) reached the configured home pose: %s", target)
    return target
