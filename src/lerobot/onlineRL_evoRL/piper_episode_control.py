"""Episode-boundary controls shared by single- and dual-Piper actors."""

from __future__ import annotations

import logging
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


def home_arms_to_default(
    robot: Any,
    teleop: Any | None = None,
    *,
    target_action: dict[str, float] | None = None,
    duration_s: float = 4.0,
    fps: float = 30.0,
) -> dict[str, float]:
    """Move single/dual followers and leaders smoothly to the captured start pose."""
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
    logging.info("Leader and follower arm(s) returned to the captured episode-start pose.")
    return target
