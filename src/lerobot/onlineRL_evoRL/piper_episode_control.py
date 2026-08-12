#!/usr/bin/env python

"""Piper arm episode boundary controls for online RL."""

from __future__ import annotations

import logging
import time
from typing import Any

from lerobot.utils.piper_sdk import PIPER_JOINT_NAMES

_PIPER_HOME_JOINTS = (0, 0, 0, 0, 0, 0)
_PIPER_HOME_GRIPPER = 70000
_PIPER_HOME_SPEED = 50
_PIPER_HOME_SLOW_SPEED = 20
_PIPER_HOME_SETTLE_S = 3.0
_PIPER_HOME_SMOOTH_DURATION_S = 4.0
_PIPER_HOME_SMOOTH_STEP_DT_S = 0.02


def _read_arm_raw_joints(arm: Any) -> tuple[list[int], int]:
    joint_msg = arm.GetArmJointMsgs()
    joint_state = getattr(joint_msg, "joint_state", None)
    joints = [int(getattr(joint_state, joint_name, 0)) for joint_name in PIPER_JOINT_NAMES]

    gripper_msg = arm.GetArmGripperMsgs()
    gripper_state = getattr(gripper_msg, "gripper_state", None)
    gripper = abs(int(getattr(gripper_state, "grippers_angle", 0)))
    return joints, gripper


def _hold_piper_arm(arm: Any, joints: list[int], gripper: int, speed: int = _PIPER_HOME_SPEED) -> None:
    arm.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    arm.JointCtrl(*joints)
    arm.GripperCtrl(gripper, 1000, 0x01, 0)


def hold_arms_current_pose(robot: Any, teleop: Any) -> None:
    held_any = False

    if teleop is not None and not isinstance(teleop, list):
        leader_arm = getattr(teleop, "arm", None)
        if leader_arm is not None and hasattr(leader_arm, "JointCtrl"):
            joints, gripper = _read_arm_raw_joints(leader_arm)
            if hasattr(teleop, "set_manual_control"):
                try:
                    teleop.set_manual_control(False)
                except Exception:
                    logging.exception("Failed to switch leader to command mode before holding pose.")
            _hold_piper_arm(leader_arm, joints, gripper)
            held_any = True

    follower_arm = getattr(robot, "arm", None)
    if follower_arm is not None and hasattr(follower_arm, "JointCtrl"):
        joints, gripper = _read_arm_raw_joints(follower_arm)
        _hold_piper_arm(follower_arm, joints, gripper)
        held_any = True

    if held_any:
        logging.info("Episode ended: leader + follower holding current pose until next episode.")
    else:
        logging.warning("Hold-current-pose skipped: arms do not expose a Piper JointCtrl interface.")


def _home_piper_arm(arm: Any, speed: int = _PIPER_HOME_SPEED) -> None:
    arm.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    arm.JointCtrl(*_PIPER_HOME_JOINTS)
    arm.GripperCtrl(_PIPER_HOME_GRIPPER, 1000, 0x01, 0)


def _home_leader_arm_smooth(
    arm: Any,
    mit_mode: int,
    speed: int = _PIPER_HOME_SLOW_SPEED,
    duration_s: float = _PIPER_HOME_SMOOTH_DURATION_S,
    step_dt_s: float = _PIPER_HOME_SMOOTH_STEP_DT_S,
) -> None:
    goal = list(_PIPER_HOME_JOINTS)
    start, _ = _read_arm_raw_joints(arm)
    steps = max(int(duration_s / step_dt_s), 1)
    for step in range(1, steps + 1):
        ratio = step / steps
        command = [int(start[i] + (goal[i] - start[i]) * ratio) for i in range(6)]
        if command[4] < -70000:
            command[4] = -70000
        arm.MotionCtrl_2(0x01, 0x01, speed, mit_mode)
        arm.JointCtrl(*command)
        time.sleep(step_dt_s)
    arm.GripperCtrl(_PIPER_HOME_GRIPPER, 1000, 0x01, 0)


def home_arms_to_default(robot: Any, teleop: Any, speed: int = _PIPER_HOME_SLOW_SPEED) -> None:
    homed_any = False

    leader_arm = None
    leader_mit_mode = 0x00
    if teleop is not None and not isinstance(teleop, list):
        candidate = getattr(teleop, "arm", None)
        if candidate is not None and hasattr(candidate, "JointCtrl"):
            if hasattr(teleop, "set_manual_control"):
                try:
                    teleop.set_manual_control(False)
                except Exception:
                    logging.exception("Failed to switch leader to command mode before homing.")
            leader_cfg = getattr(teleop, "config", None)
            leader_mit_mode = 0xAD if getattr(leader_cfg, "command_high_follow", False) else 0x00
            leader_arm = candidate

    follower_arm = getattr(robot, "arm", None)
    if follower_arm is not None and hasattr(follower_arm, "JointCtrl"):
        _home_piper_arm(follower_arm, speed=speed)
        homed_any = True

    if leader_arm is not None:
        _home_leader_arm_smooth(leader_arm, mit_mode=leader_mit_mode, speed=speed)
        homed_any = True

    if homed_any:
        time.sleep(_PIPER_HOME_SETTLE_S)
        logging.info("Leader + follower slowly homed to default all-zero pose.")
    else:
        logging.warning("Home skipped: arms do not expose a Piper JointCtrl interface.")
