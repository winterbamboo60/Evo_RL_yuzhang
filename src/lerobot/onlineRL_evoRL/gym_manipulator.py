"""Current-LeRobot hardware session used exclusively by actor_new.

Action dimensions come from robot.action_features. This handles Piper and
BiPiper without assuming a robot.bus object or a fixed joint count.
"""

from __future__ import annotations

from typing import Any

from lerobot.cameras import opencv, realsense  # noqa: F401
from lerobot.robots import bi_piper_follower, make_robot_from_config, piper_follower  # noqa: F401
from lerobot.teleoperators import bi_piper_leader, make_teleoperator_from_config, piper_leader  # noqa: F401
from lerobot.utils.robot_utils import precise_sleep


class RobotEnv:
    """Connection and action adapter for current Robot/Teleoperator APIs."""

    def __init__(self, robot: Any, *, reset_time_s: float = 0.0) -> None:
        self.robot = robot
        self.reset_time_s = max(float(reset_time_s), 0.0)
        self.current_step = 0
        self.last_raw_observation: dict[str, Any] | None = None
        if not robot.is_connected:
            robot.connect()
        self.initial_action = self.current_pose_action()

    @property
    def action_keys(self) -> list[str]:
        return list(self.robot.action_features)

    def read_raw_observation(self) -> dict[str, Any]:
        self.last_raw_observation = self.robot.get_observation()
        return self.last_raw_observation

    def current_pose_action(self, observation: dict[str, Any] | None = None) -> dict[str, float]:
        observation = observation or self.read_raw_observation()
        missing = [key for key in self.action_keys if key not in observation]
        if missing:
            raise KeyError(f"Robot observation is missing action pose keys: {missing}")
        return {key: float(observation[key]) for key in self.action_keys}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        missing = [key for key in self.action_keys if key not in action]
        if missing:
            raise ValueError(f"Action is missing robot keys: {missing}")
        sent = self.robot.send_action({key: float(action[key]) for key in self.action_keys})
        self.current_step += 1
        return sent

    def reset(self) -> dict[str, Any]:
        self.current_step = 0
        if self.reset_time_s:
            precise_sleep(self.reset_time_s)
        return self.read_raw_observation()

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()


def make_robot_env(cfg: Any, *, connect_teleop: bool = True) -> tuple[RobotEnv, Any | None]:
    """Create the robot and optionally create/connect configured leader arm(s).

    connect_teleop=False is the online equivalent of --can0.control false:
    CAN0/CAN2 are not touched.
    """
    if getattr(cfg, "name", "real_robot") == "gym_hil":
        raise ValueError("actor_new currently targets real Piper/BiPiper hardware, not gym_hil")
    if cfg.robot is None:
        raise ValueError("Online actor requires env.robot")
    robot = make_robot_from_config(cfg.robot)
    reset_cfg = getattr(cfg.processor, "reset", None)
    env = RobotEnv(robot, reset_time_s=getattr(reset_cfg, "reset_time_s", 0.0))
    if not connect_teleop:
        return env, None
    if cfg.teleop is None:
        raise ValueError("can0.control=true requires env.teleop")
    teleop = make_teleoperator_from_config(cfg.teleop)
    if not teleop.is_connected:
        teleop.connect()
    return env, teleop
