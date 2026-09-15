"""Compatibility export of current joint observation processors."""

from lerobot.rl.joint_observations_processor import *  # noqa: F403
from lerobot.rl.joint_observations_processor import JointVelocityProcessorStep, MotorCurrentProcessorStep

__all__ = ["JointVelocityProcessorStep", "MotorCurrentProcessorStep"]
