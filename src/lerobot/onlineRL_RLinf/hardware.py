"""Single-follower Piper control with leader mirroring and dry-run doubles."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import torch

from .config import CameraConfig, HardwareConfig
from .safety import safe_action


class PiperHardware(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def observation(self) -> dict: ...
    def send_automatic(self, action: torch.Tensor) -> torch.Tensor: ...
    def send_human(self) -> torch.Tensor: ...
    def set_manual_control(self, enabled: bool) -> None: ...
    def reset_to(self, pose: torch.Tensor) -> None: ...


class FakePiperHardware:
    """Deterministic hardware double; it never opens CAN devices."""

    def __init__(self, cfg: HardwareConfig, action_dim: int = 7, display_data: bool = False):
        self.cfg, self.action_dim = cfg, action_dim
        self.position = torch.zeros(action_dim)
        self.leader_action = torch.zeros(action_dim)
        self.manual = False
        self.connected = False
        self.automatic_calls = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def observation(self) -> dict:
        return {
            "state": self.position.clone(),
            "top_image": torch.zeros(3, 32, 32),
            "wrist_image": torch.zeros(3, 32, 32),
        }

    def send_automatic(self, action: torch.Tensor) -> torch.Tensor:
        executed = safe_action(action, self.action_dim, self.cfg.action_limit)
        self.position, self.leader_action = executed.clone(), executed.clone()
        self.automatic_calls += 1
        return executed

    def send_human(self) -> torch.Tensor:
        self.position = safe_action(self.leader_action, self.action_dim, self.cfg.action_limit)
        return self.position.clone()

    def set_manual_control(self, enabled: bool) -> None:
        self.manual = enabled

    def reset_to(self, pose: torch.Tensor) -> None:
        self.position = safe_action(pose, self.action_dim, self.cfg.action_limit)
        self.leader_action = self.position.clone()


class RealPiperHardware:
    """Thin adapter over existing LeRobot Piper follower/leader implementations."""

    def __init__(self, cfg: HardwareConfig, action_dim: int = 7, display_data: bool = False):
        self.cfg, self.action_dim = cfg, action_dim
        self.display_data = display_data
        self.follower = None
        self.leader = None
        self.top_camera = None
        self.wrist_camera = None
        self.initial_pose: torch.Tensor | None = None
        self.leader_feedback_calls = 0
        self._sync_pool: ThreadPoolExecutor | None = None

    @staticmethod
    def _open_camera(camera_cfg: CameraConfig, name: str):
        """Open a LeRobot OpenCV camera with its background read thread."""
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        camera = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=camera_cfg.index_or_path,
                width=camera_cfg.width,
                height=camera_cfg.height,
                fps=camera_cfg.fps,
            )
        )
        try:
            camera.connect()
        except BaseException as exc:
            raise RuntimeError(f"Unable to open {name} camera: {camera_cfg.index_or_path!r}.") from exc
        return camera

    @staticmethod
    def _read_camera(camera, name: str):
        """Return the latest RGB uint8 HWC frame from LeRobot's async camera reader."""
        frame = camera.async_read()
        if frame is None:
            raise RuntimeError(f"Failed to capture a frame from {name} camera.")
        return frame

    @staticmethod
    def _hold_sdk_arm_current_pose(arm, speed: int) -> bool:
        from lerobot.utils.piper_sdk import PIPER_JOINT_NAMES

        joint_msg = arm.GetArmJointMsgs()
        joint_state = getattr(joint_msg, "joint_state", None)
        if joint_state is None:
            return False
        joints = [int(getattr(joint_state, joint_name, 0)) for joint_name in PIPER_JOINT_NAMES]
        gripper_msg = arm.GetArmGripperMsgs()
        gripper_state = getattr(gripper_msg, "gripper_state", None)
        gripper = abs(int(getattr(gripper_state, "grippers_angle", 0))) if gripper_state is not None else 0
        arm.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        arm.JointCtrl(*joints)
        arm.GripperCtrl(gripper, 1000, 0x01, 0x00)
        return True

    def _hold_follower_current_pose(self) -> None:
        if self.follower is not None and self._hold_sdk_arm_current_pose(self.follower.arm, self.cfg.follower_speed_ratio):
            logging.info("Follower holding current pose after connect.")

    def _prime_leader_feedback(self) -> None:
        if self.leader is None:
            return
        self.leader.set_manual_control(False)
        action = self.leader.get_action()
        self.leader.send_feedback(action)
        self.leader_feedback_calls += 1
        logging.info("Leader feedback primed with current pose after connect.")

    @staticmethod
    def _action_dict(action: torch.Tensor) -> dict[str, float]:
        from lerobot.utils.piper_sdk import PIPER_ACTION_KEYS

        return {key: float(value) for key, value in zip(PIPER_ACTION_KEYS, action.tolist(), strict=True)}

    @staticmethod
    def _tensor_from_action(action: dict) -> torch.Tensor:
        from lerobot.utils.piper_sdk import PIPER_ACTION_KEYS

        return torch.tensor([float(action[key]) for key in PIPER_ACTION_KEYS], dtype=torch.float32)

    @staticmethod
    def _require_calibrated(device, command: str) -> None:
        if not device.is_calibrated:
            raise RuntimeError(
                f"{device} has no valid calibration file. Run `{command}` before starting onlineRL."
            )

    def connect(self) -> None:
        if (
            self.follower is not None
            and self.leader is not None
            and self.top_camera is not None
            and self.wrist_camera is not None
            and self._sync_pool is not None
            and getattr(self.follower, "is_connected", False)
            and getattr(self.leader, "is_connected", False)
        ):
            logging.info("RealPiperHardware already connected; reusing existing connection.")
            return

        from lerobot.robots.piper_follower import PiperFollower
        from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
        from lerobot.teleoperators.piper_leader import PiperLeader
        from lerobot.teleoperators.piper_leader.config_piper_leader import PiperLeaderConfig

        self.follower = PiperFollower(
            PiperFollowerConfig(
                port=self.cfg.follower_port,
                id=self.cfg.follower_id,
                speed_ratio=self.cfg.follower_speed_ratio,
            )
        )
        self.leader = PiperLeader(
            PiperLeaderConfig(
                port=self.cfg.leader_port,
                id=self.cfg.leader_id,
                command_speed_ratio=self.cfg.leader_speed_ratio,
                manual_control=False,
            )
        )
        self._require_calibrated(
            self.follower,
            f"lerobot-calibrate --robot.type=piper_follower --robot.port={self.cfg.follower_port} --robot.id={self.follower.id}",
        )
        self._require_calibrated(
            self.leader,
            f"lerobot-calibrate --teleop.type=piper_leader --teleop.port={self.cfg.leader_port} --teleop.id={self.leader.id}",
        )
        self.leader.connect(calibrate=False)
        self._prime_leader_feedback()
        self.follower.connect(calibrate=False)
        self._hold_follower_current_pose()
        if self.display_data:
            from lerobot.utils.visualization_utils import init_rerun

            init_rerun(session_name="onlineRL")
        try:
            self.top_camera = self._open_camera(self.cfg.top_camera, "top")
            self.wrist_camera = self._open_camera(self.cfg.wrist_camera, "wrist")
        except BaseException:
            self.disconnect()
            raise
        self._sync_pool = ThreadPoolExecutor(max_workers=2)
        self.initial_pose = self._state_tensor(self.follower.get_observation())

    @staticmethod
    def _state_tensor(observation: dict) -> torch.Tensor:
        from lerobot.utils.piper_sdk import PIPER_ACTION_KEYS

        return torch.tensor([float(observation[key]) for key in PIPER_ACTION_KEYS], dtype=torch.float32)

    def disconnect(self) -> None:
        for camera in (self.top_camera, self.wrist_camera):
            if camera is not None:
                camera.disconnect()
        self.top_camera = None
        self.wrist_camera = None
        if self._sync_pool is not None:
            self._sync_pool.shutdown(wait=True)
            self._sync_pool = None
        if self.display_data:
            import rerun as rr

            rr.rerun_shutdown()
        for device in (self.leader, self.follower):
            if device is not None and device.is_connected:
                device.disconnect()

    def observation(self) -> dict:
        if self.follower is None:
            raise RuntimeError("Hardware is not connected.")
        if self.top_camera is None or self.wrist_camera is None:
            raise RuntimeError("Hardware cameras are not connected.")
        raw = self.follower.get_observation()
        state = self._state_tensor(raw)
        top_image = self._read_camera(self.top_camera, "top")
        wrist_image = self._read_camera(self.wrist_camera, "wrist")
        if self.display_data:
            from lerobot.utils.visualization_utils import log_rerun_data

            log_rerun_data(
                observation={"state": state.numpy(), "images.top": top_image, "images.wrist": wrist_image}
            )
        return {
            "state": state,
            "top_image": top_image,
            "wrist_image": wrist_image,
            "raw": raw,
        }

    def send_automatic(self, action: torch.Tensor) -> torch.Tensor:
        if self.follower is None or self.leader is None:
            raise RuntimeError("Hardware is not connected.")
        if self._sync_pool is None:
            raise RuntimeError("Hardware sync pool is not connected.")
        action = safe_action(action, self.action_dim, self.cfg.action_limit)
        command = self._action_dict(action)
        follower_future = self._sync_pool.submit(self.follower.send_action, command)
        leader_future = self._sync_pool.submit(self.leader.send_feedback, command)
        actual = follower_future.result()
        leader_future.result()
        self.leader_feedback_calls += 1
        if self.leader_feedback_calls <= 3 or self.leader_feedback_calls % 10 == 0:
            logging.info("Leader feedback command sent: count=%d", self.leader_feedback_calls)
        return self._tensor_from_action(actual)

    def send_human(self) -> torch.Tensor:
        if self.follower is None or self.leader is None:
            raise RuntimeError("Hardware is not connected.")
        leader_action = self.leader.get_action()
        actual = self.follower.send_action(leader_action)
        return self._tensor_from_action(actual)

    def set_manual_control(self, enabled: bool) -> None:
        if self.leader is not None and hasattr(self.leader, "set_manual_control"):
            self.leader.set_manual_control(enabled)

    def reset_to(self, pose: torch.Tensor) -> None:
        if self.follower is None or self.leader is None:
            raise RuntimeError("Hardware is not connected.")
        start = self.observation()["state"]
        steps = max(1, int(self.cfg.reset_duration_s * 20))
        for idx in range(1, steps + 1):
            action = start + (pose - start) * (idx / steps)
            self.send_automatic(action)
            time.sleep(self.cfg.reset_duration_s / steps)


def make_hardware(cfg: HardwareConfig, dry_run: bool, display_data: bool = False) -> PiperHardware:
    return FakePiperHardware(cfg, display_data=display_data) if dry_run else RealPiperHardware(cfg, display_data=display_data)
