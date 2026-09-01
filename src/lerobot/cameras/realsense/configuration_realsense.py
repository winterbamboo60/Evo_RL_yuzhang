# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Literal

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class RealSenseCameraConfig(CameraConfig):
    """Configuration class for Intel RealSense cameras.

    This class provides specialized configuration options for Intel RealSense cameras,
    including support for depth sensing and device identification via serial number or name.

    Example configurations for Intel RealSense D405:
    ```python
    # Basic configurations
    RealSenseCameraConfig("0123456789", 30, 1280, 720)  # 1280x720 @ 30FPS
    RealSenseCameraConfig("0123456789", 60, 640, 480)  # 640x480 @ 60FPS

    # Advanced configurations
    RealSenseCameraConfig("0123456789", 30, 640, 480, use_depth=True)  # With depth sensing
    RealSenseCameraConfig("0123456789", 30, 640, 480, rotation=Cv2Rotation.ROTATE_90)  # With 90° rotation
    ```

    Attributes:
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        serial_number_or_name: Unique serial number or human-readable name to identify the camera.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        use_depth: Whether to enable depth stream. Defaults to False.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        exposure_mode: Exposure control mode selected in code.
        manual_exposure_us: Exposure time used by manual mode, in microseconds.
        manual_gain: Sensor gain used by manual mode.
        auto_exposure_limit_us: Maximum exposure time used by auto mode, in microseconds.
        auto_gain_limit: Maximum sensor gain used by auto mode.
        auto_exposure_roi: Optional auto-exposure ROI as [min_x, min_y, max_x, max_y].
        white_balance_kelvin: Optional fixed white balance. None keeps automatic white balance.

    Note:
        - Either name or serial_number must be specified.
        - Depth stream configuration (if enabled) will use the same FPS as the color stream.
        - The actual resolution and FPS may be adjusted by the camera to the nearest supported mode.
        - For `fps`, `width` and `height`, either all of them need to be set, or none of them.
    """

    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    # Change these defaults to tune all D405 cameras without changing launch commands.
    exposure_mode: Literal["device_default", "auto", "manual"] = "device_default"
    manual_exposure_us: int = 20000
    manual_gain: int = 16
    auto_exposure_limit_us: int = 10000
    auto_gain_limit: int = 32
    auto_exposure_roi: list[int] | None = None
    white_balance_kelvin: int | None = None

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )

        if self.exposure_mode not in ("device_default", "auto", "manual"):
            raise ValueError("`exposure_mode` must be one of: device_default, auto, or manual.")

        positive_controls = {
            "manual_exposure_us": self.manual_exposure_us,
            "manual_gain": self.manual_gain,
            "auto_exposure_limit_us": self.auto_exposure_limit_us,
            "auto_gain_limit": self.auto_gain_limit,
        }
        if self.white_balance_kelvin is not None:
            positive_controls["white_balance_kelvin"] = self.white_balance_kelvin
        for name, value in positive_controls.items():
            if value <= 0:
                raise ValueError(f"`{name}` must be greater than zero.")

        if self.auto_exposure_roi is not None:
            if len(self.auto_exposure_roi) != 4:
                raise ValueError("`auto_exposure_roi` must be [min_x, min_y, max_x, max_y].")
            min_x, min_y, max_x, max_y = self.auto_exposure_roi
            if min_x < 0 or min_y < 0 or min_x >= max_x or min_y >= max_y:
                raise ValueError("`auto_exposure_roi` must contain non-negative coordinates with min < max.")
