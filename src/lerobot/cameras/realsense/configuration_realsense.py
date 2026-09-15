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
        use_rgb: Whether to enable the color stream. Defaults to True.
        use_depth: Whether to enable depth stream. Defaults to False.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        exposure: Manual exposure value for the color sensor. When set, auto-exposure is
            disabled and this fixed value is used. Valid ranges are camera-model specific
            and reported if the value is rejected. Defaults to None (leave unchanged).
        gain: Manual gain value for the color sensor. When set, auto-exposure is disabled
            and this fixed gain is used, which also freezes exposure at its current value
            when no exposure is configured. Valid ranges are camera-model specific and
            reported if the value is rejected. Defaults to None (leave unchanged).
        white_balance: Manual white balance value for the color sensor. When set, auto
            white balance is disabled and this fixed value is used. Valid ranges are
            camera-model specific and reported if the value is rejected. Defaults to None
            (leave unchanged).

    Note:
        - Either name or serial_number must be specified.
        - At least one of `use_rgb` or `use_depth` must be enabled.
        - Depth stream configuration (if enabled) will use the same FPS as the color stream.
        - The actual resolution and FPS may be adjusted by the camera to the nearest supported mode.
        - For `fps`, `width` and `height`, either all of them need to be set, or none of them.
    """

    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    use_rgb: bool = True
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    exposure: int | None = None
    gain: int | None = None
    white_balance: int | None = None

    # Extended controls used by the EvoRL recording rigs. Values are passed in
    # native RealSense SDK units; query the device range with lerobot-find-cameras.
    exposure_mode: str = "device_default"
    auto_exposure_limit: int | None = None
    auto_gain_limit: int | None = None
    auto_exposure_roi: tuple[int, int, int, int] | list[int] | None = None

    # Backwards-compatible aliases for the 0901 launch scripts. New configs should
    # prefer exposure, gain, white_balance and auto_exposure_limit.
    manual_exposure_us: int | None = None
    manual_gain: int | None = None
    auto_exposure_limit_us: int | None = None
    white_balance_kelvin: int | None = None

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if self.exposure_mode not in {"device_default", "auto", "manual"}:
            raise ValueError("`exposure_mode` must be device_default, auto, or manual.")

        aliases = (
            ("manual_exposure_us", "exposure"),
            ("manual_gain", "gain"),
            ("auto_exposure_limit_us", "auto_exposure_limit"),
            ("white_balance_kelvin", "white_balance"),
        )
        for alias_name, current_name in aliases:
            alias_value = getattr(self, alias_name)
            current_value = getattr(self, current_name)
            if alias_value is not None and current_value is not None and alias_value != current_value:
                raise ValueError(
                    f"Conflicting RealSense values: `{alias_name}`={alias_value} and "
                    f"`{current_name}`={current_value}."
                )
            if current_value is None and alias_value is not None:
                setattr(self, current_name, alias_value)

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of `use_rgb` or `use_depth` must be enabled.")

        manual_color_options = {
            "exposure": self.exposure,
            "gain": self.gain,
            "white_balance": self.white_balance,
        }
        configured_color_options = [name for name, value in manual_color_options.items() if value is not None]
        if configured_color_options and not self.use_rgb:
            raise ValueError(
                "Manual color sensor options require `use_rgb=True`. "
                f"Configured options: {configured_color_options}."
            )

        positive_controls = {
            "exposure": self.exposure,
            "white_balance": self.white_balance,
            "auto_exposure_limit": self.auto_exposure_limit,
        }
        for name, value in positive_controls.items():
            if value is not None and value <= 0:
                raise ValueError(f"`{name}` must be greater than zero.")
        for name, value in {"gain": self.gain, "auto_gain_limit": self.auto_gain_limit}.items():
            if value is not None and value < 0:
                raise ValueError(f"`{name}` must be non-negative.")

        if self.auto_exposure_roi is not None:
            if len(self.auto_exposure_roi) != 4:
                raise ValueError("`auto_exposure_roi` must be [min_x, min_y, max_x, max_y].")
            min_x, min_y, max_x, max_y = self.auto_exposure_roi
            if min_x < 0 or min_y < 0 or min_x >= max_x or min_y >= max_y:
                raise ValueError("`auto_exposure_roi` must contain non-negative coordinates with min < max.")
            if (
                self.width is not None
                and self.height is not None
                and (max_x > self.width or max_y > self.height)
            ):
                raise ValueError("`auto_exposure_roi` must fit inside the configured capture size.")

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
