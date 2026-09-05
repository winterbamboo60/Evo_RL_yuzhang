#!/usr/bin/env python

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

"""
Helper to find the camera devices available in your system.

Example:

```shell
python -m lerobot.find_cameras
```
"""

# NOTE(Steven): RealSense can also be identified/opened as OpenCV cameras. If you know the camera is a RealSense, use the `lerobot.find_cameras realsense` flag to avoid confusion.
# NOTE(Steven): macOS cameras sometimes report different FPS at init time, not an issue here as we don't specify FPS when opening the cameras, but the information displayed might not be truthful.

import argparse
import concurrent.futures
import json
import logging
import math
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)


_RESERVED_REALSENSE_CONFIG_FIELDS = {"serial_number_or_name", "color_mode"}
_INTEGER_REALSENSE_CONFIG_FIELDS = {
    "fps",
    "width",
    "height",
    "rotation",
    "warmup_s",
    "manual_exposure_us",
    "manual_gain",
    "auto_exposure_limit_us",
    "auto_gain_limit",
    "white_balance_kelvin",
}
_NULLABLE_INTEGER_REALSENSE_CONFIG_FIELDS = {"fps", "width", "height", "white_balance_kelvin"}


def parse_camera_configs(value: str) -> dict[str, dict[str, Any]]:
    """Parse and validate RealSense overrides keyed by camera serial number."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict) or not parsed:
        raise argparse.ArgumentTypeError("must be a non-empty JSON object keyed by RealSense serial number")

    allowed_fields = {field.name for field in fields(RealSenseCameraConfig)}
    allowed_override_fields = allowed_fields - _RESERVED_REALSENSE_CONFIG_FIELDS
    camera_configs: dict[str, dict[str, Any]] = {}

    for raw_serial, raw_overrides in parsed.items():
        if not isinstance(raw_serial, str) or not raw_serial.strip():
            raise argparse.ArgumentTypeError("each camera serial must be a non-empty string")
        serial = raw_serial.strip()
        if serial in camera_configs:
            raise argparse.ArgumentTypeError(f"duplicate camera serial after trimming: {serial}")
        if not isinstance(raw_overrides, dict):
            raise argparse.ArgumentTypeError(f"camera {serial}: configuration must be a JSON object")

        reserved_fields = sorted(set(raw_overrides) & _RESERVED_REALSENSE_CONFIG_FIELDS)
        if reserved_fields:
            raise argparse.ArgumentTypeError(
                f"camera {serial}: fields are controlled by the CLI and cannot be overridden: "
                f"{', '.join(reserved_fields)}"
            )

        unknown_fields = sorted(set(raw_overrides) - allowed_override_fields)
        if unknown_fields:
            raise argparse.ArgumentTypeError(
                f"camera {serial}: unknown configuration fields: {', '.join(unknown_fields)}"
            )

        for field_name in _INTEGER_REALSENSE_CONFIG_FIELDS & set(raw_overrides):
            field_value = raw_overrides[field_name]
            if field_value is None and field_name in _NULLABLE_INTEGER_REALSENSE_CONFIG_FIELDS:
                continue
            if not isinstance(field_value, int) or isinstance(field_value, bool):
                raise argparse.ArgumentTypeError(f"camera {serial}: {field_name} must be an integer")

        if "use_depth" in raw_overrides and not isinstance(raw_overrides["use_depth"], bool):
            raise argparse.ArgumentTypeError(f"camera {serial}: use_depth must be true or false")

        roi = raw_overrides.get("auto_exposure_roi")
        if roi is not None and (
            not isinstance(roi, list)
            or any(not isinstance(coordinate, int) or isinstance(coordinate, bool) for coordinate in roi)
        ):
            raise argparse.ArgumentTypeError(
                f"camera {serial}: auto_exposure_roi must be a JSON array containing integers"
            )

        if raw_overrides.get("warmup_s", 0) < 0:
            raise argparse.ArgumentTypeError(f"camera {serial}: warmup_s must be non-negative")
        for field_name in ("fps", "width", "height"):
            field_value = raw_overrides.get(field_name)
            if field_value is not None and field_value <= 0:
                raise argparse.ArgumentTypeError(f"camera {serial}: {field_name} must be greater than zero")

        try:
            RealSenseCameraConfig(
                serial_number_or_name=serial,
                color_mode=ColorMode.RGB,
                **raw_overrides,
            )
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"camera {serial}: {exc}") from exc

        camera_configs[serial] = raw_overrides

    return camera_configs


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point CLI value."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def find_all_opencv_cameras() -> List[Dict[str, Any]]:
    """
    Finds all available OpenCV cameras plugged into the system.

    Returns:
        A list of all available OpenCV cameras with their metadata.
    """
    all_opencv_cameras_info: List[Dict[str, Any]] = []
    logger.info("Searching for OpenCV cameras...")
    try:
        opencv_cameras = OpenCVCamera.find_cameras()
        for cam_info in opencv_cameras:
            all_opencv_cameras_info.append(cam_info)
        logger.info(f"Found {len(opencv_cameras)} OpenCV cameras.")
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")

    return all_opencv_cameras_info


def find_all_realsense_cameras() -> List[Dict[str, Any]]:
    """
    Finds all available RealSense cameras plugged into the system.

    Returns:
        A list of all available RealSense cameras with their metadata.
    """
    all_realsense_cameras_info: List[Dict[str, Any]] = []
    logger.info("Searching for RealSense cameras...")
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        for cam_info in realsense_cameras:
            all_realsense_cameras_info.append(cam_info)
        logger.info(f"Found {len(realsense_cameras)} RealSense cameras.")
    except ImportError:
        logger.warning("Skipping RealSense camera search: pyrealsense2 library not found or not importable.")
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")

    return all_realsense_cameras_info


def find_and_print_cameras(camera_type_filter: str | None = None) -> List[Dict[str, Any]]:
    """
    Finds available cameras based on an optional filter and prints their information.

    Args:
        camera_type_filter: Optional string to filter cameras ("realsense" or "opencv").
                            If None, lists all cameras.

    Returns:
        A list of all available cameras matching the filter, with their metadata.
    """
    all_cameras_info: List[Dict[str, Any]] = []

    if camera_type_filter:
        camera_type_filter = camera_type_filter.lower()

    if camera_type_filter is None or camera_type_filter == "opencv":
        all_cameras_info.extend(find_all_opencv_cameras())
    if camera_type_filter is None or camera_type_filter == "realsense":
        all_cameras_info.extend(find_all_realsense_cameras())

    if not all_cameras_info:
        if camera_type_filter:
            logger.warning(f"No {camera_type_filter} cameras were detected.")
        else:
            logger.warning("No cameras (OpenCV or RealSense) were detected.")
    else:
        print("\n--- Detected Cameras ---")
        for i, cam_info in enumerate(all_cameras_info):
            print(f"Camera #{i}:")
            for key, value in cam_info.items():
                if key == "default_stream_profile" and isinstance(value, dict):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                else:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    return all_cameras_info


def save_image(
    img_array: np.ndarray,
    camera_identifier: str | int,
    images_dir: Path,
    camera_type: str,
):
    """
    Saves a single image to disk using Pillow. Handles color conversion if necessary.
    """
    try:
        img = Image.fromarray(img_array, mode="RGB")

        safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")
        filename_prefix = f"{camera_type.lower()}_{safe_identifier}"
        filename = f"{filename_prefix}.png"

        path = images_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        logger.info(f"Saved image: {path}")
    except Exception as e:
        logger.error(f"Failed to save image for camera {camera_identifier} (type {camera_type}): {e}")


def create_camera_instance(
    cam_meta: Dict[str, Any],
    camera_configs: dict[str, dict[str, Any]] | None = None,
) -> Dict[str, Any] | None:
    """Create and connect to a camera instance based on metadata."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance = None
    overrides = (camera_configs or {}).get(str(cam_id), {})
    profile_name = "configured profile" if overrides else "default profile"

    logger.info(f"Preparing {cam_type} ID {cam_id} with {profile_name}")

    try:
        if cam_type == "OpenCV":
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = OpenCVCamera(cv_config)
        elif cam_type == "RealSense":
            rs_config = RealSenseCameraConfig(
                serial_number_or_name=str(cam_id),
                color_mode=ColorMode.RGB,
                **overrides,
            )
            instance = RealSenseCamera(rs_config)
        else:
            logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
            return None

        if instance:
            logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
            instance.connect(warmup=cam_type == "RealSense")
            return {"instance": instance, "meta": cam_meta}
    except Exception as e:
        logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
        if instance and instance.is_connected:
            instance.disconnect()
        return None


def process_camera_image(
    cam_dict: Dict[str, Any], output_dir: Path, current_time: float
) -> concurrent.futures.Future | None:
    """Capture and process an image from a single camera."""
    cam = cam_dict["instance"]
    meta = cam_dict["meta"]
    cam_type_str = str(meta.get("type", "unknown"))
    cam_id_str = str(meta.get("id", "unknown"))

    try:
        image_data = cam.read()

        return save_image(
            image_data,
            cam_id_str,
            output_dir,
            cam_type_str,
        )
    except TimeoutError:
        logger.warning(
            f"Timeout reading from {cam_type_str} camera {cam_id_str} at time {current_time:.2f}s."
        )
    except Exception as e:
        logger.error(f"Error reading from {cam_type_str} camera {cam_id_str}: {e}")
    return None


def cleanup_cameras(cameras_to_use: List[Dict[str, Any]]):
    """Disconnect all cameras."""
    logger.info(f"Disconnecting {len(cameras_to_use)} cameras...")
    for cam_dict in cameras_to_use:
        try:
            if cam_dict["instance"] and cam_dict["instance"].is_connected:
                cam_dict["instance"].disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}")


def select_configured_realsense_cameras(
    all_camera_metadata: list[dict[str, Any]],
    camera_configs: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Select requested RealSense devices and fail if any configured serial is absent."""
    if camera_configs is None:
        return all_camera_metadata

    requested_serials = set(camera_configs)
    detected_serials = {
        str(cam_meta.get("id"))
        for cam_meta in all_camera_metadata
        if cam_meta.get("type") == "RealSense"
    }
    missing_serials = sorted(requested_serials - detected_serials)
    if missing_serials:
        detected_text = ", ".join(sorted(detected_serials)) or "none"
        raise RuntimeError(
            "Configured RealSense camera(s) not detected: "
            f"{', '.join(missing_serials)}. Detected serials: {detected_text}."
        )

    ignored_serials = sorted(detected_serials - requested_serials)
    if ignored_serials:
        logger.info("Ignoring unconfigured RealSense camera(s): %s", ", ".join(ignored_serials))

    return [
        cam_meta
        for cam_meta in all_camera_metadata
        if cam_meta.get("type") == "RealSense" and str(cam_meta.get("id")) in requested_serials
    ]


def save_images_from_all_cameras(
    output_dir: Path,
    record_time_s: float = 2.0,
    camera_type: str | None = None,
    camera_configs: dict[str, dict[str, Any]] | None = None,
):
    """
    Connects to detected cameras (optionally filtered by type) and saves images from each.
    Uses default stream profiles for width, height, and FPS.

    Args:
        output_dir: Directory to save images.
        record_time_s: Duration in seconds to record images.
        camera_type: Optional string to filter cameras ("realsense" or "opencv").
                            If None, uses all detected cameras.
        camera_configs: Optional RealSense configuration overrides keyed by serial number.
                        When provided, only the listed RealSense devices are used.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving images to {output_dir}")
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)
    all_camera_metadata = select_configured_realsense_cameras(all_camera_metadata, camera_configs)

    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot save images.")
        return

    cameras_to_use = []
    failed_camera_ids = []
    for cam_meta in all_camera_metadata:
        camera_instance = create_camera_instance(cam_meta, camera_configs)
        if camera_instance:
            cameras_to_use.append(camera_instance)
        else:
            failed_camera_ids.append(str(cam_meta.get("id")))

    if camera_configs is not None and failed_camera_ids:
        cleanup_cameras(cameras_to_use)
        raise RuntimeError(f"Failed to connect configured camera(s): {', '.join(failed_camera_ids)}.")

    if not cameras_to_use:
        logger.warning("No cameras could be connected. Aborting image save.")
        return

    logger.info(f"Starting image capture for {record_time_s} seconds from {len(cameras_to_use)} cameras.")
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras_to_use) * 2) as executor:
        try:
            while time.perf_counter() - start_time < record_time_s:
                futures = []
                current_capture_time = time.perf_counter()

                for cam_dict in cameras_to_use:
                    future = process_camera_image(cam_dict, output_dir, current_capture_time)
                    if future:
                        futures.append(future)

                if futures:
                    concurrent.futures.wait(futures)

        except KeyboardInterrupt:
            logger.info("Capture interrupted by user.")
        finally:
            print("\nFinalizing image saving...")
            executor.shutdown(wait=True)
            cleanup_cameras(cameras_to_use)
            logger.info(f"Image capture finished. Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified camera utility script for listing cameras and capturing images."
    )

    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Specify camera type to capture from (e.g., 'realsense', 'opencv'). Captures from all if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="Directory to save images. Default: outputs/captured_images",
    )
    parser.add_argument(
        "--record-time-s",
        type=positive_float,
        default=6.0,
        help="Time duration to attempt capturing frames. Default: 6 seconds.",
    )
    parser.add_argument(
        "--camera-configs",
        type=parse_camera_configs,
        default=None,
        help=(
            "Strict JSON object mapping RealSense serial numbers to per-camera configuration overrides. "
            "Only listed devices are captured; width, height, and fps must be supplied together."
        ),
    )
    args = parser.parse_args()
    if args.camera_configs is not None and args.camera_type != "realsense":
        parser.error("--camera-configs requires camera_type 'realsense'")

    try:
        save_images_from_all_cameras(**vars(args))
    except RuntimeError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")


if __name__ == "__main__":
    main()
