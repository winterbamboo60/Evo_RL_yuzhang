#!/usr/bin/env python
"""Load VLA, select a task with YOLO, then execute with human intervention."""

from __future__ import annotations

import importlib
import logging
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

_WORKER_FLAG = "--yolo-worker"
_WORKER_AUTHKEY = b"lerobot-yolo-task-v1"
_WORKER_MODE = __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == _WORKER_FLAG

YOLO_COMMAND_COLORS = {1: "R", 2: "B", 3: "Y"}
YOLO_COMMAND_VIDEO_TEXT = {
    1: "CMD 1: GRAB RED",
    2: "CMD 2: GRAB BLUE",
    3: "CMD 3: GRAB YELLOW",
}
YOLO_RESET_VIDEO_TEXT = "CMD 4: RETURN HOME"
_RESET_TO_DETECTION = object()
IDLE_PROMPT = (
    "\n[YOLO] 请输入命令：\n"
    "  1 = 抓取红色方块对应位置的杯子\n"
    "  2 = 抓取蓝色方块对应位置的杯子\n"
    "  3 = 抓取黄色方块对应位置的杯子\n"
    "  VLA 执行中按 4 = 停止并返回初始状态\n"
    "请选择 [1/2/3]: "
)


def _valid_order(order: Any) -> bool:
    return isinstance(order, list) and len(order) == 3 and set(order) == set(YOLO_COMMAND_COLORS.values())


def _restart_yolo_monitor(monitor: Any, order: list[str]) -> None:
    if not _valid_order(order):
        raise ValueError(f"Cannot restart YOLO with invalid order: {order!r}")
    monitor.restart_initial_order_detection(order)


def _apply_yolo_control(monitor: Any, packet: dict[str, Any]) -> bool:
    """Mirror inference_final's detection/freeze transitions in the worker."""
    command = packet.get("command")
    if command == "start_task":
        monitor.freeze_cup_labels()
        return False
    if command == "restart":
        _restart_yolo_monitor(monitor, packet.get("order"))
        return True
    raise ValueError(f"Unsupported YOLO worker command: {packet!r}")


def task_for_command(
    command: str,
    order: list[str],
    task_templates: dict[str, str],
    identity_ambiguous: bool = False,
) -> str | None:
    if identity_ambiguous or not _valid_order(order):
        return None
    color = YOLO_COMMAND_COLORS.get(int(command)) if command in {"1", "2", "3"} else None
    if color not in order:
        return None
    position = ("left", "middle", "right")[order.index(color)]
    return task_templates[position]


def _process_yolo_frame(
    monitor: Any,
    frame: Any,
    capture_timestamp: float,
    fps: float,
    *,
    detection_enabled: bool = True,
    overlay_text: str | None = None,
    cv2_module: Any = None,
):
    """Detect before selection; record without inference during VLA execution."""
    raw_frame = frame.copy() if monitor.video_path else None
    if detection_enabled:
        _, _, cup_count, detected_order = monitor.process(frame)
        if detected_order is not None:
            monitor.lock_cup_labels(detected_order)
        identity_ambiguous = bool(monitor.identity_ambiguous)
    else:
        cup_count, detected_order, identity_ambiguous = 0, None, False
        if overlay_text:
            cv2_module.putText(
                frame, overlay_text, (12, 38), cv2_module.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5
            )
            cv2_module.putText(
                frame, overlay_text, (12, 38), cv2_module.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
    if monitor.video_path:
        monitor.write_video_frames(raw_frame, frame, capture_timestamp, fps)
    return cup_count, detected_order, identity_ambiguous


def _run_yolo_worker(socket_path: str) -> None:
    """Run only YOLO in its own environment; no robot or LeRobot recording imports."""
    from multiprocessing.connection import Listener

    connection = None
    monitor = None
    try:
        with Listener(socket_path, family="AF_UNIX", authkey=_WORKER_AUTHKEY) as listener:
            connection = listener.accept()
            settings = connection.recv()
            module_name = settings.pop("module")
            display = settings.pop("display")

            import cv2
            import numpy as np

            module = importlib.import_module(module_name)
            yolo_cfg = module.InferenceConfig()
            for name, value in settings.items():
                if not hasattr(yolo_cfg, name):
                    raise AttributeError(f"{module_name}.InferenceConfig has no field {name!r}")
                setattr(yolo_cfg, name, value)
            monitor = module.TrackCameraMonitor(yolo_cfg)
            detection_enabled = True
            window_name = "YoloVla blocks (q/Esc to quit)"
            if display:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
            connection.send({"ok": True})

            while True:
                packet = connection.recv()
                if packet is None:
                    break
                if isinstance(packet, dict):
                    detection_enabled = _apply_yolo_control(monitor, packet)
                    connection.send({"ok": True})
                    continue
                shape, frame_bytes, color_mode, capture_timestamp, fps, overlay_text = packet
                if color_mode not in {"rgb", "bgr"}:
                    raise ValueError(f"Unsupported YOLO camera color mode: {color_mode!r}")
                image = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(shape)
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if color_mode == "rgb" else image.copy()

                cup_count, detected_order, identity_ambiguous = _process_yolo_frame(
                    monitor,
                    frame,
                    capture_timestamp,
                    fps,
                    detection_enabled=detection_enabled,
                    overlay_text=overlay_text,
                    cv2_module=cv2,
                )

                quit_requested = False
                if display and overlay_text is None:
                    cv2.putText(
                        frame,
                        f"cups: {cup_count}  order: {detected_order or '(detecting)'}",
                        (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                if display:
                    cv2.imshow(window_name, frame)
                    quit_requested = cv2.waitKey(1) & 0xFF in (ord("q"), 27)
                connection.send(
                    {
                        "ok": True,
                        "order": list(detected_order) if detected_order is not None else None,
                        "cup_count": cup_count,
                        "identity_ambiguous": identity_ambiguous,
                        "quit": quit_requested,
                    }
                )
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        if connection is not None:
            with suppress(Exception):
                connection.send({"ok": False, "error": traceback.format_exc()})
        raise
    finally:
        if monitor is not None:
            monitor.close()
        with suppress(Exception):
            import cv2

            cv2.destroyAllWindows()
        if connection is not None:
            connection.close()


if not _WORKER_MODE:
    from multiprocessing.connection import Client

    import numpy as np

    from lerobot.configs import parser
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.datasets.utils import combine_feature_dicts
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor import make_default_processors
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.robots import make_robot_from_config
    from lerobot.scripts.lerobot_record import RecordConfig, _home_arms_to_default
    from lerobot.scripts.recording_hil import PolicySyncDualArmExecutor
    from lerobot.scripts.recording_loop import record_loop
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.utils.control_utils import init_keyboard_listener
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.utils import init_logging

    @dataclass
    class YoloHumanInloopRecordConfig(RecordConfig):
        # Reuse inference_final.py's detector/tracker in its native environment.
        yolo_python: str = "/home/hpc/miniconda3/envs/lerobot_lzh/bin/python"
        yolo_module: str = "lerobot.scripts.inference.inference_final"
        yolo_working_dir: str = "/home/hpc/lzh/cup_swap_tracking"
        yolo_camera: str = "top"
        yolo_weights: str = "/home/hpc/lzh/cup_swap_tracking/yoloe-26n-seg.pt"
        yolo_cup_weights: str | None = None
        yolo_tracking_dir: str = "/home/hpc/lzh/cup_swap_tracking_pkg_2/src"
        yolo_device: str = "0"
        yolo_display: bool = True
        yolo_out_video: str | None = None
        yolo_out_csv: str | None = None
        yolo_conf: float = 0.02
        yolo_imgsz: int = 1280
        yolo_block_min_ratio: float = 0.02
        yolo_block_max_ratio: float = 0.08
        yolo_nms_dist: float = 30.0
        yolo_anchor_max_dist: float = 50.0
        yolo_anchor_confirm_hits: int = 3
        yolo_anchor_candidate_max_missed: int = 90
        yolo_startup_timeout_s: float = 180.0
        yolo_frame_timeout_s: float = 30.0
        yolo_detection_timeout_s: float = 0.0  # 0 means wait until the operator aborts.
        yolo_task_left: str = "Grab the left cup"
        yolo_task_middle: str = "Take the middle cup away"
        yolo_task_right: str = "Pick up the cup on the right"

        def __post_init__(self) -> None:
            super().__post_init__()
            if shutil.which(self.yolo_python) is None:
                raise FileNotFoundError(
                    f"YOLO Python does not exist or is not executable: {self.yolo_python}"
                )
            required = [
                Path(self.yolo_weights),
                Path(self.yolo_tracking_dir) / "track_video_blocks.py",
                Path(self.yolo_working_dir) / "mobileclip2_b.ts",
            ]
            if self.yolo_cup_weights:
                required.append(Path(self.yolo_cup_weights))
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"YOLO file does not exist: {', '.join(missing)}")
            if not self.yolo_camera:
                raise ValueError("`yolo_camera` must name one configured robot camera.")
            if (
                self.yolo_startup_timeout_s <= 0
                or self.yolo_frame_timeout_s <= 0
                or self.yolo_detection_timeout_s < 0
            ):
                raise ValueError(
                    "YOLO startup/frame timeouts must be > 0 and detection timeout must be >= 0."
                )
            if self.yolo_conf < 0 or self.yolo_imgsz <= 0:
                raise ValueError("`yolo_conf` must be >= 0 and `yolo_imgsz` must be > 0.")
            if not all((self.yolo_task_left, self.yolo_task_middle, self.yolo_task_right)):
                raise ValueError("YOLO task templates must be non-empty.")

    def _worker_settings(cfg: YoloHumanInloopRecordConfig) -> dict[str, Any]:
        return {
            "module": cfg.yolo_module,
            "display": cfg.yolo_display,
            "yolo_weights": cfg.yolo_weights,
            "yolo_cup_weights": cfg.yolo_cup_weights,
            "yolo_tracking_dir": cfg.yolo_tracking_dir,
            "yolo_device": cfg.yolo_device,
            "yolo_out_video": cfg.yolo_out_video,
            "yolo_out_csv": cfg.yolo_out_csv,
            "yolo_conf": cfg.yolo_conf,
            "yolo_imgsz": cfg.yolo_imgsz,
            "yolo_block_min_ratio": cfg.yolo_block_min_ratio,
            "yolo_block_max_ratio": cfg.yolo_block_max_ratio,
            "yolo_nms_dist": cfg.yolo_nms_dist,
            "yolo_anchor_max_dist": cfg.yolo_anchor_max_dist,
            "yolo_anchor_confirm_hits": cfg.yolo_anchor_confirm_hits,
            "yolo_anchor_candidate_max_missed": cfg.yolo_anchor_candidate_max_missed,
        }

    def _connect_to_worker(process: subprocess.Popen, socket_path: str, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"YOLO worker exited with code {process.returncode}")
            try:
                return Client(socket_path, family="AF_UNIX", authkey=_WORKER_AUTHKEY)
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.05)
        raise TimeoutError("Timed out connecting to the YOLO worker")

    def _send_yolo_frame(
        camera: Any,
        cfg: YoloHumanInloopRecordConfig,
        connection: Any,
        frame: np.ndarray,
        capture_timestamp: float,
        *,
        overlay_text: str | None = None,
    ) -> dict[str, Any]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"YOLO camera must return an HxWx3 frame, got {frame.shape}")
        camera_color_mode = getattr(camera, "color_mode", "rgb")
        color_mode = getattr(camera_color_mode, "value", camera_color_mode)
        camera_fps = float(getattr(camera, "fps", None) or 30.0)
        connection.send(
            (frame.shape, frame.tobytes(), color_mode, capture_timestamp, camera_fps, overlay_text)
        )
        if not connection.poll(cfg.yolo_frame_timeout_s):
            raise TimeoutError("Timed out processing a YOLO frame")
        response = connection.recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "YOLO frame processing failed"))
        return response

    def _send_yolo_control(
        cfg: YoloHumanInloopRecordConfig,
        connection: Any,
        command: str,
        order: list[str] | None = None,
    ) -> None:
        packet = {"command": command}
        if order is not None:
            packet["order"] = list(order)
        connection.send(packet)
        if not connection.poll(cfg.yolo_frame_timeout_s):
            raise TimeoutError(f"Timed out applying YOLO command {command!r}")
        response = connection.recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", f"YOLO command {command!r} failed"))

    def _stream_yolo_execution(
        camera: Any,
        cfg: YoloHumanInloopRecordConfig,
        connection: Any,
        stop_event: Event,
        events: dict[str, Any],
        errors: list[BaseException],
        overlay_text: str,
    ) -> None:
        """Keep the same annotated writer alive without blocking VLA control."""
        last_timestamp = None
        try:
            while not stop_event.is_set():
                capture_timestamp = getattr(camera, "latest_timestamp", None)
                if capture_timestamp is None or capture_timestamp == last_timestamp:
                    stop_event.wait(0.001)
                    continue
                frame = np.asarray(camera.read_latest(max_age_ms=1000), dtype=np.uint8)
                capture_timestamp = float(getattr(camera, "latest_timestamp", None) or capture_timestamp)
                if capture_timestamp == last_timestamp:
                    continue
                last_timestamp = capture_timestamp
                response = _send_yolo_frame(
                    camera,
                    cfg,
                    connection,
                    frame,
                    capture_timestamp,
                    overlay_text=(YOLO_RESET_VIDEO_TEXT if events.get("reset_episode") else overlay_text),
                )
                if response.get("quit"):
                    events["exit_early"] = True
                    return
        except BaseException as error:
            errors.append(error)
            logging.exception("YOLO execution-video stream stopped")

    def select_task_with_yolo(
        camera: Any,
        cfg: YoloHumanInloopRecordConfig,
        execute_task: Callable[[str, Any, str], Any] | None = None,
        on_detection_started: Callable[[], None] | None = None,
    ) -> Any:
        task_templates = {
            "left": cfg.yolo_task_left,
            "middle": cfg.yolo_task_middle,
            "right": cfg.yolo_task_right,
        }

        with tempfile.TemporaryDirectory(prefix="lerobot_yolo_") as temp_dir:
            socket_path = str(Path(temp_dir) / "worker.sock")
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            process = subprocess.Popen(
                [cfg.yolo_python, str(Path(__file__).resolve()), _WORKER_FLAG, socket_path],
                cwd=cfg.yolo_working_dir,
                env=env,
            )
            connection = None
            try:
                connection = _connect_to_worker(process, socket_path, cfg.yolo_startup_timeout_s)
                connection.send(_worker_settings(cfg))
                if not connection.poll(cfg.yolo_startup_timeout_s):
                    raise TimeoutError("Timed out while loading YOLO models")
                response = connection.recv()
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "YOLO worker failed to start"))

                print(IDLE_PROMPT, end="", flush=True)
                order: list[str] = []
                last_reported_order: list[str] = []
                identity_ambiguous = False
                started = time.monotonic()
                while True:
                    if cfg.yolo_detection_timeout_s and (
                        time.monotonic() - started >= cfg.yolo_detection_timeout_s
                    ):
                        raise TimeoutError("Timed out waiting for a YOLO task selection")

                    # ponytail: latest-frame sampling; add a producer FIFO only if YOLO falls below camera FPS.
                    frame = np.asarray(camera.async_read(timeout_ms=1000), dtype=np.uint8)
                    capture_timestamp = float(
                        getattr(camera, "latest_timestamp", None) or time.perf_counter()
                    )
                    response = _send_yolo_frame(
                        camera,
                        cfg,
                        connection,
                        frame,
                        capture_timestamp,
                    )
                    if response.get("quit"):
                        raise KeyboardInterrupt
                    if on_detection_started is not None:
                        on_detection_started()
                        on_detection_started = None
                    identity_ambiguous = bool(response.get("identity_ambiguous", False))
                    if response.get("order") is not None:
                        if not _valid_order(response["order"]):
                            raise RuntimeError(
                                f"YOLO worker returned an invalid order: {response['order']!r}"
                            )
                        order = response["order"]
                        if order != last_reported_order:
                            print(f"\n[YOLO] 当前杯子顺序：{order}")
                            print("请选择 [1/2/3]: ", end="", flush=True)
                            last_reported_order = list(order)

                    readable, _, _ = select.select([sys.stdin], [], [], 0)
                    if not readable:
                        continue
                    command = sys.stdin.readline()
                    if command == "":
                        raise KeyboardInterrupt
                    command = command.strip()
                    task = task_for_command(
                        command,
                        order,
                        task_templates,
                        identity_ambiguous=identity_ambiguous,
                    )
                    if task is None:
                        reason = "杯子身份存在歧义" if identity_ambiguous else "该颜色尚未定位"
                        print(f"[YOLO] {reason}，请等待检测稳定后重试：", end="", flush=True)
                        continue
                    print(f"[YOLO] command {command} -> task: {task!r}", flush=True)
                    if execute_task is None:
                        return task
                    frozen_order = list(order)
                    _send_yolo_control(cfg, connection, "start_task")
                    result = execute_task(task, connection, YOLO_COMMAND_VIDEO_TEXT[int(command)])
                    if result is not _RESET_TO_DETECTION:
                        return result

                    _send_yolo_control(cfg, connection, "restart", frozen_order)
                    identity_ambiguous = False
                    started = time.monotonic()
                    print(IDLE_PROMPT, end="", flush=True)
            finally:
                if connection is not None:
                    with suppress(Exception):
                        connection.send(None)
                    connection.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

    @parser.wrap()
    def yolo_human_inloop_record(cfg: YoloHumanInloopRecordConfig):
        if cfg.teleop is None:
            raise ValueError("YOLO human-in-loop execution requires `teleop` config.")
        if cfg.policy is None or not cfg.policy.pretrained_path:
            raise ValueError("YOLO human-in-loop execution requires `--policy.path`.")

        camera_configs = getattr(cfg.robot, "cameras", {})
        if cfg.yolo_camera not in camera_configs:
            raise ValueError(
                f"YOLO camera {cfg.yolo_camera!r} is not configured; "
                f"available cameras: {sorted(camera_configs)}"
            )
        init_logging(log_file=os.path.join(os.getcwd(), "inloop_execute.log"), file_level="INFO")
        robot = make_robot_from_config(cfg.robot)
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )
        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        )

        def load_vla():
            logging.info("Loading VLA model after the first YOLO frame: %s", cfg.policy.pretrained_path)
            with tempfile.TemporaryDirectory(prefix="lerobot_runtime_meta_") as temp_meta_dir:
                runtime_meta = LeRobotDatasetMetadata.create(
                    repo_id="runtime/inference",
                    fps=cfg.dataset.fps,
                    root=Path(temp_meta_dir) / "metadata",
                    robot_type=robot.name,
                    features=dataset_features,
                    use_videos=True,
                )
                policy = make_policy(cfg.policy, ds_meta=runtime_meta)
                preprocessor, postprocessor = make_pre_post_processors(
                    policy_cfg=cfg.policy,
                    pretrained_path=cfg.policy.pretrained_path,
                    dataset_stats=rename_stats(runtime_meta.stats, cfg.dataset.rename_map),
                    preprocessor_overrides={
                        "device_processor": {"device": cfg.policy.device},
                        "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                    },
                )
            logging.info("VLA model loaded; no dataset will be created and Rerun is disabled.")
            return policy, preprocessor, postprocessor

        startup_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="startup")
        vla_future = None
        home_future = None
        listener = None
        policy_sync_executor = None
        try:
            robot.connect()
            camera = robot.cameras[cfg.yolo_camera]

            def connect_teleop_and_home() -> None:
                teleop.connect()
                _home_arms_to_default(robot, teleop)

            def start_background_setup() -> None:
                nonlocal home_future, vla_future
                logging.info("First YOLO frame processed; starting VLA load and arm homing.")
                vla_future = startup_executor.submit(load_vla)
                home_future = startup_executor.submit(connect_teleop_and_home)

            def execute_selected_task(
                task: str,
                connection: Any | None = None,
                overlay_text: str = "",
            ):
                nonlocal listener, policy_sync_executor
                cfg.dataset.single_task = task
                logging.info("YOLO selected execution task: %s", task)
                listener, events = init_keyboard_listener(
                    intervention_toggle_key=cfg.intervention_toggle_key,
                    reset_episode_key="4",
                )
                if policy_sync_executor is None:
                    policy_sync_executor = PolicySyncDualArmExecutor(
                        robot=robot,
                        teleop=teleop,
                        parallel_dispatch=cfg.policy_sync_parallel,
                    )
                stream_stop = Event()
                stream_errors: list[BaseException] = []
                stream_thread = None
                if connection is not None and cfg.yolo_out_video:
                    stream_thread = Thread(
                        target=_stream_yolo_execution,
                        args=(camera, cfg, connection, stream_stop, events, stream_errors, overlay_text),
                        name="yolo-execution-video",
                        daemon=True,
                    )
                    stream_thread.start()
                try:
                    if home_future is None or vla_future is None:
                        raise RuntimeError("VLA/arm startup did not begin after the first YOLO frame")
                    if not (home_future.done() and vla_future.done()):
                        logging.info("Task selected; waiting for arm homing and VLA loading.")
                    home_future.result()
                    policy, preprocessor, postprocessor = vla_future.result()
                    result = record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        dataset=None,
                        dataset_features=dataset_features,
                        teleop=teleop,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        control_time_s=cfg.dataset.episode_time_s,
                        single_task=task,
                        display_data=False,
                        policy_sync_executor=policy_sync_executor,
                        intervention_state_machine_enabled=True,
                        acp_inference=cfg.acp_inference,
                        communication_retry_timeout_s=cfg.communication_retry_timeout_s,
                        communication_retry_interval_s=cfg.communication_retry_interval_s,
                    )
                    reset_requested = bool(events.get("reset_episode"))
                    if reset_requested:
                        logging.info("Command 4: resetting policy and returning both arms home.")
                        policy.reset()
                        preprocessor.reset()
                        postprocessor.reset()
                        _home_arms_to_default(robot, teleop)
                        time.sleep(2.0)
                finally:
                    stream_stop.set()
                    if stream_thread is not None:
                        stream_thread.join()
                    if listener is not None and hasattr(listener, "stop"):
                        listener.stop()
                    listener = None
                if stream_errors:
                    raise stream_errors[0]
                return _RESET_TO_DETECTION if reset_requested else result

            return select_task_with_yolo(
                camera,
                cfg,
                execute_task=execute_selected_task,
                on_detection_started=start_background_setup,
            )
        finally:
            startup_executor.shutdown(wait=True)
            if policy_sync_executor is not None:
                policy_sync_executor.shutdown()
            if robot.is_connected:
                robot.disconnect()
            if teleop.is_connected:
                teleop.disconnect()
            if listener is not None and hasattr(listener, "stop"):
                listener.stop()

    def self_check() -> None:
        templates = {
            "left": "left task",
            "middle": "middle task",
            "right": "right task",
        }
        order = ["B", "Y", "R"]
        assert task_for_command("1", order, templates) == "right task"
        assert task_for_command("2", order, templates) == "left task"
        assert task_for_command("3", order, templates) == "middle task"
        assert task_for_command("4", order, templates) is None
        assert task_for_command("01", order, templates) is None
        assert task_for_command("1", order, templates, identity_ambiguous=True) is None

        class FakeMonitor:
            video_path = "annotated.mp4"
            identity_ambiguous = False
            process_calls = 0

            def process(self, frame):
                self.process_calls += 1
                frame[0, 0] = 255
                return [], [], 3, ["R", "B", "Y"]

            def lock_cup_labels(self, detected_order):
                self.locked_order = list(detected_order)

            def write_video_frames(self, raw_frame, annotated_frame, timestamp, fps):
                self.video_args = raw_frame.copy(), annotated_frame.copy(), timestamp, fps

            def restart_initial_order_detection(self, order):
                self.restarted_order = list(order)

            def freeze_cup_labels(self):
                self.frozen = True

        monitor = FakeMonitor()
        cup_count, detected_order, identity_ambiguous = _process_yolo_frame(
            monitor,
            np.zeros((2, 2, 3), dtype=np.uint8),
            12.5,
            30.0,
        )
        raw_frame, annotated_frame, timestamp, fps = monitor.video_args
        assert cup_count == 3 and detected_order == ["R", "B", "Y"]
        assert not identity_ambiguous and monitor.locked_order == detected_order
        assert not raw_frame[0, 0].any() and annotated_frame[0, 0].all()
        assert timestamp == 12.5 and fps == 30.0
        assert not _apply_yolo_control(monitor, {"command": "start_task"})
        assert monitor.frozen
        assert _apply_yolo_control(
            monitor,
            {"command": "restart", "order": ["B", "Y", "R"]},
        )
        assert monitor.restarted_order == ["B", "Y", "R"]

        _process_yolo_frame(
            monitor,
            np.zeros((2, 2, 3), dtype=np.uint8),
            12.75,
            30.0,
            detection_enabled=False,
        )
        assert monitor.process_calls == 1

        fake_cv2 = SimpleNamespace(
            FONT_HERSHEY_SIMPLEX=0,
            putText=lambda frame, *_args: frame.__setitem__((0, 0), 255),
        )

        cup_count, detected_order, identity_ambiguous = _process_yolo_frame(
            monitor,
            np.zeros((2, 2, 3), dtype=np.uint8),
            13.0,
            30.0,
            detection_enabled=False,
            overlay_text=YOLO_COMMAND_VIDEO_TEXT[2],
            cv2_module=fake_cv2,
        )
        raw_frame, annotated_frame, _, _ = monitor.video_args
        assert cup_count == 0 and detected_order is None and not identity_ambiguous
        assert monitor.process_calls == 1
        assert not raw_frame[0, 0].any() and annotated_frame[0, 0].all()

        class FakeCamera:
            fps = 30
            color_mode = "rgb"
            latest_timestamp = 13.0

            def read_latest(self, max_age_ms):
                assert max_age_ms == 1000
                return np.zeros((2, 2, 3), dtype=np.uint8)

        class FakeConnection:
            def send(self, packet):
                self.packet = packet

            def poll(self, timeout):
                assert timeout == 1.0
                return True

            def recv(self):
                return {"ok": True, "quit": True}

        execution_events = {"exit_early": False, "reset_episode": True}
        stream_errors = []
        connection = FakeConnection()
        _stream_yolo_execution(
            FakeCamera(),
            type("FakeConfig", (), {"yolo_frame_timeout_s": 1.0})(),
            connection,
            Event(),
            execution_events,
            stream_errors,
            YOLO_COMMAND_VIDEO_TEXT[2],
        )
        assert execution_events["exit_early"] and not stream_errors
        assert connection.packet[0] == (2, 2, 3) and connection.packet[3] == 13.0
        assert connection.packet[5] == YOLO_RESET_VIDEO_TEXT

        events = {"exit_early": True, "reset_episode": True}

        def passthrough(value):
            return value

        fake_robot = type("FakeRobot", (), {"action_features": {"joint.pos": float}})()
        assert (
            record_loop(
                robot=fake_robot,
                events=events,
                fps=30,
                teleop_action_processor=passthrough,
                robot_action_processor=passthrough,
                robot_observation_processor=passthrough,
                policy=object(),
                dataset=None,
                dataset_features={"action": {"names": ["joint.pos"]}},
                control_time_s=1,
            )
            is None
        )
        assert not events["exit_early"] and events["reset_episode"]
        assert not events["episode_timeout"]
        print("self-check: ok")

    def main() -> None:
        if sys.argv[1:] == ["--self-check"]:
            self_check()
            return
        register_third_party_plugins()
        yolo_human_inloop_record()


if __name__ == "__main__":
    if _WORKER_MODE:
        if len(sys.argv) != 3:
            raise SystemExit(f"usage: {Path(__file__).name} {_WORKER_FLAG} SOCKET_PATH")
        _run_yolo_worker(sys.argv[2])
    else:
        main()
