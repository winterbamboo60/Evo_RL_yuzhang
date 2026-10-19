# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execute recording code with fake hardware and the real preview sender."""

import ast
import copy
import logging
import math
import multiprocessing
import sys
import threading
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/lerobot/scripts/lerobot_record.py"
sys.path.insert(0, str(ROOT / "src"))
from lerobot.utils.async_visualization import AsyncVisualization  # noqa: E402


def load_recording_node(name, namespace):
    """Run the actual class/function without importing optional hardware SDKs."""
    tree = ast.parse(SOURCE.read_text())
    node = copy.deepcopy(next(node for node in tree.body if getattr(node, "name", None) == name))
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name]


class BlockingBackend:
    """Spawn-safe backend that stays blocked until the test releases it."""

    def __init__(self, entered, release, calls):
        self.entered, self.release, self.calls = entered, release, calls

    def init_visualization(self, *args, **kwargs):
        pass

    def log_visualization_data(self, *args, **kwargs):
        with self.calls.get_lock():
            self.calls.value += 1
        self.entered.set()
        self.release.wait(10)

    def shutdown_visualization(self, *args, **kwargs):
        pass


class RecordingPreviewTests(unittest.TestCase):
    def config_class(self):
        namespace = {
            "__name__": __name__,
            "dataclass": dataclass,
            "np": SimpleNamespace(isfinite=math.isfinite),
            "EVORL_INTERVENTION_KEY": "c",
            "EVORL_SUCCESS_KEY": "b",
            "EVORL_FAILURE_KEY": "f",
            "EVORL_RERECORD_KEY": "a",
            "EVORL_RESET_KEY": "r",
        }
        return load_recording_node("RecordConfig", namespace)

    def test_default_config_uses_async_preview_at_independent_rate(self):
        config = self.config_class()(robot=SimpleNamespace(type="fake"), dataset=object(), teleop=object())
        self.assertTrue(config.display_async)
        self.assertEqual(config.display_fps, 15.0)
        config = self.config_class()(
            robot=SimpleNamespace(type="fake"),
            dataset=object(),
            teleop=object(),
            display_async=False,
            display_fps=12.0,
        )
        self.assertFalse(config.display_async)
        for value in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "display_fps"):
                self.config_class()(
                    robot=SimpleNamespace(type="fake"), dataset=object(), teleop=object(), display_fps=value
                )

    def loop_fixture(self, *, preview=None, display_data=True, backend=None, frames=8):
        clock = SimpleNamespace(now=0)
        sent, recorded, displayed = [], [], []
        events = {"exit_early": False, "stop_recording": False}

        class Teleop:
            def get_action(self):
                return {"joint.pos": len(sent) + 1}

        class Timer:
            def tick(self):
                pass

            def section(self, name):
                return nullcontext()

            def wait(self):
                clock.now += 1

        def send(action):
            sent.append(dict(action))
            return {"joint.pos": -1000}  # Physical return must not replace recorded action coordinates.

        def visualize(mode, **kwargs):
            displayed.append((mode, kwargs))
            if backend is not None:
                backend(mode, **kwargs)

        namespace = {
            "Teleoperator": Teleop,
            "time": SimpleNamespace(perf_counter=lambda: clock.now),
            "logging": logging,
            "ACTION": "action",
            "OBS_STR": "observation",
            "EVORL_POLICY_ACTION_FIELD": "policy",
            "build_dataset_frame": lambda features, values, prefix: {prefix: dict(values)},
            "log_visualization_data": visualize,
        }
        load_recording_node("_log_record_telemetry", namespace)
        loop = load_recording_node("record_loop", namespace)
        robot = SimpleNamespace(
            name="fake", get_observation=lambda: {"joint.pos": len(sent)}, send_action=send
        )
        dataset = SimpleNamespace(fps=30, features={}, add_frame=lambda frame: recorded.append(frame))
        kwargs = {
            "robot": robot,
            "events": events,
            "fps": 30,
            "teleop_action_processor": lambda item: item[0],
            "robot_action_processor": lambda item: item[0],
            "robot_observation_processor": lambda obs: obs,
            "dataset": dataset,
            "teleop": Teleop(),
            "control_time_s": frames,
            "single_task": "pour",
            "display_data": display_data,
            "display_mode": "foxglove",
            "display_compressed_images": True,
            "timer": Timer(),
            "preview": preview,
        }
        return loop, kwargs, sent, recorded, displayed

    def assert_complete_recording(self, sent, recorded, frames=8):
        self.assertEqual(sent, [{"joint.pos": value} for value in range(1, frames + 1)])
        self.assertEqual(len(recorded), frames)
        self.assertEqual([frame["action"] for frame in recorded], sent)
        self.assertEqual([frame["observation"]["joint.pos"] for frame in recorded], list(range(frames)))
        self.assertTrue(all(frame["task"] == "pour" for frame in recorded))

    def test_rejected_previews_keep_every_dataset_frame_and_robot_action(self):
        submissions = []
        preview = SimpleNamespace(submit=lambda **kwargs: submissions.append(kwargs) or False)
        loop, kwargs, sent, recorded, displayed = self.loop_fixture(preview=preview)
        loop(**kwargs)
        self.assert_complete_recording(sent, recorded)
        self.assertEqual(len(submissions), 8)
        self.assertTrue(all(item["observation"]["intervention"] is True for item in submissions))
        self.assertEqual([item["action"] for item in submissions], sent)
        self.assertEqual(displayed, [])

    def test_synchronous_preview_still_logs_manual_telemetry(self):
        loop, kwargs, sent, recorded, displayed = self.loop_fixture()
        loop(**kwargs)
        self.assert_complete_recording(sent, recorded)
        self.assertEqual(len(displayed), 8)
        for mode, values in displayed:
            self.assertEqual(mode, "foxglove")
            self.assertTrue(values["observation"]["intervention"])
            self.assertTrue(values["compress_images"])
        self.assertEqual([values["action"] for _, values in displayed], sent)

    def test_disabled_display_does_not_submit(self):
        preview = SimpleNamespace(submit=lambda **kwargs: self.fail("Display is disabled"))
        loop, kwargs, sent, recorded, displayed = self.loop_fixture(preview=preview, display_data=False)
        loop(**kwargs)
        self.assert_complete_recording(sent, recorded)
        self.assertEqual(displayed, [])

    def test_real_blocked_backend_does_not_delay_recording_loop(self):
        context = multiprocessing.get_context("spawn")
        entered, release = context.Event(), context.Event()
        calls = context.Value("i", 0)
        sender = AsyncVisualization(
            "rerun", max_fps=100000, _backend_factory=BlockingBackendFactory(entered, release, calls)
        )
        failures = []
        completed = threading.Event()
        thread = None
        try:
            sender.start()
            self.assertTrue(sender.submit(observation={"joint.pos": -1}))
            self.assertTrue(entered.wait(5), "Backend never received its initial preview")
            loop, kwargs, sent, recorded, displayed = self.loop_fixture(preview=sender)

            def run_loop():
                try:
                    loop(**kwargs)
                except BaseException as error:
                    failures.append(error)
                finally:
                    completed.set()

            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
            self.assertTrue(completed.wait(2), "Recording waited for the blocked preview backend")
            self.assertEqual(failures, [])
            self.assertFalse(release.is_set())
            self.assertEqual(calls.value, 1)
            self.assert_complete_recording(sent, recorded)
            self.assertEqual(displayed, [])
        finally:
            release.set()
            if thread is not None:
                thread.join(timeout=5)
            sender.close()

    def record_fixture(self, *, failure=None, display_async=True, episodes=1):
        trace = []

        class Preview:
            def __init__(self, mode, **kwargs):
                trace.append(("preview_config", mode, kwargs))

            def start(self):
                trace.append("preview_start")
                if failure == "start":
                    raise RuntimeError("preview initialization failed")

            def close(self):
                trace.append("preview_close")

        class Device:
            name = "fake"
            action_features = {"joint.pos": float}
            observation_features = {"joint.pos": float}
            cameras = {}
            is_connected = False

            def __init__(self, name):
                self.kind = name

            def connect(self):
                trace.append(f"{self.kind}_connect")
                if failure == "connect" and self.kind == "robot":
                    raise RuntimeError("robot connection failed")
                self.is_connected = True

            def disconnect(self):
                trace.append(f"{self.kind}_disconnect")
                self.is_connected = False

        dataset = SimpleNamespace(num_episodes=0, writer=SimpleNamespace(episode_buffer={"size": 1}))

        def save(**kwargs):
            dataset.num_episodes += 1
            trace.append("save")

        dataset.save_episode = save
        dataset.finalize = lambda: trace.append("finalize")

        def create(*args, **kwargs):
            trace.append("create")
            if failure == "dataset":
                raise RuntimeError("dataset creation failed")
            return dataset

        def loop(**kwargs):
            trace.append(("loop_preview", kwargs["preview"]))
            if failure == "loop":
                raise RuntimeError("control loop failed")

        events = {"stop_recording": False, "rerecord_episode": False, "exit_early": False}
        robot, teleop = Device("robot"), Device("teleop")
        timer = SimpleNamespace(
            log_episode_summary=lambda *args: None, restart=lambda: None, log_run_summary=lambda: None
        )
        namespace = {
            "sys": sys,
            "logging": logging,
            "init_logging": lambda: None,
            "pformat": repr,
            "asdict": lambda cfg: {},
            "AsyncVisualization": Preview,
            "make_robot_from_config": lambda cfg: robot,
            "make_teleoperator_from_config": lambda cfg: teleop,
            "create_initial_features": lambda **kwargs: {},
            "aggregate_pipeline_dataset_features": lambda **kwargs: {},
            "combine_feature_dicts": lambda *args: {},
            "CycleTimer": lambda fps: timer,
            "LeRobotDataset": SimpleNamespace(create=create),
            "init_keyboard_listener": lambda **kwargs: (
                SimpleNamespace(stop=lambda: trace.append("listener_stop")),
                events,
            ),
            "VideoEncodingManager": lambda dataset: nullcontext(),
            "nullcontext": nullcontext,
            "_episode_start_countdown": lambda *args: True,
            "record_loop": loop,
            "log_say": lambda *args, **kwargs: None,
            "init_visualization": lambda *args, **kwargs: trace.append("sync_init"),
            "shutdown_visualization": lambda *args: trace.append("sync_close"),
        }
        record = load_recording_node("record", namespace)
        dataset_config = SimpleNamespace(
            repo_id="recording",
            fps=30,
            video=False,
            num_episodes=episodes,
            root=None,
            num_image_writer_processes=0,
            num_image_writer_threads_per_camera=4,
            video_encoding_batch_size=1,
            rgb_encoder=None,
            depth_encoder=None,
            encoder_threads=2,
            streaming_encoding=True,
            encoder_queue_maxsize=30,
            episode_time_s=1,
            reset_time_s=1,
            single_task="pour",
            push_to_hub=False,
            stamp_repo_id=lambda: None,
        )
        config = self.config_class()(
            robot=SimpleNamespace(type="fake"),
            dataset=dataset_config,
            teleop=object(),
            display_data=True,
            display_async=display_async,
            display_ip="127.0.0.1",
            display_port=9876,
            display_fps=12,
            play_sounds=False,
        )
        kwargs = {
            "teleop_action_processor": lambda item: item[0],
            "robot_action_processor": lambda item: item[0],
            "robot_observation_processor": lambda obs: obs,
        }
        return record, config, kwargs, trace, dataset

    def test_async_lifecycle_starts_before_hardware_and_closes_after_success(self):
        record, config, kwargs, trace, dataset = self.record_fixture(episodes=2)
        self.assertIs(record(config, **kwargs), dataset)
        self.assertLess(trace.index("preview_start"), trace.index("teleop_connect"))
        self.assertLess(trace.index("preview_start"), trace.index("robot_connect"))
        loops = [item[1] for item in trace if isinstance(item, tuple) and item[0] == "loop_preview"]
        self.assertEqual(len(loops), 3)  # Two recording phases plus one reset phase.
        self.assertTrue(all(preview is loops[0] and preview is not None for preview in loops))
        self.assertEqual(trace.count("preview_close"), 1)
        mode, values = trace[0][1:]
        self.assertEqual(mode, "rerun")
        self.assertEqual(
            values,
            {
                "session_name": "recording",
                "ip": "127.0.0.1",
                "port": 9876,
                "compress_images": True,
                "max_fps": 12,
            },
        )
        self.assertNotIn("sync_init", trace)
        self.assertNotIn("sync_close", trace)

    def test_async_lifecycle_closes_on_start_dataset_connect_and_control_errors(self):
        for failure in ("start", "dataset", "connect", "loop"):
            with self.subTest(failure=failure):
                record, config, kwargs, trace, _ = self.record_fixture(failure=failure)
                with self.assertRaises(RuntimeError):
                    record(config, **kwargs)
                self.assertEqual(trace.count("preview_close"), 1)
                if failure in {"connect", "loop"}:
                    self.assertIn("finalize", trace)
                    self.assertIn("teleop_disconnect", trace)
                if failure == "loop":
                    self.assertIn("robot_disconnect", trace)
                    self.assertIn("listener_stop", trace)

    def test_synchronous_lifecycle_remains_available(self):
        record, config, kwargs, trace, dataset = self.record_fixture(display_async=False)
        self.assertIs(record(config, **kwargs), dataset)
        self.assertIn("sync_init", trace)
        self.assertIn("sync_close", trace)
        self.assertNotIn("preview_start", trace)
        self.assertTrue(
            all(item[1] is None for item in trace if isinstance(item, tuple) and item[0] == "loop_preview")
        )


class BlockingBackendFactory:
    def __init__(self, entered, release, calls):
        self.entered, self.release, self.calls = entered, release, calls

    def __call__(self):
        return BlockingBackend(self.entered, self.release, self.calls)


if __name__ == "__main__":
    unittest.main()
