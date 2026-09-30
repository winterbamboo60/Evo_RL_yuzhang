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

"""Hardware-free state-machine checks, runnable with the standard library alone."""
import ast
import copy
import logging
import math
from pathlib import Path
import threading
import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/lerobot/scripts/lerobot_record.py"
KEYBOARD = ROOT / "src/lerobot/utils/keyboard_input.py"
if not SOURCE.exists():  # Allow review/testing of staged files before installing them.
    SOURCE = Path(__file__).with_name("lerobot_record.py")
    KEYBOARD = Path(__file__).with_name("keyboard_input.py")


def load_function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    node = copy.deepcopy(node)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class EpisodeHoldTests(unittest.TestCase):
    def namespace(self):
        return dict(threading=threading, time=time, logging=logging, np=SimpleNamespace(ceil=math.ceil))

    def test_hold_repeats_input_and_joins_before_next_recording(self):
        ns = self.namespace()
        hold = contextmanager(load_function(SOURCE, "_hold_episode_action", ns))
        sent = []
        enough = threading.Event()
        def send(action):
            sent.append(action.copy())
            if len(sent) >= 3:
                enough.set()
            return {"left_joint_1.pos": 999}  # Raw physical return must never be replayed.
        target = {"left_joint_1.pos": 12, "right_gripper.pos": 30}
        with hold(SimpleNamespace(send_action=send), target, 200):
            self.assertTrue(enough.wait(1))
        count = len(sent)
        time.sleep(.02)
        self.assertEqual(len(sent), count)
        self.assertTrue(all(action == target for action in sent))

    def test_hold_propagates_send_failure(self):
        hold = contextmanager(load_function(SOURCE, "_hold_episode_action", self.namespace()))
        def send(action):
            raise RuntimeError("CAN failure")
        with self.assertRaisesRegex(RuntimeError, "CAN failure"):
            with hold(SimpleNamespace(send_action=send), {"joint": 1}, 30):
                pass

    def test_countdown_is_two_seconds_and_can_be_cancelled(self):
        clock = SimpleNamespace(now=0.)
        ns = self.namespace()
        ns["time"] = SimpleNamespace(monotonic=lambda: clock.now,
            sleep=lambda duration: setattr(clock, "now", clock.now + duration))
        countdown = load_function(SOURCE, "_episode_start_countdown", ns)
        with self.assertLogs(level="INFO") as logs:
            self.assertTrue(countdown({"stop_recording": False}, 2., 4))
        self.assertAlmostEqual(clock.now, 2.)
        self.assertEqual(sum("[倒计时] 2" in line for line in logs.output), 1)
        self.assertEqual(sum("[倒计时] 1" in line for line in logs.output), 1)
        self.assertTrue(any("第 5 个 episode" in line for line in logs.output))
        self.assertFalse(countdown({"stop_recording": True}, 2., 4))
        self.assertAlmostEqual(clock.now, 2.)

    def test_idle_keys_do_not_change_outcome_but_escape_still_works(self):
        callbacks = []
        def apply(key, events):
            if key == "esc":
                events["stop_recording"] = True
        ns = dict(time=time, logger=logging.getLogger(__name__), INTERVENTION_TOGGLE_COOLDOWN_S=.2,
            apply_recording_control=apply,
            create_key_listener=lambda callback, **kwargs: callbacks.append(callback))
        listener = load_function(KEYBOARD, "init_keyboard_listener", ns)
        _, events = listener(episode_success_key="b", episode_failure_key="f", rerecord_episode_key="a",
            reset_episode_key="r", intervention_toggle_key="c")
        events["recording_active"] = False
        for key in ("b", "f", "a", "r", "c"):
            callbacks[0](key)
        self.assertIsNone(events["episode_outcome"])
        self.assertFalse(events["rerecord_episode"])
        self.assertFalse(events["reset_episode"])
        callbacks[0]("esc")
        self.assertTrue(events["stop_recording"])
        events["recording_active"] = True
        callbacks[0]("b")
        self.assertEqual(events["episode_outcome"], "success")

    def test_episode_flow_saves_before_countdown_and_reuses_discarded_index(self):
        # Execute the real episode orchestration with hardware/dataset boundaries replaced.
        tree = ast.parse(SOURCE.read_text())
        record = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "record")
        loop = next(n for n in ast.walk(record) if isinstance(n, ast.While) and
            isinstance(n.test, ast.BoolOp) and ast.unparse(n.test).startswith("recorded_episodes"))
        events = dict(stop_recording=False, rerecord_episode=False, exit_early=False)
        trace = []
        dataset = SimpleNamespace(num_episodes=0, writer=SimpleNamespace(episode_buffer={"size": 3}))
        def save(**kwargs):
            trace.append(("save", dataset.num_episodes, kwargs["episode_metadata"]["episode_success"]))
            dataset.num_episodes += 1
        dataset.save_episode = save
        dataset.clear_episode_buffer = lambda: trace.append(("discard", dataset.num_episodes))
        attempts = iter(("a", "r", "b", "f"))
        def record_loop(**kwargs):
            self.assertIs(kwargs["dataset"], dataset)  # No unrecorded 3-second teleop loop.
            self.assertTrue(events["recording_active"])
            key = next(attempts)
            trace.append(("record", dataset.num_episodes, key))
            kwargs["last_action"].update(joint=20)
            if key in ("a", "r"):
                events["rerecord_episode"] = True
                events["reset_episode"] = key == "r"
            else:
                events["episode_outcome"] = "success" if key == "b" else "failure"
        @contextmanager
        def hold(robot, action, fps):
            self.assertFalse(events["recording_active"])
            trace.append(("hold", dict(action)))
            yield
        def countdown(events, delay, index):
            self.assertEqual(delay, 2)
            trace.append(("countdown", index))
            return True
        def home(*args, **kwargs):
            trace.append(("home",))
            return {"joint": 0}
        ns = dict(events=events, dataset=dataset, recorded_episodes=0, last_action={"joint": 0},
            cfg=SimpleNamespace(hold_between_episodes=True, episode_start_delay_s=2, play_sounds=False,
                enable_evorl_controls=True, display_data=False, display_mode="rerun", episode_success_key="b",
                episode_failure_key="f", dataset=SimpleNamespace(num_episodes=2, fps=30,
                    episode_time_s=120000, single_task="pour")),
            robot=object(), teleop=object(), teleop_action_processor=None, robot_action_processor=None,
            robot_observation_processor=None, display_compressed_images=False,
            timer=SimpleNamespace(log_episode_summary=lambda *a: None, restart=lambda: None),
            logging=logging, log_say=lambda *a: None, nullcontext=nullcontext,
            _hold_episode_action=hold, _episode_start_countdown=countdown, record_loop=record_loop,
            _home_piper_followers=home, normalize_episode_success_label=lambda x: x)
        events["recording_active"] = False
        exec(compile(ast.Module(body=[copy.deepcopy(loop)], type_ignores=[]), str(SOURCE), "exec"), ns)
        self.assertEqual([x for x in trace if x[0] == "countdown"],
            [("countdown", 0), ("countdown", 0), ("countdown", 0), ("countdown", 1)])
        self.assertEqual([x for x in trace if x[0] == "save"],
            [("save", 0, "success"), ("save", 1, "failure")])
        self.assertLess(trace.index(("save", 0, "success")), trace.index(("countdown", 1)))
        self.assertEqual(sum(x[0] == "home" for x in trace), 1)


if __name__ == "__main__":
    unittest.main()
