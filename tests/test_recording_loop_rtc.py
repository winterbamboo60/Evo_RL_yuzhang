#!/usr/bin/env python

import unittest
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.scripts.recording_loop import _RTCActionChunkRunner, _RTCChunkPrediction, record_loop
from lerobot.utils.constants import ACTION


class _IdentityProcessor:
    def __call__(self, value):
        return value

    def reset(self):
        pass


class _ChunkPolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(device="cpu", use_amp=False)
        self.calls = []
        self.second_started = Event()
        self.second_release = Event()
        self.third_started = Event()
        self.third_release = Event()

    def predict_action_chunk(self, _observation, *, inference_delay, prev_chunk_left_over):
        call_index = len(self.calls)
        self.calls.append(
            {
                "thread": get_ident(),
                "delay": inference_delay,
                "previous": (
                    None if prev_chunk_left_over is None else prev_chunk_left_over.detach().clone()
                ),
            }
        )
        if call_index == 1:
            self.second_started.set()
            assert self.second_release.wait(2)
        elif call_index == 2:
            self.third_started.set()
            assert self.third_release.wait(2)

        start = call_index * 100
        return torch.arange(start, start + 6, dtype=torch.float32).reshape(1, 6, 1)


class RTCActionChunkRunnerTest(unittest.TestCase):
    def test_background_merge_and_stale_result_invalidation(self):
        policy = _ChunkPolicy()
        runner = _RTCActionChunkRunner(
            policy=policy,
            preprocessor=_IdentityProcessor(),
            postprocessor=_IdentityProcessor(),
            rtc=RTCConfig(enabled=True, execution_horizon=2),
            fps=30,
            queue_threshold=3,
            task="test",
            robot_type="test_robot",
        )
        observation = {"observation.state": np.array([0.0], dtype=np.float32)}

        try:
            first_action = runner.get_action(observation)
            self.assertEqual(first_action.shape, (1, 1))
            old_actions = [first_action.item()]
            old_actions.extend(runner.get_action(observation).item() for _ in range(3))
            self.assertTrue(policy.second_started.wait(2))
            old_actions.append(runner.get_action(observation).item())
            self.assertEqual(old_actions, [0.0, 1.0, 2.0, 3.0, 4.0])

            policy.second_release.set()
            assert runner.future is not None
            runner.future.result(timeout=2)
            self.assertEqual(runner.get_action(observation).item(), 102.0)
            torch.testing.assert_close(
                policy.calls[1]["previous"], torch.tensor([[3.0], [4.0], [5.0]])
            )

            self.assertEqual(runner.get_action(observation).item(), 103.0)
            self.assertTrue(policy.third_started.wait(2))
            runner.invalidate()
            policy.third_release.set()
            runner.wait_for_idle()
            self.assertTrue(runner.action_queue.empty())

            self.assertEqual(runner.get_action(observation).item(), 300.0)
            self.assertIsNone(policy.calls[3]["previous"])
            self.assertEqual(policy.calls[3]["delay"], 0)
            self.assertTrue(all(call["thread"] != get_ident() for call in policy.calls))
        finally:
            policy.second_release.set()
            policy.third_release.set()
            runner.close()

    def test_custom_actor_chunk_uses_execution_prefix_fusion(self):
        policy = _ChunkPolicy()
        predictor_calls = []

        def actor_chunk_predictor(observation, inference_delay, previous_actions, task, robot_type):
            predictor_calls.append(
                (observation, inference_delay, previous_actions.detach().clone(), task, robot_type)
            )
            return _RTCChunkPrediction(
                actions=torch.arange(100, 106, dtype=torch.float32).reshape(1, 6, 1),
                apply_prefix_fusion=True,
            )

        runner = _RTCActionChunkRunner(
            policy=policy,
            preprocessor=_IdentityProcessor(),
            postprocessor=_IdentityProcessor(),
            rtc=RTCConfig(enabled=True, execution_horizon=4),
            fps=30,
            queue_threshold=3,
            task="actor-task",
            robot_type="test_robot",
            chunk_predictor=actor_chunk_predictor,
        )
        previous = torch.arange(6, dtype=torch.float32).reshape(6, 1)

        try:
            original, processed, snapshot, apply_fusion, _ = runner._infer_chunk(
                {"observation.state": np.array([0.0], dtype=np.float32)},
                2,
                previous,
                "actor-task",
                "test_robot",
            )
            self.assertIsNone(processed)
            self.assertTrue(apply_fusion)
            torch.testing.assert_close(snapshot, previous)
            self.assertEqual(predictor_calls[0][1], 2)
            torch.testing.assert_close(predictor_calls[0][2], previous)
            self.assertEqual(predictor_calls[0][3:], ("actor-task", "test_robot"))

            fused = runner._fuse_action_prefix(original, snapshot, real_delay=2)
            expected = torch.tensor([[0.0], [1.0], [35.333332], [69.666664], [104.0], [105.0]])
            torch.testing.assert_close(fused, expected)
        finally:
            runner.close()

    def test_disabled_rtc_keeps_synchronous_select_action_path(self):
        events = {"exit_early": False, "toggle_intervention": False}

        class _Robot:
            robot_type = "test_robot"
            action_features = {"joint.pos": float}

            def __init__(self):
                self.sent_actions = []

            def get_observation(self):
                events["exit_early"] = True
                return {"joint.pos": 0.0}

            def send_action(self, action, *, add_offset):
                self.sent_actions.append((action, add_offset))
                return action

        class _Policy:
            config = SimpleNamespace(device="cpu", use_amp=False)

            def __init__(self):
                self.select_calls = 0

            def reset(self):
                pass

            def select_action(self, _observation):
                self.select_calls += 1
                return torch.tensor([[7.0]])

            def predict_action_chunk(self, _observation, **_kwargs):
                raise AssertionError("RTC chunk inference must stay disabled")

        robot = _Robot()
        policy = _Policy()
        dataset_features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["joint.pos"],
            },
            ACTION: {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        }

        with patch(
            "lerobot.scripts.recording_loop._RTCActionChunkRunner",
            side_effect=AssertionError("RTC runner must not be created"),
        ):
            record_loop(
                robot=robot,
                events=events,
                fps=1000,
                teleop_action_processor=_IdentityProcessor(),
                robot_action_processor=lambda value: value[0],
                robot_observation_processor=_IdentityProcessor(),
                policy=policy,
                preprocessor=_IdentityProcessor(),
                postprocessor=_IdentityProcessor(),
                single_task="test",
                control_time_s=1,
                dataset_features=dataset_features,
                rtc=RTCConfig(enabled=False),
            )

        self.assertEqual(policy.select_calls, 1)
        self.assertEqual(robot.sent_actions, [({"joint.pos": 7.0}, True)])


if __name__ == "__main__":
    unittest.main()
