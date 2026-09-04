import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lerobot.onlineRL_evoRL.actor_new import ActorVLARuntime, OnlineActorRuntime
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_STATE


class _FakeRunner:
    def __init__(self, action=None, events=None):
        self.action = action if action is not None else torch.tensor([[9.0]])
        self.events = events if events is not None else []
        self.calls = []

    def get_action(self, observation):
        self.calls.append(observation)
        return self.action

    def invalidate(self):
        self.events.append("invalidate")

    def wait_for_idle(self):
        self.events.append("wait")

    def close(self):
        self.events.append("close")


class _Resettable:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def reset(self):
        self.events.append(self.name)


class OnlineActorRTCTest(unittest.TestCase):
    def test_rtc_select_action_bypasses_legacy_actor_deque(self):
        runtime = OnlineActorRuntime.__new__(OnlineActorRuntime)
        runtime.cfg = SimpleNamespace(rtc=RTCConfig(enabled=True))
        runtime.control = SimpleNamespace(use_actor=True, actor_available=True)
        runtime.policy = object()
        runtime.preprocessor = object()
        runtime.postprocessor = object()
        runtime._actor_actions = deque([torch.tensor([[123.0]])])
        runner = _FakeRunner(action=torch.tensor([[7.0]]))
        runtime._get_or_create_rtc_runner = lambda _robot_type: runner

        action = runtime.select_action({"frame": 1}, "robot", torch.device("cpu"))

        torch.testing.assert_close(action, torch.tensor([[7.0]]))
        self.assertEqual(runner.calls, [{"frame": 1}])
        self.assertEqual(len(runtime._actor_actions), 1)

    def test_disabled_rtc_keeps_actor_deque_chunk_consumption(self):
        actor_calls = []

        class _Policy:
            def extract_rlt_features(self, _batch):
                return {
                    "z_rl": torch.tensor([[1.0]]),
                    "proprio": torch.tensor([[2.0]]),
                    "ref_action": torch.zeros(1, 3, 1),
                }

            def actor(self, *args, **kwargs):
                actor_calls.append((args, kwargs))
                return torch.tensor([[[1.0], [2.0], [3.0]]])

        runtime = OnlineActorRuntime.__new__(OnlineActorRuntime)
        runtime.cfg = SimpleNamespace(
            rtc=RTCConfig(enabled=False),
            env=SimpleNamespace(task="test"),
        )
        runtime.control = SimpleNamespace(use_actor=True, actor_available=True)
        runtime.policy = _Policy()
        runtime.policy_cfg = SimpleNamespace(
            device="cpu",
            use_amp=False,
            actor_rollout_deterministic=True,
        )
        runtime.preprocessor = lambda observation: observation
        runtime.postprocessor = lambda action: action
        runtime._actor_actions = deque()

        with patch(
            "lerobot.onlineRL_evoRL.actor_new.prepare_observation_for_inference",
            return_value={"prepared": torch.tensor([[0.0]])},
        ):
            first = runtime.select_action({}, "robot", torch.device("cpu"))
            second = runtime.select_action({}, "robot", torch.device("cpu"))

        torch.testing.assert_close(first, torch.tensor([[1.0]]))
        torch.testing.assert_close(second, torch.tensor([[2.0]]))
        self.assertEqual(len(actor_calls), 1)
        self.assertEqual(len(runtime._actor_actions), 1)

    def test_actor_rtc_chunk_passes_previous_actions_and_requests_final_fusion(self):
        calls = {}

        class _Policy:
            def predict_action_chunk_with_rlt(self, batch, **kwargs):
                calls["batch"] = batch
                calls["kwargs"] = kwargs
                return {
                    "z_rl": torch.tensor([[1.0]]),
                    "actions": torch.zeros(1, 3, 1),
                }

            def actor(self, z_rl, proprio, ref_action, **kwargs):
                calls["actor"] = (z_rl, proprio, ref_action, kwargs)
                return torch.tensor([[[4.0], [5.0], [6.0]]])

        runtime = OnlineActorRuntime.__new__(OnlineActorRuntime)
        runtime.control = SimpleNamespace(use_actor=True, actor_available=True)
        runtime.policy = _Policy()
        runtime.policy_cfg = SimpleNamespace(
            device="cpu",
            proprio_dim=2,
            actor_rollout_deterministic=True,
        )
        runtime.preprocessor = lambda _observation: {
            OBS_STATE: torch.tensor([[10.0, 20.0, 30.0]])
        }
        previous = torch.tensor([[1.0], [2.0], [3.0]])

        with patch(
            "lerobot.onlineRL_evoRL.actor_new.prepare_observation_for_inference",
            return_value={},
        ):
            prediction = runtime._predict_rtc_chunk(
                {},
                inference_delay=2,
                previous_actions=previous,
                task="task",
                robot_type="robot",
            )

        self.assertTrue(prediction.apply_prefix_fusion)
        torch.testing.assert_close(prediction.actions, torch.tensor([[[4.0], [5.0], [6.0]]]))
        self.assertEqual(calls["kwargs"]["inference_delay"], 2)
        torch.testing.assert_close(calls["kwargs"]["prev_chunk_left_over"], previous)
        torch.testing.assert_close(calls["actor"][1], torch.tensor([[10.0, 20.0]]))
        self.assertTrue(calls["actor"][3]["deterministic"])

    def test_reset_waits_for_rtc_worker_before_resetting_components(self):
        events = []
        runtime = ActorVLARuntime.__new__(ActorVLARuntime)
        runtime._rtc_runner = _FakeRunner(events=events)
        runtime.policy = _Resettable("policy", events)
        runtime.preprocessor = _Resettable("preprocessor", events)
        runtime.postprocessor = _Resettable("postprocessor", events)

        runtime.reset_action_state()

        self.assertEqual(
            events,
            ["invalidate", "wait", "policy", "preprocessor", "postprocessor"],
        )


if __name__ == "__main__":
    unittest.main()
