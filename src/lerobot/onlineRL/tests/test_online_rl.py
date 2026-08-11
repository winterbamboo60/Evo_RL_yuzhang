from __future__ import annotations

import random
import sys
from builtins import __import__ as real_import
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from lerobot.onlineRL import compare_stage1_parity, keyboard as keyboard_module, run
from lerobot.onlineRL.actor_runtime import ActorRuntime
from lerobot.onlineRL.config import OnlineRLConfig
from lerobot.onlineRL.hardware import FakePiperHardware, RealPiperHardware
from lerobot.onlineRL.keyboard import ControlState, KeyboardController
from lerobot.onlineRL.learner_runtime import Learner
from lerobot.onlineRL.protocol import Episode, Transition
from lerobot.onlineRL.replay import EpisodeReplayBuffer
from lerobot.onlineRL.safety import safe_action
from lerobot.onlineRL.stage1_rlt_adapter import _PIPER_IMAGE_KEYS, _PIPER_TRAINING_ACTION_DIM, _image_batch
from lerobot.onlineRL.vla import FakeFeatureExtractor


def make_cfg() -> OnlineRLConfig:
    cfg = OnlineRLConfig()
    cfg.model.z_dim = 8
    cfg.model.hidden_dim = 16
    cfg.model.ref_num_action_chunks = 4
    cfg.model.num_action_chunks = 2
    cfg.replay.batch_size = 2
    cfg.replay.update_epoch = 1
    cfg.replay.max_cached_episodes = 2
    cfg.replay.sample_window_episodes = 2
    cfg.runtime.learner_device = "cpu"
    cfg.runtime.actor_device = "cpu"
    cfg.validate()
    return cfg


def make_episode(cfg: OnlineRLConfig, identifier: str, success: bool) -> Episode:
    feature = FakeFeatureExtractor(cfg.model)({"state": torch.zeros(7)})
    action = torch.zeros(1, cfg.model.num_action_chunks, 7)
    transition = Transition(
        z_rl=feature.z_rl, proprio=feature.proprio, ref_chunk=feature.ref_chunk, action=action,
        next_z_rl=feature.z_rl, next_proprio=feature.proprio, next_ref_chunk=feature.ref_chunk,
        reward=float(success), terminal=True, truncated=not success,
        action_source="actor", intervention=False,
    )
    return Episode(identifier, [transition], "success" if success else "timeout", 0)


def test_keyboard_state_machine() -> None:
    controller, state = KeyboardController(), ControlState()
    controller.push("b")
    controller.push("i")
    controller.push("s")
    state = controller.poll(state)
    assert state.action_source == "actor"
    assert state.human_control is True
    assert state.terminal_event == "success"
    controller.push("f")
    assert controller.poll(state).terminal_event == "failure"
    controller.push("r")
    assert controller.poll(state).reset_requested is True


def test_keyboard_listener_is_optional_without_display() -> None:
    def import_without_pynput(name: str, *args: object, **kwargs: object) -> object:
        if name == "pynput":
            raise ImportError("failed to acquire X connection")
        return real_import(name, *args, **kwargs)

    with (
        patch.object(sys.stdin, "isatty", return_value=False),
        patch("builtins.__import__", side_effect=import_without_pynput),
    ):
        assert KeyboardController().start_pynput_listener() is None


def test_keyboard_listener_prefers_current_tty() -> None:
    controller = KeyboardController()
    listener = Mock()
    listener_cls = Mock(return_value=listener)
    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch.object(keyboard_module, "TTYKeyboardListener", listener_cls),
    ):
        assert controller.start_pynput_listener() is listener
    listener_cls.assert_called_once_with(controller)
    listener.start.assert_called_once()


def test_episode_replay_is_episode_counted() -> None:
    cfg = make_cfg()
    replay = EpisodeReplayBuffer(capacity=2, sample_window=2)
    replay.add(make_episode(cfg, "one", False))
    replay.add(make_episode(cfg, "two", True))
    replay.add(make_episode(cfg, "three", True))
    assert len(replay) == 2
    assert replay.episode_ids == ["two", "three"]
    assert replay.sample(3)[0].action.shape == (1, 2, 7)


def test_learner_waits_for_two_complete_episodes() -> None:
    cfg = make_cfg()
    learner = Learner(cfg)
    assert learner.ingest(make_episode(cfg, "one", False)) is None
    weights = learner.ingest(make_episode(cfg, "two", True))
    assert weights is not None
    assert weights.version >= 1


def test_fake_hardware_only_returns_follower_action() -> None:
    cfg = make_cfg()
    hardware = FakePiperHardware(cfg.hardware)
    hardware.connect()
    returned = hardware.send_automatic(torch.ones(7) * 99)
    assert torch.all(returned <= cfg.hardware.action_limit)
    hardware.leader_action = torch.ones(7) * 99
    assert torch.all(hardware.send_human() <= cfg.hardware.action_limit)
    observation = hardware.observation()
    assert {"state", "top_image", "wrist_image"}.issubset(observation)


def test_safe_action_preserves_piper_degree_scale() -> None:
    action = torch.tensor([50.0, 147.0, -142.0, 100.0, 74.0, 122.0, 98.0])
    assert torch.equal(safe_action(action, 7, OnlineRLConfig().hardware.action_limit), action)


def test_actor_runtime_infers_once_per_action_chunk() -> None:
    cfg = make_cfg()
    cfg.runtime.max_episode_steps = 4
    cfg.hardware.control_hz = 1000.0
    hardware = FakePiperHardware(cfg.hardware)

    class CountingExtractor(FakeFeatureExtractor):
        def __init__(self):
            super().__init__(cfg.model)
            self.calls = 0

        def __call__(self, observation: dict):
            self.calls += 1
            return super().__call__(observation)

    extractor = CountingExtractor()
    episode = ActorRuntime(cfg, hardware, extractor, KeyboardController()).rollout_episode()
    assert episode is not None
    assert len(episode.transitions) == 2
    assert extractor.calls == 2


def test_actor_runtime_refills_vla_actions_below_low_watermark() -> None:
    cfg = make_cfg()
    cfg.model.ref_num_action_chunks = 50
    cfg.model.num_action_chunks = 50
    cfg.runtime.max_episode_steps = 41
    cfg.hardware.control_hz = 1000.0
    hardware = FakePiperHardware(cfg.hardware)

    class CountingExtractor(FakeFeatureExtractor):
        def __init__(self):
            super().__init__(cfg.model)
            self.calls = 0

        def __call__(self, observation: dict):
            self.calls += 1
            return super().__call__(observation)

    extractor = CountingExtractor()
    episode = ActorRuntime(cfg, hardware, extractor, KeyboardController()).rollout_episode()
    assert episode is not None
    assert len(episode.transitions) == 3
    assert extractor.calls == 2


def test_actor_runtime_caches_generated_vla_actions() -> None:
    cfg = make_cfg()
    runtime = ActorRuntime(cfg, FakePiperHardware(cfg.hardware), FakeFeatureExtractor(cfg.model), KeyboardController())
    features = FakeFeatureExtractor(cfg.model)({"state": torch.zeros(7)})

    runtime._fill_action_queue("vla", features)

    assert len(runtime._action_queue) == cfg.model.num_action_chunks
    action, queued_features = runtime._action_queue.popleft()
    assert torch.equal(action, features.ref_chunk[0, 0])
    assert queued_features is features
    assert len(runtime._action_queue) == cfg.model.num_action_chunks - 1


def test_actor_runtime_smooths_automatic_actions() -> None:
    cfg = make_cfg()
    cfg.runtime.max_episode_steps = 2
    cfg.hardware.control_hz = 1000.0

    class JumpExtractor(FakeFeatureExtractor):
        def __call__(self, observation: dict):
            features = super().__call__(observation)
            features.ref_chunk[0, 0] = torch.zeros(7)
            features.ref_chunk[0, 1] = torch.ones(7) * 10
            return features

    hardware = FakePiperHardware(cfg.hardware)
    episode = ActorRuntime(cfg, hardware, JumpExtractor(cfg.model), KeyboardController()).rollout_episode()
    assert episode is not None
    assert torch.allclose(episode.transitions[0].action[0, 0, :6], torch.zeros(6))
    assert torch.allclose(episode.transitions[0].action[0, 1, :6], torch.ones(6) * 5)


def test_stage1_image_batch_outputs_channels_first() -> None:
    hwc = torch.zeros(480, 640, 3, dtype=torch.uint8)
    bhwc = torch.zeros(1, 480, 640, 3, dtype=torch.uint8)
    chw = torch.zeros(3, 480, 640)
    assert _image_batch(hwc).shape == (1, 3, 480, 640)
    assert _image_batch(bhwc).shape == (1, 3, 480, 640)
    assert _image_batch(chw).shape == (1, 3, 480, 640)
    assert _image_batch(hwc).max() <= 1.0


def test_stage1_uses_piper_training_image_slots() -> None:
    assert _PIPER_IMAGE_KEYS == (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )
    assert _PIPER_TRAINING_ACTION_DIM == 32


def test_real_hardware_reuses_sync_pool_for_policy_actions() -> None:
    from lerobot.utils.piper_sdk import PIPER_ACTION_KEYS

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class Pool:
        def __init__(self):
            self.calls = 0

        def submit(self, fn, *args, **kwargs):
            self.calls += 1
            return ImmediateFuture(fn(*args, **kwargs))

    hardware = RealPiperHardware(make_cfg().hardware)
    pool = Pool()
    sent = {key: 1.0 for key in PIPER_ACTION_KEYS}
    hardware._sync_pool = pool
    hardware.follower = Mock(send_action=Mock(return_value=sent))
    hardware.leader = Mock(send_feedback=Mock(return_value=None))

    returned = hardware.send_automatic(torch.ones(7))

    assert pool.calls == 2
    assert returned.shape == (7,)
    hardware.follower.send_action.assert_called_once()
    hardware.leader.send_feedback.assert_called_once()


def test_real_hardware_connect_reuses_existing_camera_session() -> None:
    hardware = RealPiperHardware(make_cfg().hardware)
    hardware.follower = Mock(is_connected=True)
    hardware.leader = Mock(is_connected=True)
    hardware.top_camera = Mock()
    hardware.wrist_camera = Mock()
    hardware._sync_pool = Mock()

    with patch.object(hardware, "_open_camera", side_effect=AssertionError("camera reopened")):
        hardware.connect()


def test_real_hardware_requires_existing_calibration_before_connect() -> None:
    class Device:
        def __init__(self, calibrated: bool, identifier: str) -> None:
            self.is_calibrated = calibrated
            self.id = identifier
            self.connect = Mock()

        def __str__(self) -> str:
            return self.id

    class Config:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    follower = Device(True, "follower-id")
    leader = Device(False, "leader-id")
    follower_module = ModuleType("lerobot.robots.piper_follower")
    follower_module.PiperFollower = Mock(return_value=follower)
    follower_config_module = ModuleType("lerobot.robots.piper_follower.config_piper_follower")
    follower_config_module.PiperFollowerConfig = Config
    leader_module = ModuleType("lerobot.teleoperators.piper_leader")
    leader_module.PiperLeader = Mock(return_value=leader)
    leader_config_module = ModuleType("lerobot.teleoperators.piper_leader.config_piper_leader")
    leader_config_module.PiperLeaderConfig = Config

    with patch.dict(
        sys.modules,
        {
            "lerobot.robots.piper_follower": follower_module,
            "lerobot.robots.piper_follower.config_piper_follower": follower_config_module,
            "lerobot.teleoperators.piper_leader": leader_module,
            "lerobot.teleoperators.piper_leader.config_piper_leader": leader_config_module,
        },
    ):
        try:
            RealPiperHardware(make_cfg().hardware).connect()
        except RuntimeError as error:
            assert "lerobot-calibrate --teleop.type=piper_leader" in str(error)
        else:
            raise AssertionError("Expected missing leader calibration to abort before connect.")

    follower_module.PiperFollower.assert_called_once()
    leader_module.PiperLeader.assert_called_once()
    assert follower_module.PiperFollower.call_args.args[0].id == "my_piper_follower"
    assert leader_module.PiperLeader.call_args.args[0].id == "my_piper_leader"
    follower.connect.assert_not_called()
    leader.connect.assert_not_called()


def test_real_run_requires_two_camera_sources() -> None:
    cfg = make_cfg()
    cfg.runtime.dry_run = False
    try:
        cfg.validate()
    except ValueError as error:
        assert "top_camera.index_or_path" in str(error)
    else:
        raise AssertionError("Expected real hardware validation to require camera sources.")


def test_episode_countdown_sleeps_once_per_second() -> None:
    with patch.object(run.time, "sleep") as sleep:
        run._countdown_before_episode(3)
    assert sleep.call_count == 3
    sleep.assert_called_with(1.0)


def test_actor_mode_persists_without_learner() -> None:
    cfg = make_cfg()
    cfg.runtime.mode = "actor"
    actor, hardware = Mock(), Mock()
    episode = make_episode(cfg, "actor-only", True)
    actor.rollout_episode.return_value = episode
    actor.persist_outbox.return_value = Path("actor-only.pt")
    with (
        patch.object(run, "load_config", return_value=cfg),
        patch.object(run, "make_hardware", return_value=hardware),
        patch.object(run, "load_feature_extractor", return_value=Mock()),
        patch.object(run, "ActorRuntime", return_value=actor),
        patch.object(run, "Learner", side_effect=AssertionError("actor mode created a learner")),
        patch.object(sys, "argv", ["run", "--episodes", "1"]),
    ):
        run.main()
    actor.persist_outbox.assert_called_once_with(episode)
    hardware.disconnect.assert_called_once()


def test_random_parity_sample_keeps_full_future_action_chunk() -> None:
    episodes = {
        0: {"actions": [[0.0]] * 2, "frame_index": [0, 1]},
        1: {"actions": [[0.0]] * 5, "frame_index": [10, 11, 12, 13, 14]},
    }
    old_total = compare_stage1_parity._total_episodes
    old_read = compare_stage1_parity._read_episode_table
    compare_stage1_parity._total_episodes = lambda _path: len(episodes)
    compare_stage1_parity._read_episode_table = lambda _path, episode: episodes[episode]
    try:
        episode_index, episode, frames = compare_stage1_parity._read_random_sample(Path("dataset"), 4, random.Random(1))
    finally:
        compare_stage1_parity._total_episodes = old_total
        compare_stage1_parity._read_episode_table = old_read

    assert episode_index == 1
    row = episode["frame_index"].index(frames[0])
    assert row <= len(episode["actions"]) - 4
