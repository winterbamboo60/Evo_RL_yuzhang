from threading import Event
from types import SimpleNamespace

import torch

from lerobot.onlineRL_evoRL import gym_manipulator
from lerobot.onlineRL_evoRL.actor_new import ActorVLARuntime, OnlineActorRuntime
from lerobot.onlineRL_evoRL.piper_episode_control import hold_arms_current_pose
from lerobot.onlineRL_evoRL.rtc_action_runner import RTCActionChunkRunner
from lerobot.policies.rtc.configuration_rtc import RTCConfig


class _Robot:
    action_features = {"left_joint.pos": float, "right_joint.pos": float}
    observation_features = action_features
    robot_type = "bi_piper_follower"

    def __init__(self):
        self.is_connected = False
        self.actions = []

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        return {"left_joint.pos": 1.0, "right_joint.pos": 2.0}

    def send_action(self, action):
        self.actions.append(dict(action))
        return action


class _Teleop:
    def __init__(self):
        self.feedback = []
        self.manual = None

    def set_manual_control(self, enabled):
        self.manual = enabled

    def send_feedback(self, action):
        self.feedback.append(dict(action))


def test_can_control_false_never_builds_or_connects_leader(monkeypatch):
    robot = _Robot()
    monkeypatch.setattr(gym_manipulator, "make_robot_from_config", lambda _cfg: robot)
    built_teleop = []
    monkeypatch.setattr(
        gym_manipulator,
        "make_teleoperator_from_config",
        lambda _cfg: built_teleop.append(True),
    )
    cfg = SimpleNamespace(
        name="real_robot",
        robot=object(),
        teleop=object(),
        processor=SimpleNamespace(reset=None),
    )

    env, teleop = gym_manipulator.make_robot_env(cfg, connect_teleop=False)

    assert env.robot is robot
    assert teleop is None
    assert built_teleop == []


def test_hold_pose_supports_prefixed_bimanual_features_without_disabling():
    robot, teleop = _Robot(), _Teleop()
    held = hold_arms_current_pose(robot, teleop)
    assert held == {"left_joint.pos": 1.0, "right_joint.pos": 2.0}
    assert robot.actions[-1] == held
    assert teleop.feedback[-1] == held
    assert teleop.manual is False


def test_rtc_inflight_chunk_is_discarded_after_intervention_invalidation():
    started, release = Event(), Event()

    def predict(*_args):
        started.set()
        assert release.wait(timeout=2)
        return torch.ones(1, 4, 2)

    policy = SimpleNamespace(config=SimpleNamespace(device="cpu", use_amp=False))
    runner = RTCActionChunkRunner(
        policy=policy,
        postprocessor=lambda actions: actions,
        rtc=RTCConfig(enabled=True, mode="guided", execution_horizon=2),
        fps=30,
        queue_threshold=2,
        task="pick",
        robot_type="bi_piper_follower",
        chunk_predictor=predict,
    )
    runner._start_inference({"observation.state": torch.zeros(2)})
    assert started.wait(timeout=2)
    runner.pause()
    runner.invalidate_pending_actions()
    release.set()
    runner.wait_for_idle()
    assert runner.action_queue.empty()
    runner.close()


def test_online_actor_rtc_uses_shared_chunk_runner(monkeypatch):
    runtime = object.__new__(OnlineActorRuntime)
    runtime.cfg = SimpleNamespace(rtc=SimpleNamespace(enabled=True))
    calls = []

    def shared_rtc_select(self, observation_frame, robot_type, device):
        calls.append((observation_frame, robot_type, device))
        return torch.tensor([[3.0]])

    monkeypatch.setattr(ActorVLARuntime, "select_action", shared_rtc_select)
    observation = {"observation.state": torch.zeros(1)}

    result = runtime.select_action(observation, "bi_piper_follower", torch.device("cpu"))

    assert torch.equal(result, torch.tensor([[3.0]]))
    assert calls == [(observation, "bi_piper_follower", torch.device("cpu"))]
