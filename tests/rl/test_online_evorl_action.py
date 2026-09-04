from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.train import ActorOnlyConfig, OnlineTransitionConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.onlineRL_evoRL.actor_new import ActorEpisodeWriter, ActorVLARuntime
from lerobot.onlineRL_evoRL.compact_transition import (
    SLIDING_WINDOW_TRANSITIONS,
    make_compact_episode,
)
from lerobot.onlineRL_evoRL.gym_manipulator import (
    RobotEnv,
    create_transition,
    step_env_and_process_transition,
)
from lerobot.processor import TransitionKey
from lerobot.scripts.recording_hil import PolicySyncDualArmExecutor
from lerobot.utils.constants import ACTION, OBS_STATE


def test_actor_only_save_format_validation():
    assert ActorOnlyConfig().save_format == "lerobot"
    assert ActorOnlyConfig(save_format="transition").save_format == "transition"
    assert ActorOnlyConfig(save_format="lerobot").save_format == "lerobot"
    with pytest.raises(ValueError, match="actor_only.save_format"):
        ActorOnlyConfig(save_format="invalid")


def test_online_transition_stride_validation():
    assert OnlineTransitionConfig().sliding_window_stride == 1
    assert OnlineTransitionConfig(sliding_window_stride=2).sliding_window_stride == 2
    with pytest.raises(ValueError, match="sliding_window_stride"):
        OnlineTransitionConfig(sliding_window_stride=0)


def test_compact_episode_computes_features_only_at_stride_boundaries():
    class Policy:
        def __init__(self):
            self.observation_indices = []

        def predict_action_chunk_with_rlt(self, batch):
            state = batch[OBS_STATE].float()
            self.observation_indices.extend(int(value) for value in state[:, 0].tolist())
            return {
                "z_rl": torch.cat((state, state + 100), dim=-1),
                "actions": state.unsqueeze(1).repeat(1, 4, 1),
            }

    policy = Policy()
    runtime = ActorVLARuntime.__new__(ActorVLARuntime)
    runtime.policy = policy
    runtime.policy_cfg = SimpleNamespace(chunk_size=4, proprio_dim=1)
    runtime.preprocessor = lambda raw: {
        OBS_STATE: raw[OBS_STATE].reshape(1, -1).float(),
        ACTION: raw[ACTION].reshape(1, -1).float(),
    }
    runtime.fingerprint = SimpleNamespace(resolved_path="/tmp/base")
    runtime.cfg = SimpleNamespace(online_transition=SimpleNamespace(sliding_window_stride=2))

    transitions = []
    for index in range(7):
        transitions.append(
            {
                "state": {OBS_STATE: torch.tensor([[float(index)]])},
                ACTION: torch.tensor([[index + 0.5]]),
                "reward": float(index),
                "next_state": {OBS_STATE: torch.tensor([[float(index + 1)]])},
                "done": index == 6,
                "truncated": False,
                "complementary_info": {"is_intervention": index % 2 == 0},
            }
        )

    payload = runtime.build_compact_episode(
        transitions,
        metadata={"task": "test"},
        batch_size=3,
    )

    assert policy.observation_indices == [0, 2, 4, 6, 7]
    assert payload["transition_layout"] == SLIDING_WINDOW_TRANSITIONS
    assert payload["sliding_window_stride"] == 2
    assert payload["primitive_steps"] == 7
    assert len(payload["transitions"]) == 4
    assert payload["transitions"][0]["state"]["z_rl"][0, 0] == 0
    assert payload["transitions"][0]["next_state"]["z_rl"][0, 0] == 4
    assert payload["transitions"][0]["target_action_chunk"][:, 0].tolist() == [
        0.5,
        1.5,
        2.5,
        3.5,
    ]


def test_single_env_sends_and_records_unbatched_policy_action():
    class FakeRobot:
        is_connected = True
        cameras = {}
        action_features = {
            **{f"joint_{i}.pos": float for i in range(1, 7)},
            "gripper.pos": float,
        }

        def get_observation(self):
            return dict.fromkeys(self.action_features, 0.0)

        def send_action(self, action, add_offset=False):
            self.sent_action = action
            self.sent_add_offset = add_offset
            return dict.fromkeys(action, -1.0)

    class FakeTeleop:
        def send_feedback(self, action, add_offset=False):
            self.feedback = action
            self.feedback_add_offset = add_offset

    robot = FakeRobot()
    env = RobotEnv(robot, reset_time_s=0)
    teleop = FakeTeleop()
    executor = PolicySyncDualArmExecutor(robot=robot, teleop=teleop, parallel_dispatch=True)
    env.action_sender = executor
    env.action_add_offset = True
    policy_action = torch.arange(7, dtype=torch.float32).unsqueeze(0)

    transition = step_env_and_process_transition(
        env=env,
        transition=create_transition(observation={}),
        action=policy_action,
        env_processor=lambda value: value,
        action_processor=lambda value: value,
    )
    executor.shutdown()

    assert robot.sent_action == dict(zip(robot.action_features, range(7), strict=True))
    assert teleop.feedback == robot.sent_action
    assert robot.sent_add_offset and teleop.feedback_add_offset
    assert torch.equal(transition[TransitionKey.ACTION], policy_action.squeeze(0))


def test_actor_episode_writer_save_formats(tmp_path: Path):
    action = torch.arange(7, dtype=torch.float32).unsqueeze(0)
    transition = {
        "state": {OBS_STATE: action.clone()},
        "action": action.clone(),
        "reward": 1.0,
        "next_state": {OBS_STATE: action.clone()},
        "done": True,
        "truncated": False,
        "complementary_info": {
            "policy_action": action.clone(),
            "is_intervention": torch.tensor([0.0]),
            "intervention_state": torch.tensor([0.0]),
        },
    }
    metadata = {
        "task": "test task",
        "episode_outcome": "success",
        "actor_policy_path": "/tmp/test-policy",
    }
    robot = SimpleNamespace(
        name="piper_follower",
        action_features={
            **{f"joint_{i}.pos": float for i in range(1, 7)},
            "gripper.pos": float,
        },
    )

    compact = make_compact_episode(
        transitions=[
            {
                "state": {
                    "z_rl": torch.zeros(1, 2),
                    "proprio": action.clone(),
                    "ref_action": action.unsqueeze(1),
                },
                "next_state": {
                    "z_rl": torch.zeros(1, 2),
                    "proprio": action.clone(),
                    "ref_action": action.unsqueeze(1),
                },
                "action": action.clone(),
                "reward": 1.0,
                "done": True,
                "truncated": False,
                "complementary_info": {"is_intervention": False},
            }
        ],
        metadata=metadata,
        feature_model={"resolved_path": "/tmp/test-policy"},
    )

    for save_format in ("transition", "lerobot"):
        output_dir = tmp_path / save_format
        cfg = SimpleNamespace(
            output_dir=tmp_path,
            actor_only=SimpleNamespace(
                episode_output_dir=str(output_dir),
                save_format=save_format,
                save_episode_images=False,
                save_episode_viewer=False,
            ),
            dataset=SimpleNamespace(repo_id="local_data"),
            env=SimpleNamespace(fps=30),
        )
        writer = ActorEpisodeWriter(cfg)
        writer.configure_robot(robot)
        writer.save_episode(
            transitions=[transition],
            metadata=metadata,
            compact_episode=compact if save_format == "transition" else None,
        )
        writer.finalize()

    assert (tmp_path / "transition/episode_000000/compact_episode.pt").is_file()
    assert (tmp_path / "lerobot/meta/info.json").is_file()
    assert any((tmp_path / "lerobot/data").rglob("*.parquet"))
    dataset = LeRobotDataset(repo_id="local_data", root=tmp_path / "lerobot")
    assert dataset.num_episodes == 1
    assert torch.equal(dataset[0][ACTION], action.squeeze(0))
