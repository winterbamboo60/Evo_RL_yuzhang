from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.onlineRL_evoRL.actor import ActorEpisodeWriter
from lerobot.onlineRL_evoRL.gym_manipulator import (
    RobotEnv,
    create_transition,
    step_env_and_process_transition,
)
from lerobot.processor import TransitionKey
from lerobot.scripts.recording_hil import PolicySyncDualArmExecutor
from lerobot.utils.constants import ACTION, OBS_STATE


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
        writer.save_episode(transitions=[transition], metadata=metadata)
        writer.finalize()

    assert (tmp_path / "transition/episode_000000/transitions.pt").is_file()
    assert (tmp_path / "lerobot/meta/info.json").is_file()
    assert any((tmp_path / "lerobot/data").rglob("*.parquet"))
    dataset = LeRobotDataset(repo_id="local_data", root=tmp_path / "lerobot")
    assert dataset.num_episodes == 1
    assert torch.equal(dataset[0][ACTION], action.squeeze(0))
