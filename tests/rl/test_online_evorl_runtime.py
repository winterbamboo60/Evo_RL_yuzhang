import json
import time
from threading import Event
from types import SimpleNamespace

import torch

from lerobot.onlineRL_evoRL import gym_manipulator, learner as evorl_learner
from lerobot.onlineRL_evoRL.actor_new import ActorVLARuntime, OnlineActorRuntime
from lerobot.onlineRL_evoRL.compact_transition import (
    SLIDING_WINDOW_TRANSITIONS,
    make_compact_episode,
    save_compact_episode,
)
from lerobot.onlineRL_evoRL.extract_offline_features import partition_episodes
from lerobot.onlineRL_evoRL.learner import _restore_replay_buffer, _save_replay_buffer
from lerobot.onlineRL_evoRL.learner_algorithm import HeadOnlyRLTChunkAlgorithm
from lerobot.onlineRL_evoRL.offline_pretraining import (
    OfflineFrame,
    append_offline_episode,
    build_offline_compact_transitions,
    load_compact_replay,
)
from lerobot.onlineRL_evoRL.piper_episode_control import (
    PoseHoldController,
    default_home_action,
    hold_arms_current_pose,
)
from lerobot.onlineRL_evoRL.rtc_action_runner import RTCActionChunkRunner
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION, OBS_STATE


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


def test_episode_pose_holder_refreshes_terminal_action_until_stopped():
    robot, teleop = _Robot(), _Teleop()
    holder = PoseHoldController(robot, teleop, fps=200)

    held = holder.start()
    time.sleep(0.03)
    holder.stop()

    assert held == {"left_joint.pos": 1.0, "right_joint.pos": 2.0}
    assert len(robot.actions) >= 2
    assert all(action == held for action in robot.actions)
    assert teleop.manual is False


def test_default_home_action_zeros_joints_and_opens_all_grippers():
    robot = SimpleNamespace(
        action_features={
            "left_joint_1.pos": float,
            "left_gripper.pos": float,
            "right_joint_1.pos": float,
            "right_gripper.pos": float,
        }
    )

    assert default_home_action(robot, max_gripper_pos=100.0) == {
        "left_joint_1.pos": 0.0,
        "left_gripper.pos": 100.0,
        "right_joint_1.pos": 0.0,
        "right_gripper.pos": 100.0,
    }


def test_head_only_learner_builds_rlt_heads_without_a_vla():
    def feature(size):
        return SimpleNamespace(shape=(size,))

    policy_config = SimpleNamespace(
        output_features={"action": feature(2)},
        input_features={"observation.state": feature(3)},
        rlt_embed_dim=4,
        rlt_num_rl_tokens=1,
        chunk_size=2,
    )
    algorithm_config = SimpleNamespace(
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        num_critics=2,
        fixed_std=0.1,
        reference_dropout_prob=0.0,
    )

    algorithm = HeadOnlyRLTChunkAlgorithm(policy_config, algorithm_config, device="cpu")

    assert algorithm.policy is None
    assert sum(parameter.numel() for parameter in algorithm.actor.parameters()) > 0


def test_compact_replay_checkpoint_saves_only_live_rows_and_restores_capacity(tmp_path):
    source = ReplayBuffer(
        capacity=5,
        device="cpu",
        storage_device="cpu",
        state_keys=("z_rl",),
        use_drq=False,
    )
    for value in range(3):
        tensor = torch.tensor([[float(value)]])
        source.add(
            state={"z_rl": tensor},
            action=tensor,
            reward=float(value),
            next_state={"z_rl": tensor + 1},
            done=False,
            truncated=False,
        )
    path = tmp_path / "compact_replay.pt"

    _save_replay_buffer(source, path)
    payload = torch.load(path, weights_only=True)
    restored = ReplayBuffer(
        capacity=5,
        device="cpu",
        storage_device="cpu",
        state_keys=("z_rl",),
        use_drq=False,
    )
    _restore_replay_buffer(restored, path)

    assert payload["states"]["z_rl"].shape[0] == 3
    assert restored.states["z_rl"].shape[0] == 5
    assert restored.size == 3
    restored.add(
        state={"z_rl": torch.tensor([[3.0]])},
        action=torch.tensor([[3.0]]),
        reward=3.0,
        next_state={"z_rl": torch.tensor([[4.0]])},
        done=False,
        truncated=False,
    )
    assert restored.size == 4


def test_offline_episode_is_successful_and_every_valid_action_is_intervention():
    replay = ReplayBuffer(
        capacity=2,
        device="cpu",
        storage_device="cpu",
        state_keys=("z_rl", "proprio", "ref_action"),
        use_drq=False,
    )
    frames = []
    for index in range(3):
        frames.append(
            OfflineFrame(
                state={
                    "z_rl": torch.full((1, 2), float(index)),
                    "proprio": torch.full((2,), float(index)),
                    "ref_action": torch.zeros(2, 2),
                },
                action_chunk=torch.tensor(
                    [[float(index), 0.0], [float(min(index + 1, 2)), 0.0]]
                ),
                valid_action_mask=torch.tensor([True, index < 2]),
            )
        )

    added = append_offline_episode(replay, frames, horizon=2, stride=2)

    assert added == len(replay) == 2
    torch.testing.assert_close(
        replay.complementary_info["reward_chunk"],
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
    )
    assert torch.equal(
        replay.complementary_info["intervene_flags"],
        replay.complementary_info["valid_action_mask"].bool(),
    )
    assert replay.dones.tolist() == [False, True]


def test_offline_compact_batch_updates_actor_and_critic_heads():
    def feature(size):
        return SimpleNamespace(shape=(size,))

    policy_config = SimpleNamespace(
        output_features={ACTION: feature(2)},
        input_features={OBS_STATE: feature(3)},
        rlt_embed_dim=4,
        rlt_num_rl_tokens=1,
        chunk_size=2,
    )
    algorithm_config = SimpleNamespace(
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        num_critics=2,
        fixed_std=0.1,
        reference_dropout_prob=0.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        actor_update_interval=1,
        discount=0.96,
        critic_target_update_weight=0.005,
        q_weight=0.1,
        bc_weight=5.0,
        grad_clip_norm=10.0,
    )
    algorithm = HeadOnlyRLTChunkAlgorithm(policy_config, algorithm_config, device="cpu")
    algorithm.make_optimizers_and_scheduler()
    replay = ReplayBuffer(
        capacity=2,
        device="cpu",
        storage_device="cpu",
        state_keys=("z_rl", "proprio", "ref_action"),
        use_drq=False,
    )
    frames = [
        OfflineFrame(
            state={
                "z_rl": torch.full((1, 4), float(index)),
                "proprio": torch.full((3,), float(index)),
                "ref_action": torch.zeros(2, 2),
            },
            action_chunk=torch.full((2, 2), 0.1 * (index + 1)),
            valid_action_mask=torch.tensor([True, index < 2]),
        )
        for index in range(3)
    ]
    append_offline_episode(replay, frames, horizon=2, stride=2)
    actor_before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in algorithm.critic_ensemble.parameters()]

    stats = algorithm.update(iter([replay.sample(2)]))

    assert "loss_actor" in stats.losses
    assert "loss_critic" in stats.losses
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, algorithm.actor.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, algorithm.critic_ensemble.parameters(), strict=True)
    )


def test_multigpu_extraction_partitions_whole_episodes_without_overlap():
    episodes = list(range(10))
    shards = [partition_episodes(episodes, rank, 3) for rank in range(3)]

    assert shards == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert sorted(episode for shard in shards for episode in shard) == episodes


def test_learner_loads_actor_format_compact_dataset_without_raw_lerobot_data(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    compact_root = tmp_path / "compact"
    compact_root.mkdir()
    transition_count = 0
    for episode_index in range(2):
        frames = [
            OfflineFrame(
                state={
                    "z_rl": torch.full((1, 4), float(frame_index)),
                    "proprio": torch.full((3,), float(frame_index)),
                    "ref_action": torch.zeros(2, 2),
                },
                action_chunk=torch.full((2, 2), 0.1 * (frame_index + 1)),
                valid_action_mask=torch.tensor([True, frame_index < 2]),
            )
            for frame_index in range(3)
        ]
        transitions = build_offline_compact_transitions(frames, horizon=2, stride=2)
        transition_count += len(transitions)
        metadata = {
            "task": "pick",
            "episode_id": f"offline:test:{episode_index}",
            "source_episode_index": episode_index,
            "primitive_steps": len(frames),
            "num_transitions": len(transitions),
        }
        payload = make_compact_episode(
            transitions=transitions,
            metadata=metadata,
            feature_model={"resolved_path": str(model_path)},
            transition_layout=SLIDING_WINDOW_TRANSITIONS,
            sliding_window_stride=2,
            primitive_steps=len(frames),
        )
        episode_dir = compact_root / f"episode_{episode_index:06d}"
        save_compact_episode(payload, episode_dir / "compact_episode.pt")
        (episode_dir / "metadata.json").write_text(json.dumps(metadata))
    (compact_root / "manifest.json").write_text(
        json.dumps({"completed_episodes": 2, "total_transitions": transition_count})
    )

    cfg = SimpleNamespace(
        offline_pretraining=SimpleNamespace(compact_dataset_path=str(compact_root)),
        policy=SimpleNamespace(pretrained_path=str(model_path), device="cpu"),
        algorithm=SimpleNamespace(
            state_keys=("z_rl", "proprio", "ref_action"),
            storage_device="cpu",
        ),
    )

    class Ingestor:
        ingest_transition_payload = HeadOnlyRLTChunkAlgorithm.ingest_transition_payload

    replay = load_compact_replay(cfg, Ingestor())

    assert len(replay) == transition_count == 4
    assert torch.equal(
        replay.complementary_info["intervene_flags"].bool(),
        replay.complementary_info["valid_action_mask"].bool(),
    )


def test_compact_episode_uses_vla_inference_image_layout_and_normalized_actions():
    processed_inputs = []

    def preprocessor(model_input):
        processed_inputs.append(model_input)
        assert model_input["observation.images.front"].shape == (1, 3, 6, 8)
        assert model_input["observation.images.front"].dtype == torch.float32
        result = dict(model_input)
        result[ACTION] = (model_input[ACTION] + 10).unsqueeze(0)
        return result

    class FakePolicy:
        def __init__(self):
            self.image_batch_shapes = []

        def predict_action_chunk_with_rlt(self, batch):
            self.image_batch_shapes.append(tuple(batch["observation.images.front"].shape))
            batch_size = batch[ACTION].shape[0]
            return {
                "z_rl": torch.zeros(batch_size, 1, 4),
                "actions": batch[ACTION].unsqueeze(1).repeat(1, 2, 1),
            }

    def observation(pixel_value):
        return {
            OBS_STATE: torch.arange(4, dtype=torch.float32),
            "observation.images.front": torch.full((6, 8, 3), pixel_value, dtype=torch.uint8),
        }

    policy = FakePolicy()
    runtime = object.__new__(ActorVLARuntime)
    runtime.policy = policy
    runtime.preprocessor = preprocessor
    runtime.policy_cfg = SimpleNamespace(chunk_size=2, device="cpu", proprio_dim=4)
    runtime.cfg = SimpleNamespace(online_transition=SimpleNamespace(sliding_window_stride=1))
    runtime.fingerprint = SimpleNamespace(resolved_path="/tmp/test-policy", files={})
    transitions = [
        {
            "state": observation(255),
            ACTION: torch.tensor([1.0, 2.0]),
            "reward": 0.0,
            "next_state": observation(127),
            "done": False,
            "truncated": False,
            "complementary_info": {"intervention": False},
        },
        {
            "state": observation(127),
            ACTION: torch.tensor([3.0, 4.0]),
            "reward": 1.0,
            "next_state": observation(0),
            "done": True,
            "truncated": False,
            "complementary_info": {"intervention": False},
        },
    ]

    compact = runtime.build_compact_episode(
        transitions,
        {"task": "catch cube"},
        batch_size=2,
        robot_type="bi_piper_follower",
    )

    assert policy.image_batch_shapes == [(2, 3, 6, 8), (1, 3, 6, 8)]
    assert all(item["task"] == "catch cube" for item in processed_inputs)
    assert all(item["robot_type"] == "bi_piper_follower" for item in processed_inputs)
    assert torch.equal(compact["transitions"][0][ACTION], torch.tensor([[11.0, 12.0]]))
    assert transitions[0]["state"]["observation.images.front"].shape == (6, 8, 3)
    assert transitions[0]["state"]["observation.images.front"].dtype == torch.uint8


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


def test_rtc_guided_second_chunk_uses_only_local_input_gradients():
    rtc = RTCConfig(enabled=True, mode="guided", execution_horizon=2)
    processor = RTCProcessor(rtc)
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    callback_grad_modes = []
    denoiser_grad_modes = []
    has_previous_actions = []

    def predict(_observation, inference_delay, previous_actions, _task, _robot_type):
        callback_grad_modes.append(torch.is_grad_enabled())
        has_previous_actions.append(previous_actions is not None)
        x_t = torch.full((1, 4, 2), 0.5)

        def denoise(input_x):
            denoiser_grad_modes.append(torch.is_grad_enabled())
            return input_x * parameter

        velocity = processor.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=previous_actions,
            inference_delay=inference_delay,
            time=0.5,
            original_denoise_step_partial=denoise,
            execution_horizon=rtc.execution_horizon,
        )
        return x_t - 0.5 * velocity

    policy = SimpleNamespace(config=SimpleNamespace(device="cpu", use_amp=False))
    runner = RTCActionChunkRunner(
        policy=policy,
        postprocessor=lambda actions: actions,
        rtc=rtc,
        fps=30,
        queue_threshold=3,
        task="pick",
        robot_type="bi_piper_follower",
        chunk_predictor=predict,
    )
    try:
        first = runner.get_action({"observation.state": torch.zeros(2)})
        second = runner.get_action({"observation.state": torch.ones(2)})
        runner.wait_for_idle()
    finally:
        runner.close()

    assert first.shape == second.shape == (1, 2)
    assert has_previous_actions == [False, True]
    assert callback_grad_modes == [False, False]
    assert denoiser_grad_modes == [False, True]
    assert parameter.grad is None


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


def test_evorl_learner_initializes_and_closes_tensorboard_logger(monkeypatch, tmp_path):
    events = {}

    class Config(SimpleNamespace):
        def to_dict(self):
            return {}

    class FakeTensorBoardLogger:
        def __init__(self, cfg):
            events["initialized_with"] = cfg
            events["logger"] = self
            self.closed = False

        def finish(self):
            self.closed = True

    cfg = Config(
        output_dir=str(tmp_path / "learner"),
        job_name="tensorboard-test",
        seed=1000,
        algorithm=SimpleNamespace(concurrency=SimpleNamespace(learner="threads")),
        wandb=SimpleNamespace(enable=False, project=None),
        tensorboard=SimpleNamespace(enable=True, log_dir=str(tmp_path / "tensorboard")),
    )
    shutdown_event = Event()

    monkeypatch.setattr(evorl_learner, "_validate_startup_config", lambda _cfg: None)
    monkeypatch.setattr(evorl_learner, "init_logging", lambda **_kwargs: None)
    monkeypatch.setattr(evorl_learner, "set_seed", lambda _seed: None)
    monkeypatch.setattr(
        evorl_learner,
        "ProcessSignalHandler",
        lambda *_args, **_kwargs: SimpleNamespace(shutdown_event=shutdown_event),
    )
    monkeypatch.setattr(evorl_learner, "TensorBoardLogger", FakeTensorBoardLogger)

    def start_runtime(runtime_cfg, wandb_logger, tensorboard_logger, runtime_shutdown_event):
        events["runtime_args"] = (
            runtime_cfg,
            wandb_logger,
            tensorboard_logger,
            runtime_shutdown_event,
        )

    monkeypatch.setattr(evorl_learner, "_start_runtime", start_runtime)

    evorl_learner.train(cfg)

    logger = events["logger"]
    assert events["initialized_with"] is cfg
    assert events["runtime_args"] == (cfg, None, logger, shutdown_event)
    assert logger.closed
