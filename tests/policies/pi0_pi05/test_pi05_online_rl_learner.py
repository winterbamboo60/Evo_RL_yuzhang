import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.onlineRL_evoRL import learner as server_learner
from lerobot.onlineRL_evoRL.compact_transition import (
    SLIDING_WINDOW_TRANSITIONS,
    bytes_to_episode_payload,
    compact_episode_to_bytes,
    make_compact_episode,
    save_compact_episode,
)
from lerobot.policies.pi05_onlineRL.configuration_pi05_online_rl import PI05OnlineRLConfig
from lerobot.policies.pi05_onlineRL.learner import (
    _add_compact_episode,
    _build_offline_batch,
    _move_training_state,
    _offline_annotations,
    _online_batch_sizes,
    _save_online_replay_checkpoint,
    _training_phase,
    _validate_online_task,
)


class _Episodes(list):
    column_names = ["episode_success", "dataset_to_index"]


class _Dataset:
    features = {"complementary_info.collector_policy_id": {"dtype": "string"}}
    hf_dataset = {
        "index": [torch.tensor(i) for i in range(4)],
        "episode_index": [torch.tensor(0), torch.tensor(0), torch.tensor(1), torch.tensor(1)],
        "complementary_info.collector_policy_id": ["human", "policy", "human", "policy"],
    }
    meta = SimpleNamespace(
        total_episodes=2,
        episodes=_Episodes(
            [
                {"episode_success": "success", "dataset_to_index": 2},
                {"episode_success": "failure", "dataset_to_index": 4},
            ]
        ),
    )

    def __len__(self):
        return 4


def test_offline_annotations_use_collector_and_episode_metadata():
    annotations = _offline_annotations(_Dataset())
    assert annotations["is_human"].tolist() == [True, False, True, False]
    assert annotations["terminal"].tolist() == [False, True, False, True]
    assert annotations["terminal_reward"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert annotations["episode_end"].tolist() == [2, 4]


def test_online_episode_metadata_task_must_match_dataset_meta():
    assert _validate_online_task("a", {"a", "b"}) == "a"
    with pytest.raises(ValueError, match="non-empty task"):
        _validate_online_task(None, {"a", "b"})
    with pytest.raises(ValueError, match="meta/tasks.parquet"):
        _validate_online_task("unknown", {"a"})


def test_offline_batch_builds_reward_intervention_and_masks():
    class Policy:
        config = SimpleNamespace(chunk_size=3, input_features={"observation.state": object()})

        def extract_rlt_features(self, batch):
            state = batch["observation.state"].float()
            return {"z_rl": state, "proprio": state, "ref_action": batch["action"].float()}

    raw = {
        "index": torch.tensor([0]),
        "episode_index": torch.tensor([0]),
        "observation.state": torch.tensor([[[1.0], [4.0]]]),
        "action": torch.tensor([[[1.0], [2.0], [3.0]]]),
        "action_is_pad": torch.tensor([[False, False, False]]),
        "task": ["a"],
    }
    annotations = {
        "is_human": torch.tensor([True, False, True]),
        "terminal": torch.tensor([False, False, True]),
        "terminal_reward": torch.tensor([0.0, 0.0, 1.0]),
        "episode_end": torch.tensor([3]),
    }
    batch = _build_offline_batch(raw, annotations, Policy(), lambda value: value)
    assert batch["reward"].tolist() == [[0.0, 0.0, 1.0]]
    assert batch["intervene_flags"].tolist() == [[True, False, True]]
    assert batch["next_valid_action_mask"].tolist() == [[False, False, False]]
    assert batch["done"].tolist() == [1.0]
    assert batch["state"]["z_rl"].tolist() == [[1.0]]
    assert batch["next_state"]["z_rl"].tolist() == [[4.0]]


def test_compact_episode_round_trip_and_replay_insert(tmp_path):
    class Replay:
        storage_device = "cpu"

        def __init__(self):
            self.items = []

        def add(self, **transition):
            self.items.append(transition)

    policy = SimpleNamespace(
        config=SimpleNamespace(
            pretrained_path="/tmp/base",
            device="cuda",
            z_dim=2,
            proprio_dim=7,
            chunk_size=2,
        ),
        action_dim=7,
    )
    state = {
        "z_rl": torch.zeros(1, 2),
        "proprio": torch.zeros(1, 7),
        "ref_action": torch.zeros(1, 2, 7),
    }
    payload = make_compact_episode(
        transitions=[
            {
                "state": state,
                "next_state": state,
                "action": torch.zeros(1, 7),
                "reward": 1.0,
                "done": True,
                "truncated": False,
                "complementary_info": {"is_intervention": True},
            }
        ],
        metadata={"task": "a"},
        feature_model={"resolved_path": "/tmp/base"},
    )
    local_path = tmp_path / "compact_episode.pt"
    save_compact_episode(payload, local_path)
    local_payload = torch.load(local_path, weights_only=True)
    online_payload = bytes_to_episode_payload(compact_episode_to_bytes(payload))
    legacy_payload = dict(payload)
    legacy_payload["schema_version"] = 1
    for key in ("transition_layout", "sliding_window_stride", "primitive_steps"):
        legacy_payload.pop(key)

    assert local_payload.keys() == online_payload.keys()
    assert torch.equal(
        local_payload["transitions"][0]["state"]["z_rl"],
        online_payload["transitions"][0]["state"]["z_rl"],
    )
    for decoded in (local_payload, online_payload, legacy_payload):
        replay = Replay()
        count, metadata = _add_compact_episode(decoded, replay, policy, {"a"})
        assert count == len(replay.items) == 1
        assert metadata["task"] == "a"
        assert replay.items[0]["intervene_flags"].tolist() == [True, False]


def test_compact_sliding_windows_are_inserted_without_second_windowing():
    class Replay:
        storage_device = "cpu"

        def __init__(self):
            self.items = []

        def add(self, **transition):
            self.items.append(transition)

    policy = SimpleNamespace(
        config=SimpleNamespace(
            pretrained_path="/tmp/base",
            z_dim=2,
            proprio_dim=7,
            chunk_size=2,
        ),
        action_dim=7,
    )

    def state(value):
        return {
            "z_rl": torch.full((1, 2), float(value)),
            "proprio": torch.full((1, 7), float(value)),
            "ref_action": torch.full((1, 2, 7), float(value)),
        }

    transitions = []
    for index in range(2):
        transitions.append(
            {
                "state": state(index),
                "next_state": state(index + 2),
                "action": torch.full((1, 7), float(index + 1)),
                "target_action_chunk": torch.full((2, 7), float(index + 9)),
                "reward": torch.tensor([1.0, 2.0]),
                "intervene_flags": torch.tensor([True, False]),
                "valid_action_mask": torch.ones(2),
                "next_valid_action_mask": torch.zeros(2),
                "done": index == 1,
                "truncated": False,
                "complementary_info": {"is_intervention": index == 0},
            }
        )
    payload = make_compact_episode(
        transitions=transitions,
        metadata={"task": "a"},
        feature_model={"resolved_path": "/tmp/base"},
        transition_layout=SLIDING_WINDOW_TRANSITIONS,
        sliding_window_stride=2,
        primitive_steps=4,
    )

    replay = Replay()
    count, _ = _add_compact_episode(payload, replay, policy, {"a"})

    assert count == len(replay.items) == 2
    assert torch.equal(replay.items[0]["target_action_chunk"], transitions[0]["target_action_chunk"])
    assert torch.equal(replay.items[1]["target_action_chunk"], transitions[1]["target_action_chunk"])


def test_pi05_online_rl_step_parameters_are_unambiguous():
    config = PI05OnlineRLConfig()
    assert config.observation_delta_indices == [0, config.chunk_size]
    assert config.online_steps == 1_000_000
    assert config.online_updates_per_episode == 100
    assert config.online_only_after_initialization is False
    assert config.offload_to_cpu_while_waiting is False
    enabled = PI05OnlineRLConfig(offload_to_cpu_while_waiting=True)
    assert enabled.offload_to_cpu_while_waiting is True
    assert config.actor_update_interval == 1
    assert not hasattr(config, "max_actor_interaction_steps")
    assert not hasattr(config, "offline_buffer_capacity")
    assert not hasattr(config, "min_online_replay_size")


def test_two_stage_training_schedule():
    assert _training_phase(199, offline_steps=200, online_budget=0) == "offline_initialization"
    assert _training_phase(200, offline_steps=200, online_budget=0) is None
    assert _training_phase(200, offline_steps=200, online_budget=100) == "online"


def test_training_state_device_round_trip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = torch.nn.Linear(2, 1).to(device)
    optimizer = torch.optim.Adam(policy.parameters())
    policy(torch.ones(1, 2, device=device)).sum().backward()
    optimizer.step()
    step_device = optimizer.state[policy.weight]["step"].device

    _move_training_state(policy, {"actor": optimizer}, "cpu")
    assert policy.weight.device.type == "cpu"
    assert optimizer.state[policy.weight]["exp_avg"].device.type == "cpu"
    assert optimizer.state[policy.weight]["step"].device == step_device

    _move_training_state(policy, {"actor": optimizer}, device)
    optimizer.zero_grad()
    policy(torch.ones(1, 2, device=device)).sum().backward()
    optimizer.step()
    assert policy.weight.device == device
    assert optimizer.state[policy.weight]["exp_avg"].device == device


def test_online_phase_batch_composition():
    assert _online_batch_sizes(16, online_only=False) == (8, 8)
    assert _online_batch_sizes(16, online_only=True) == (16, 0)
    assert _online_batch_sizes(1, online_only=False) == (1, 0)
    with pytest.raises(ValueError, match="online_updates_per_episode"):
        PI05OnlineRLConfig(online_updates_per_episode=0)


class _ReplayExporter:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def to_lerobot_dataset(self, repo_id, fps, root):
        root = Path(root)
        root.mkdir(parents=True)
        (root / "marker.txt").write_text("new")
        if self.fail:
            raise RuntimeError("export failed")


def test_online_replay_checkpoint_replaces_existing_only_after_export(tmp_path):
    checkpoint_dir = tmp_path / "001200"
    old_root = checkpoint_dir / "replay_online"
    old_root.mkdir(parents=True)
    (old_root / "marker.txt").write_text("old")

    published = _save_online_replay_checkpoint(
        _ReplayExporter(), checkpoint_dir, fps=30
    )

    assert published == old_root
    assert (published / "marker.txt").read_text() == "new"
    previous = list(checkpoint_dir.glob("replay_online.previous-*"))
    assert len(previous) == 1
    assert (previous[0] / "marker.txt").read_text() == "old"


def test_online_replay_checkpoint_preserves_existing_on_export_failure(tmp_path):
    checkpoint_dir = tmp_path / "001200"
    old_root = checkpoint_dir / "replay_online"
    old_root.mkdir(parents=True)
    (old_root / "marker.txt").write_text("old")

    with pytest.raises(RuntimeError, match="export failed"):
        _save_online_replay_checkpoint(
            _ReplayExporter(fail=True), checkpoint_dir, fps=30
        )

    assert (old_root / "marker.txt").read_text() == "old"
    assert len(list(checkpoint_dir.glob(".replay_online.tmp-*"))) == 1


def test_learner_shutdown_sets_event_before_join(monkeypatch):
    queues = []
    workers = []
    shutdown_event = threading.Event()

    class FakeQueue:
        def __init__(self):
            self.closed = False
            self.cancelled = False
            queues.append(self)

        def close(self):
            self.closed = True

        def cancel_join_thread(self):
            self.cancelled = True

    class FakeWorker:
        def __init__(self, **kwargs):
            self.alive = True
            self.join_timeout = None
            workers.append(self)

        def start(self):
            pass

        def join(self, timeout=None):
            assert shutdown_event.is_set()
            assert not any(queue.closed for queue in queues)
            self.join_timeout = timeout
            self.alive = False

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(server_learner, "Queue", FakeQueue)
    monkeypatch.setattr(server_learner, "use_threads", lambda _: True)
    monkeypatch.setattr(threading, "Thread", FakeWorker)
    monkeypatch.setattr(server_learner, "add_actor_information_and_train", lambda **_: None)

    server_learner.start_learner_threads(object(), None, shutdown_event)

    assert workers[0].join_timeout == server_learner.SHUTDOWN_TIMEOUT + 1
    assert all(queue.closed and queue.cancelled for queue in queues)
