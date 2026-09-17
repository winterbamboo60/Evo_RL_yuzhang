import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch

import lerobot.onlineRL_evoRL.actor_new as actor_new
from lerobot.onlineRL_evoRL.actor_new import (
    ActorControl,
    ActorEpisodeWriter,
    ActorEpisodeWriters,
    ActorKeyboardController,
    ActorVLARuntime,
    OnlineActorRuntime,
    TaskHotkeys,
    _build_actor_control,
    _find_actor_checkpoint,
    _log_unhandled_exception,
    load_task_hotkeys,
)
from lerobot.onlineRL_evoRL.keyboard_control import KeyboardState
from lerobot.utils.recording_annotations import (
    EVORL_COLLECTOR_POLICY_ID_FIELD,
    EVORL_INTERVENTION_FIELD,
    EVORL_POLICY_ACTION_FIELD,
    EVORL_STATE_FIELD,
)
from lerobot.utils.utils import init_logging


class _Resettable:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def reset_action_state(self):
        self.reset()


def _control():
    cfg = SimpleNamespace(env=SimpleNamespace(task="Grab the left cup"))
    tasks = TaskHotkeys(
        default_key="3",
        tasks={
            "1": "Pick up the cup on the right",
            "2": "Take the middle cup away",
            "3": "Grab the left cup",
        },
    )
    control = ActorControl(cfg=cfg, task_hotkeys=tasks, dataset_meta=SimpleNamespace())
    control.runtime = _Resettable()
    control.smoother = _Resettable()
    return control


def test_v_switch_requires_actor_and_clears_cached_actions():
    control = _control()
    controller = ActorKeyboardController(control)
    state = KeyboardState()

    controller.push("v")
    controller.poll(state)
    assert control.use_actor is False

    control.actor_available = True
    controller._last_actor2_event_t = 0.0
    controller.push("V")
    controller.poll(state)
    assert control.use_actor is True
    assert control.runtime.reset_count == 1
    assert control.smoother.reset_count == 1

    controller._last_actor2_event_t = 0.0
    controller.push("v")
    controller.poll(state)
    assert control.use_actor is False


def test_online_actor_mode_starts_on_vla_until_v_is_pressed():
    cfg = SimpleNamespace(actor_mode="online_actor", dataset=None, task_hotkeys_path=None)

    control = _build_actor_control(cfg)

    assert control.use_actor is False


def test_gpu_reacquire_loads_actor_without_selecting_it(monkeypatch):
    control = ActorControl(cfg=SimpleNamespace(), use_actor=False)
    runtime = object.__new__(OnlineActorRuntime)
    runtime.control = control
    runtime.policy_cfg = SimpleNamespace(device="cpu")
    loaded = []

    monkeypatch.setattr(
        ActorVLARuntime,
        "reload",
        lambda self: setattr(self, "policy", object()),
    )
    monkeypatch.setattr(
        OnlineActorRuntime,
        "_build_algorithm",
        lambda self: setattr(
            self,
            "algorithm",
            SimpleNamespace(load_weights=lambda weights, device: loaded.append((weights, device))),
        ),
    )
    monkeypatch.setattr(OnlineActorRuntime, "reset_action_state", lambda self: None)

    weights = {"policy": {"weight": torch.tensor(1.0)}}
    runtime.acquire_gpu(weights)

    assert loaded == [(weights, "cpu")]
    assert control.actor_available is True
    assert control.use_actor is False


def test_task_hotkey_uses_rerecord_home_semantics_without_changing_action_mode():
    control = _control()
    control.actor_available = True
    control.use_actor = True
    controller = ActorKeyboardController(control)
    state = KeyboardState()

    controller.push("1")
    controller.poll(state)

    assert control.cfg.env.task == "Pick up the cup on the right"
    assert control.use_actor is True
    assert state.reset_episode and state.rerecord_episode and state.exit_episode
    assert control.runtime.reset_count == 1
    assert control.smoother.reset_count == 1


def test_task_config_and_actor_checkpoint_discovery(tmp_path):
    task_path = tmp_path / "tasks.json"
    task_path.write_text('{"default_key":"1","tasks":{"1":"Pick up the cup on the right"}}')
    assert load_task_hotkeys(task_path).default_key == "1"

    actor_file = tmp_path / "checkpoints/last/actor_critic.pt"
    actor_file.parent.mkdir(parents=True)
    actor_file.touch()
    assert _find_actor_checkpoint(tmp_path) == actor_file


def test_unhandled_exception_traceback_is_written():
    with TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "actor_2.log"
        init_logging(log_file=log_file)
        try:
            raise RuntimeError("actor_2 logging check")
        except RuntimeError:
            _log_unhandled_exception("main thread", *sys.exc_info())

        text = log_file.read_text()
        assert "Traceback (most recent call last)" in text
        assert "RuntimeError: actor_2 logging check" in text
        logging.shutdown()
        logging.getLogger().handlers.clear()


def test_run_actor_does_not_revalidate_after_cli_creates_log_dir(tmp_path, monkeypatch):
    output_dir = tmp_path / "actor_output"
    validate_calls = 0

    def validate():
        nonlocal validate_calls
        validate_calls += 1
        if output_dir.is_dir():
            raise FileExistsError(f"Output directory {output_dir} already exists")

    cfg = SimpleNamespace(
        actor_mode="online_actor",
        save_format="transition",
        actor_only=SimpleNamespace(enabled=False, save_format="transition"),
        online_transition=SimpleNamespace(enabled=True),
        actor_checkpoint_path=None,
        output_dir=output_dir,
        algorithm=SimpleNamespace(
            actor_learner_config=SimpleNamespace(learner_host="127.0.0.1", learner_port=50051),
        ),
        validate=validate,
    )

    actor_new._normalize_actor_config(cfg)
    (output_dir / "logs").mkdir(parents=True)
    monkeypatch.setattr(
        actor_new,
        "learner_service_client",
        lambda **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(actor_new, "establish_learner_connection", lambda *_args: False)

    with pytest.raises(ConnectionError, match="Failed to establish connection with learner"):
        actor_new.run_actor_online(cfg, shutdown_event=object(), actor_control=object())

    assert validate_calls == 1


def test_actor_lerobot_writer_uses_canonical_legacy_fields(tmp_path):
    frames = []
    saved_metadata = []
    dataset = SimpleNamespace(
        features={"observation.state": {"dtype": "float32"}},
        root=tmp_path / "actor_dataset",
        add_frame=frames.append,
        save_episode=lambda episode_metadata: saved_metadata.append(episode_metadata),
    )
    writer = ActorEpisodeWriter.__new__(ActorEpisodeWriter)
    writer._ensure_lerobot_dataset = lambda _transition: dataset
    transition = {
        "state": {"observation.state": torch.tensor([[1.0]])},
        "action": torch.tensor([[0.8]]),
        "reward": 1.0,
        "next_state": {"observation.state": torch.tensor([[2.0]])},
        "done": True,
        "truncated": False,
        "complementary_info": {
            "is_intervention": True,
            "policy_action": torch.tensor([[0.3]]),
        },
    }

    result = writer._save_lerobot_episode(
        [transition],
        {
            "episode_outcome": "success",
            "task": "pick",
            "actor_policy_path": "/models/pi05-checkpoint",
        },
    )

    assert result == dataset.root
    torch.testing.assert_close(frames[0][EVORL_POLICY_ACTION_FIELD], torch.tensor([0.3]))
    assert frames[0][EVORL_INTERVENTION_FIELD].tolist() == [1.0]
    assert frames[0][EVORL_STATE_FIELD].tolist() == [1.0]
    assert frames[0][EVORL_COLLECTOR_POLICY_ID_FIELD] == "human"
    assert saved_metadata == [{"episode_success": "success"}]


def test_actor_episode_writers_fan_out_the_same_episode(tmp_path):
    calls = []

    class FakeWriter:
        def __init__(self, save_format):
            self.save_format = save_format

        def configure_robot(self, robot):
            calls.append((self.save_format, "configure", robot))

        def save_episode(self, *, transitions, metadata, compact_episode):
            calls.append((self.save_format, "save", transitions, metadata, compact_episode))
            return tmp_path / self.save_format

        def finalize(self):
            calls.append((self.save_format, "finalize"))

    group = ActorEpisodeWriters([FakeWriter("transition"), FakeWriter("lerobot")])
    robot = object()
    transitions = [{"frame": 1}]
    metadata = {"episode_outcome": "success"}
    compact = {"schema": "evorl_compact_transition_v1"}

    group.configure_robot(robot)
    paths = group.save_episode(transitions=transitions, metadata=metadata, compact_episode=compact)
    group.finalize()

    assert paths == {"transition": tmp_path / "transition", "lerobot": tmp_path / "lerobot"}
    assert [call[:2] for call in calls] == [
        ("transition", "configure"),
        ("lerobot", "configure"),
        ("transition", "save"),
        ("lerobot", "save"),
        ("transition", "finalize"),
        ("lerobot", "finalize"),
    ]
