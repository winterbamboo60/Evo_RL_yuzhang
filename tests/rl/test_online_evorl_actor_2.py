import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from lerobot.onlineRL_evoRL.actor_2 import (
    Actor2Control,
    Actor2KeyboardController,
    TaskHotkeys,
    _find_actor_checkpoint,
    _log_unhandled_exception,
    load_task_hotkeys,
)
from lerobot.onlineRL_evoRL.keyboard_control import KeyboardState
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
    control = Actor2Control(cfg=cfg, task_hotkeys=tasks, dataset_meta=SimpleNamespace())
    control.runtime = _Resettable()
    control.smoother = _Resettable()
    return control


def test_b_switch_requires_actor_and_clears_cached_actions():
    control = _control()
    controller = Actor2KeyboardController(control)
    state = KeyboardState()

    controller.push("b")
    controller.poll(state)
    assert control.use_actor is False

    control.actor_available = True
    controller._last_actor2_event_t = 0.0
    controller.push("B")
    controller.poll(state)
    assert control.use_actor is True
    assert control.runtime.reset_count == 1
    assert control.smoother.reset_count == 1

    controller._last_actor2_event_t = 0.0
    controller.push("b")
    controller.poll(state)
    assert control.use_actor is False


def test_task_hotkey_uses_rerecord_home_semantics_without_changing_action_mode():
    control = _control()
    control.actor_available = True
    control.use_actor = True
    controller = Actor2KeyboardController(control)
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
    task_path.write_text(
        '{"default_key":"1","tasks":{"1":"Pick up the cup on the right"}}'
    )
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
