from queue import Queue

import torch

from lerobot.onlineRL_evoRL.chunk_transition import build_sliding_window_transitions
from lerobot.onlineRL_evoRL.compact_transition import (
    SCHEMA_VERSION,
    compact_episode_to_bytes,
    make_compact_episode,
)
from lerobot.onlineRL_evoRL.keyboard_control import KeyboardController, KeyboardState
from lerobot.onlineRL_evoRL.wire import decode_transition_payload, enqueue_decoded_payload
from lerobot.transport.utils import transitions_to_bytes


def _state(value: float):
    return {
        "z_rl": torch.tensor([[value, value]]),
        "proprio": torch.tensor([[value]]),
        "ref_action": torch.tensor([[[value], [value]]]),
    }


def test_compact_episode_uses_existing_transition_wire_without_conversion():
    primitive = [
        {
            "state": _state(0),
            "next_state": _state(1),
            "action": torch.tensor([[0.5]]),
            "reward": 1.0,
            "done": True,
            "truncated": False,
            "complementary_info": {"is_intervention": True},
        }
    ]
    compact = make_compact_episode(
        transitions=build_sliding_window_transitions(primitive, horizon=2),
        metadata={"task": "test"},
        feature_model={"resolved_path": "/tmp/policy"},
        transition_layout="sliding_windows",
        primitive_steps=1,
    )

    decoded = decode_transition_payload(compact_episode_to_bytes(compact))
    assert decoded.kind == "compact_episode"
    assert decoded.value["schema_version"] == SCHEMA_VERSION
    assert decoded.value["transitions"][0]["intervene_flags"].tolist() == [True, False]


def test_ordinary_transition_payload_stays_ordinary():
    transitions = [{"state": {}, "action": torch.zeros(1)}]
    decoded = decode_transition_payload(transitions_to_bytes(transitions))
    assert decoded.kind == "transitions"
    assert len(decoded.value) == 1


def test_payload_router_does_not_reencode_wire_data():
    transition_queue, compact_queue = Queue(), Queue()
    payload = transitions_to_bytes([{"state": {}, "action": torch.zeros(1)}])
    assert enqueue_decoded_payload(
        payload,
        transition_queue=transition_queue,
        compact_episode_queue=compact_queue,
    ) == "transitions"
    assert transition_queue.get()[0]["action"].shape == (1,)
    assert compact_queue.empty()


def test_episode_control_hotkeys_are_preserved():
    controller = KeyboardController()
    state = KeyboardState()
    for key in ("c", "b"):
        controller.push(key)
    state = controller.poll(state)
    assert state.toggle_intervention
    assert state.episode_outcome == "success"
    assert state.exit_episode


def test_abandon_and_failure_use_a_and_f():
    controller = KeyboardController()
    abandoned = controller.poll(KeyboardState())
    controller.push("a")
    abandoned = controller.poll(abandoned)
    assert abandoned.rerecord_episode and abandoned.exit_episode
    controller.push("f")
    failed = controller.poll(KeyboardState())
    assert failed.episode_outcome == "failure" and failed.exit_episode
