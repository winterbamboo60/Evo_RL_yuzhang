"""Payload adapter above the unchanged LeRobot gRPC byte transport.

Both ordinary transition lists and EvoRL compact episodes are serialized with
``torch.save`` and carried by the same ``Transition`` chunk stream.  This module
only identifies the decoded object; it does not alter protobuf messages,
chunking, RPC methods or bytes on the wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from lerobot.transport.utils import bytes_to_transitions

from .compact_transition import is_compact_episode, validate_compact_episode

CONTROL_FIELD = "_evorl_control"
ACTOR_GPU_RELEASED = "actor_gpu_released"
HANDOFF_ID_FIELD = "handoff_id"
WEIGHT_HANDOFF_ID_FIELD = "_evorl_handoff_id"
WEIGHT_UPDATE_STEP_FIELD = "_evorl_update_step"


@dataclass(frozen=True)
class DecodedTransitionPayload:
    """A validated payload routed to the matching learner queue."""

    kind: Literal["transitions", "compact_episode"]
    value: Any


def make_control_message(kind: str, *, handoff_id: int, **values: Any) -> dict[str, Any]:
    """Build a versioned control message carried by the interaction stream."""
    if handoff_id <= 0:
        raise ValueError("handoff_id must be positive")
    return {
        CONTROL_FIELD: kind,
        "schema_version": 1,
        HANDOFF_ID_FIELD: handoff_id,
        **values,
    }


def parse_control_message(value: Any) -> tuple[str, int] | None:
    """Return ``(kind, handoff_id)`` for an EvoRL control message."""
    if not isinstance(value, dict) or CONTROL_FIELD not in value:
        return None
    kind = value.get(CONTROL_FIELD)
    handoff_id = value.get(HANDOFF_ID_FIELD)
    if not isinstance(kind, str) or not isinstance(handoff_id, int) or handoff_id <= 0:
        raise ValueError("Invalid EvoRL control message")
    return kind, handoff_id


def decode_transition_payload(buffer: bytes) -> DecodedTransitionPayload:
    """Decode an existing transition-stream payload without changing its format."""
    value = bytes_to_transitions(buffer)
    if is_compact_episode(value):
        return DecodedTransitionPayload("compact_episode", validate_compact_episode(value))
    if not isinstance(value, list):
        raise ValueError(
            "Transition payload must contain a transition list or an EvoRL compact episode"
        )
    return DecodedTransitionPayload("transitions", value)


def enqueue_decoded_payload(buffer: bytes, *, transition_queue, compact_episode_queue) -> str:
    """Route a received payload after LearnerService has delivered its bytes."""
    decoded = decode_transition_payload(buffer)
    queue = compact_episode_queue if decoded.kind == "compact_episode" else transition_queue
    queue.put(decoded.value)
    return decoded.kind
