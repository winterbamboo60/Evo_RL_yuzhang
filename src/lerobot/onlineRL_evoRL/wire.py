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


@dataclass(frozen=True)
class DecodedTransitionPayload:
    kind: Literal["transitions", "compact_episode"]
    value: Any


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
