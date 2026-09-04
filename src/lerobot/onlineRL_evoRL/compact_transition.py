"""Versioned compact episode payload shared by the actor and PI05 learner."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

SCHEMA_NAME = "pi05_compact_episode"
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, SCHEMA_VERSION)
FEATURE_KEYS = ("z_rl", "proprio", "ref_action")
PRIMITIVE_TRANSITIONS = "primitive"
SLIDING_WINDOW_TRANSITIONS = "sliding_windows"


def make_compact_episode(
    *,
    transitions: list[dict],
    metadata: dict[str, Any],
    feature_model: dict[str, Any],
    transition_layout: str = PRIMITIVE_TRANSITIONS,
    sliding_window_stride: int = 1,
    primitive_steps: int | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "feature_model": feature_model,
        "transition_layout": transition_layout,
        "sliding_window_stride": sliding_window_stride,
        "primitive_steps": primitive_steps,
        "transitions": transitions,
    }
    validate_compact_episode(payload)
    return payload


def is_compact_episode(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("schema") == SCHEMA_NAME


def validate_compact_episode(payload: Any) -> dict[str, Any]:
    if not is_compact_episode(payload):
        raise ValueError(f"Expected compact episode schema {SCHEMA_NAME!r}")
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported compact episode schema version: {payload.get('schema_version')!r}")
    transition_layout = payload.get("transition_layout", PRIMITIVE_TRANSITIONS)
    if transition_layout not in {PRIMITIVE_TRANSITIONS, SLIDING_WINDOW_TRANSITIONS}:
        raise ValueError(f"Unsupported compact transition layout: {transition_layout!r}")
    sliding_window_stride = payload.get("sliding_window_stride", 1)
    if not isinstance(sliding_window_stride, int) or sliding_window_stride <= 0:
        raise ValueError("Compact episode sliding_window_stride must be a positive integer")
    primitive_steps = payload.get("primitive_steps")
    if primitive_steps is not None and (not isinstance(primitive_steps, int) or primitive_steps <= 0):
        raise ValueError("Compact episode primitive_steps must be a positive integer when provided")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("task"):
        raise ValueError("Compact episode metadata must contain a non-empty task")
    feature_model = payload.get("feature_model")
    if not isinstance(feature_model, dict) or not feature_model.get("resolved_path"):
        raise ValueError("Compact episode must identify the feature model resolved_path")
    transitions = payload.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("Compact episode transitions must be a non-empty list")
    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            raise ValueError(f"Compact transition {index} must be a mapping")
        for state_name in ("state", "next_state"):
            state = transition.get(state_name)
            if not isinstance(state, dict):
                raise ValueError(f"Compact transition {index} is missing {state_name}")
            missing = [key for key in FEATURE_KEYS if not isinstance(state.get(key), torch.Tensor)]
            if missing:
                raise ValueError(
                    f"Compact transition {index} {state_name} is missing tensor features {missing}"
                )
        for key in ("action", "reward", "done", "truncated"):
            if key not in transition:
                raise ValueError(f"Compact transition {index} is missing {key}")
        if transition_layout == SLIDING_WINDOW_TRANSITIONS:
            missing = [
                key
                for key in (
                    "target_action_chunk",
                    "intervene_flags",
                    "valid_action_mask",
                    "next_valid_action_mask",
                )
                if not isinstance(transition.get(key), torch.Tensor)
            ]
            if missing:
                raise ValueError(
                    f"Compact sliding-window transition {index} is missing tensor fields {missing}"
                )
    return payload


def compact_episode_to_bytes(payload: dict[str, Any]) -> bytes:
    validate_compact_episode(payload)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def bytes_to_episode_payload(buffer: bytes) -> Any:
    return torch.load(io.BytesIO(buffer), weights_only=True)


def save_compact_episode(payload: dict[str, Any], path: Path) -> None:
    validate_compact_episode(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
