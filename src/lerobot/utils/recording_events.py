#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config-driven per-episode quality events (e.g. ``bad_depth`` / ``bad_edge``).

A quality event marks a *single frame* (``step``) of an episode as exhibiting a
problem the human operator noticed while recording. Each event has the shape::

    {"step": 11, "event": "bad_depth"}

Which events exist and which keyboard key triggers which events is **not**
hardcoded: it is described by an external JSON config file (see
:class:`EventConfig`). A single key may map to several events (e.g. ``"a"`` ->
``["bad_depth", "bad_edge"]``), in which case each event is recorded separately.

Two complementary representations are produced for each episode:

* **Per-frame flags** written into the dataset through ``add_frame`` as
  ``complementary_info.<event>`` features (one ``float32`` flag per frame, per
  event). This keeps the events aligned with the rest of the trajectory.
* **An episode-level event list** (``[{"step", "event"}, ...]``) serialized into
  episode metadata for convenient inspection.

Example config JSON::

    {
        "events": ["bad_depth", "bad_edge"],
        "key_bindings": {
            "d": ["bad_depth"],
            "e": ["bad_edge"],
            "a": ["bad_depth", "bad_edge"]
        }
    }
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Per-frame dataset features live under this prefix; the keyboard marker flags
# shared through the `events` dict live under the other prefix.
FEATURE_PREFIX = "complementary_info."
MARKER_PREFIX = "mark_"


def event_feature_key(event: str) -> str:
    """Return the per-frame dataset feature key for an event (e.g. ``complementary_info.bad_depth``)."""
    return f"{FEATURE_PREFIX}{event}"


def event_marker_key(event: str) -> str:
    """Return the shared `events`-dict marker key for an event (e.g. ``mark_bad_depth``)."""
    return f"{MARKER_PREFIX}{event}"


@dataclass
class EventConfig:
    """Declarative configuration of the recordable quality events and their hotkeys.

    Attributes:
        events: All event names that are listened for, in declaration order.
        key_bindings: Map ``{key_char: [event_names]}``. A key may map to several
            events; pressing it records each of them on the current frame.
    """

    events: list[str]
    key_bindings: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize + validate the event list.
        normalized_events: list[str] = []
        for event in self.events:
            name = str(event).strip().lower()
            if not name:
                raise ValueError("Event names must be non-empty strings.")
            if name in normalized_events:
                raise ValueError(f"Duplicate event name in config: '{name}'.")
            normalized_events.append(name)
        if not normalized_events:
            raise ValueError("`events` must contain at least one event name.")
        self.events = normalized_events
        event_set = set(self.events)

        # Normalize + validate the key bindings.
        normalized_bindings: dict[str, list[str]] = {}
        for key, bound_events in self.key_bindings.items():
            key_char = str(key).strip().lower()
            if len(key_char) != 1:
                raise ValueError(f"Each key binding must be a single character, got '{key}'.")
            if key_char in normalized_bindings:
                raise ValueError(f"Duplicate key binding for '{key_char}'.")
            if isinstance(bound_events, str):
                bound_events = [bound_events]
            resolved: list[str] = []
            for event in bound_events:
                name = str(event).strip().lower()
                if name not in event_set:
                    raise ValueError(
                        f"Key '{key_char}' maps to unknown event '{name}'. Known events: {self.events}."
                    )
                if name not in resolved:
                    resolved.append(name)
            if not resolved:
                raise ValueError(f"Key '{key_char}' must map to at least one event.")
            normalized_bindings[key_char] = resolved
        self.key_bindings = normalized_bindings

    # ----- constructors ---------------------------------------------------

    @classmethod
    def from_dict(cls, payload: dict) -> EventConfig:
        if not isinstance(payload, dict):
            raise ValueError(f"Event config must be a JSON object, got {type(payload)}.")
        if "events" not in payload:
            raise ValueError("Event config must contain an `events` list.")
        return cls(
            events=list(payload["events"]),
            key_bindings=dict(payload.get("key_bindings", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> EventConfig:
        """Load an :class:`EventConfig` from a JSON file."""
        path = Path(path)
        with open(path) as f:
            payload = json.load(f)
        try:
            return cls.from_dict(payload)
        except ValueError as error:
            raise ValueError(f"Invalid event config at {path}: {error}") from error

    # ----- serialization --------------------------------------------------

    def to_dict(self) -> dict:
        return {"events": list(self.events), "key_bindings": {k: list(v) for k, v in self.key_bindings.items()}}

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    # ----- derived maps ---------------------------------------------------

    @property
    def feature_keys(self) -> dict[str, str]:
        """``{event_name: dataset_feature_key}``."""
        return {event: event_feature_key(event) for event in self.events}

    @property
    def marker_keys(self) -> dict[str, str]:
        """``{event_name: events_dict_marker_key}``."""
        return {event: event_marker_key(event) for event in self.events}

    # ----- helpers used during recording ----------------------------------

    def normalize_event(self, event: str) -> str:
        name = str(event).strip().lower()
        if name not in self.events:
            raise ValueError(f"`event` must be one of {self.events}, got '{event}'.")
        return name

    def build_features(self) -> dict[str, dict]:
        """Return the per-frame dataset feature schema for all configured events."""
        return {
            event_feature_key(event): {"dtype": "float32", "shape": (1,), "names": [event]}
            for event in self.events
        }

    def flags(self, active_events: Iterable[str]) -> dict[str, np.ndarray]:
        """Per-frame feature flags for a frame's active events.

        Every event feature is always present (defaulting to ``0.0``) so frames
        stay schema-consistent for ``add_frame`` validation; active events are
        set to ``1.0``.
        """
        active = {self.normalize_event(e) for e in active_events}
        return {
            event_feature_key(event): np.array([1.0 if event in active else 0.0], dtype=np.float32)
            for event in self.events
        }

    def index_events_by_step(self, events: Sequence[dict] | None) -> dict[int, set[str]]:
        """Group an event list into ``{step: {event_names}}`` for per-frame lookup."""
        by_step: dict[int, set[str]] = {}
        for event in events or []:
            step = int(event["step"])
            name = self.normalize_event(event["event"])
            by_step.setdefault(step, set()).add(name)
        return by_step

    def serialize(self, events: Sequence[dict] | None) -> str:
        """Serialize an episode's events to a compact, parquet-safe JSON string."""
        normalized = [
            {"step": int(e["step"]), "event": self.normalize_event(e["event"])} for e in (events or [])
        ]
        normalized.sort(key=lambda e: (e["step"], e["event"]))
        return json.dumps(normalized)
