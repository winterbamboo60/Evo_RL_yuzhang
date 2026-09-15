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

"""Helpers for recording-time annotations and per-step policy-source tracing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

EPISODE_SUCCESS = "success"
EPISODE_FAILURE = "failure"
VALID_EPISODE_SUCCESS_LABELS = {EPISODE_SUCCESS, EPISODE_FAILURE}

# Shared recording controls. Both pure-manual recording and policy-assisted
# EvoRL recording import these defaults so the two modes cannot silently drift.
EVORL_SUCCESS_KEY = "b"
EVORL_FAILURE_KEY = "f"
EVORL_INTERVENTION_KEY = "c"
EVORL_RERECORD_KEY = "a"
EVORL_RESET_KEY = "r"

# Canonical EvoRL dataset annotations. These names and dtypes intentionally
# match the 0901 human-in-the-loop datasets so pure-human, policy-assisted and
# online-RL recordings can be merged without rewriting their frame schema.
EVORL_POLICY_ACTION_FIELD = "complementary_info.policy_action"
EVORL_INTERVENTION_FIELD = "complementary_info.is_intervention"
EVORL_STATE_FIELD = "complementary_info.state"
EVORL_COLLECTOR_POLICY_ID_FIELD = "complementary_info.collector_policy_id"

EVORL_STATE_POLICY = 0.0
EVORL_STATE_ACTIVE = 1.0
EVORL_STATE_RELEASE = 2.0

EPISODE_SUCCESS_FIELD_ALIASES = ("episode_success", "success", "episode_outcome", "outcome")
INTERVENTION_FIELD_ALIASES = (EVORL_INTERVENTION_FIELD, "intervention", "is_intervention")
ACP_INDICATOR_FIELD_ALIASES = (
    "complementary_info.acp_indicator",
    "complementary_info.indicator_field",
    "acp_indicator",
)


def build_evorl_dataset_features(action_feature: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the merge-stable 0901 EvoRL annotation schema for one action feature."""
    action_shape = tuple(action_feature["shape"])
    action_names = action_feature.get("names")
    if action_names is not None:
        action_names = list(action_names)

    return {
        EVORL_POLICY_ACTION_FIELD: {
            "dtype": "float32",
            "shape": action_shape,
            "names": action_names,
        },
        EVORL_INTERVENTION_FIELD: {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_intervention"],
        },
        EVORL_STATE_FIELD: {
            "dtype": "float32",
            "shape": (1,),
            "names": ["state"],
        },
        EVORL_COLLECTOR_POLICY_ID_FIELD: {
            "dtype": "string",
            "shape": (1,),
            "names": ["collector_policy_id"],
        },
    }


def normalize_episode_success_label(label: str | None) -> str | None:
    """Normalize a user-provided episode label to canonical lowercase values."""
    if label is None:
        return None
    normalized = label.strip().lower()
    if normalized not in VALID_EPISODE_SUCCESS_LABELS:
        raise ValueError(
            f"`episode_success` must be one of {sorted(VALID_EPISODE_SUCCESS_LABELS)}, got '{label}'."
        )
    return normalized


def resolve_episode_success_label(
    explicit_label: str | None,
    default_label: str | None = None,
    require_label: bool = False,
) -> str | None:
    """Resolve the final episode-success label from explicit and default values."""
    explicit = normalize_episode_success_label(explicit_label)
    if explicit is not None:
        return explicit

    default = normalize_episode_success_label(default_label)
    if default is not None:
        return default

    if require_label:
        raise ValueError(
            "Missing `episode_success` label. Use success/failure hotkeys or set `default_episode_success`."
        )
    return None


def infer_collector_policy_id(policy_cfg: Any | None) -> str:
    """Infer a stable policy identifier for frame-level provenance."""
    if policy_cfg is None:
        return "human"

    pretrained_path = getattr(policy_cfg, "pretrained_path", None)
    if pretrained_path:
        return str(pretrained_path)

    policy_type = getattr(policy_cfg, "type", None)
    if policy_type:
        return str(policy_type)

    return "policy"


def infer_collector_policy_version(policy_cfg: Any | None) -> str:
    """Infer a compact policy-version string suitable for metadata columns."""
    if policy_cfg is None:
        return "human"

    pretrained_path = getattr(policy_cfg, "pretrained_path", None)
    if pretrained_path:
        return Path(str(pretrained_path)).name

    policy_type = getattr(policy_cfg, "type", None)
    if policy_type:
        return str(policy_type)

    return "policy"


def resolve_collector_policy_id(
    *,
    intervention_enabled: bool,
    is_intervention: bool,
    selected_from_policy: bool,
    policy_id: str,
    human_id: str,
) -> str:
    """Resolve frame-level `collector_policy_id` from control mode and source."""
    if intervention_enabled:
        return human_id if is_intervention else policy_id
    return policy_id if selected_from_policy else human_id


def _unwrap_scalar_annotation(value: Any) -> Any:
    """Unwrap scalar tensors/arrays and legacy one-element list columns."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_scalar_annotation(value[0])
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            converted = value.tolist()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _unwrap_scalar_annotation(converted)
    return value


def resolve_episode_success_from_mapping(
    values: dict[str, Any],
    *,
    preferred_field: str = "episode_success",
    default_label: str | None = None,
    require_label: bool = False,
) -> str | None:
    """Resolve success/failure from both 0901 and current episode schemas."""
    fields = (preferred_field, *EPISODE_SUCCESS_FIELD_ALIASES)
    for field in dict.fromkeys(fields):
        value = values.get(field)
        if value is None:
            continue
        # New recording metadata may call the field ``outcome`` and include
        # aborted/reset values. Those are not silently treated as success.
        value = _unwrap_scalar_annotation(value)
        if isinstance(value, bool):
            return EPISODE_SUCCESS if value else EPISODE_FAILURE
        if isinstance(value, (int, float)) and value in (0, 1):
            return EPISODE_SUCCESS if bool(value) else EPISODE_FAILURE
        normalized = str(value).strip().lower()
        if normalized in VALID_EPISODE_SUCCESS_LABELS:
            return normalized
        if normalized in {"true", "yes", "1"}:
            return EPISODE_SUCCESS
        if normalized in {"false", "no", "0", "failed", "fail"}:
            return EPISODE_FAILURE
    return resolve_episode_success_label(None, default_label=default_label, require_label=require_label)


def resolve_intervention_value(values: dict[str, Any], default: bool = False) -> bool:
    """Read intervention state from the current or legacy frame schema."""
    for field in INTERVENTION_FIELD_ALIASES:
        if field in values and values[field] is not None:
            value = values[field]
            value = _unwrap_scalar_annotation(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "yes", "1", "intervention", "human"}:
                    return True
                if normalized in {"false", "no", "0", "policy", "none", ""}:
                    return False
            return bool(value)
    return bool(default)
