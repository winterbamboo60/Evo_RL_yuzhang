#!/usr/bin/env python

from types import SimpleNamespace

import pytest

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.value_train import ValueTrainPipelineConfig
from lerobot.utils.recording_annotations import (
    EPISODE_FAILURE,
    EPISODE_SUCCESS,
    infer_collector_policy_id,
    infer_collector_policy_version,
    normalize_episode_success_label,
    resolve_collector_policy_id,
    resolve_episode_success_from_mapping,
    resolve_episode_success_label,
    resolve_intervention_value,
)


def test_value_train_uses_standard_train_config_schema():
    assert issubclass(ValueTrainPipelineConfig, TrainPipelineConfig)


def test_episode_success_aliases_cover_legacy_and_current_recordings():
    assert resolve_episode_success_from_mapping({"episode_success": "success"}) == "success"
    assert resolve_episode_success_from_mapping({"success": "failure"}) == "failure"
    assert resolve_episode_success_from_mapping({"episode_outcome": "success"}) == "success"
    assert resolve_episode_success_from_mapping({"success": True}) == "success"
    assert resolve_episode_success_from_mapping({"success": 0}) == "failure"
    assert resolve_episode_success_from_mapping({"episode_success": [True]}) == "success"
    assert (
        resolve_episode_success_from_mapping(
            {"outcome": "aborted"}, default_label="failure", require_label=True
        )
        == "failure"
    )


def test_intervention_aliases_cover_legacy_and_current_recordings():
    assert resolve_intervention_value({"intervention": True})
    assert resolve_intervention_value({"complementary_info.is_intervention": 1.0})
    assert resolve_intervention_value({"is_intervention": "true"})
    assert not resolve_intervention_value({"is_intervention": "false"})
    assert not resolve_intervention_value({"complementary_info.is_intervention": [0.0]})
    assert resolve_intervention_value({"complementary_info.is_intervention": [1.0]})
    assert not resolve_intervention_value({})


def test_legacy_recording_annotation_api_is_preserved():
    assert normalize_episode_success_label("SUCCESS") == EPISODE_SUCCESS
    assert normalize_episode_success_label("failure") == EPISODE_FAILURE
    assert normalize_episode_success_label(None) is None
    with pytest.raises(ValueError):
        normalize_episode_success_label("maybe")

    assert resolve_episode_success_label("success", default_label="failure") == EPISODE_SUCCESS
    assert resolve_episode_success_label(None, default_label="failure") == EPISODE_FAILURE
    assert resolve_episode_success_label(None, require_label=False) is None
    with pytest.raises(ValueError):
        resolve_episode_success_label(None, require_label=True)

    policy_cfg = SimpleNamespace(pretrained_path="org/act_v1", type="act")
    assert infer_collector_policy_id(policy_cfg) == "org/act_v1"
    assert infer_collector_policy_version(policy_cfg) == "act_v1"
    assert infer_collector_policy_id(None) == "human"
    assert infer_collector_policy_version(None) == "human"

    assert resolve_collector_policy_id(
        intervention_enabled=True,
        is_intervention=True,
        selected_from_policy=True,
        policy_id="act_v1",
        human_id="human",
    ) == "human"
    assert resolve_collector_policy_id(
        intervention_enabled=True,
        is_intervention=False,
        selected_from_policy=True,
        policy_id="act_v1",
        human_id="human",
    ) == "act_v1"
    assert resolve_collector_policy_id(
        intervention_enabled=False,
        is_intervention=False,
        selected_from_policy=False,
        policy_id="act_v1",
        human_id="human",
    ) == "human"
