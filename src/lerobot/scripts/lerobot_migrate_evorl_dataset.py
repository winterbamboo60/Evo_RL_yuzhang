#!/usr/bin/env python

"""Non-destructively normalize EvoRL datasets or export the historical bool schema."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, DEFAULT_FEATURES
from lerobot.utils.recording_annotations import (
    EVORL_COLLECTOR_POLICY_ID_FIELD,
    EVORL_INTERVENTION_FIELD,
    EVORL_POLICY_ACTION_FIELD,
    EVORL_STATE_ACTIVE,
    EVORL_STATE_FIELD,
    EVORL_STATE_POLICY,
    build_evorl_dataset_features,
    resolve_episode_success_from_mapping,
    resolve_intervention_value,
)
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

_OLD_POLICY_ACTION = EVORL_POLICY_ACTION_FIELD
_OLD_INTERVENTION = EVORL_INTERVENTION_FIELD
_OLD_STATE = EVORL_STATE_FIELD
_OLD_COLLECTOR_POLICY_ID = EVORL_COLLECTOR_POLICY_ID_FIELD
_CURRENT_INTERVENTION = "intervention"
_CURRENT_POLICY_ACTION = "policy_action"
_CURRENT_COLLECTOR_POLICY_ID = "collector_policy_id"
_CURRENT_EPISODE_OUTCOME = "episode_outcome"
_EPISODE_RESERVED = {"episode_index", "tasks", "length"}
_EPISODE_RESERVED_PREFIXES = ("stats/", "meta/", "data/", "videos/", "dataset_")


def _load_source(root: Path) -> LeRobotDataset:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: missing {info_path}")
    import json

    info = json.loads(info_path.read_text())
    repo_id = str(info.get("repo_id") or f"local/{root.name}")
    return LeRobotDataset(repo_id=repo_id, root=root)


def _destination_features(source: LeRobotDataset, direction: str) -> dict[str, dict]:
    features = {
        key: dict(value) for key, value in source.meta.features.items() if key not in DEFAULT_FEATURES
    }
    if direction == "to-current":
        for key in (
            _OLD_POLICY_ACTION,
            _OLD_INTERVENTION,
            _OLD_STATE,
            _OLD_COLLECTOR_POLICY_ID,
            _CURRENT_POLICY_ACTION,
            _CURRENT_COLLECTOR_POLICY_ID,
            _CURRENT_EPISODE_OUTCOME,
        ):
            features.pop(key, None)
        features[_CURRENT_INTERVENTION] = {"dtype": "bool", "shape": (1,), "names": None}
    elif direction in {"to-0901", "to-canonical"}:
        for key in (
            _CURRENT_INTERVENTION,
            _CURRENT_POLICY_ACTION,
            _CURRENT_COLLECTOR_POLICY_ID,
            _CURRENT_EPISODE_OUTCOME,
        ):
            features.pop(key, None)
        features.update(build_evorl_dataset_features(features[ACTION]))
    else:
        raise ValueError(f"Unsupported migration direction: {direction}")
    return features


def _scalar_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    elif isinstance(value, (list, tuple)) and len(value) == 1:
        return _scalar_value(value[0])
    return value


def _frame_value(value: Any, feature: dict) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if feature["dtype"] in {"image", "video"} and value.ndim == 3:
            if value.shape[0] in (1, 3, 4):
                value = value.permute(1, 2, 0)
            value = value.numpy()
            if np.issubdtype(value.dtype, np.floating) and value.max(initial=0) <= 1.0:
                value = np.rint(value * 255).astype(np.uint8)
            return value
        return value.numpy()
    return value


def _convert_frame(
    item: dict,
    features: dict[str, dict],
    direction: str,
    task: str,
    collector_policy_id: str = "policy",
) -> dict:
    frame: dict[str, Any] = {}
    intervention = resolve_intervention_value(item)
    has_intervention_annotation = any(key in item for key in (_OLD_INTERVENTION, _CURRENT_INTERVENTION))

    for key, feature in features.items():
        if key == _CURRENT_INTERVENTION:
            frame[key] = np.array([intervention], dtype=bool)
        elif key == _OLD_POLICY_ACTION:
            if _OLD_POLICY_ACTION in item:
                frame[key] = _frame_value(item[_OLD_POLICY_ACTION], feature)
            elif _CURRENT_POLICY_ACTION in item:
                frame[key] = _frame_value(item[_CURRENT_POLICY_ACTION], feature)
            elif not has_intervention_annotation or intervention:
                frame[key] = np.zeros(tuple(feature["shape"]), dtype=np.float32)
            else:
                frame[key] = _frame_value(item[ACTION], feature)
        elif key == _OLD_INTERVENTION:
            frame[key] = np.array([float(intervention)], dtype=np.float32)
        elif key == _OLD_STATE:
            if _OLD_STATE in item:
                state = float(_scalar_value(item[_OLD_STATE]))
            else:
                state = EVORL_STATE_ACTIVE if intervention else EVORL_STATE_POLICY
            frame[key] = np.array([state], dtype=np.float32)
        elif key == _OLD_COLLECTOR_POLICY_ID:
            if _OLD_COLLECTOR_POLICY_ID in item:
                collector = _scalar_value(item[_OLD_COLLECTOR_POLICY_ID])
            elif _CURRENT_COLLECTOR_POLICY_ID in item:
                collector = _scalar_value(item[_CURRENT_COLLECTOR_POLICY_ID])
            elif not has_intervention_annotation:
                collector = "human"
            else:
                collector = "human" if intervention else collector_policy_id
            frame[key] = str(collector)
        elif key in item:
            frame[key] = _frame_value(item[key], feature)
        else:
            raise KeyError(f"Source frame is missing required feature `{key}` for {direction} export.")
    frame["task"] = str(item.get("task", task))
    return frame


def _user_episode_metadata(episode: dict[str, Any]) -> dict[str, str | bool | int | float | None]:
    metadata: dict[str, str | bool | int | float | None] = {}
    for key, value in episode.items():
        if key in _EPISODE_RESERVED or key.startswith(_EPISODE_RESERVED_PREFIXES):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, bool, int, float)):
            metadata[key] = value
    outcome = resolve_episode_success_from_mapping(episode)
    if outcome is not None:
        metadata["episode_success"] = outcome
        for alias in ("success", "episode_outcome", "outcome"):
            metadata.pop(alias, None)
    return metadata


def migrate_dataset(
    source_root: Path,
    destination_root: Path,
    destination_repo_id: str,
    direction: str,
    collector_policy_id: str = "policy",
) -> None:
    """Copy a dataset into a fresh destination; never changes the source tree."""
    source_root = source_root.expanduser().resolve()
    destination_root = destination_root.expanduser().resolve()
    if source_root == destination_root:
        raise ValueError("Source and destination must be different directories.")
    if destination_root.exists():
        raise FileExistsError(f"Destination already exists; refusing to overwrite: {destination_root}")

    source = _load_source(source_root)
    features = _destination_features(source, direction)
    destination = LeRobotDataset.create(
        repo_id=destination_repo_id,
        fps=source.fps,
        root=destination_root,
        robot_type=source.meta.robot_type,
        features=features,
        use_videos=bool(source.meta.video_keys),
        image_writer_threads=max(1, len(source.meta.camera_keys)),
    )

    try:
        for episode_index in range(source.meta.total_episodes):
            episode = source.meta.episodes[episode_index]
            start = int(episode["dataset_from_index"])
            stop = int(episode["dataset_to_index"])
            tasks = episode.get("tasks") or [""]
            fallback_task = str(tasks[0])
            for frame_index in range(start, stop):
                destination.add_frame(
                    _convert_frame(
                        source[frame_index],
                        features,
                        direction,
                        fallback_task,
                        collector_policy_id=collector_policy_id,
                    )
                )
            destination.save_episode(episode_metadata=_user_episode_metadata(episode))
            logger.info("Migrated episode %d/%d", episode_index + 1, source.meta.total_episodes)
    finally:
        destination.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--destination-repo-id", required=True)
    parser.add_argument(
        "--direction",
        choices=("to-canonical", "to-current", "to-0901"),
        default="to-canonical",
        help="Use to-canonical for the shared 0901 field schema in current v3 storage.",
    )
    parser.add_argument(
        "--collector-policy-id",
        default="policy",
        help="Policy identifier used for non-intervention frames when the source lacks this field.",
    )
    args = parser.parse_args()
    init_logging()
    migrate_dataset(**vars(args))


if __name__ == "__main__":
    main()
