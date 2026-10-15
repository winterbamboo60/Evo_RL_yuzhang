#!/usr/bin/env python

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.io_utils import load_info, write_info, write_table_one_row_group_per_episode
from lerobot.datasets.utils import DatasetInfo
from lerobot.scripts.lerobot_value_infer import (
    _binarize_advantages,
    _compute_dense_rewards_from_targets,
    _compute_n_step_advantages,
    _compute_task_thresholds,
    _write_columns_in_place,
    _write_marked_dataset,
)


def test_compute_dense_rewards_from_targets_terminal_handling():
    episode_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    frame_indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    targets = np.array([-0.6, -0.4, -0.2, -0.8, -0.5], dtype=np.float32)

    rewards = _compute_dense_rewards_from_targets(targets, episode_indices, frame_indices)
    expected = np.array([-0.2, -0.2, -0.2, -0.3, -0.5], dtype=np.float32)
    assert np.allclose(rewards, expected)


def test_compute_n_step_advantages_simple_case():
    rewards = np.array([-0.2, -0.2, -0.2], dtype=np.float32)
    values = np.array([-0.5, -0.3, -0.1], dtype=np.float32)
    episode_indices = np.array([0, 0, 0], dtype=np.int64)
    frame_indices = np.array([0, 1, 2], dtype=np.int64)

    advantages = _compute_n_step_advantages(
        rewards=rewards,
        values=values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        n_step=2,
    )

    expected = np.array([0.0, -0.1, -0.1], dtype=np.float32)
    assert np.allclose(advantages, expected)


def test_compute_task_thresholds_and_binarize_with_interventions():
    task_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    advantages = np.array([-0.4, -0.1, 0.3, -0.2, 0.2], dtype=np.float32)
    interventions = np.array([0, 1, 0, 0, 0], dtype=np.float32)

    thresholds = _compute_task_thresholds(task_indices, advantages, positive_ratio=0.5)

    indicators = _binarize_advantages(
        task_indices=task_indices,
        advantages=advantages,
        thresholds=thresholds,
        interventions=interventions,
        force_intervention_positive=True,
    )

    assert indicators.tolist() == [0, 1, 1, 0, 1]


def test_write_columns_in_place_preserves_current_parquet_layout(tmp_path: Path):
    write_info(
        DatasetInfo(
            codebase_version="v3.0",
            fps=30,
            features={
                "index": {"dtype": "int64", "shape": (1,), "names": None},
                "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
            },
            total_episodes=2,
            total_frames=3,
        ),
        tmp_path,
    )
    parquet_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    parquet_path.parent.mkdir(parents=True)
    table = pa.table({"index": [0, 1, 2], "episode_index": [0, 0, 1]})
    write_table_one_row_group_per_episode(table, parquet_path)

    _write_columns_in_place(
        dataset_root=tmp_path,
        absolute_indices=np.array([0, 1, 2], dtype=np.int64),
        columns={"complementary_info.value": np.array([-0.5, -0.25, 0.0], dtype=np.float32)},
        feature_infos={
            "complementary_info.value": {"dtype": "float32", "shape": (1,), "names": None}
        },
    )

    rewritten = pq.read_table(parquet_path)
    assert rewritten["complementary_info.value"].to_pylist() == [-0.5, -0.25, 0.0]
    assert pq.read_metadata(parquet_path).num_row_groups == 2
    assert load_info(tmp_path).features["complementary_info.value"]["shape"] == (1,)


def test_marked_dataset_is_independent_and_preserves_source(tmp_path: Path):
    source = tmp_path / "original"
    source.mkdir()
    write_info(
        DatasetInfo(
            codebase_version="v3.0",
            fps=30,
            features={"index": {"dtype": "int64", "shape": (1,), "names": None}},
            total_episodes=1,
            total_frames=2,
        ),
        source,
    )
    parquet = source / "data/chunk-000/file-000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.table({"index": [0, 1], "episode_index": [0, 0]}), parquet)
    video = source / "videos/camera/video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video fixture")
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    columns = {
        "complementary_info.value": np.array([-0.5, 0.0], dtype=np.float32),
        "complementary_info.advantage": np.array([-0.1, 0.2], dtype=np.float32),
        "complementary_info.acp_indicator": np.array([0, 1], dtype=np.int64),
    }
    features = {
        key: {"dtype": "int64" if "indicator" in key else "float32", "shape": (1,), "names": None}
        for key in columns
    }
    destination = _write_marked_dataset(source, np.array([0, 1]), columns, features)
    assert destination == tmp_path / "original_RLMarked"
    after = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert before == after
    marked = pq.read_table(destination / parquet.relative_to(source))
    for key, values in columns.items():
        np.testing.assert_allclose(marked[key].to_numpy(), values)
        assert key in load_info(destination).features
    copied_video = destination / video.relative_to(source)
    assert copied_video.read_bytes() == video.read_bytes()
    copied_video.write_bytes(b"changed copy")
    assert video.read_bytes() == b"video fixture"
    with pytest.raises(FileExistsError, match="already exists"):
        _write_marked_dataset(source, np.array([0, 1]), columns, features)


def test_failed_marking_does_not_publish_partial_dataset(tmp_path: Path, monkeypatch):
    from lerobot.scripts import lerobot_value_infer

    source = tmp_path / "original"
    source.mkdir()
    (source / "sentinel").write_text("unchanged")

    def fail(*args, **kwargs):
        raise RuntimeError("annotation failed")

    monkeypatch.setattr(lerobot_value_infer, "_write_columns_in_place", fail)
    with pytest.raises(RuntimeError, match="annotation failed"):
        _write_marked_dataset(source, np.array([0]), {}, {})
    assert list(tmp_path.iterdir()) == [source]
    assert (source / "sentinel").read_text() == "unchanged"
