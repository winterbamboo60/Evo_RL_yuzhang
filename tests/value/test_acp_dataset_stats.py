#!/usr/bin/env python

from types import SimpleNamespace

from lerobot.rl.acp_dataset_stats import compute_acp_indicator_stats
from lerobot.rl.acp_tags import ACP_NEGATIVE_TAG, ACP_POSITIVE_TAG, build_acp_tagged_task


class DummyHFDataset:
    def __init__(self, columns: dict[str, list]):
        self._columns = columns
        self.column_names = list(columns.keys())

    def __getitem__(self, key: str):
        return self._columns[key]


def test_compute_acp_indicator_stats_from_meta_stats():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            stats={
                "complementary_info.acp_indicator": {
                    "mean": [0.3],
                    "count": [10],
                }
            }
        )
    )

    stats = compute_acp_indicator_stats(dataset, "complementary_info.acp_indicator")

    assert stats is not None
    assert stats.source == "meta.stats"
    assert stats.positive_ratio == 0.3
    assert stats.total_count == 10
    assert stats.positive_count == 3
    assert stats.invalid_count == -1


def test_compute_acp_indicator_stats_from_dataset_scan():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(stats={}),
        hf_dataset=DummyHFDataset(
            {
                "complementary_info.acp_indicator": [[1], [0], [1], [2]],
            }
        ),
    )

    stats = compute_acp_indicator_stats(dataset, "complementary_info.acp_indicator")

    assert stats is not None
    assert stats.source == "hf_dataset_scan"
    assert stats.total_count == 4
    assert stats.positive_count == 2
    assert stats.positive_ratio == 0.5
    assert stats.invalid_count == 1


def test_compute_acp_indicator_stats_returns_none_when_missing_field():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(stats={}),
        hf_dataset=DummyHFDataset({"task": ["pick", "place"]}),
    )

    stats = compute_acp_indicator_stats(dataset, "complementary_info.acp_indicator")
    assert stats is None


def test_compute_acp_indicator_stats_prefers_meta_stats():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            stats={
                "complementary_info.acp_indicator": {
                    "mean": [0.75],
                    "count": [8],
                }
            }
        ),
        hf_dataset=DummyHFDataset({"complementary_info.acp_indicator": [0, 0, 0, 0]}),
    )

    stats = compute_acp_indicator_stats(dataset, "complementary_info.acp_indicator")

    assert stats is not None
    assert stats.source == "meta.stats"
    assert stats.positive_ratio == 0.75
    assert stats.total_count == 8
    assert stats.positive_count == 6


def test_compute_acp_indicator_stats_accepts_0901_field_alias():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(stats={}),
        hf_dataset=DummyHFDataset(
            {"complementary_info.indicator_field": [1, 0, 1]}
        ),
    )

    stats = compute_acp_indicator_stats(dataset, "complementary_info.acp_indicator")

    assert stats is not None
    assert stats.indicator_field == "complementary_info.indicator_field"
    assert stats.positive_ratio == 2 / 3


def test_acp_tags_preserve_0901_prompt_format():
    assert build_acp_tagged_task("pick", is_positive=True) == f"pick\n{ACP_POSITIVE_TAG}"
    assert build_acp_tagged_task("place", is_positive=False) == f"place\n{ACP_NEGATIVE_TAG}"
    assert build_acp_tagged_task(None, is_positive=True) == ACP_POSITIVE_TAG
