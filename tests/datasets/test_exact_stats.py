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
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.exact_stats import compute_exact_numeric_stats

QUANTILES = {"q01": 0.01, "q10": 0.1, "q50": 0.5, "q90": 0.9, "q99": 0.99}


def write_parquet_parts(root, columns, cuts):
    """Place consecutive parts in distinct chunks, independently of episode boundaries."""
    table = pa.table(columns)
    start = 0
    for chunk_index, end in enumerate(cuts):
        path = root / "data" / f"chunk-{chunk_index:03d}" / "file-000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.slice(start, end - start), path)
        start = end
    assert start == len(table)


def manual_statistics(values, shape):
    """Use Python sorting and centered math.fsum rather than production reductions."""
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    result = {metric: [] for metric in ("min", "max", "mean", "std", *QUANTILES)}
    for dimension in range(values.shape[1]):
        column = [float(value) for value in values[:, dimension]]
        ordered = sorted(column)
        mean = math.fsum(column) / len(column)
        result["min"].append(ordered[0])
        result["max"].append(ordered[-1])
        result["mean"].append(mean)
        result["std"].append(math.sqrt(math.fsum((value - mean) ** 2 for value in column) / len(column)))
        for name, probability in QUANTILES.items():
            position = (len(ordered) - 1) * probability
            lower = math.floor(position)
            upper = math.ceil(position)
            result[name].append(ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]))
    return {
        **{metric: np.asarray(value).reshape(shape) for metric, value in result.items()},
        "count": np.asarray([len(values)]),
    }


def assert_statistics(actual, expected, shape):
    assert set(actual) == set(expected)
    for metric, value in expected.items():
        assert np.asarray(actual[metric]).shape == ((1,) if metric == "count" else tuple(shape))
        if metric == "count":
            np.testing.assert_array_equal(actual[metric], value)
        else:
            np.testing.assert_allclose(actual[metric], value, rtol=1e-12, atol=1e-12)


def test_exact_statistics_use_all_rows_and_group_episodes_across_chunks(tmp_path):
    # The middle episode crosses every chunk; its five frames outweigh the first two frames.
    episode_ids = [4, 4, 8, 8, 8, 8, 8, 12]
    scalar = [0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 500.0]
    vector = [[value, 2.0 * (index + 1)] for index, value in enumerate(scalar)]
    matrix = [[[value, value + 1], [index, -index]] for index, value in enumerate(scalar)]
    columns = {
        "episode_index": episode_ids,
        "index": list(range(8)),
        "scalar": scalar,
        "true_scalar": scalar,
        "vector": vector,
        "matrix": matrix,
        "constant": [7.0] * 8,
        "unused": [None] * 8,
    }
    features = {
        "index": {"dtype": "int64", "shape": (1,)},
        "scalar": {"dtype": "float64", "shape": (1,)},
        "true_scalar": {"dtype": "float64", "shape": ()},
        "vector": {"dtype": "float64", "shape": (2,)},
        "matrix": {"dtype": "float64", "shape": (2, 2)},
        "constant": {"dtype": "float64", "shape": (1,)},
    }
    write_parquet_parts(tmp_path, columns, [3, 6, 8])

    global_stats, episode_stats = compute_exact_numeric_stats(tmp_path, features)

    assert set(global_stats) == set(features)
    assert set(episode_stats) == {4, 8, 12}
    for key, feature in features.items():
        assert_statistics(
            global_stats[key], manual_statistics(columns[key], feature["shape"]), feature["shape"]
        )
        for episode_index in episode_stats:
            selected = [
                value
                for value, episode in zip(columns[key], episode_ids, strict=True)
                if episode == episode_index
            ]
            assert set(episode_stats[episode_index]) == set(features)
            assert_statistics(
                episode_stats[episode_index][key],
                manual_statistics(selected, feature["shape"]),
                feature["shape"],
            )

    assert float(np.asarray(global_stats["scalar"]["q50"])[0]) == 100.0
    assert float(np.asarray(global_stats["scalar"]["q90"])[0]) == pytest.approx(220.0)
    np.testing.assert_array_equal(global_stats["index"]["max"], [7])
    np.testing.assert_array_equal(episode_stats[12]["scalar"]["std"], [0.0])
    np.testing.assert_array_equal(global_stats["constant"]["std"], [0.0])


def test_exact_statistics_preserve_small_variance_at_large_mean(tmp_path):
    # E[x**2] - E[x]**2 loses precision here; a sample standard deviation also differs.
    columns = {"episode_index": [0, 0, 1, 1], "value": [1e8 + offset for offset in range(4)]}
    features = {"value": {"dtype": "float64", "shape": (1,)}}
    write_parquet_parts(tmp_path, columns, [1, 3, 4])

    global_stats, episode_stats = compute_exact_numeric_stats(tmp_path, features)

    np.testing.assert_allclose(global_stats["value"]["mean"], [1e8 + 1.5], rtol=0, atol=0)
    np.testing.assert_allclose(global_stats["value"]["std"], [math.sqrt(1.25)], rtol=0, atol=1e-12)
    for episode_index in (0, 1):
        np.testing.assert_array_equal(episode_stats[episode_index]["value"]["count"], [2])
        np.testing.assert_allclose(episode_stats[episode_index]["value"]["std"], [0.5], rtol=0, atol=1e-12)


def test_exact_statistics_group_disjoint_episode_runs(tmp_path):
    episode_ids = [3, 9, 3, 9, 3, 9]
    values = [0.0, 1.0, 10.0, 2.0, 20.0, 3.0]
    features = {"value": {"dtype": "float64", "shape": (1,)}}
    write_parquet_parts(tmp_path, {"episode_index": episode_ids, "value": values}, [2, 5, 6])

    global_stats, episode_stats = compute_exact_numeric_stats(tmp_path, features)

    assert_statistics(global_stats["value"], manual_statistics(values, (1,)), (1,))
    for episode_index in (3, 9):
        selected = [
            value for value, episode in zip(values, episode_ids, strict=True) if episode == episode_index
        ]
        assert_statistics(episode_stats[episode_index]["value"], manual_statistics(selected, (1,)), (1,))


def test_exact_statistics_cross_read_and_moment_batch_boundaries(tmp_path):
    # A full first batch and a tail both contain whole 0,1,2,3 cycles.
    count = 65540
    columns = {
        "episode_index": [17] * 65537 + [32] * 3,
        "value": (1e8 + np.arange(count) % 4).tolist(),
        "flag": (np.arange(count) % 2 == 1).tolist(),
    }
    features = {
        "value": {"dtype": "float64", "shape": (1,)},
        "flag": {"dtype": "bool", "shape": (1,)},
    }
    write_parquet_parts(tmp_path, columns, [65538, count])

    global_stats, episode_stats = compute_exact_numeric_stats(tmp_path, features)

    np.testing.assert_array_equal(global_stats["value"]["count"], [count])
    np.testing.assert_allclose(global_stats["value"]["mean"], [1e8 + 1.5], rtol=0, atol=0)
    np.testing.assert_allclose(global_stats["value"]["std"], [math.sqrt(1.25)], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(episode_stats[17]["value"]["count"], [65537])
    np.testing.assert_array_equal(episode_stats[32]["value"]["count"], [3])
    assert_statistics(global_stats["flag"], manual_statistics(columns["flag"], (1,)), (1,))
    for episode_index in (17, 32):
        selected = [
            value
            for value, episode in zip(columns["flag"], columns["episode_index"], strict=True)
            if episode == episode_index
        ]
        assert_statistics(episode_stats[episode_index]["flag"], manual_statistics(selected, (1,)), (1,))


@pytest.mark.parametrize(
    "values,shape",
    [
        ([1.0, None], (1,)),
        ([1.0, float("nan")], (1,)),
        ([1.0, float("inf")], (1,)),
        ([[1.0, 2.0], [None, 4.0]], (2,)),
        ([[1.0, 2.0], [3.0, float("nan")]], (2,)),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2,)),
        ([[1.0, 2.0], [3.0]], (2,)),
        ([1.0, 2.0], (2,)),
        ([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]], (2, 2)),
    ],
    ids=[
        "null",
        "nan",
        "infinity",
        "nested-null",
        "nested-nan",
        "wrong-width",
        "ragged",
        "scalar-as-vector",
        "wrong-matrix-shape",
    ],
)
def test_exact_statistics_reject_invalid_values_or_shapes(tmp_path, values, shape):
    write_parquet_parts(tmp_path, {"episode_index": [0, 0], "value": values}, [2])

    with pytest.raises(ValueError):
        compute_exact_numeric_stats(tmp_path, {"value": {"dtype": "float64", "shape": shape}})


def test_exact_statistics_reject_null_episode_indices(tmp_path):
    write_parquet_parts(tmp_path, {"episode_index": [0, None], "value": [1.0, 2.0]}, [2])

    with pytest.raises(ValueError):
        compute_exact_numeric_stats(tmp_path, {"value": {"dtype": "float64", "shape": (1,)}})


@pytest.mark.parametrize(
    "values,dtype",
    [([1.25, 2.75], "int64"), ([1.0, 2.0], "float32"), ([1, 2], "float64")],
    ids=["fractional-float-as-integer", "float64-as-float32", "integer-as-float"],
)
def test_exact_statistics_reject_declared_dtype_mismatch(tmp_path, values, dtype):
    write_parquet_parts(tmp_path, {"episode_index": [0, 0], "value": values}, [2])

    with pytest.raises(ValueError):
        compute_exact_numeric_stats(tmp_path, {"value": {"dtype": dtype, "shape": (1,)}})
