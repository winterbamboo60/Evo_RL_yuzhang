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

"""Exact numeric statistics for offline dataset editing."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .compute_stats import DEFAULT_QUANTILES

_BATCH_SIZE = 65536


def _basic_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Merge centered batch moments in float64, avoiding E[x²] - E[x]² cancellation."""
    count = 0
    mean = np.zeros(values.shape[1:], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for start in range(0, len(values), _BATCH_SIZE):
        batch = np.asarray(values[start : start + _BATCH_SIZE], dtype=np.float64)
        batch_count = len(batch)
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        delta = batch_mean - mean
        total = count + batch_count
        m2 += batch_m2 + delta * delta * (count * batch_count / total)
        mean += delta * (batch_count / total)
        count = total
    return {
        "min": np.asarray(values.min(axis=0)).copy(),
        "max": np.asarray(values.max(axis=0)).copy(),
        "mean": mean,
        "std": np.sqrt(m2 / count),
        "count": np.array([count], dtype=np.int64),
    }


def _add_quantiles(stats: dict[str, np.ndarray], values: np.ndarray) -> None:
    # Partition the disposable disk-backed array in place; no histogram estimates.
    quantiles = np.quantile(values, DEFAULT_QUANTILES, axis=0, method="linear", overwrite_input=True)
    for q, value in zip(DEFAULT_QUANTILES, quantiles, strict=True):
        stats[f"q{int(q * 100):02d}"] = np.asarray(value).copy()


def _numeric_batch(column: pa.Array, shape: tuple[int, ...], dtype: np.dtype, key: str) -> np.ndarray:
    if column.null_count:
        raise ValueError(f"Null values in numeric feature {key!r}")
    try:
        value_type = column.type
        while (
            pa.types.is_list(value_type)
            or pa.types.is_fixed_size_list(value_type)
            or pa.types.is_large_list(value_type)
        ):
            value_type = value_type.value_type
        if np.dtype(value_type.to_pandas_dtype()) != dtype:
            raise ValueError(f"Invalid dtype for feature {key!r}: {value_type}, expected {dtype}")
        if (
            pa.types.is_list(column.type)
            or pa.types.is_fixed_size_list(column.type)
            or pa.types.is_large_list(column.type)
        ):
            array = np.asarray(column.to_pylist())
        else:
            array = column.to_numpy(zero_copy_only=False)
            if shape == (1,):
                array = array.reshape(-1, 1)
        if array.shape != (len(column), *shape):
            raise ValueError(
                f"Invalid shape for feature {key!r}: {array.shape}, expected {(len(column), *shape)}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in numeric feature {key!r}")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric data for feature {key!r}: {error}") from error
    return array


def compute_exact_numeric_stats(
    root: Path, features: dict
) -> tuple[dict[str, dict[str, np.ndarray]], dict[int, dict[str, dict[str, np.ndarray]]]]:
    """Compute full-data and per-episode stats without decoding images or videos.

    Every numeric row is included, including bookkeeping fields after reindexing.
    Quantiles use linear interpolation at (N - 1) * q; std uses ddof=0. One feature
    at a time is stored in a temporary memory map, so exact quantiles do not require
    loading all features into RAM. Episode boundaries may cross Parquet files.
    """
    root = Path(root)
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {root / 'data'}")
    parquet_files = [pq.ParquetFile(path) for path in files]
    total_rows = sum(file.metadata.num_rows for file in parquet_files)
    if not total_rows:
        raise ValueError(f"No data rows found in {root / 'data'}")

    # Keep spans rather than an N-row episode index array. Coalesce runs across
    # batches/files, and support episodes appearing in multiple disjoint runs.
    episode_spans: dict[int, list[tuple[int, int]]] = {}
    offset = 0
    for file in parquet_files:
        for batch in file.iter_batches(batch_size=_BATCH_SIZE, columns=["episode_index"]):
            column = batch.column(0)
            if column.null_count or not pa.types.is_integer(column.type):
                raise ValueError("episode_index must contain non-null integer values")
            ids = column.to_numpy(zero_copy_only=False)
            boundaries = np.concatenate(([0], np.flatnonzero(ids[1:] != ids[:-1]) + 1, [len(ids)]))
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                if start == stop:
                    continue
                spans = episode_spans.setdefault(int(ids[start]), [])
                new_span = (offset + int(start), offset + int(stop))
                if spans and spans[-1][1] == new_span[0]:
                    spans[-1] = (spans[-1][0], new_span[1])
                else:
                    spans.append(new_span)
            offset += len(ids)

    global_stats = {}
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {ep: {} for ep in episode_spans}
    with TemporaryDirectory(prefix="lerobot-exact-stats-") as temporary_dir:
        for key, feature in features.items():
            if feature["dtype"] in ("image", "video", "string", "language"):
                continue
            shape = tuple(feature["shape"])
            dtype = np.dtype(feature["dtype"])
            # Preserve integer extrema exactly; floats are promoted before moments
            # and quantile interpolation, including float32 constant features.
            storage_dtype = dtype if dtype.kind in "iu" else np.dtype("float64")
            values = np.memmap(
                Path(temporary_dir) / "values.dat", mode="w+", dtype=storage_dtype, shape=(total_rows, *shape)
            )
            try:
                offset = 0
                for path, file in zip(files, parquet_files, strict=True):
                    if key not in file.schema_arrow.names:
                        raise ValueError(f"Missing numeric feature {key!r} in {path}")
                    for batch in file.iter_batches(batch_size=_BATCH_SIZE, columns=[key]):
                        array = _numeric_batch(batch.column(0), shape, dtype, key)
                        values[offset : offset + len(array)] = array
                        offset += len(array)

                stats = _basic_stats(values)
                for ep, spans in episode_spans.items():
                    if len(spans) == 1:
                        start, stop = spans[0]
                        ep_values = values[start:stop]
                        ep_stats = _basic_stats(ep_values)
                        _add_quantiles(ep_stats, ep_values)
                    else:
                        ep_count = sum(stop - start for start, stop in spans)
                        ep_values = np.memmap(
                            Path(temporary_dir) / "episode.dat",
                            mode="w+",
                            dtype=storage_dtype,
                            shape=(ep_count, *shape),
                        )
                        try:
                            ep_offset = 0
                            for start, stop in spans:
                                ep_values[ep_offset : ep_offset + stop - start] = values[start:stop]
                                ep_offset += stop - start
                            ep_stats = _basic_stats(ep_values)
                            _add_quantiles(ep_stats, ep_values)
                        finally:
                            ep_values._mmap.close()
                    episode_stats[ep][key] = ep_stats
                _add_quantiles(stats, values)
                global_stats[key] = stats
            finally:
                values._mmap.close()
    return global_stats, episode_stats
