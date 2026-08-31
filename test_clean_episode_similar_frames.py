#!/usr/bin/env python3
"""Runnable end-to-end check for clean_episode_similar_frames.py."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_video(path: Path, colors: list[int], pts: list[int], fps: int) -> None:
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(
            "h264", rate=fps, options={"g": "2", "crf": "18", "preset": "ultrafast"}
        )
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for color, timestamp in zip(colors, pts, strict=True):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), color, dtype=np.uint8), format="rgb24")
            frame.pts = timestamp
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def make_dataset(root: Path) -> Path:
    source = root / "toy"
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    states = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    data = pa.table(
        {
            "observation.state": pa.array([[value] for value in states], type=pa.list_(pa.float32(), 1)),
            "action": pa.array([[value] for value in range(10)], type=pa.list_(pa.float32(), 1)),
            "timestamp": pa.array([index / 10 for index in range(7)] + [index / 10 for index in range(3)], pa.float32()),
            "frame_index": pa.array(list(range(7)) + list(range(3)), pa.int64()),
            "episode_index": pa.array([0] * 7 + [1] * 3, pa.int64()),
            "index": pa.array(range(10), pa.int64()),
            "task_index": pa.array([0] * 10, pa.int64()),
        }
    )
    pq.write_table(data, source / "data/chunk-000/file-000.parquet")

    video_key = "observation.images.camera"
    meta = pa.table(
        {
            "episode_index": pa.array([0, 1], pa.int64()),
            "tasks": pa.array([["test"], ["test"]], pa.list_(pa.string())),
            "length": pa.array([7, 3], pa.int64()),
            "data/chunk_index": pa.array([0, 0], pa.int64()),
            "data/file_index": pa.array([0, 0], pa.int64()),
            "dataset_from_index": pa.array([0, 7], pa.int64()),
            "dataset_to_index": pa.array([7, 10], pa.int64()),
            "episode_success": pa.array(["yes", "no"]),
            f"videos/{video_key}/chunk_index": pa.array([0, 0], pa.int64()),
            f"videos/{video_key}/file_index": pa.array([0, 0], pa.int64()),
            f"videos/{video_key}/from_timestamp": pa.array([0.0, 1.2], pa.float64()),
            f"videos/{video_key}/to_timestamp": pa.array([0.7, 1.5], pa.float64()),
            "meta/episodes/chunk_index": pa.array([0, 0], pa.int64()),
            "meta/episodes/file_index": pa.array([0, 0], pa.int64()),
        }
    )
    pq.write_table(meta, source / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["test"]}), source / "meta/tasks.parquet")
    (source / "meta/stats.json").write_text("{}\n", encoding="utf-8")

    features = {
        "observation.state": {"dtype": "float32", "shape": [1], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [1], "names": ["action"]},
        video_key: {
            "dtype": "video",
            "shape": [24, 32, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 24,
                "video.width": 32,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 10,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "total_episodes": 2,
        "total_frames": 10,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": 10,
        "splits": {"train": "0:2"},
        "custom": "keep-me",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (source / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    write_video(
        source / f"videos/{video_key}/chunk-000/file-000.mp4",
        list(range(0, 200, 20)),
        list(range(7)) + [12, 13, 14],
        10,
    )
    return source


def read_video(path: Path) -> tuple[list[float], list[float]]:
    timestamps, means = [], []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            timestamps.append(float(frame.pts * frame.time_base))
            means.append(float(frame.to_ndarray(format="rgb24").mean()))
    return timestamps, means


def main() -> None:
    script = Path(__file__).with_name("clean_episode_similar_frames.py")
    with tempfile.TemporaryDirectory(prefix="clean-similar-test-") as directory:
        source = make_dataset(Path(directory))
        source_video = source / "videos/observation.images.camera/chunk-000/file-000.mp4"
        source_hash = hashlib.sha256(source_video.read_bytes()).hexdigest()
        result = subprocess.run(
            [sys.executable, str(script), str(source), "--threshold", "0", "--apply", "--vcodec", "h264"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        output = source.with_name("toy_cleaned")
        data = pq.read_table(sorted((output / "data").rglob("file-*.parquet")))
        meta = pq.read_table(sorted((output / "meta/episodes").rglob("file-*.parquet")))
        info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
        assert data["observation.state"].to_pylist() == [[0.0], [1.0], [2.0], [2.0], [3.0], [4.0]]
        assert data["action"].to_pylist() == [[1.0], [4.0], [5.0], [6.0], [8.0], [9.0]]
        assert np.asarray(data["frame_index"]).tolist() == [0, 1, 2, 3, 0, 1]
        assert np.allclose(np.asarray(data["timestamp"]), [0, 0.1, 0.2, 0.3, 0, 0.1])
        assert meta["length"].to_pylist() == [4, 2]
        assert meta["episode_success"].to_pylist() == ["yes", "no"]
        assert np.allclose(meta["videos/observation.images.camera/from_timestamp"], [0.0, 0.4])
        assert np.allclose(meta["videos/observation.images.camera/to_timestamp"], [0.4, 0.6])
        assert info["custom"] == "keep-me" and info["splits"] == {"train": "0:2"}
        timestamps, means = read_video(output / "videos/observation.images.camera/chunk-000/file-000.mp4")
        assert np.allclose(timestamps, np.arange(6) / 10)
        assert np.allclose(means, [20, 80, 100, 120, 160, 180], atol=8)
        assert hashlib.sha256(source_video.read_bytes()).hexdigest() == source_hash
        assert "planned_remove_frames=4" in result.stdout
    print("clean_episode_similar_frames: end-to-end check passed")


if __name__ == "__main__":
    main()
