#!/usr/bin/env python
"""Copy a LeRobot-style dataset and replace every episode task label.

The source dataset is never modified. The script first copies the full dataset
directory to a new location, then rewrites task metadata in the copy:

    - meta/tasks.parquet is replaced with a single task row.
    - meta/info.json total_tasks is set to 1.
    - every data parquet task_index column is set to 0.
    - if meta/episodes parquet files exist, their tasks/task_index columns are
      updated as well.

Default example:
    python /home/yz/datasets/replace_dataset_task.py

Custom example:
    python /home/yz/datasets/replace_dataset_task.py \
        /home/yz/datasets/V6_task2_checked \
        --task xx \
        --output-suffix _xx
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path("/mnt/cfs/0z9lxh/yuzhang/datasets/20260910_cube_catch_rollout_v3-2_merged")
DEFAULT_TASK = "Grab the moving blocks on the conveyor belt"
DEFAULT_OUTPUT_SUFFIX = "_rawTask"


def _default_output_path(source: Path, suffix: str) -> Path:
    return source.with_name(f"{source.name}{suffix}")


def _prepare_output(source: Path, output: Path, overwrite: bool) -> None:
    if not source.is_dir():
        raise NotADirectoryError(f"Source dataset directory not found: {source}")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Source does not look like a LeRobot dataset: {source}")
    if output.resolve() == source.resolve():
        raise ValueError("Output directory must be different from the source directory.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)


def _write_single_task_table(dataset_dir: Path, task: str) -> None:
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing task metadata: {tasks_path}")

    # LeRobot v3 stores the task string as the pandas index and task_index as a column.
    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([task]))
    tasks.to_parquet(tasks_path, index=True)


def _update_info(dataset_dir: Path) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = 1
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n")


def _rewrite_task_index_in_data(dataset_dir: Path) -> int:
    changed = 0
    for parquet_path in sorted((dataset_dir / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        if "task_index" in df.columns:
            df["task_index"] = 0
            df.to_parquet(parquet_path, index=False)
            changed += 1
    return changed


def _rewrite_episode_metadata(dataset_dir: Path, task: str) -> int:
    episodes_dir = dataset_dir / "meta" / "episodes"
    if not episodes_dir.is_dir():
        return 0

    changed = 0
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        did_change = False
        if "tasks" in df.columns:
            df["tasks"] = [[task] for _ in range(len(df))]
            did_change = True
        if "task_index" in df.columns:
            df["task_index"] = 0
            did_change = True
        if did_change:
            df.to_parquet(parquet_path, index=False)
            changed += 1
    return changed


def replace_dataset_task(source: Path, output: Path, task: str, overwrite: bool) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    _prepare_output(source, output, overwrite)

    shutil.copytree(source, output)
    _write_single_task_table(output, task)
    _update_info(output)
    data_files_changed = _rewrite_task_index_in_data(output)
    episode_files_changed = _rewrite_episode_metadata(output, task)

    print(f"Source dataset : {source}")
    print(f"Output dataset : {output}")
    print(f"New task       : {task!r}")
    print(f"Data files updated     : {data_files_changed}")
    print(f"Episode meta updated   : {episode_files_changed}")
    print("Original dataset was not modified.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_SOURCE),
        help=f"Source dataset directory. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help=f"New task string. Default: {DEFAULT_TASK!r}")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dataset directory. Default: <source><output-suffix>",
    )
    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help=f"Suffix used for the default output directory. Default: {DEFAULT_OUTPUT_SUFFIX!r}",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists.")
    args = parser.parse_args()
    args.source = Path(args.source)
    if args.output is None:
        args.output = _default_output_path(args.source, args.output_suffix)
    return args


def main() -> None:
    args = parse_args()
    replace_dataset_task(args.source, args.output, args.task, args.overwrite)


if __name__ == "__main__":
    main()
