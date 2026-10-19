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

from types import SimpleNamespace
from unittest.mock import Mock, call

import draccus
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts import lerobot_edit_dataset as edit_dataset


@pytest.mark.parametrize("old", [False, True], ids=["exact_default", "legacy_stats"])
def test_source_dir_merge_dispatches_statistics_once(tmp_path, monkeypatch, old):
    """The compatibility CLI delegates the selected stats mode to merge_datasets."""
    source_dir = tmp_path / "sources"
    source_roots = [source_dir / "session-a", source_dir / "session-b"]
    for root in source_roots:
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}")
    output_dir = tmp_path / "merged"
    cli_args = [
        "--repo_id",
        str(output_dir),
        "--operation.type",
        "merge",
        "--operation.source_dir",
        str(source_dir),
    ]
    if old:
        cli_args.extend(["--operation.old", "true"])
    cfg = draccus.parse(edit_dataset.EditDatasetConfig, args=cli_args)

    source_datasets = [object(), object()]
    load_dataset = Mock(side_effect=source_datasets)
    merge = Mock(return_value=SimpleNamespace(meta=SimpleNamespace(total_episodes=2, total_frames=5)))
    recompute = Mock()
    monkeypatch.setattr(edit_dataset, "LeRobotDataset", load_dataset)
    monkeypatch.setattr(edit_dataset, "merge_datasets", merge)
    monkeypatch.setattr(edit_dataset, "recompute_stats", recompute)

    edit_dataset.handle_merge(cfg)

    assert load_dataset.call_args_list == [
        call(repo_id="session-a", root=source_roots[0]),
        call(repo_id="session-b", root=source_roots[1]),
    ]
    merge.assert_called_once_with(
        source_datasets,
        output_repo_id="local/merged",
        output_dir=output_dir,
        concatenate_videos=True,
        concatenate_data=True,
        recompute_statistics=not old,
    )
    recompute.assert_not_called()


def test_source_dir_merge_refuses_existing_output_before_loading(tmp_path, monkeypatch):
    """An existing output must be preserved before loading or merging sources."""
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    marker = output_dir / "existing.txt"
    marker.write_text("preserve this output")
    cfg = draccus.parse(
        edit_dataset.EditDatasetConfig,
        args=[
            "--repo_id",
            str(output_dir),
            "--operation.type",
            "merge",
            "--operation.source_dir",
            str(tmp_path / "sources"),
        ],
    )
    load_dataset = Mock()
    merge = Mock()
    recompute = Mock()
    monkeypatch.setattr(edit_dataset, "LeRobotDataset", load_dataset)
    monkeypatch.setattr(edit_dataset, "merge_datasets", merge)
    monkeypatch.setattr(edit_dataset, "recompute_stats", recompute)

    with pytest.raises(FileExistsError, match="Merge output directory already exists"):
        edit_dataset.handle_merge(cfg)

    load_dataset.assert_not_called()
    merge.assert_not_called()
    recompute.assert_not_called()
    assert marker.read_text() == "preserve this output"
