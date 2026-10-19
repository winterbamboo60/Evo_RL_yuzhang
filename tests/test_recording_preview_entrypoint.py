"""Exercise the bimanual launcher using stubs, without devices or real recording logs."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def bimanual_entrypoint(tmp_path):
    project = tmp_path / "project"
    script = project / "scripts" / "RL_data_bimanual.sh"
    script.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts" / script.name
    shutil.copy2(source, script)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    capture = tmp_path / "argv.bin"
    for name in ("python", "lerobot-rollout"):
        stub = stub_dir / name
        stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\0\' "${0##*/}" "$@" > "$PREVIEW_ARGV_PATH"\n')
        stub.chmod(0o755)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    env = dict(os.environ, PATH=f"{stub_dir}{os.pathsep}{os.environ['PATH']}", PREVIEW_ARGV_PATH=str(capture))
    required = [
        "--dataset.root",
        str(tmp_path / "dataset"),
        "--dataset.single_task",
        "pour water",
        "--top_camera.index_or_path",
        "18",
        "--left_wrist_camera.index_or_path",
        "22",
        "--right_wrist_camera.index_or_path",
        "10",
    ]

    def run(*args):
        result = subprocess.run(
            ["bash", str(script), *required, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        argv = capture.read_bytes().decode().rstrip("\0").split("\0") if capture.exists() else []
        return result, argv

    return run, checkpoint


def test_manual_preview_defaults(bimanual_entrypoint):
    run, _ = bimanual_entrypoint
    result, argv = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert argv[:3] == ["python", "-m", "lerobot.scripts.lerobot_record_cli"]
    assert "--display_data=true" in argv
    assert "--display_fps=15" in argv
    assert "--dataset.fps=30" in argv
    assert "--display_compressed_images=false" in argv
    assert "--dataset.rgb_encoder.vcodec=libsvtav1" in argv


def test_preview_and_codec_overrides(bimanual_entrypoint):
    run, _ = bimanual_entrypoint
    result, argv = run(
        "--display_data",
        "false",
        "--display_fps=7.5",
        "--display_compressed_images",
        "true",
        "--dataset.rgb_encoder.vcodec=h264",
        "--fps=20",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--display_data=false" in argv
    assert "--display_fps=7.5" in argv
    assert "--display_compressed_images=true" in argv
    assert "--dataset.rgb_encoder.vcodec=h264" in argv
    assert "--dataset.fps=20" in argv
    cameras = [arg for arg in argv if ".cameras=" in arg]
    assert len(cameras) == 2
    assert all("type: opencv" in arg and "fps: 30" in arg for arg in cameras)


@pytest.mark.parametrize("display_data", ["true", "false"])
def test_rollout_does_not_receive_record_only_flags(bimanual_entrypoint, display_data):
    run, checkpoint = bimanual_entrypoint
    result, argv = run(
        "--policy.path",
        str(checkpoint),
        "--display_data",
        display_data,
        "--display_fps",
        "12",
        "--fps",
        "20",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert argv[0] == "lerobot-rollout"
    assert f"--display_data={display_data}" in argv
    assert "--fps=30" in argv
    assert not any(arg.startswith("--dataset.fps=") for arg in argv)
    assert not any(arg.startswith(("--display_fps=", "--display_async=")) for arg in argv)
    assert ("原有同步路径" in result.stdout + result.stderr) == (display_data == "true")


@pytest.mark.parametrize(
    "args",
    [
        ("--display_data", "yes"),
        ("--display_compressed_images=falsee",),
        ("--display_fps", "0"),
        ("--display_fps", "0.0"),
        ("--display_fps", "-1"),
        ("--display_fps", "nan"),
        ("--display_fps", "inf"),
        ("--display_fps", "abc"),
        ("--dataset.rgb_encoder.vcodec=",),
        ("--display_data",),
        ("--display_fps",),
        ("--display_compressed_images",),
        ("--dataset.rgb_encoder.vcodec",),
        ("--unknown_preview_option", "true"),
    ],
)
def test_invalid_preview_options_do_not_start_recorder(bimanual_entrypoint, args):
    run, _ = bimanual_entrypoint
    result, argv = run(*args)
    assert result.returncode != 0
    assert "[错误]" in result.stdout + result.stderr
    assert argv == []
