#!/usr/bin/env python3
"""Read-only smoke checks for an unpacked Evo 0911 environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


CORE_MODULES = (
    "accelerate",
    "av",
    "cv2",
    "datasets",
    "grpc",
    "lerobot",
    "numpy",
    "piper_sdk",
    "pyrealsense2",
    "torch",
    "torchvision",
    "transformers",
)

CORE_DISTRIBUTIONS = (
    "accelerate",
    "av",
    "datasets",
    "grpcio",
    "lerobot",
    "numpy",
    "opencv-python-headless",
    "piper_sdk",
    "pyrealsense2",
    "torch",
    "torchvision",
    "transformers",
)

REQUIRED_COMMANDS = (
    "lerobot-find-cameras",
    "lerobot-record",
    "lerobot-setup-can",
    "lerobot-train",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument(
        "--expect-lerobot-root",
        type=Path,
        help="Fail unless lerobot.__file__ is below this path.",
    )
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def is_below(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def main() -> int:
    args = parse_args()
    env_root = args.env_root.resolve()

    if Path(sys.prefix).resolve() != env_root:
        raise RuntimeError(f"sys.prefix={sys.prefix!r}, expected {str(env_root)!r}")

    imported = {}
    for module_name in CORE_MODULES:
        imported[module_name] = importlib.import_module(module_name)

    versions = {
        name: importlib.metadata.version(name) for name in CORE_DISTRIBUTIONS
    }
    lerobot_file = Path(imported["lerobot"].__file__).resolve()
    if args.expect_lerobot_root and not is_below(
        lerobot_file, args.expect_lerobot_root
    ):
        raise RuntimeError(
            f"lerobot loaded from {lerobot_file}, expected it below "
            f"{args.expect_lerobot_root.resolve()}"
        )

    missing_commands = [
        command
        for command in REQUIRED_COMMANDS
        if not (env_root / "bin" / command).is_file()
    ]
    if missing_commands:
        raise RuntimeError(f"missing LeRobot commands: {', '.join(missing_commands)}")

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if pip_check.returncode:
        raise RuntimeError(
            "pip check failed:\n" + (pip_check.stdout + pip_check.stderr).strip()
        )

    torch = imported["torch"]
    cuda_available = bool(torch.cuda.is_available())
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")

    result = {
        "status": "ok",
        "env_root": str(env_root),
        "python": sys.version.split()[0],
        "lerobot_file": str(lerobot_file),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "versions": versions,
    }
    if cuda_available:
        result["cuda_device"] = torch.cuda.get_device_name(0)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
