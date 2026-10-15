"""Exercise the launcher with real checkpoint discovery and a stub training stack."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "RL_train.sh"


class ResumeLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "run"
        self.dataset = self.root / "dataset"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "meta/info.json").write_text("{}")
        self.initial = self.root / "initial"
        self.initial.mkdir()
        (self.initial / "config.json").write_text('{"type": "pi05"}')
        (self.initial / "model.safetensors").touch()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        python = bin_dir / "python"
        # Run stdlib-only metadata inspection for real. Stub only the expensive
        # LeRobot imports and training launch, recording the final CLI arguments.
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1] == '-':\n"
            "    source = sys.stdin.read()\n"
            "    if 'import lerobot' not in source and 'from lerobot' not in source:\n"
            "        sys.argv = sys.argv[1:]\n"
            "        exec(compile(source, '<launcher>', 'exec'))\n"
            "else:\n"
            "    Path(os.environ['RECORDED_CMD']).write_text(json.dumps(sys.argv[1:]))\n"
        )
        python.chmod(0o755)
        self.record = self.root / "command.json"
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", RECORDED_CMD=str(self.record))

    def checkpoint(self, name, step, complete=True):
        checkpoint = self.output / "train/checkpoints" / name
        model = checkpoint / "pretrained_model"
        state = checkpoint / "training_state"
        model.mkdir(parents=True)
        state.mkdir()
        (model / "config.json").write_text('{"type": "pi05"}')
        (model / "model.safetensors").touch()
        (model / "train_config.json").write_text('{"scheduler": {"type": "cosine"}}')
        (state / "training_step.json").write_text(json.dumps({"step": step}))
        if complete:
            for name in (
                "rng_state.safetensors", "optimizer_state.safetensors",
                "optimizer_param_groups.json", "scheduler_state.json",
            ):
                (state / name).touch()
        return model / "train_config.json"

    def run_launcher(self, *args):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--OUTPUT_DIR", str(self.output), "--DATASET_ROOT",
             str(self.dataset), "--steps", "100", *args],
            env=self.env, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_resume_highest_saved_step_ignoring_partial_and_stale_last(self):
        self.checkpoint("000020", 20)
        latest = self.checkpoint("000080", 80)
        self.checkpoint("000090", 90, complete=False)
        (self.output / "train/checkpoints/last").symlink_to("000020")
        self.run_launcher()
        command = json.loads(self.record.read_text())
        self.assertIn(f"--config_path={latest}", command)
        self.assertIn("--resume=true", command)
        self.assertIn("--steps=100", command)
        self.assertFalse(any(arg.startswith("--policy.path=") for arg in command))
        self.assertFalse(list(self.output.glob("train.incomplete.*")))

    def test_target_reached_skips_validation_and_launch(self):
        self.checkpoint("000100", 100)
        result = self.run_launcher("--DATASET_ROOT", "/missing/dataset")
        self.assertIn("已达到预定总步数", result.stdout)
        self.assertFalse(self.record.exists())

    def test_no_checkpoint_archives_old_directory_and_starts_at_zero(self):
        self.checkpoint("000090", 90, complete=False)
        self.run_launcher("--history_pretrained_path", str(self.initial))
        command = json.loads(self.record.read_text())
        self.assertIn(f"--policy.path={self.initial}", command)
        self.assertNotIn("--resume=true", command)
        archives = list(self.output.glob("train.incomplete.*/train/checkpoints/000090"))
        self.assertEqual(len(archives), 1)
        self.assertFalse((self.output / "train").exists())

    def test_fresh_run_uses_initial_policy(self):
        self.run_launcher("--history_pretrained_path", str(self.initial))
        self.assertIn(f"--policy.path={self.initial}", json.loads(self.record.read_text()))

    def test_dry_run_does_not_archive_or_launch(self):
        (self.output / "train").mkdir(parents=True)
        self.run_launcher("--history_pretrained_path", str(self.initial), "--dry_run")
        self.assertTrue((self.output / "train").is_dir())
        self.assertFalse(list(self.output.glob("train.incomplete.*")))
        self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
