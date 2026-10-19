"""Keep preview startup independent of the recording and training import graph."""

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_python(*args, extra_paths=()):
    env = dict(os.environ)
    paths = [str(path) for path in (*extra_paths, ROOT / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    result = subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_record_cli_does_not_load_recording_or_torch():
    _run_python(
        "-c",
        textwrap.dedent(
            """
            import sys

            class RejectRecordingImports:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in ('torch', 'lerobot.scripts.lerobot_record'):
                        raise AssertionError('CLI import loaded ' + fullname)
                    return None

            sys.meta_path.insert(0, RejectRecordingImports())
            from lerobot.scripts.lerobot_record_cli import main

            assert callable(main)
            assert 'lerobot.scripts.lerobot_record' not in sys.modules
            assert 'torch' not in sys.modules
            """
        ),
    )


def test_console_entrypoint_spawn_does_not_repeat_recording_import(tmp_path):
    # A console-script launcher imports its target before the __main__ guard.
    # Spawn reruns that launcher in the child. Only the parent's call to main()
    # may enter the expensive recording module; the worker needs neither it nor
    # any SDK. Report the child's import graph through the backend's init call.
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    module_name, function_name = config["project"]["scripts"]["lerobot-record"].split(":")
    helper = tmp_path / "preview_startup_helper.py"
    helper.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys
            from pathlib import Path

            class Backend:
                def __init__(self, report_path):
                    self.report_path = report_path

                def __call__(self):
                    return self

                def init_visualization(self, *args, **kwargs):
                    Path(self.report_path).write_text(json.dumps({
                        'pid': os.getpid(),
                        'recording_loaded': 'lerobot.scripts.lerobot_record' in sys.modules,
                        'processor_loaded': 'lerobot.processor' in sys.modules,
                        'configs_loaded': 'lerobot.configs' in sys.modules,
                    }))

                def log_visualization_data(self, *args, **kwargs):
                    pass

                def shutdown_visualization(self, *args, **kwargs):
                    pass
            """
        )
    )
    shim = tmp_path / "lerobot-record"
    shim.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys
            from pathlib import Path
            from types import ModuleType

            class RejectRecordingImports:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in ('lerobot.configs', 'lerobot.processor',
                                    'lerobot.scripts.lerobot_record'):
                        raise AssertionError('Spawn launcher imported ' + fullname)
                    return None

            sys.meta_path.insert(0, RejectRecordingImports())
            ENTRYPOINT_IMPORT

            if __name__ == '__main__':
                # Substitute just the recording entrypoint, retaining its real
                # lazy CLI import and the real spawn/shared-memory sender.
                record = ModuleType('lerobot.scripts.lerobot_record')

                def recording_main():
                    from lerobot.utils.async_visualization import AsyncVisualization
                    from preview_startup_helper import Backend

                    report_path = sys.argv[1]
                    preview = AsyncVisualization('rerun', _backend_factory=Backend(report_path))
                    try:
                        preview.start()
                        report = json.loads(Path(report_path).read_text())
                        assert report['pid'] != os.getpid()
                        assert report['recording_loaded'] is False
                        assert report['processor_loaded'] is False
                        assert report['configs_loaded'] is False
                    finally:
                        preview.close()

                record.main = recording_main
                sys.modules[record.__name__] = record
                main()
            """
        ).replace("ENTRYPOINT_IMPORT", f"from {module_name} import {function_name} as main")
    )
    _run_python(str(shim), str(tmp_path / "worker.json"), extra_paths=(tmp_path,))


def test_preview_backend_import_does_not_load_recording_or_training_dependencies():
    _run_python(
        "-c",
        textwrap.dedent(
            """
            import sys

            class RejectHeavyImports:
                def find_spec(self, fullname, path=None, target=None):
                    blocked = ('lerobot.configs', 'lerobot.processor', 'lerobot.lerobot_types',
                               'lerobot.robots', 'lerobot.teleoperators',
                               'lerobot.scripts.lerobot_record')
                    if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
                        raise AssertionError('Preview startup imported ' + fullname)
                    return None

            sys.meta_path.insert(0, RejectHeavyImports())
            from lerobot.utils.async_visualization import _load_backend

            backend = _load_backend()
            assert callable(backend.init_visualization)
            assert callable(backend.log_visualization_data)
            assert callable(backend.shutdown_visualization)
            """
        ),
    )
