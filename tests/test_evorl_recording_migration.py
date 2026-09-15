from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.datasets.dataset_writer import _validate_extra_episode_metadata
from lerobot.rollout.configs import EvoRLEpisodicStrategyConfig
from lerobot.rollout.strategies import evorl_episodic as evorl_module
from lerobot.rollout.strategies.evorl_episodic import EvoRLEpisodicStrategy
from lerobot.scripts import lerobot_migrate_evorl_dataset as migration_module, lerobot_record as record_module
from lerobot.scripts.lerobot_migrate_evorl_dataset import (
    _CURRENT_INTERVENTION,
    _OLD_COLLECTOR_POLICY_ID,
    _OLD_INTERVENTION,
    _OLD_POLICY_ACTION,
    _OLD_STATE,
    _convert_frame,
    _destination_features,
)
from lerobot.utils import keyboard_input
from lerobot.utils.recording_annotations import (
    EVORL_FAILURE_KEY,
    EVORL_INTERVENTION_KEY,
    EVORL_RERECORD_KEY,
    EVORL_RESET_KEY,
    EVORL_SUCCESS_KEY,
)


def test_realsense_legacy_aliases_map_to_current_fields():
    cfg = RealSenseCameraConfig(
        serial_number_or_name="123",
        exposure_mode="manual",
        manual_exposure_us=14000,
        manual_gain=16,
        white_balance_kelvin=3700,
    )
    assert cfg.exposure == 14000
    assert cfg.gain == 16
    assert cfg.white_balance == 3700


def test_realsense_rejects_conflicting_aliases():
    with pytest.raises(ValueError, match="Conflicting"):
        RealSenseCameraConfig(
            serial_number_or_name="123",
            exposure=120,
            manual_exposure_us=140,
        )


def test_episode_metadata_validation_is_scalar_and_reserves_writer_keys():
    assert _validate_extra_episode_metadata({"episode_success": "success"}) == {"episode_success": "success"}
    with pytest.raises(ValueError, match="reserved"):
        _validate_extra_episode_metadata({"data/chunk_index": 9})
    with pytest.raises(TypeError, match="parquet-safe scalar"):
        _validate_extra_episode_metadata({"episode_success": {"nested": True}})


def test_evorl_keyboard_controls(monkeypatch):
    captured = {}

    def fake_listener(dispatch, **_kwargs):
        captured["dispatch"] = dispatch
        return SimpleNamespace(stop=lambda: None)

    monkeypatch.setattr(keyboard_input, "create_key_listener", fake_listener)
    _listener, events = keyboard_input.init_keyboard_listener(
        intervention_toggle_key=EVORL_INTERVENTION_KEY,
        episode_success_key=EVORL_SUCCESS_KEY,
        episode_failure_key=EVORL_FAILURE_KEY,
        rerecord_episode_key=EVORL_RERECORD_KEY,
        reset_episode_key=EVORL_RESET_KEY,
    )
    captured["dispatch"]("C")
    assert events["toggle_intervention"] is True
    captured["dispatch"]("B")
    assert events["episode_outcome"] == "success"
    captured["dispatch"]("F")
    assert events["episode_outcome"] == "failure"
    events["rerecord_episode"] = False
    events["exit_early"] = False
    captured["dispatch"]("A")
    assert events["rerecord_episode"] is True
    assert events["reset_episode"] is False
    events["rerecord_episode"] = False
    events["exit_early"] = False
    captured["dispatch"]("Left")
    assert events["rerecord_episode"] is False
    captured["dispatch"]("R")
    assert events["reset_episode"] is True
    assert events["rerecord_episode"] is True
    captured["dispatch"]("Esc")
    assert events["stop_recording"] is True


def test_manual_and_vla_recording_share_hotkey_defaults():
    record_fields = record_module.RecordConfig.__dataclass_fields__
    rollout_cfg = EvoRLEpisodicStrategyConfig()

    assert record_fields["episode_success_key"].default == rollout_cfg.success_key == EVORL_SUCCESS_KEY
    assert record_fields["episode_failure_key"].default == rollout_cfg.failure_key == EVORL_FAILURE_KEY
    assert record_fields["rerecord_episode_key"].default == rollout_cfg.rerecord_key == EVORL_RERECORD_KEY
    assert (
        record_fields["intervention_toggle_key"].default
        == rollout_cfg.intervention_key
        == EVORL_INTERVENTION_KEY
    )
    assert record_fields["reset_episode_key"].default == rollout_cfg.reset_key == EVORL_RESET_KEY


@pytest.mark.parametrize("script_name", ["RL_data.sh", "RL_data_bimanual.sh"])
def test_recording_entrypoints_share_hotkeys_and_project_log_dir(script_name):
    script = (Path(__file__).resolve().parents[1] / "scripts" / script_name).read_text()

    assert 'LOG_DIR="${PROJECT_ROOT}/output/logs"' in script
    assert 'exec > >(tee -a "${LOG_FILE}") 2>&1' in script
    assert 'SUCCESS_KEY="b"' in script
    assert 'FAILURE_KEY="f"' in script
    assert 'INTERVENTION_KEY="c"' in script
    assert 'RERECORD_KEY="a"' in script
    assert 'RESET_KEY="r"' in script
    assert '"--episode_success_key=${SUCCESS_KEY}"' in script
    assert '"--rerecord_episode_key=${RERECORD_KEY}"' in script
    assert '"--strategy.success_key=${SUCCESS_KEY}"' in script
    assert '"--strategy.rerecord_key=${RERECORD_KEY}"' in script


def test_rl_data_vla_defaults_to_inference_only_without_can0():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "RL_data.sh").read_text()

    common_block = script.split("COMMON=(", 1)[1].split("TELEOP=(", 1)[0]
    assert 'VLA_CAN0_CONTROL="false"' in script
    assert "--can0.control|--vla.can0_control" in script
    assert "--teleop.port=can0" not in common_block
    assert "CMD+=(--strategy.enable_intervention=false)" in script
    assert '"${TELEOP[@]}"' in script
    assert "can0 不连接、不发送动作，C 已禁用" in script


def test_rl_data_rtc_uses_guided_backend_and_validates_queue_settings():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "RL_data.sh").read_text()

    assert '--rtc.enabled) RTC_ENABLED="$2"' in script
    assert '--rtc.execution_horizon) RTC_EXECUTION_HORIZON="$2"' in script
    assert '--rtc_action_queue_threshold|--rtc.queue_threshold)' in script
    assert '--inference.type=rtc' in script
    assert '--inference.rtc.mode=guided' in script
    assert '"--inference.rtc.execution_horizon=${RTC_EXECUTION_HORIZON}"' in script
    assert '"--inference.queue_threshold=${RTC_QUEUE_THRESHOLD}"' in script
    assert "--rtc.execution_horizon 必须是正整数" in script
    assert "--rtc_action_queue_threshold/--rtc.queue_threshold 必须是非负整数" in script


def test_rl_data_bimanual_vla_defaults_to_inference_only_without_leaders():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "RL_data_bimanual.sh").read_text()

    common_block = script.split("COMMON=(", 1)[1].split("TELEOP=(", 1)[0]
    assert 'VLA_CAN0_CONTROL="false"' in script
    assert "--can0.control|--vla.can0_control" in script
    assert "--teleop.type=bi_piper_leader" not in common_block
    assert '"${TELEOP[@]}"' in script
    assert "CMD+=(--strategy.enable_intervention=false)" in script
    assert "--strategy.enable_intervention=true" in script
    assert "can0/can2 不连接、不发送动作，C 已禁用" in script


def test_rl_data_bimanual_rtc_uses_guided_backend_and_validates_queue_settings():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "RL_data_bimanual.sh").read_text()

    assert '--rtc.enabled) RTC_ENABLED="$2"' in script
    assert '--rtc.execution_horizon) RTC_EXECUTION_HORIZON="$2"' in script
    assert '--rtc_action_queue_threshold|--rtc.queue_threshold)' in script
    assert '--inference.type=rtc' in script
    assert '--inference.rtc.mode=guided' in script
    assert '"--inference.rtc.execution_horizon=${RTC_EXECUTION_HORIZON}"' in script
    assert '"--inference.queue_threshold=${RTC_QUEUE_THRESHOLD}"' in script
    assert "--rtc.execution_horizon 必须是正整数" in script
    assert "--rtc_action_queue_threshold/--rtc.queue_threshold 必须是非负整数" in script
    assert "双臂 RTC 已启用" in script


def test_evorl_inference_only_does_not_require_teleop_or_register_c(monkeypatch):
    captured = {}

    def fake_listener(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stop=lambda: None), {}

    cfg = EvoRLEpisodicStrategyConfig(enable_intervention=False)
    strategy = EvoRLEpisodicStrategy(cfg)
    strategy._init_engine = lambda _ctx: None
    monkeypatch.setattr(evorl_module, "init_keyboard_listener", fake_listener)

    strategy.setup(SimpleNamespace(hardware=SimpleNamespace(teleop=None)))

    assert cfg.teleop_is_required() is False
    assert captured["intervention_toggle_key"] is None
    assert captured["episode_success_key"] == EVORL_SUCCESS_KEY
    assert captured["episode_failure_key"] == EVORL_FAILURE_KEY
    assert captured["rerecord_episode_key"] == EVORL_RERECORD_KEY
    assert captured["reset_episode_key"] == EVORL_RESET_KEY


def test_evorl_reset_without_can0_only_moves_follower(monkeypatch):
    calls = []
    robot = SimpleNamespace(
        get_observation=lambda: {
            "joint_1.pos": 3.0,
            "observation.images.top": np.zeros((2, 2, 3), dtype=np.uint8),
        }
    )
    processors = SimpleNamespace(
        teleop_action_processor=lambda pair: pair[0],
        robot_action_processor=lambda pair: pair[0],
    )
    ctx = SimpleNamespace(
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=None),
        processors=processors,
    )
    strategy = EvoRLEpisodicStrategy(EvoRLEpisodicStrategyConfig(enable_intervention=False))
    strategy.reset_control_state = lambda: calls.append(("reset",))
    monkeypatch.setattr(
        evorl_module,
        "follower_smooth_move_to",
        lambda robot_arg, current, target, *, duration_s: calls.append(
            (robot_arg, current, target, duration_s)
        ),
    )

    strategy._reset_to_home(ctx, duration_s=1.5)

    assert calls[0] == (robot, {"joint_1.pos": 3.0}, {"joint_1.pos": 0.0}, 1.5)
    assert calls[1] == ("reset",)


def test_evorl_reset_with_actuated_leader_keeps_it_enabled(monkeypatch):
    calls = []
    robot = SimpleNamespace(get_observation=lambda: {"joint_1.pos": 3.0})
    teleop = SimpleNamespace(
        feedback_features={"joint_1.pos": float},
        disable_torque=lambda: calls.append(("disable",)),
        enable_torque=lambda: calls.append(("enable",)),
        get_action=lambda: {"joint_1.pos": 3.0},
    )
    processors = SimpleNamespace(
        teleop_action_processor=lambda pair: pair[0],
        robot_action_processor=lambda pair: pair[0],
    )
    ctx = SimpleNamespace(
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=teleop),
        processors=processors,
    )
    strategy = EvoRLEpisodicStrategy(EvoRLEpisodicStrategyConfig())
    strategy.reset_control_state = lambda: calls.append(("reset",))
    monkeypatch.setattr(evorl_module, "follower_smooth_move_to", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evorl_module,
        "teleop_smooth_move_to",
        lambda teleop_arg, target, *, duration_s: calls.append(
            ("move_leader", teleop_arg, target, duration_s)
        ),
    )

    strategy._reset_to_home(ctx, duration_s=1.5)

    assert calls == [("move_leader", teleop, {"joint_1.pos": 0.0}, 1.5), ("reset",)]


def test_rollout_context_rollback_disconnects_devices_after_dataset_error():
    from lerobot.rollout.context import _RolloutBuildResources

    calls = []

    class BrokenDataset:
        def finalize(self):
            calls.append("dataset")
            raise RuntimeError("finalize failed")

    robot = SimpleNamespace(is_connected=True, disconnect=lambda: calls.append("robot"))
    teleop = SimpleNamespace(is_connected=True, disconnect=lambda: calls.append("teleop"))

    _RolloutBuildResources(robot=robot, teleop=teleop, dataset=BrokenDataset()).close()

    assert calls == ["dataset", "teleop", "robot"]


def test_rollout_hardware_cleanup_continues_after_engine_error():
    calls = []

    class BrokenEngine:
        def stop(self):
            calls.append("engine")
            raise RuntimeError("stop failed")

    robot = SimpleNamespace(is_connected=True, disconnect=lambda: calls.append("robot"))
    teleop = SimpleNamespace(is_connected=True, disconnect=lambda: calls.append("teleop"))
    strategy = EvoRLEpisodicStrategy(EvoRLEpisodicStrategyConfig())
    strategy._engine = BrokenEngine()
    hardware = SimpleNamespace(
        robot_wrapper=SimpleNamespace(inner=robot),
        teleop=teleop,
        initial_position=None,
    )

    with pytest.raises(RuntimeError, match="stop failed"):
        strategy._teardown_hardware(hardware, return_to_initial_position=False)

    assert calls == ["engine", "robot", "teleop"]


def test_evorl_strategy_declares_legacy_compatible_features():
    cfg = EvoRLEpisodicStrategyConfig()
    action_feature = {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}
    features = cfg.build_extra_dataset_features({"action": action_feature})

    assert set(features) == {
        _OLD_POLICY_ACTION,
        _OLD_INTERVENTION,
        _OLD_STATE,
        _OLD_COLLECTOR_POLICY_ID,
    }
    assert features[_OLD_POLICY_ACTION] == action_feature
    assert features[_OLD_INTERVENTION]["dtype"] == "float32"
    assert features[_OLD_INTERVENTION]["names"] == ["is_intervention"]
    assert features[_OLD_STATE]["dtype"] == "float32"
    assert features[_OLD_COLLECTOR_POLICY_ID]["dtype"] == "string"


def test_converter_maps_legacy_intervention_to_current():
    source = SimpleNamespace(
        meta=SimpleNamespace(
            features={
                "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
                _OLD_INTERVENTION: {"dtype": "float32", "shape": (1,), "names": None},
                _OLD_STATE: {"dtype": "float32", "shape": (1,), "names": None},
            }
        )
    )
    features = _destination_features(source, "to-current")
    frame = _convert_frame(
        {
            "action": np.array([1.0, 2.0], dtype=np.float32),
            _OLD_INTERVENTION: np.array([1.0], dtype=np.float32),
            "task": "pick",
        },
        features,
        "to-current",
        "fallback",
    )
    assert _OLD_INTERVENTION not in features
    assert frame[_CURRENT_INTERVENTION].tolist() == [True]
    assert frame["task"] == "pick"


def test_converter_loads_v3_dataset_without_repo_id(tmp_path, monkeypatch):
    source_root = tmp_path / "legacy_dataset"
    (source_root / "meta").mkdir(parents=True)
    (source_root / "meta" / "info.json").write_text('{"codebase_version": "v3.0"}')
    captured = {}

    def fake_dataset(*, repo_id, root):
        captured.update(repo_id=repo_id, root=root)
        return "dataset"

    monkeypatch.setattr(migration_module, "LeRobotDataset", fake_dataset)

    assert migration_module._load_source(source_root) == "dataset"
    assert captured == {"repo_id": "local/legacy_dataset", "root": source_root}


def test_converter_maps_current_intervention_to_legacy():
    source = SimpleNamespace(
        meta=SimpleNamespace(
            features={
                "action": {"dtype": "float32", "shape": (1,), "names": ["a"]},
                _CURRENT_INTERVENTION: {"dtype": "bool", "shape": (1,), "names": None},
            }
        )
    )
    features = _destination_features(source, "to-canonical")
    frame = _convert_frame(
        {
            "action": np.array([1.0], dtype=np.float32),
            _CURRENT_INTERVENTION: np.array([True]),
        },
        features,
        "to-canonical",
        "pick",
    )
    assert frame[_OLD_INTERVENTION].tolist() == [1.0]
    assert frame[_OLD_STATE].tolist() == [1.0]
    assert frame[_OLD_POLICY_ACTION].tolist() == [0.0]
    assert frame[_OLD_COLLECTOR_POLICY_ID] == "human"


def test_converter_canonicalizes_pure_manual_data():
    source = SimpleNamespace(
        meta=SimpleNamespace(features={"action": {"dtype": "float32", "shape": (1,), "names": ["a"]}})
    )
    features = _destination_features(source, "to-canonical")
    frame = _convert_frame(
        {"action": np.array([2.0], dtype=np.float32)},
        features,
        "to-canonical",
        "pick",
    )

    assert frame[_OLD_POLICY_ACTION].tolist() == [0.0]
    assert frame[_OLD_INTERVENTION].tolist() == [0.0]
    assert frame[_OLD_STATE].tolist() == [0.0]
    assert frame[_OLD_COLLECTOR_POLICY_ID] == "human"


def test_converter_canonicalizes_non_intervention_policy_data():
    source = SimpleNamespace(
        meta=SimpleNamespace(
            features={
                "action": {"dtype": "float32", "shape": (1,), "names": ["a"]},
                _CURRENT_INTERVENTION: {"dtype": "bool", "shape": (1,), "names": None},
            }
        )
    )
    features = _destination_features(source, "to-canonical")
    frame = _convert_frame(
        {
            "action": np.array([2.0], dtype=np.float32),
            _CURRENT_INTERVENTION: np.array([False]),
        },
        features,
        "to-canonical",
        "pick",
        collector_policy_id="pi05-checkpoint",
    )

    assert frame[_OLD_POLICY_ACTION].tolist() == [2.0]
    assert frame[_OLD_INTERVENTION].tolist() == [0.0]
    assert frame[_OLD_STATE].tolist() == [0.0]
    assert frame[_OLD_COLLECTOR_POLICY_ID] == "pi05-checkpoint"


def test_dataset_wrapper_forwards_episode_metadata():
    from lerobot.datasets import LeRobotDataset

    calls = []
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.writer = SimpleNamespace(
        save_episode=lambda episode_data, parallel_encoding, *, episode_metadata: calls.append(
            (episode_data, parallel_encoding, episode_metadata)
        )
    )
    dataset._require_writer = lambda _method: None
    dataset.save_episode(episode_metadata={"episode_success": "failure"})
    assert calls == [(None, True, {"episode_success": "failure"})]


def test_piper_configs_are_registered_and_bimanual_feedback_is_prefixed():
    from lerobot.robots import RobotConfig
    from lerobot.teleoperators import TeleoperatorConfig
    from lerobot.teleoperators.bi_piper_leader import BiPiperLeader

    assert "piper_follower" in RobotConfig.get_known_choices()
    assert "bi_piper_follower" in RobotConfig.get_known_choices()
    assert "piper_leader" in TeleoperatorConfig.get_known_choices()
    assert "bi_piper_leader" in TeleoperatorConfig.get_known_choices()

    leader = BiPiperLeader.__new__(BiPiperLeader)
    leader.left_arm = SimpleNamespace(feedback_features={"joint_1.pos": float})
    leader.right_arm = SimpleNamespace(feedback_features={"joint_1.pos": float})
    assert leader.feedback_features == {
        "left_joint_1.pos": float,
        "right_joint_1.pos": float,
    }


def test_manual_record_rerun_telemetry_includes_intervention(monkeypatch):
    captured = {}

    def fake_log(display_mode, **kwargs):
        captured["display_mode"] = display_mode
        captured.update(kwargs)

    monkeypatch.setattr(record_module, "log_visualization_data", fake_log)
    record_module._log_record_telemetry(
        "rerun",
        observation={"joint.pos": 1.0},
        action={"joint.pos": 2.0},
        compress_images=True,
    )

    assert captured["display_mode"] == "rerun"
    assert captured["observation"] == {"joint.pos": 1.0, "intervention": True}
    assert captured["action"] == {"joint.pos": 2.0}
    assert captured["compress_images"] is True


def test_evorl_rerun_telemetry_includes_live_intervention_state():
    strategy = EvoRLEpisodicStrategy(EvoRLEpisodicStrategyConfig())
    captured = {}
    strategy._log_telemetry = lambda observation, action, runtime: captured.update(
        observation=observation, action=action, runtime=runtime
    )
    runtime = SimpleNamespace(cfg=SimpleNamespace(display_data=True, display_mode="rerun"))

    strategy._log_evorl_telemetry(
        {"joint.pos": 1.0},
        {"joint.pos": 2.0},
        runtime,
        intervening=False,
    )

    assert captured["observation"] == {"joint.pos": 1.0, "intervention": False}
    assert captured["action"] == {"joint.pos": 2.0}
    assert captured["runtime"] is runtime
