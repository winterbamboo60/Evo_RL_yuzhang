from pathlib import Path

import draccus
import pytest

import lerobot.onlineRL_evoRL.learner as learner_module
from lerobot.envs import env_to_policy_features
from lerobot.onlineRL_evoRL.actor_new import ActorPipelineConfig
from lerobot.onlineRL_evoRL.config import (
    ActorGPUHandoffConfig,
    LearnerGPUHandoffConfig,
    LearnerPipelineConfig,
)
from lerobot.onlineRL_evoRL.learner import _validate_startup_config
from lerobot.onlineRL_evoRL.preflight import PreflightError, _validate_pair

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "src/lerobot/onlineRL_evoRL/configs"
EXPERIMENTS = (
    "pi05_base_cup_catch_v2_0819_35k",
    "pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k",
    "pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k",
)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_migrated_actor_configs_use_current_schema(experiment, tmp_path):
    path = CONFIG_ROOT / "actor" / f"Actor_onlineRL_transition_{experiment}.json"
    cfg = draccus.parse(ActorPipelineConfig, path, args=[])
    cfg.output_dir = tmp_path / f"actor_{experiment}"

    cfg.validate()

    env_features = env_to_policy_features(cfg.env)
    assert cfg.actor_mode == "online_actor"
    assert cfg.save_format == "transition"
    assert cfg.policy.type == "pi05_rlt"
    assert cfg.algorithm.type == "rlt_chunk"
    assert set(env_features) == set(cfg.policy.input_features) | set(cfg.policy.output_features)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_migrated_learner_configs_use_optional_offline_startup(experiment, tmp_path):
    path = CONFIG_ROOT / "learner" / f"Leanrer_onlineRL_transition_{experiment}.json"
    cfg = draccus.parse(LearnerPipelineConfig, path, args=[])
    cfg.output_dir = tmp_path / f"learner_{experiment}"

    cfg.validate()

    if cfg.offline_pretraining.enabled:
        assert cfg.dataset is None
        assert cfg.offline_pretraining.steps > 0
        assert cfg.offline_pretraining.compact_dataset_path
    else:
        assert cfg.dataset is None
    assert cfg.policy.type == "pi05_rlt"
    assert cfg.algorithm.type == "rlt_chunk"
    assert cfg.algorithm.policy_config is cfg.policy
    assert cfg.online_ratio == 1.0


def test_learner_cli_parses_config_path_with_dataclass_annotation(monkeypatch):
    experiment = "pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k"
    path = CONFIG_ROOT / "learner" / f"Leanrer_onlineRL_transition_{experiment}.json"
    received = {}

    monkeypatch.setattr("sys.argv", ["learner", "--config_path", str(path)])
    monkeypatch.setattr(learner_module, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(learner_module, "use_threads", lambda cfg: True)
    monkeypatch.setattr(
        learner_module,
        "train",
        lambda cfg, job_name=None: received.update(cfg=cfg, job_name=job_name),
    )

    learner_module.train_cli()

    assert isinstance(received["cfg"], LearnerPipelineConfig)
    assert received["job_name"] == received["cfg"].job_name


def test_dual_arm_handoff_configs_use_the_same_dynamic_threshold():
    experiment = "pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k"
    actor = draccus.parse(
        ActorPipelineConfig,
        CONFIG_ROOT / "actor" / f"Actor_onlineRL_transition_{experiment}.json",
        args=[],
    )
    learner = draccus.parse(
        LearnerPipelineConfig,
        CONFIG_ROOT / "learner" / f"Leanrer_onlineRL_transition_{experiment}.json",
        args=[],
    )

    assert actor.gpu_handoff.update_quota_threshold == learner.gpu_handoff.update_quota_threshold
    assert actor.gpu_handoff.updates_per_episode == learner.gpu_handoff.updates_per_episode
    assert (
        actor.gpu_handoff.update_quota_threshold % actor.gpu_handoff.updates_per_episode == 0
    )


def test_handoff_threshold_is_configurable_instead_of_fixed_to_200():
    actor = ActorGPUHandoffConfig(update_quota_threshold=120, updates_per_episode=4)
    learner = LearnerGPUHandoffConfig(update_quota_threshold=120, updates_per_episode=4)

    assert actor.update_quota_threshold == learner.update_quota_threshold == 120


def test_preflight_rejects_only_a_cross_process_threshold_mismatch():
    learner = {
        "policy": {},
        "algorithm": {},
        "gpu_handoff": {
            "enabled": True,
            "update_quota_threshold": 120,
            "updates_per_episode": 4,
        },
    }
    actor = {
        "policy": {},
        "algorithm": {},
        "gpu_handoff": {
            "enabled": True,
            "update_quota_threshold": 200,
            "updates_per_episode": 4,
        },
    }

    with pytest.raises(PreflightError, match="update_quota_threshold"):
        _validate_pair(learner, actor)


def test_learner_allows_logs_only_restart_but_protects_existing_checkpoint(tmp_path):
    experiment = "pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k"
    cfg = draccus.parse(
        LearnerPipelineConfig,
        CONFIG_ROOT / "learner" / f"Leanrer_onlineRL_transition_{experiment}.json",
        args=[],
    )
    cfg.output_dir = tmp_path / "learner"
    (cfg.output_dir / "logs").mkdir(parents=True)

    _validate_startup_config(cfg)

    (cfg.output_dir / "checkpoints" / "last").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="resume=true"):
        _validate_startup_config(cfg)


def test_dual_arm_actor_saves_compact_and_lerobot_copies(tmp_path):
    experiment = "pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k"
    path = CONFIG_ROOT / "actor" / f"Actor_onlineRL_transition_{experiment}.json"
    cfg = draccus.parse(ActorPipelineConfig, path, args=[])
    cfg.output_dir = tmp_path / "actor"

    cfg.validate()

    assert cfg.online_transition.save_local_copy is True
    assert cfg.online_transition.save_lerobot_copy is True
    assert cfg.online_transition.lerobot_output_dir

    cfg.online_transition.save_local_copy = False
    with pytest.raises(ValueError, match="save_local_copy=true"):
        cfg.validate()
