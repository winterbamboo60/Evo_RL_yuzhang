from pathlib import Path

import draccus
import pytest

from lerobot.envs import env_to_policy_features
from lerobot.onlineRL_evoRL.actor_new import ActorPipelineConfig
from lerobot.rl.train_rl import TrainRLServerPipelineConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "src/lerobot/onlineRL_evoRL/configs"
EXPERIMENTS = (
    "pi05_base_cup_catch_v2_0819_35k",
    "pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k",
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
def test_migrated_learner_configs_are_online_only(experiment, tmp_path):
    path = CONFIG_ROOT / "learner" / f"Leanrer_onlineRL_transition_{experiment}.json"
    cfg = draccus.parse(TrainRLServerPipelineConfig, path, args=[])
    cfg.output_dir = tmp_path / f"learner_{experiment}"

    cfg.validate()

    assert cfg.dataset is None
    assert cfg.policy.type == "pi05_rlt"
    assert cfg.algorithm.type == "rlt_chunk"
    assert cfg.algorithm.policy_config is cfg.policy
    assert cfg.online_ratio == 1.0
