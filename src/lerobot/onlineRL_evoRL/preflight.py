"""Static preflight checks for the paired EvoRL actor/learner configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

POLICY_KEYS = (
    "type",
    "input_features",
    "output_features",
    "chunk_size",
    "n_action_steps",
    "action_feature_names",
    "rlt_input_dim",
    "rlt_embed_dim",
    "rlt_num_rl_tokens",
    "rlt_prefix_seq_len",
    "rlt_num_layers",
    "rlt_num_heads",
    "rlt_image_only",
    "rlt_use_mask",
)
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


class PreflightError(ValueError):
    """Raised when a config pair cannot run as one online-RL job."""


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise PreflightError(f"JSON file does not exist: {resolved}")
    try:
        value = json.loads(resolved.read_text())
    except json.JSONDecodeError as exc:
        raise PreflightError(f"Invalid JSON in {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"Expected a JSON object in {resolved}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _shape(feature: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in feature.get("shape", ()))


def _validate_model(config: dict[str, Any], label: str) -> Path:
    policy = config.get("policy") or {}
    _require(policy.get("type") == "pi05_rlt", f"{label}: policy.type must be pi05_rlt")
    path_value = policy.get("pretrained_path")
    _require(bool(path_value), f"{label}: policy.pretrained_path is required")
    model_dir = Path(path_value).expanduser().resolve()
    _require(model_dir.is_dir(), f"{label}: model directory does not exist: {model_dir}")
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    _require(not missing, f"{label}: model directory is missing {missing}")

    saved = _load_json(model_dir / "config.json")
    for key in POLICY_KEYS:
        if key in policy:
            _require(
                saved.get(key) == policy.get(key),
                f"{label}: policy.{key} differs from {model_dir / 'config.json'}",
            )
    _require(saved.get("type") == "pi05_rlt", f"{label}: checkpoint is not a pi05_rlt model")
    return model_dir


def _normalized_dataset_shape(feature: dict[str, Any], policy_feature: dict[str, Any]) -> tuple[int, ...]:
    shape = _shape(feature)
    if policy_feature.get("type") == "VISUAL" and len(shape) == 3:
        return (shape[2], shape[0], shape[1])
    return shape


def _validate_dataset(config: dict[str, Any], label: str = "actor") -> Path:
    dataset = config.get("dataset") or {}
    root_value = dataset.get("root")
    _require(bool(root_value), f"{label}: dataset.root is required")
    root = Path(root_value).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    _require(info_path.is_file(), f"{label}: dataset metadata is missing: {info_path}")
    _require(tasks_path.is_file(), f"{label}: task metadata is missing: {tasks_path}")
    info = _load_json(info_path)
    _require(
        info.get("fps") == (config.get("env") or {}).get("fps"),
        f"{label}: dataset/env fps mismatch",
    )

    dataset_features = info.get("features") or {}
    policy = config.get("policy") or {}
    configured_features = {
        **(policy.get("input_features") or {}),
        **(policy.get("output_features") or {}),
    }
    for key, feature in configured_features.items():
        _require(key in dataset_features, f"{label}: dataset is missing feature {key}")
        actual_shape = _normalized_dataset_shape(dataset_features[key], feature)
        _require(
            actual_shape == _shape(feature),
            f"{label}: {key} shape mismatch: dataset={actual_shape}, policy={_shape(feature)}",
        )

    dataset_action_names = (dataset_features.get("action") or {}).get("names")
    configured_action_names = policy.get("action_feature_names")
    _require(
        dataset_action_names == configured_action_names,
        f"{label}: action_feature_names do not match dataset metadata",
    )
    return root


def _validate_algorithm(config: dict[str, Any], label: str) -> None:
    algorithm = config.get("algorithm") or {}
    _require(algorithm.get("type") == "rlt_chunk", f"{label}: algorithm.type must be rlt_chunk")
    _require(config.get("online_ratio", 0.5) == 1.0, f"{label}: online_ratio must be 1.0")
    _require(
        int(algorithm.get("online_step_before_learning", 0)) > 0,
        f"{label}: online_step_before_learning must be positive",
    )
    _require(
        int(algorithm.get("online_buffer_capacity", 0))
        >= int(algorithm.get("online_step_before_learning", 0)),
        f"{label}: online_buffer_capacity must cover the warmup",
    )
    transport = algorithm.get("actor_learner_config") or {}
    _require(bool(transport.get("learner_host")), f"{label}: learner_host is required")
    port = int(transport.get("learner_port", 0))
    _require(0 < port < 65536, f"{label}: invalid learner_port {port}")


def _validate_handoff(config: dict[str, Any], label: str) -> None:
    handoff = config.get("gpu_handoff")
    if handoff is None:
        return
    _require(isinstance(handoff.get("enabled"), bool), f"{label}: gpu_handoff.enabled is required")
    threshold = int(handoff.get("update_quota_threshold", 0))
    updates_per_episode = int(handoff.get("updates_per_episode", 0))
    _require(threshold > 0, f"{label}: gpu_handoff threshold must be positive")
    _require(updates_per_episode > 0, f"{label}: updates_per_episode must be positive")
    _require(
        threshold % updates_per_episode == 0,
        f"{label}: gpu_handoff threshold must be divisible by updates_per_episode",
    )


def _validate_learner(learner: dict[str, Any]) -> Path:
    offline = learner.get("offline_pretraining") or {}
    _require(learner.get("dataset") is None, "learner: dataset must be null")
    if offline.get("enabled"):
        compact_value = offline.get("compact_dataset_path")
        _require(
            bool(compact_value),
            "learner: offline pretraining requires compact_dataset_path",
        )
        compact_root = Path(compact_value).expanduser().resolve()
        manifest_path = compact_root / "manifest.json"
        _require(
            manifest_path.is_file(),
            f"learner: compact manifest is missing: {manifest_path}",
        )
        manifest = _load_json(manifest_path)
        expected_episodes = int(manifest.get("expected_episodes", 0))
        completed_episodes = int(manifest.get("completed_episodes", 0))
        _require(
            expected_episodes > 0 and completed_episodes == expected_episodes,
            "learner: compact extraction is incomplete: "
            f"completed={completed_episodes}, expected={expected_episodes}",
        )
        compact_files = list(compact_root.glob("episode_*/compact_episode.pt"))
        _require(
            len(compact_files) == completed_episodes,
            "learner: compact episode file count differs from manifest: "
            f"files={len(compact_files)}, manifest={completed_episodes}",
        )
        _require(int(offline.get("steps", 0)) > 0, "learner: offline steps must be positive")
        _require(
            int(offline.get("feature_batch_size", 0)) > 0,
            "learner: offline feature_batch_size must be positive",
        )
        _require(
            int(offline.get("sliding_window_stride", 0)) > 0,
            "learner: offline sliding_window_stride must be positive",
        )
    _validate_algorithm(learner, "learner")
    _validate_handoff(learner, "learner")
    return _validate_model(learner, "learner")


def _validate_actor(actor: dict[str, Any]) -> tuple[Path, Path]:
    _require(actor.get("actor_mode") == "online_actor", "actor: actor_mode must be online_actor")
    _require(actor.get("save_format") == "transition", "actor: save_format must be transition")
    online_transition = actor.get("online_transition") or {}
    _require(
        bool(online_transition.get("enabled")),
        "actor: online_transition.enabled must be true",
    )
    _require(
        bool(online_transition.get("save_local_copy")),
        "actor: online_transition.save_local_copy must be true",
    )
    compact_dir = online_transition.get("episode_output_dir")
    _require(bool(compact_dir), "actor: online_transition.episode_output_dir is required")
    if online_transition.get("save_lerobot_copy"):
        lerobot_dir = online_transition.get("lerobot_output_dir")
        _require(bool(lerobot_dir), "actor: online_transition.lerobot_output_dir is required")
        _require(
            Path(compact_dir).expanduser().resolve() != Path(lerobot_dir).expanduser().resolve(),
            "actor: compact and LeRobot output directories must differ",
        )
    _require(
        bool((actor.get("actor_vla_policy") or {}).get("enabled")),
        "actor: actor_vla_policy.enabled must be true",
    )
    _validate_algorithm(actor, "actor")
    _validate_handoff(actor, "actor")
    model_dir = _validate_model(actor, "actor")
    dataset_root = _validate_dataset(actor, "actor")

    env = actor.get("env") or {}
    robot = env.get("robot") or {}
    teleop = env.get("teleop") or {}
    _require(robot.get("type") == "bi_piper_follower", "actor: robot.type must be bi_piper_follower")
    _require(teleop.get("type") == "bi_piper_leader", "actor: teleop.type must be bi_piper_leader")
    ports = (
        (robot.get("left_arm_config") or {}).get("port"),
        (robot.get("right_arm_config") or {}).get("port"),
        (teleop.get("left_arm_config") or {}).get("port"),
        (teleop.get("right_arm_config") or {}).get("port"),
    )
    _require(all(ports), "actor: all four Piper CAN ports must be configured")
    _require(len(set(ports)) == 4, f"actor: Piper CAN ports must be unique, got {ports}")

    chunk_size = int((actor.get("policy") or {}).get("chunk_size", 0))
    rtc_horizon = int((actor.get("rtc") or {}).get("execution_horizon", 0))
    _require(0 < rtc_horizon <= chunk_size, "actor: RTC execution_horizon must be in [1, chunk_size]")
    return model_dir, dataset_root


def _validate_pair(learner: dict[str, Any], actor: dict[str, Any]) -> None:
    for key in POLICY_KEYS:
        _require(
            (learner.get("policy") or {}).get(key) == (actor.get("policy") or {}).get(key),
            f"actor/learner policy mismatch at {key}",
        )
    _require(learner.get("algorithm") == actor.get("algorithm"), "actor/learner algorithm configs differ")
    learner_handoff = learner.get("gpu_handoff") or {}
    actor_handoff = actor.get("gpu_handoff") or {}
    for key in ("enabled", "update_quota_threshold", "updates_per_episode"):
        _require(
            learner_handoff.get(key) == actor_handoff.get(key),
            f"actor/learner gpu_handoff mismatch at {key}",
        )
    _require(
        (learner.get("env") or {}).get("task") == (actor.get("env") or {}).get("task"),
        "actor/learner task prompts differ",
    )
    _require(
        (learner.get("env") or {}).get("fps") == (actor.get("env") or {}).get("fps"),
        "actor/learner fps differs",
    )
    learner_model = Path((learner.get("policy") or {}).get("pretrained_path", "")).expanduser().resolve()
    actor_model = Path((actor.get("policy") or {}).get("pretrained_path", "")).expanduser().resolve()
    actor_vla_model = (
        Path((actor.get("actor_vla_policy") or {}).get("policy_path", "")).expanduser().resolve()
    )
    _require(
        learner_model == actor_model == actor_vla_model,
        "actor policy, actor_vla_policy and learner must use the same pretrained model",
    )
    _require(
        Path(actor.get("actor_checkpoint_path", "")).expanduser().resolve()
        == Path(learner.get("output_dir", "")).expanduser().resolve(),
        "actor_checkpoint_path must match learner output_dir",
    )


def run_preflight(
    learner_config: str | Path,
    actor_config: str | Path,
    mode: str = "both",
) -> dict[str, Path]:
    """Validate local artifacts and the selected actor/learner config pair."""
    learner = _load_json(learner_config)
    actor = _load_json(actor_config)
    result: dict[str, Path] = {}
    if mode in {"learner", "both"}:
        result["model"] = _validate_learner(learner)
    if mode in {"actor", "both"}:
        actor_model, dataset = _validate_actor(actor)
        result["model"] = actor_model
        result["dataset"] = dataset
    # Every launcher invocation receives both JSON paths. Validate the pair even
    # when starting one side so independently launched processes cannot disagree.
    _validate_pair(learner, actor)
    return result


def main() -> None:
    """Run the command-line preflight checker."""
    parser = argparse.ArgumentParser(description="Validate EvoRL actor/learner configs without hardware")
    parser.add_argument("--learner-config", required=True)
    parser.add_argument("--actor-config", required=True)
    parser.add_argument("--mode", choices=("learner", "actor", "both"), default="both")
    args = parser.parse_args()
    checked = run_preflight(args.learner_config, args.actor_config, args.mode)
    details = ", ".join(f"{key}={value}" for key, value in sorted(checked.items()))
    print(f"[onlineRL_evoRL preflight] OK ({args.mode}): {details}")


if __name__ == "__main__":
    main()
