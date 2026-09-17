"""Standalone, resumable multi-GPU export of Actor-compatible PI05-RLT features."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import draccus
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lerobot.configs.default import DatasetConfig
from lerobot.datasets import LeRobotDatasetMetadata, make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_batch_keys
from lerobot.utils.constants import ACTION

from .compact_transition import (
    SCHEMA_NAME,
    SLIDING_WINDOW_TRANSITIONS,
    make_compact_episode,
    save_compact_episode,
    validate_compact_episode,
)
from .config import LearnerPipelineConfig
from .offline_pretraining import (
    COMPACT_EPISODE_NAME,
    COMPACT_MANIFEST_NAME,
    OfflineFrame,
    build_offline_compact_transitions,
)


@dataclass(frozen=True)
class DistributedContext:
    """torchrun rank information used for episode sharding and device choice."""

    rank: int
    local_rank: int
    world_size: int


def _distributed_context() -> DistributedContext:
    return DistributedContext(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )


def partition_episodes(episodes: list[int], rank: int, world_size: int) -> list[int]:
    """Assign complete episodes deterministically; never split an episode across GPUs."""
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed rank/world_size: {rank}/{world_size}")
    return episodes[rank::world_size]


def _parse_episode_spec(value: str | None, total_episodes: int) -> list[int]:
    if not value:
        return list(range(total_episodes))
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            fields = part.split(":")
            if len(fields) not in {2, 3}:
                raise ValueError(f"Invalid episode range: {part}")
            start = int(fields[0]) if fields[0] else 0
            stop = int(fields[1]) if fields[1] else total_episodes
            step = int(fields[2]) if len(fields) == 3 and fields[2] else 1
            result.extend(range(start, stop, step))
        else:
            result.append(int(part))
    unique = sorted(set(result))
    invalid = [episode for episode in unique if not 0 <= episode < total_episodes]
    if invalid:
        raise ValueError(f"Episode indices outside [0,{total_episodes}): {invalid[:10]}")
    return unique


def _storage_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _resolve_device(requested: str, context: DistributedContext) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requires CUDA")
        return f"cuda:{context.local_rank}"
    if requested.startswith("cuda") and context.world_size > 1:
        raise ValueError("Under torchrun use --device=cuda so each process selects LOCAL_RANK")
    return requested


def _load_config(path: Path) -> LearnerPipelineConfig:
    with draccus.config_type("json"):
        return draccus.parse(LearnerPipelineConfig, path, args=[])


def _loader(dataset: Any, *, batch_size: int, num_workers: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    return DataLoader(dataset, **kwargs)


def _task_at(tasks: Any, row: int, fallback: str) -> str:
    if isinstance(tasks, (list, tuple)):
        return str(tasks[row])
    if isinstance(tasks, str):
        return tasks
    return fallback


def _model_fingerprint(load_path: Path, contract_path: Path) -> dict[str, Any]:
    files = {}
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        candidate = load_path / name
        if candidate.is_file():
            files[name] = {"size": candidate.stat().st_size}
    return {"resolved_path": str(contract_path.resolve()), "files": files}


def _existing_episode_is_valid(
    episode_dir: Path,
    *,
    episode_index: int,
    expected_model_path: Path,
    stride: int,
) -> bool:
    payload_path = episode_dir / COMPACT_EPISODE_NAME
    metadata_path = episode_dir / "metadata.json"
    if not payload_path.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text())
    metadata_valid = (
        int(metadata.get("source_episode_index", -1)) == episode_index
        and int(metadata.get("sliding_window_stride", -1)) == stride
        and Path(metadata.get("feature_model_resolved_path", "")).expanduser().resolve()
        == expected_model_path.resolve()
    )
    if metadata_valid:
        return True
    # Compatibility fallback for a compact payload written before these two
    # identity fields were added to metadata.json.
    payload = validate_compact_episode(torch.load(payload_path, map_location="cpu", weights_only=True))
    return (
        int(metadata.get("source_episode_index", -1)) == episode_index
        and int(payload.get("sliding_window_stride", -1)) == stride
        and Path(payload["feature_model"]["resolved_path"]).expanduser().resolve()
        == expected_model_path.resolve()
    )


def _save_episode_atomic(
    output_dir: Path,
    *,
    episode_index: int,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    rank: int,
) -> Path:
    final_dir = output_dir / f"episode_{episode_index:06d}"
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite compact episode: {final_dir}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".episode_{episode_index:06d}.rank{rank}.", dir=output_dir)
    )
    try:
        save_compact_episode(payload, temporary / COMPACT_EPISODE_NAME)
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        temporary.rename(final_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_dir


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def _write_global_manifest(
    output_dir: Path,
    *,
    source_repo_id: str,
    source_root: str,
    feature_model: dict[str, Any],
    stride: int,
    storage_dtype: str,
    world_size: int,
    requested_episodes: list[int],
) -> None:
    metadata_files = sorted(output_dir.glob("episode_*/metadata.json"))
    metadata = [json.loads(path.read_text()) for path in metadata_files]
    completed_indices = sorted(int(item["source_episode_index"]) for item in metadata)
    if completed_indices != requested_episodes:
        missing = sorted(set(requested_episodes) - set(completed_indices))
        extra = sorted(set(completed_indices) - set(requested_episodes))
        raise RuntimeError(
            "Compact output does not exactly match the requested episodes: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    _write_json_atomic(
        output_dir / COMPACT_MANIFEST_NAME,
        {
            "schema": SCHEMA_NAME,
            "schema_version": 2,
            "transition_layout": SLIDING_WINDOW_TRANSITIONS,
            "source_repo_id": source_repo_id,
            "source_root": source_root,
            "feature_model": feature_model,
            "sliding_window_stride": stride,
            "storage_dtype": storage_dtype,
            "world_size": world_size,
            "expected_episodes": len(requested_episodes),
            "completed_episodes": len(metadata),
            "total_transitions": sum(int(item["num_transitions"]) for item in metadata),
            "total_source_frames": sum(int(item["primitive_steps"]) for item in metadata),
            "episode_indices": completed_indices,
            "requested_episode_indices": requested_episodes,
        },
    )


def extract(args: argparse.Namespace) -> None:
    """Extract the rank-local episode shard and finalize a shared manifest."""
    context = _distributed_context()
    distributed = context.world_size > 1
    if distributed:
        dist.init_process_group("gloo", timeout=timedelta(hours=24))
    try:
        cfg = _load_config(Path(args.config_path))
        configured_model_path = Path(cfg.policy.pretrained_path).expanduser()
        load_model_path = Path(args.policy_path).expanduser() if args.policy_path else configured_model_path
        contract_model_path = (
            Path(args.feature_model_path).expanduser()
            if args.feature_model_path
            else configured_model_path
        )
        device = _resolve_device(args.device, context)
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        cfg.policy.device = device
        cfg.policy.pretrained_path = str(load_model_path)
        cfg.dataset = DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            revision=args.dataset_revision,
            streaming=False,
        )
        metadata = LeRobotDatasetMetadata(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
        )
        requested_episodes = _parse_episode_spec(args.episodes, metadata.total_episodes)
        if args.max_episodes is not None:
            requested_episodes = requested_episodes[: args.max_episodes]
        assigned = partition_episodes(requested_episodes, context.rank, context.world_size)
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for episode_index in assigned:
            episode_dir = output_dir / f"episode_{episode_index:06d}"
            if args.resume and episode_dir.exists() and _existing_episode_is_valid(
                episode_dir,
                episode_index=episode_index,
                expected_model_path=contract_model_path,
                stride=args.stride,
            ):
                continue
            if episode_dir.exists():
                raise FileExistsError(
                    f"Existing episode is incomplete or incompatible: {episode_dir}. "
                    "Move it away before retrying."
                )
            pending.append(episode_index)

        feature_model = _model_fingerprint(load_model_path, contract_model_path)
        if pending:
            cfg.dataset.episodes = pending
            dataset = make_dataset(cfg)
            policy_cfg = deepcopy(cfg.policy)
            policy = make_policy(policy_cfg, env_cfg=cfg.env, rename_map=cfg.rename_map).eval()
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=str(load_model_path),
                preprocessor_overrides={"device_processor": {"device": device}},
            )
            extract_features = getattr(policy, "extract_rlt_features", None)
            if not callable(extract_features):
                raise TypeError(f"{type(policy).__name__} does not implement extract_rlt_features()")
            loader = _loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
            dtype = _storage_dtype(args.storage_dtype)
            frames: list[OfflineFrame] = []
            current_episode: int | None = None
            current_task = cfg.env.task
            completed = 0

            def flush_episode() -> None:
                nonlocal frames, completed
                if current_episode is None or not frames:
                    return
                transitions = build_offline_compact_transitions(
                    frames,
                    horizon=int(policy_cfg.chunk_size),
                    stride=args.stride,
                )
                episode_metadata = {
                    "task": current_task,
                    "episode_id": f"offline:{cfg.dataset.repo_id}:{current_episode}",
                    "episode_index": current_episode,
                    "source_episode_index": current_episode,
                    "source_repo_id": cfg.dataset.repo_id,
                    "source_root": str(Path(cfg.dataset.root).expanduser()),
                    "episode_outcome": "success",
                    "episode_success": True,
                    "all_intervention": True,
                    "primitive_steps": len(frames),
                    "num_transitions": len(transitions),
                    "sliding_window_stride": args.stride,
                    "feature_model_resolved_path": feature_model["resolved_path"],
                    "transition_schema": SCHEMA_NAME,
                }
                payload = make_compact_episode(
                    transitions=transitions,
                    metadata=episode_metadata,
                    feature_model=feature_model,
                    transition_layout=SLIDING_WINDOW_TRANSITIONS,
                    sliding_window_stride=args.stride,
                    primitive_steps=len(frames),
                )
                _save_episode_atomic(
                    output_dir,
                    episode_index=current_episode,
                    payload=payload,
                    metadata=episode_metadata,
                    rank=context.rank,
                )
                frames = []
                completed += 1

            progress = tqdm(
                total=len(dataset),
                desc=f"GPU {context.rank}/{context.world_size} z_rl",
                unit="frame",
                position=context.rank,
                dynamic_ncols=True,
            )
            try:
                with torch.inference_mode():
                    for batch in loader:
                        episode_indices = torch.as_tensor(batch["episode_index"]).reshape(-1).tolist()
                        tasks = batch.get("task")
                        action_pad = batch.get(f"{ACTION}_is_pad")
                        for camera_key in dataset.meta.camera_keys:
                            if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                                batch[camera_key] = batch[camera_key].float().div_(255.0)
                        processed = preprocessor(rename_batch_keys(batch, cfg.rename_map))
                        features = extract_features(processed)
                        actions = processed[ACTION]
                        horizon = int(policy_cfg.chunk_size)
                        if actions.ndim != 3 or actions.shape[1] != horizon:
                            raise ValueError(
                                f"Expected normalized action [B,{horizon},A], got {tuple(actions.shape)}"
                            )
                        valid_masks = (
                            torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
                            if action_pad is None
                            else ~torch.as_tensor(action_pad, device=actions.device).bool()
                        )
                        cpu_features = {
                            key: value.detach().to(device="cpu", dtype=dtype)
                            for key, value in features.items()
                        }
                        cpu_actions = actions.detach().to(device="cpu", dtype=dtype)
                        cpu_masks = valid_masks.detach().cpu()
                        for row, source_episode in enumerate(episode_indices):
                            source_episode = int(source_episode)
                            if current_episode is None:
                                current_episode = source_episode
                                current_task = _task_at(tasks, row, cfg.env.task)
                            elif source_episode != current_episode:
                                flush_episode()
                                current_episode = source_episode
                                current_task = _task_at(tasks, row, cfg.env.task)
                            frames.append(
                                OfflineFrame(
                                    state={key: value[row] for key, value in cpu_features.items()},
                                    action_chunk=cpu_actions[row],
                                    valid_action_mask=cpu_masks[row],
                                )
                            )
                        progress.update(len(episode_indices))
                        progress.set_postfix(
                            episode=current_episode,
                            saved=completed,
                            refresh=False,
                        )
                flush_episode()
            finally:
                progress.close()
                del loader, preprocessor, postprocessor, policy, dataset
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _write_json_atomic(
                output_dir / f"rank_{context.rank:03d}_manifest.json",
                {
                    "rank": context.rank,
                    "world_size": context.world_size,
                    "assigned_episodes": assigned,
                    "completed_this_run": completed,
                },
            )
        if distributed:
            dist.barrier()
        if context.rank == 0:
            _write_global_manifest(
                output_dir,
                source_repo_id=args.dataset_repo_id,
                source_root=args.dataset_root,
                feature_model=feature_model,
                stride=args.stride,
                storage_dtype=args.storage_dtype,
                world_size=context.world_size,
                requested_episodes=requested_episodes,
            )
            manifest = json.loads((output_dir / COMPACT_MANIFEST_NAME).read_text())
            print(
                "Offline compact extraction complete: "
                f"episodes={manifest['completed_episodes']} "
                f"transitions={manifest['total_transitions']} output={output_dir}"
            )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract PI05-RLT compact episodes; launch with torchrun for one process per GPU."
    )
    parser.add_argument("--config-path", required=True, help="Learner JSON supplying policy/env schema")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-repo-id", default="local/offline_feature_extraction")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-path", help="Actual checkpoint path on this extraction machine")
    parser.add_argument(
        "--feature-model-path",
        help="Path identity written to payloads; defaults to policy.pretrained_path in config",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--storage-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", help="Comma/range syntax, e.g. 0:1300 or 0:100,200:300:2")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    """Parse standalone extraction arguments and run the export."""
    args = _parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.stride <= 0:
        raise ValueError("batch-size/stride must be positive and num-workers must be non-negative")
    extract(args)


if __name__ == "__main__":
    main()
