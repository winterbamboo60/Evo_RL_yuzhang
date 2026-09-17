"""Actor-compatible compact offline transitions and learner replay loading."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION

from .compact_transition import validate_compact_episode
from .config import LearnerPipelineConfig

COMPACT_MANIFEST_NAME = "manifest.json"
COMPACT_EPISODE_NAME = "compact_episode.pt"


@dataclass
class OfflineFrame:
    """Frozen RLT features and normalized action target for one source frame."""

    state: dict[str, torch.Tensor]
    action_chunk: torch.Tensor
    valid_action_mask: torch.Tensor


def _batched_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0) for key, value in state.items()}


def build_offline_compact_transitions(
    frames: list[OfflineFrame],
    *,
    horizon: int,
    stride: int,
) -> list[dict[str, Any]]:
    """Build Actor-compatible all-success, all-intervention sliding windows."""
    if not frames:
        return []
    if horizon <= 0 or stride <= 0:
        raise ValueError("offline horizon and stride must be positive")

    transitions = []
    episode_length = len(frames)
    for start in range(0, episode_length, stride):
        frame = frames[start]
        target_action_chunk = frame.action_chunk[:horizon]
        valid_action_mask = frame.valid_action_mask[:horizon].to(dtype=torch.float32)
        if target_action_chunk.shape[0] != horizon or valid_action_mask.shape[0] != horizon:
            raise ValueError(
                "offline dataset action delta indices must provide exactly policy.chunk_size entries"
            )

        reward_chunk = torch.zeros(horizon, dtype=target_action_chunk.dtype)
        terminal_offset = episode_length - 1 - start
        if terminal_offset < horizon:
            reward_chunk[terminal_offset] = 1.0

        next_start = start + horizon
        if next_start < episode_length:
            next_frame = frames[next_start]
            next_valid_action_mask = next_frame.valid_action_mask[:horizon].to(dtype=torch.float32)
        else:
            next_frame = frames[-1]
            next_valid_action_mask = torch.zeros_like(valid_action_mask)

        transitions.append(
            {
                "state": _batched_state(frame.state),
                ACTION: target_action_chunk[0].unsqueeze(0),
                "target_action_chunk": target_action_chunk,
                "reward": reward_chunk,
                "intervene_flags": valid_action_mask.bool(),
                "valid_action_mask": valid_action_mask,
                "next_valid_action_mask": next_valid_action_mask,
                "next_state": _batched_state(next_frame.state),
                "done": next_start >= episode_length,
                "truncated": False,
                "complementary_info": {
                    "intervention": True,
                },
            }
        )
    return transitions


def append_offline_episode(
    replay_buffer: ReplayBuffer,
    frames: list[OfflineFrame],
    *,
    horizon: int,
    stride: int,
) -> int:
    """Append one all-success, all-intervention episode to compact replay."""
    transitions = build_offline_compact_transitions(
        frames,
        horizon=horizon,
        stride=stride,
    )
    for transition in transitions:
        replay_buffer.add(
            state=transition["state"],
            action=transition[ACTION],
            reward=float(torch.as_tensor(transition["reward"]).sum().item()),
            next_state=transition["next_state"],
            done=transition["done"],
            truncated=transition["truncated"],
            complementary_info={
                "target_action_chunk": transition["target_action_chunk"].unsqueeze(0),
                "reward_chunk": transition["reward"].unsqueeze(0),
                "intervene_flags": transition["intervene_flags"].unsqueeze(0),
                "valid_action_mask": transition["valid_action_mask"].unsqueeze(0),
                "next_valid_action_mask": transition["next_valid_action_mask"].unsqueeze(0),
            },
        )
    return len(transitions)


def _compact_episode_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob(f"episode_*/{COMPACT_EPISODE_NAME}"))
    if not paths:
        raise FileNotFoundError(f"No Actor-compatible compact episodes found under {root}")
    return paths


def _compact_transition_capacity(root: Path, paths: list[Path]) -> int:
    manifest_path = root / COMPACT_MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        total = int(manifest.get("total_transitions", 0))
        episodes = int(manifest.get("completed_episodes", manifest.get("total_episodes", 0)))
        if total > 0 and episodes == len(paths):
            return total
    logging.warning(
        "[LEARNER][OFFLINE] compact manifest missing/stale; scanning %d episode files",
        len(paths),
    )
    return sum(
        len(validate_compact_episode(torch.load(path, map_location="cpu", weights_only=True))["transitions"])
        for path in paths
    )


def load_compact_replay(
    cfg: LearnerPipelineConfig,
    algorithm: Any,
    *,
    shutdown_event: Any | None = None,
) -> ReplayBuffer:
    """Load pre-extracted Actor-format episodes without constructing PI05."""
    configured_path = cfg.offline_pretraining.compact_dataset_path
    if not configured_path:
        raise ValueError("offline_pretraining.compact_dataset_path is required")
    root = Path(configured_path).expanduser().resolve()
    paths = _compact_episode_paths(root)
    capacity = _compact_transition_capacity(root, paths)
    replay_buffer = ReplayBuffer(
        capacity=capacity,
        device=str(cfg.policy.device),
        state_keys=cfg.algorithm.state_keys,
        storage_device=cfg.algorithm.storage_device,
        optimize_memory=False,
        use_drq=False,
    )
    expected_model = str(Path(cfg.policy.pretrained_path).expanduser().resolve())
    loaded_transitions = 0
    progress = tqdm(
        paths,
        desc="Offline compact loading",
        unit="episode",
        dynamic_ncols=True,
    )
    try:
        for path in progress:
            if shutdown_event is not None and shutdown_event.is_set():
                raise KeyboardInterrupt("offline compact loading interrupted")
            payload = validate_compact_episode(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            feature_model = payload["feature_model"].get("resolved_path")
            if str(Path(feature_model).expanduser().resolve()) != expected_model:
                raise ValueError(
                    f"Compact episode {path} uses feature model {feature_model}, expected {expected_model}"
                )
            payload_size = len(payload["transitions"])
            if loaded_transitions + payload_size > capacity:
                raise ValueError(
                    f"Compact manifest capacity {capacity} is smaller than episode contents"
                )
            if not algorithm.ingest_transition_payload(payload, replay_buffer):
                raise ValueError(f"Compact episode was not accepted by rlt_chunk: {path}")
            loaded_transitions += payload_size
            progress.set_postfix(transitions=loaded_transitions, refresh=False)
    finally:
        progress.close()
    if loaded_transitions != capacity or len(replay_buffer) != capacity:
        raise ValueError(
            "Compact dataset size mismatch: "
            f"manifest={capacity}, loaded={loaded_transitions}, replay={len(replay_buffer)}"
        )
    logging.info(
        "[LEARNER][OFFLINE] loaded %d episodes / %d compact transitions from %s",
        len(paths),
        loaded_transitions,
        root,
    )
    return replay_buffer
