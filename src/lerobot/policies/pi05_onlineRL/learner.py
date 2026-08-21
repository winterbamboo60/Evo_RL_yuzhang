"""Learner-only runtime for the frozen PI05+RLT chunk actor/critic."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import torch

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import cycle
from lerobot.onlineRL_evoRL.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.onlineRL_evoRL.compact_transition import (
    SCHEMA_NAME,
    bytes_to_episode_payload,
    is_compact_episode,
    validate_compact_episode,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.transport.utils import bytes_to_python_object
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.utils.recording_annotations import normalize_episode_success_label
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_training_state,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_transition_to_device

from .chunk_transition import sliding_windows, split_episodes
from .modeling_pi05_online_rl import PI05OnlineRLPolicy

_COLLECTOR_POLICY_ID = "complementary_info.collector_policy_id"
_FEATURE_KEYS = ("z_rl", "proprio", "ref_action")


def _merge_processed(items: list[dict]) -> dict:
    first = items[0]
    return {
        key: torch.cat([item[key] for item in items], dim=0)
        for key, value in first.items()
        if isinstance(value, torch.Tensor)
    }


def _validate_online_task(task: str | None, valid_tasks: set[str]) -> str:
    if not task:
        raise ValueError("Online episode metadata must carry a non-empty task")
    if task not in valid_tasks:
        raise ValueError(f"Online task {task!r} is not declared in dataset meta/tasks.parquet")
    return task


@torch.no_grad()
def _enrich_episode(
    episode: list[dict],
    policy: PI05OnlineRLPolicy,
    preprocessor,
    task: str,
    batch_size: int,
) -> list[dict]:
    """Normalize one online episode and cache its frozen VLA/RLT outputs."""
    if not episode:
        return []

    observations = [transition["state"] for transition in episode]
    observations.append(episode[-1]["next_state"])
    actions = [transition[ACTION] for transition in episode]
    actions.append(episode[-1][ACTION])

    processed = []
    for observation, action in zip(observations, actions, strict=True):
        raw = {key: value.squeeze(0) for key, value in observation.items()}
        raw[ACTION] = action.squeeze(0)
        raw["task"] = task
        processed.append(preprocessor(raw))

    features: dict[str, list[torch.Tensor]] = {key: [] for key in _FEATURE_KEYS}
    for start in range(0, len(processed), batch_size):
        batch = _merge_processed(processed[start : start + batch_size])
        output = policy.extract_rlt_features(batch)
        for key in _FEATURE_KEYS:
            features[key].extend(output[key].split(1))

    enriched = []
    for index, transition in enumerate(episode):
        item = dict(transition)
        item["state"] = {key: features[key][index] for key in _FEATURE_KEYS}
        item["next_state"] = {key: features[key][index + 1] for key in _FEATURE_KEYS}
        item[ACTION] = processed[index][ACTION]
        enriched.append(item)
    return enriched


def _add_online_episodes(
    primitive: list[dict],
    replay: ReplayBuffer,
    policy: PI05OnlineRLPolicy,
    preprocessor,
    task: str | None,
    valid_tasks: set[str],
    batch_size: int,
) -> int:
    task = _validate_online_task(task, valid_tasks)
    episodes = split_episodes(primitive)
    if len(episodes) != 1:
        raise ValueError(f"Expected one online episode per payload, got {len(episodes)}")
    count = 0
    for episode in episodes:
        enriched = _enrich_episode(episode, policy, preprocessor, task, batch_size)
        for transition in sliding_windows(enriched, policy.config.chunk_size):
            replay.add(**transition)
            count += 1
    return count


def _add_compact_episode(
    payload: dict,
    replay: ReplayBuffer,
    policy: PI05OnlineRLPolicy,
    valid_tasks: set[str],
) -> tuple[int, dict]:
    validate_compact_episode(payload)
    metadata = payload["metadata"]
    _validate_online_task(metadata.get("task"), valid_tasks)

    expected_path = Path(policy.config.pretrained_path).expanduser().resolve()
    feature_path = Path(payload["feature_model"]["resolved_path"]).expanduser().resolve()
    if feature_path != expected_path:
        raise ValueError(
            f"Compact episode feature model {feature_path} does not match learner base {expected_path}"
        )

    primitive = [
        move_transition_to_device(transition, device=policy.config.device)
        for transition in payload["transitions"]
    ]
    episodes = split_episodes(primitive)
    if len(episodes) != 1:
        raise ValueError(f"Expected one compact online episode per payload, got {len(episodes)}")

    first_state = episodes[0][0]["state"]
    if first_state["z_rl"].flatten(1).shape[1] != policy.config.z_dim:
        raise ValueError("Compact z_rl dimension does not match learner policy.z_dim")
    if first_state["proprio"].flatten(1).shape[1] != policy.config.proprio_dim:
        raise ValueError("Compact proprio dimension does not match learner policy.proprio_dim")
    ref_action = first_state["ref_action"]
    if ref_action.shape[-2:] != (policy.config.chunk_size, policy.action_dim):
        raise ValueError("Compact ref_action shape does not match learner chunk/action dimensions")

    count = 0
    for transition in sliding_windows(episodes[0], policy.config.chunk_size):
        replay.add(**transition)
        count += 1
    return count, metadata


def _column_values(dataset, key: str) -> list:
    values = dataset.hf_dataset[key]
    return [
        value.item() if isinstance(value, torch.Tensor) and value.ndim == 0 else value
        for value in values
    ]


def _offline_annotations(dataset) -> dict[str, torch.Tensor]:
    """Build compact frame annotations without loading images into memory."""
    if len(dataset) == 0:
        raise ValueError("Offline dataset is empty")
    if _COLLECTOR_POLICY_ID not in dataset.features:
        raise ValueError(f"Offline dataset is missing {_COLLECTOR_POLICY_ID!r}")
    if "episode_success" not in dataset.meta.episodes.column_names:
        raise ValueError("Offline dataset meta/episodes is missing 'episode_success'")

    absolute_indices = [int(value) for value in _column_values(dataset, "index")]
    episode_indices = [int(value) for value in _column_values(dataset, "episode_index")]
    collector_ids = _column_values(dataset, _COLLECTOR_POLICY_ID)
    size = max(absolute_indices) + 1
    is_human = torch.zeros(size, dtype=torch.bool)
    for index, collector_id in zip(absolute_indices, collector_ids, strict=True):
        is_human[index] = collector_id == "human"

    terminal = torch.zeros(size, dtype=torch.bool)
    terminal_reward = torch.zeros(size, dtype=torch.float32)
    episode_end = torch.zeros(dataset.meta.total_episodes, dtype=torch.long)
    for episode_index in set(episode_indices):
        episode = dataset.meta.episodes[episode_index]
        outcome = normalize_episode_success_label(episode["episode_success"])
        if outcome is None:
            raise ValueError(f"Episode {episode_index} is missing episode_success")
        end = int(episode["dataset_to_index"])
        episode_end[episode_index] = end
        terminal[end - 1] = True
        terminal_reward[end - 1] = float(outcome == "success")

    return {
        "is_human": is_human,
        "terminal": terminal,
        "terminal_reward": terminal_reward,
        "episode_end": episode_end,
    }


def _slice_raw_batch(batch: dict, size: int) -> dict:
    return {
        key: value[:size] if isinstance(value, (torch.Tensor, list, tuple)) else value
        for key, value in batch.items()
    }


def _observation_batch(batch: dict, feature_keys, time_index: int) -> dict:
    raw = {key: batch[key][:, time_index] for key in feature_keys}
    raw[ACTION] = batch[ACTION]
    raw["task"] = batch["task"]
    return raw


@torch.no_grad()
def _build_offline_batch(
    raw_batch: dict,
    annotations: dict[str, torch.Tensor],
    policy: PI05OnlineRLPolicy,
    preprocessor,
) -> dict:
    horizon = policy.config.chunk_size
    indices = raw_batch["index"].long()
    episode_indices = raw_batch["episode_index"].long()
    offsets = torch.arange(horizon)
    chunk_indices = indices[:, None] + offsets
    valid_mask = ~raw_batch[f"{ACTION}_is_pad"].bool()
    safe_indices = chunk_indices.clamp_max(annotations["terminal"].numel() - 1)

    rewards = annotations["terminal_reward"][safe_indices] * valid_mask
    intervene_flags = annotations["is_human"][safe_indices] & valid_mask
    done = (annotations["terminal"][safe_indices] & valid_mask).any(dim=1)
    episode_end = annotations["episode_end"][episode_indices]
    next_length = (episode_end - (indices + horizon)).clamp(min=0, max=horizon)
    next_valid_mask = offsets[None, :] < next_length[:, None]

    feature_keys = policy.config.input_features.keys()
    current = preprocessor(_observation_batch(raw_batch, feature_keys, 0))
    next_observation = preprocessor(_observation_batch(raw_batch, feature_keys, 1))
    state = policy.extract_rlt_features(current)
    next_state = policy.extract_rlt_features(next_observation)
    target_action = current[ACTION]
    target_action = target_action * valid_mask.to(target_action.device).unsqueeze(-1)
    device = target_action.device

    return {
        "state": state,
        ACTION: target_action[:, 0],
        "target_action_chunk": target_action,
        "reward": rewards.to(device),
        "intervene_flags": intervene_flags.to(device),
        "valid_action_mask": valid_mask.to(device),
        "next_valid_action_mask": next_valid_mask.to(device),
        "next_state": next_state,
        "done": done.to(device=device, dtype=torch.float32),
        "truncated": torch.zeros_like(done, device=device, dtype=torch.float32),
        "complementary_info": None,
    }


def _offline_dataloader(cfg, dataset):
    if cfg.dataset.streaming:
        raise ValueError("pi05_online_rl currently requires dataset.streaming=false")
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=str(cfg.policy.device).startswith("cuda"),
        drop_last=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )


def _online_batch_sizes(batch_size: int, online_only: bool) -> tuple[int, int]:
    online_size = batch_size if online_only else max(1, batch_size // 2)
    return online_size, batch_size - online_size


def _training_phase(learner_step: int, offline_steps: int, online_budget: int) -> str | None:
    if learner_step < offline_steps:
        return "offline_initialization"
    return "online" if online_budget > 0 else None


_ACTOR_CRITIC_FILE = "actor_critic.pt"


def _checkpoint_dir_for_resume(cfg) -> Path | None:
    if not cfg.resume:
        return None
    return Path(cfg.output_dir) / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK


def _load_actor_critic(checkpoint_dir: Path, policy: PI05OnlineRLPolicy) -> None:
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing PI05 checkpoint manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "base_policy_path": str(Path(policy.config.pretrained_path).expanduser().resolve()),
        "chunk_size": policy.config.chunk_size,
        "z_dim": policy.config.z_dim,
        "proprio_dim": policy.config.proprio_dim,
    }
    mismatched = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatched:
        raise ValueError(f"PI05 actor-critic checkpoint manifest mismatch: {mismatched}")

    state_path = checkpoint_dir / _ACTOR_CRITIC_FILE
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing PI05 actor-critic checkpoint: {state_path}")
    state = torch.load(state_path, map_location=policy.config.device, weights_only=True)
    policy.actor.load_state_dict(state["actor"])
    policy.critic_ensemble.load_state_dict(state["critic"])
    policy.critic_target.load_state_dict(state["critic_target"])


def _save_actor_critic_checkpoint(
    checkpoint_dir: Path, step: int, cfg, policy: PI05OnlineRLPolicy, optimizers: dict
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / _ACTOR_CRITIC_FILE
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "actor": policy.actor.state_dict(),
            "critic": policy.critic_ensemble.state_dict(),
            "critic_target": policy.critic_target.state_dict(),
        },
        temporary,
    )
    temporary.replace(state_path)

    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(pretrained_dir)
    policy.config.save_pretrained(pretrained_dir)
    manifest = {
        "format_version": 1,
        "base_policy_path": str(Path(policy.config.pretrained_path).expanduser().resolve()),
        "chunk_size": policy.config.chunk_size,
        "z_dim": policy.config.z_dim,
        "proprio_dim": policy.config.proprio_dim,
    }
    (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    save_training_state(checkpoint_dir, step, optimizers, None)


def train_pi05_online_rl(
    cfg,
    wandb_logger,
    shutdown_event,
    transition_queue,
    interaction_message_queue,
    parameters_queue,
):
    """Train RL heads while the VLA and RLT backbone remains frozen."""
    del parameters_queue  # Actor deployment is intentionally deferred.
    if cfg.dataset is None:
        raise ValueError("pi05_online_rl learner requires a metadata-backed offline dataset")

    offline_dataset = make_dataset(cfg)
    valid_tasks = set(offline_dataset.meta.tasks.index)
    if not valid_tasks:
        raise ValueError("Offline dataset meta/tasks.parquet is empty")
    annotations = _offline_annotations(offline_dataset)
    offline_iterator = cycle(_offline_dataloader(cfg, offline_dataset))

    policy = make_policy(cfg=cfg.policy, ds_meta=offline_dataset.meta, env_cfg=None)
    if not isinstance(policy, PI05OnlineRLPolicy):
        raise TypeError(f"Expected PI05OnlineRLPolicy, got {type(policy).__name__}")
    policy.train()

    preprocessor, _ = make_pre_post_processors(
        cfg.policy,
        pretrained_path=str(cfg.policy.pretrained_path) if cfg.policy.pretrained_path else None,
        dataset_stats=offline_dataset.meta.stats,
    )

    device = cfg.policy.device
    resume_dir = _checkpoint_dir_for_resume(cfg)
    if resume_dir is not None:
        _load_actor_critic(resume_dir, policy)
    online_root = resume_dir / "replay_online" if resume_dir is not None else None
    if online_root is not None and online_root.exists():
        online = ReplayBuffer.from_lerobot_dataset(
            LeRobotDataset(repo_id="pi05_online_rl_online_replay", root=online_root),
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=_FEATURE_KEYS,
            storage_device=cfg.policy.storage_device,
            use_drq=False,
            optimize_memory=False,
        )
    else:
        online = ReplayBuffer(
            cfg.policy.online_buffer_capacity,
            device=device,
            storage_device=cfg.policy.storage_device,
            state_keys=_FEATURE_KEYS,
            use_drq=False,
            optimize_memory=False,
        )

    optimizers = {
        "actor": torch.optim.Adam(policy.actor.parameters(), lr=cfg.policy.actor_lr),
        "critic": torch.optim.Adam(policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr),
    }
    learner_step = load_training_state(resume_dir, optimizers, None)[0] if resume_dir else 0
    interaction_step = 0
    pending_online_episodes = deque()
    pending_online_tasks = deque()
    pending_online_update_steps = 0
    online_episode_count = 0
    if resume_dir is not None:
        runtime_state_path = resume_dir / TRAINING_STATE_DIR / "training_state.pt"
        if runtime_state_path.is_file():
            runtime_state = torch.load(runtime_state_path, weights_only=True)
            interaction_step = runtime_state.get("interaction_step", 0)
            pending_online_update_steps = runtime_state.get("pending_online_update_steps", 0)
            online_episode_count = runtime_state.get("online_episode_count", 0)
    initialization_complete_logged = False

    while shutdown_event is None or not shutdown_event.is_set():
        while not interaction_message_queue.empty():
            message = bytes_to_python_object(interaction_message_queue.get())
            interaction_step = message.get("Interaction step", interaction_step)
            if message.get("transition_schema") != SCHEMA_NAME:
                pending_online_tasks.append(message.get("task"))

        while not transition_queue.empty():
            payload = bytes_to_episode_payload(transition_queue.get())
            if is_compact_episode(payload):
                added, metadata = _add_compact_episode(payload, online, policy, valid_tasks)
                interaction_step = metadata.get("Interaction step", interaction_step)
                payload_format = SCHEMA_NAME
            else:
                pending_online_episodes.append(payload)
                continue
            if added:
                online_episode_count += 1
                pending_online_update_steps += cfg.policy.online_updates_per_episode
                logging.info(
                    "Received online episode %d (%s): added %d transitions, scheduled %d updates",
                    online_episode_count,
                    payload_format,
                    added,
                    cfg.policy.online_updates_per_episode,
                )

        while pending_online_episodes and pending_online_tasks:
            primitive = [
                move_transition_to_device(transition, device=device)
                for transition in pending_online_episodes.popleft()
            ]
            added = _add_online_episodes(
                primitive,
                online,
                policy,
                preprocessor,
                pending_online_tasks.popleft(),
                valid_tasks,
                cfg.batch_size,
            )
            if added:
                online_episode_count += 1
                pending_online_update_steps += cfg.policy.online_updates_per_episode
                logging.info(
                    "Received online episode %d (legacy_raw): added %d transitions, scheduled %d updates",
                    online_episode_count,
                    added,
                    cfg.policy.online_updates_per_episode,
                )

        training_phase = _training_phase(
            learner_step, cfg.steps, pending_online_update_steps
        )
        if training_phase != "offline_initialization" and not initialization_complete_logged:
            logging.info(
                "Offline initialization complete at learner step %d; entering online phase",
                learner_step,
            )
            initialization_complete_logged = True
        if training_phase is None:
            if shutdown_event is not None:
                shutdown_event.wait(0.05)
            else:
                time.sleep(0.05)
            continue

        if training_phase == "offline_initialization":
            raw_offline = next(offline_iterator)
            batch = _build_offline_batch(raw_offline, annotations, policy, preprocessor)
        else:
            online_size, offline_size = _online_batch_sizes(
                cfg.batch_size, cfg.policy.online_only_after_initialization
            )
            batch = online.sample(online_size)
            if offline_size:
                raw_offline = _slice_raw_batch(next(offline_iterator), offline_size)
                offline_batch = _build_offline_batch(
                    raw_offline, annotations, policy, preprocessor
                )
                batch = concatenate_batch_transitions(batch, offline_batch)
            pending_online_update_steps -= 1

        started = time.time()
        critic_output = policy(batch, model="critic")
        optimizers["critic"].zero_grad()
        critic_output["loss_critic"].backward()
        critic_norm = torch.nn.utils.clip_grad_norm_(
            policy.critic_ensemble.parameters(), cfg.policy.grad_clip_norm
        )
        optimizers["critic"].step()
        policy.soft_update_target()

        metrics = {
            "loss_critic": critic_output["loss_critic"].item(),
            "q": critic_output["q"].item(),
            "target_q": critic_output["target_q"].item(),
            "critic_grad_norm": float(critic_norm),
        }
        if learner_step % cfg.policy.actor_update_interval == 0:
            actor_output = policy(batch, model="actor")
            optimizers["actor"].zero_grad()
            actor_output["loss_actor"].backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(
                policy.actor.parameters(), cfg.policy.grad_clip_norm
            )
            optimizers["actor"].step()
            metrics.update(
                loss_actor=actor_output["loss_actor"].item(),
                bc_loss=actor_output["bc_loss"].item(),
                actor_q=actor_output["actor_q"].item(),
                actor_grad_norm=float(actor_norm),
            )

        learner_step += 1
        if learner_step % cfg.log_freq == 0:
            metrics.update(
                learner_step=learner_step,
                training_phase=training_phase,
                interaction_step=interaction_step,
                online_episode_count=online_episode_count,
                pending_online_update_steps=pending_online_update_steps,
                online_replay_buffer_size=len(online),
                update_frequency_hz=1 / max(time.time() - started, 1e-9),
            )
            logging.info("PI05 online-RL learner step %d: %s", learner_step, metrics)
            if wandb_logger:
                wandb_logger.log_dict(metrics, mode="train", custom_step_key="learner_step")

        if cfg.save_checkpoint and (
            learner_step % cfg.save_freq == 0 or learner_step == cfg.steps
        ):
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, learner_step)
            _save_actor_critic_checkpoint(
                checkpoint_dir, learner_step, cfg, policy, optimizers
            )
            training_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
            os.makedirs(training_dir, exist_ok=True)
            torch.save(
                {
                    "step": learner_step,
                    "interaction_step": interaction_step,
                    "pending_online_update_steps": pending_online_update_steps,
                    "online_episode_count": online_episode_count,
                },
                os.path.join(training_dir, "training_state.pt"),
            )
            if len(online):
                online.to_lerobot_dataset(
                    repo_id="pi05_online_rl_online_replay",
                    fps=offline_dataset.fps,
                    root=os.path.join(checkpoint_dir, "replay_online"),
                )
            update_last_checkpoint(checkpoint_dir)

    return policy
