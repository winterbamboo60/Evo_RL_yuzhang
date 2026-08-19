"""Learner-only runtime for the frozen PI05+RLT chunk actor/critic."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.onlineRL_evoRL.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.transport.utils import bytes_to_transitions
from lerobot.utils.constants import ACTION, TRAINING_STATE_DIR
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_transition_to_device

from .chunk_transition import sliding_windows, split_episodes
from .modeling_pi05_online_rl import PI05OnlineRLPolicy

_FEATURE_KEYS = ("z_rl", "proprio", "ref_action")


def _merge_processed(items: list[dict]) -> dict:
    first = items[0]
    return {
        key: torch.cat([item[key] for item in items], dim=0)
        for key, value in first.items()
        if isinstance(value, torch.Tensor)
    }


@torch.no_grad()
def _enrich_episode(
    episode: list[dict],
    policy: PI05OnlineRLPolicy,
    preprocessor,
    task: str,
) -> list[dict]:
    """Normalize actions and cache frozen VLA/RLT outputs once per primitive state."""
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
    batch_size = policy.config.feature_extract_batch_size
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


def _add_primitive_list(
    primitive: list[dict],
    replay: ReplayBuffer,
    policy: PI05OnlineRLPolicy,
    preprocessor,
    task: str,
) -> int:
    count = 0
    for episode in split_episodes(primitive):
        enriched = _enrich_episode(episode, policy, preprocessor, task)
        for transition in sliding_windows(enriched, policy.config.chunk_size):
            replay.add(**transition)
            count += 1
    return count


def _validate_offline_intervention(dataset, field: str | None) -> None:
    if field is None:
        return
    if len(dataset) == 0:
        raise ValueError("Offline dataset is empty")
    if field not in dataset[0]:
        raise ValueError(
            f"Configured offline intervention field {field!r} is missing from the dataset"
        )


def _offline_primitives(dataset, state_keys, intervention_field: str | None) -> list[dict]:
    _validate_offline_intervention(dataset, intervention_field)
    transitions = ReplayBuffer._lerobotdataset_to_transitions(dataset, state_keys=state_keys)
    if intervention_field is None:
        for transition in transitions:
            info = dict(transition.get("complementary_info") or {})
            info["is_intervention"] = False
            transition["complementary_info"] = info
    return transitions


def _sample_training_batch(online, offline, batch_size: int, online_ready: bool):
    if online_ready and offline is not None and len(offline):
        online_size = batch_size // 2
        return concatenate_batch_transitions(
            online.sample(online_size), offline.sample(batch_size - online_size)
        )
    if online_ready:
        return online.sample(batch_size)
    return offline.sample(batch_size)


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

    offline_dataset = make_dataset(cfg) if cfg.dataset is not None else None
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=offline_dataset.meta if offline_dataset is not None else None,
        env_cfg=None if offline_dataset is not None else cfg.env,
    )
    if not isinstance(policy, PI05OnlineRLPolicy):
        raise TypeError(f"Expected PI05OnlineRLPolicy, got {type(policy).__name__}")
    policy.train()

    stats = offline_dataset.meta.stats if offline_dataset is not None else None
    preprocessor, _ = make_pre_post_processors(
        cfg.policy,
        pretrained_path=str(cfg.policy.pretrained_path) if cfg.policy.pretrained_path else None,
        dataset_stats=stats,
    )

    device = cfg.policy.device
    resume_dir = (
        Path(cfg.policy.pretrained_path).parent
        if cfg.resume and cfg.policy.pretrained_path
        else None
    )
    online_root = resume_dir / "replay_online" if resume_dir is not None else None
    offline_root = resume_dir / "replay_offline" if resume_dir is not None else None

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

    offline = None
    if offline_root is not None and offline_root.exists():
        offline = ReplayBuffer.from_lerobot_dataset(
            LeRobotDataset(repo_id=cfg.dataset.repo_id, root=offline_root),
            capacity=cfg.policy.offline_buffer_capacity,
            device=device,
            state_keys=_FEATURE_KEYS,
            storage_device=cfg.policy.storage_device,
            use_drq=False,
            optimize_memory=False,
        )
    elif offline_dataset is not None:
        offline = ReplayBuffer(
            cfg.policy.offline_buffer_capacity,
            device=device,
            storage_device=cfg.policy.storage_device,
            state_keys=_FEATURE_KEYS,
            use_drq=False,
            optimize_memory=False,
        )
        primitives = _offline_primitives(
            offline_dataset,
            state_keys=cfg.policy.input_features.keys(),
            intervention_field=cfg.policy.offline_intervention_field,
        )
        added = _add_primitive_list(primitives, offline, policy, preprocessor, cfg.env.task)
        logging.info("Loaded %d offline sliding-window transitions", added)

    optimizers = {
        "actor": torch.optim.Adam(policy.actor.parameters(), lr=cfg.policy.actor_lr),
        "critic": torch.optim.Adam(policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr),
    }
    optimization_step = (
        load_training_state(resume_dir, optimizers, None)[0]
        if resume_dir is not None else 0
    )
    interaction_step = 0

    while optimization_step < cfg.policy.online_steps:
        if shutdown_event is not None and shutdown_event.is_set():
            break

        while not interaction_message_queue.empty():
            from lerobot.transport.utils import bytes_to_python_object

            message = bytes_to_python_object(interaction_message_queue.get())
            interaction_step = message.get("Interaction step", interaction_step)

        while not transition_queue.empty():
            primitive = bytes_to_transitions(buffer=transition_queue.get())
            primitive = [
                move_transition_to_device(transition, device=device) for transition in primitive
            ]
            added = _add_primitive_list(primitive, online, policy, preprocessor, cfg.env.task)
            logging.debug("Added %d online sliding-window transitions", added)

        # Offline-only training starts immediately; online-only observes the warmup.
        online_ready = len(online) >= cfg.policy.online_step_before_learning
        if not online_ready and (offline is None or len(offline) == 0):
            time.sleep(0.01)
            continue

        started = time.time()
        metrics = {}
        for _ in range(cfg.policy.utd_ratio):
            batch = _sample_training_batch(online, offline, cfg.batch_size, online_ready)
            critic_output = policy(batch, model="critic")
            optimizers["critic"].zero_grad()
            critic_output["loss_critic"].backward()
            critic_norm = torch.nn.utils.clip_grad_norm_(
                policy.critic_ensemble.parameters(), cfg.policy.grad_clip_norm
            )
            optimizers["critic"].step()
            policy.soft_update_target()

            if optimization_step % cfg.policy.policy_update_freq == 0:
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

            metrics.update(
                loss_critic=critic_output["loss_critic"].item(),
                q=critic_output["q"].item(),
                target_q=critic_output["target_q"].item(),
                critic_grad_norm=float(critic_norm),
            )

        optimization_step += 1
        if optimization_step % cfg.log_freq == 0:
            metrics.update(
                **{
                    "Optimization step": optimization_step,
                    "online_replay_buffer_size": len(online),
                    "offline_replay_buffer_size": len(offline) if offline is not None else 0,
                    "Optimization frequency loop [Hz]": 1 / max(time.time() - started, 1e-9),
                }
            )
            logging.info("PI05 online-RL step %d: %s", optimization_step, metrics)
            if wandb_logger:
                wandb_logger.log_dict(
                    metrics, mode="train", custom_step_key="Optimization step"
                )


        if cfg.save_checkpoint and (
            optimization_step % cfg.save_freq == 0
            or optimization_step == cfg.policy.online_steps
        ):
            checkpoint_dir = get_step_checkpoint_dir(
                cfg.output_dir, cfg.policy.online_steps, optimization_step
            )
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=optimization_step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizers,
                scheduler=None,
            )
            training_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
            os.makedirs(training_dir, exist_ok=True)
            torch.save(
                {"step": optimization_step, "interaction_step": interaction_step},
                os.path.join(training_dir, "training_state.pt"),
            )
            if len(online):
                online.to_lerobot_dataset(
                    repo_id="pi05_online_rl_online_replay",
                    fps=cfg.env.fps,
                    root=os.path.join(checkpoint_dir, "replay_online"),
                )
            if offline is not None and len(offline):
                offline.to_lerobot_dataset(
                    repo_id=cfg.dataset.repo_id,
                    fps=cfg.env.fps,
                    root=os.path.join(checkpoint_dir, "replay_offline"),
                )
            update_last_checkpoint(checkpoint_dir)

    return policy
