"""Independent compact-replay PI05-RLT learner owned by onlineRL_evoRL."""

# Keep runtime annotations enabled in this CLI module: parser.wrap() reads the
# first parameter annotation directly and therefore needs the dataclass object,
# not the string produced by ``from __future__ import annotations``.

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file as load_safetensors
from termcolor import colored
from torch.multiprocessing import Queue
from tqdm.auto import tqdm

from lerobot.common.train_utils import get_step_checkpoint_dir, update_last_checkpoint
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.optim import load_optimizer_state, save_optimizer_state
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.data_sources.data_mixer import DataMixer
from lerobot.rl.trainer import RLTrainer
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    state_to_bytes,
)
from lerobot.utils.constants import (
    ALGORITHM_DIR,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.import_utils import _grpc_available, require_package
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.process import ProcessSignalHandler, ensure_multiprocessing_start_method
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from .compact_transition import bytes_to_episode_payload, validate_compact_episode
from .config import LearnerPipelineConfig
from .learner_algorithm import HeadOnlyRLTChunkAlgorithm
from .learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService
from .offline_pretraining import load_compact_replay
from .wire import (
    ACTOR_GPU_RELEASED,
    WEIGHT_HANDOFF_ID_FIELD,
    WEIGHT_UPDATE_STEP_FIELD,
    parse_control_message,
)

if TYPE_CHECKING or _grpc_available:
    import grpc

    from lerobot.transport import services_pb2_grpc
else:
    grpc = None
    services_pb2_grpc = None


class OnlineReplayMixer(DataMixer):
    """Sample one compact replay without online/offline ratio mixing."""

    offline_buffer = None

    def __init__(self, online_buffer: ReplayBuffer) -> None:
        """Bind the only replay source used by this learner."""
        self.online_buffer = online_buffer

    def sample(self, batch_size: int):
        """Sample exclusively from online compact replay."""
        return self.online_buffer.sample(batch_size)

    def get_iterator(self, batch_size: int, async_prefetch: bool = True, queue_size: int = 2):
        """Yield online compact replay batches indefinitely."""
        yield from self.online_buffer.get_iterator(
            batch_size=batch_size,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
        )


@dataclass
class LearnerProgress:
    """Separately track startup pretraining and quota-driven online updates."""

    algorithm_step: int = 0
    online_step: int = 0
    offline_step: int = 0
    offline_complete: bool = False
    pending_updates: int = 0
    accepted_episode_ids: set[str] = field(default_factory=set)
    completed_handoff_id: int = 0


@dataclass
class LearnerRuntime:
    """Long-lived heads, online replay and optimizer state."""

    algorithm: HeadOnlyRLTChunkAlgorithm
    replay_buffer: ReplayBuffer
    trainer: RLTrainer
    progress: LearnerProgress


def use_threads(cfg: LearnerPipelineConfig) -> bool:
    """Return whether this learner runs its server as a thread."""
    return cfg.algorithm.concurrency.learner == "threads"


def _validate_startup_config(cfg: LearnerPipelineConfig) -> None:
    """Validate while allowing a logs-only output directory from a failed prior start."""
    output_dir = Path(cfg.output_dir)
    last_checkpoint = output_dir / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK
    if cfg.resume and not last_checkpoint.exists():
        raise RuntimeError(f"resume=true but no EvoRL checkpoint exists at {last_checkpoint}")
    if not cfg.resume and last_checkpoint.exists():
        raise RuntimeError(
            f"EvoRL checkpoint already exists at {last_checkpoint}; set resume=true or use a new output_dir"
        )
    if output_dir.exists() and not cfg.resume:
        original_output_dir = cfg.output_dir
        validation_path = output_dir.with_name(f".{output_dir.name}.validation-only")
        cfg.output_dir = validation_path
        try:
            cfg.validate()
        finally:
            cfg.output_dir = original_output_dir
        return
    cfg.validate()


def _cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _save_replay_buffer(replay_buffer: ReplayBuffer, path: Path) -> None:
    if not replay_buffer.initialized:
        return
    if replay_buffer.size < replay_buffer.capacity:
        indices = torch.arange(replay_buffer.size, device=replay_buffer.storage_device)
    else:
        indices = torch.cat(
            (
                torch.arange(
                    replay_buffer.position,
                    replay_buffer.capacity,
                    device=replay_buffer.storage_device,
                ),
                torch.arange(replay_buffer.position, device=replay_buffer.storage_device),
            )
        )
    tensor_names = (
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
        "truncateds",
        "episode_ends",
        "complementary_info",
    )
    payload = {
        "capacity": replay_buffer.capacity,
        "position": replay_buffer.position,
        "size": replay_buffer.size,
        "state_keys": list(replay_buffer.state_keys),
        "has_complementary_info": replay_buffer.has_complementary_info,
        "complementary_info_keys": list(replay_buffer.complementary_info_keys),
        **{
            name: {
                key: value[indices].detach().cpu()
                for key, value in getattr(replay_buffer, name).items()
            }
            if isinstance(getattr(replay_buffer, name), dict)
            else getattr(replay_buffer, name)[indices].detach().cpu()
            for name in tensor_names
        },
    }
    torch.save(payload, path)


def _restore_replay_buffer(replay_buffer: ReplayBuffer, path: Path) -> None:
    if not path.is_file():
        return
    payload = torch.load(path, map_location="cpu", weights_only=True)
    replay_buffer.position = int(payload["position"])
    replay_buffer.size = int(payload["size"])
    replay_buffer.has_complementary_info = bool(payload["has_complementary_info"])
    replay_buffer.complementary_info_keys = list(payload["complementary_info_keys"])
    for name in (
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
        "truncateds",
        "episode_ends",
        "complementary_info",
    ):
        value = payload[name]
        if isinstance(value, dict):
            restored = {}
            for key, tensor in value.items():
                storage = torch.empty(
                    (replay_buffer.capacity, *tensor.shape[1:]),
                    dtype=tensor.dtype,
                    device=replay_buffer.storage_device,
                )
                storage[: replay_buffer.size].copy_(tensor)
                restored[key] = storage
        else:
            restored = torch.empty(
                (replay_buffer.capacity, *value.shape[1:]),
                dtype=value.dtype,
                device=replay_buffer.storage_device,
            )
            restored[: replay_buffer.size].copy_(value)
        setattr(replay_buffer, name, restored)
    replay_buffer.position = replay_buffer.size % replay_buffer.capacity
    replay_buffer.initialized = True


def _checkpoint_dir(cfg: LearnerPipelineConfig) -> Path:
    return Path(cfg.output_dir) / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK


def _save_checkpoint(
    cfg: LearnerPipelineConfig,
    algorithm: HeadOnlyRLTChunkAlgorithm,
    replay_buffer: ReplayBuffer,
    *,
    progress: LearnerProgress,
) -> None:
    checkpoint_dir = get_step_checkpoint_dir(
        cfg.output_dir, cfg.algorithm.online_steps, progress.algorithm_step
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    algorithm.save_pretrained(checkpoint_dir / ALGORITHM_DIR)
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    training_state_dir.mkdir(parents=True, exist_ok=True)
    save_optimizer_state(algorithm.get_optimizers(), training_state_dir)
    write_json(
        {
            # Keep ``step`` for backward compatibility with earlier EvoRL checkpoints.
            "step": progress.algorithm_step,
            "algorithm_step": progress.algorithm_step,
            "online_step": progress.online_step,
            "offline_pretraining_step": progress.offline_step,
            "offline_pretraining_complete": progress.offline_complete,
            "pending_updates": progress.pending_updates,
            "accepted_episode_ids": sorted(progress.accepted_episode_ids),
            "completed_handoff_id": progress.completed_handoff_id,
        },
        training_state_dir / TRAINING_STEP,
    )
    cfg.save_pretrained(checkpoint_dir)
    _save_replay_buffer(replay_buffer, checkpoint_dir / "compact_replay.pt")
    update_last_checkpoint(checkpoint_dir)


def _load_checkpoint(
    cfg: LearnerPipelineConfig,
    algorithm: HeadOnlyRLTChunkAlgorithm,
    replay_buffer: ReplayBuffer,
) -> LearnerProgress:
    if not cfg.resume:
        return LearnerProgress(offline_complete=not cfg.offline_pretraining.enabled)
    checkpoint_dir = _checkpoint_dir(cfg)
    if not checkpoint_dir.exists():
        raise RuntimeError(f"No EvoRL checkpoint found at {checkpoint_dir}")
    algorithm_path = checkpoint_dir / ALGORITHM_DIR / SAFETENSORS_SINGLE_FILE
    algorithm.load_state_dict(load_safetensors(str(algorithm_path)), device="cpu")
    load_optimizer_state(algorithm.get_optimizers(), checkpoint_dir / TRAINING_STATE_DIR)
    state = load_json(checkpoint_dir / TRAINING_STATE_DIR / TRAINING_STEP)
    _restore_replay_buffer(replay_buffer, checkpoint_dir / "compact_replay.pt")
    algorithm_step = int(state.get("algorithm_step", state.get("step", 0)))
    offline_step = int(state.get("offline_pretraining_step", 0))
    algorithm.optimization_step = algorithm_step
    return LearnerProgress(
        algorithm_step=algorithm_step,
        online_step=int(state.get("online_step", max(algorithm_step - offline_step, 0))),
        offline_step=offline_step,
        offline_complete=bool(
            state.get("offline_pretraining_complete", not cfg.offline_pretraining.enabled)
        ),
        pending_updates=int(state.get("pending_updates", 0)),
        accepted_episode_ids=set(state.get("accepted_episode_ids", [])),
        completed_handoff_id=int(state.get("completed_handoff_id", 0)),
    )


def _start_server(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_queue: Queue,
    shutdown_event: Any,
    cfg: LearnerPipelineConfig,
) -> None:
    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.algorithm.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_queue,
        queue_get_timeout=cfg.algorithm.actor_learner_config.queue_get_timeout,
    )
    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )
    services_pb2_grpc.add_LearnerServiceServicer_to_server(service, server)
    address = (
        f"{cfg.algorithm.actor_learner_config.learner_host}:"
        f"{cfg.algorithm.actor_learner_config.learner_port}"
    )
    server.add_insecure_port(address)
    server.start()
    logging.info("[LEARNER] independent EvoRL gRPC server started at %s", address)
    shutdown_event.wait()
    server.stop(SHUTDOWN_TIMEOUT)


def _push_weights(
    parameters_queue: Queue,
    algorithm: HeadOnlyRLTChunkAlgorithm,
    *,
    handoff_id: int,
    optimization_step: int,
) -> None:
    bundle = algorithm.get_weights()
    bundle[WEIGHT_HANDOFF_ID_FIELD] = torch.tensor(handoff_id, dtype=torch.int64)
    bundle[WEIGHT_UPDATE_STEP_FIELD] = torch.tensor(optimization_step, dtype=torch.int64)
    parameters_queue.put(state_to_bytes(bundle))


def _ingest_compact_episodes(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    algorithm: HeadOnlyRLTChunkAlgorithm,
    cfg: LearnerPipelineConfig,
    accepted_episode_ids: set[str],
) -> int:
    accepted = 0
    expected_model = str(Path(cfg.policy.pretrained_path).expanduser().resolve())
    while not transition_queue.empty():
        payload = validate_compact_episode(bytes_to_episode_payload(transition_queue.get()))
        episode_id = payload["metadata"].get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Compact episode metadata must contain episode_id")
        if episode_id in accepted_episode_ids:
            logging.warning("[LEARNER] Ignoring duplicate compact episode %s", episode_id)
            continue
        feature_model = payload["feature_model"].get("resolved_path")
        if str(Path(feature_model).expanduser().resolve()) != expected_model:
            raise ValueError(
                f"Compact episode {episode_id} uses feature model {feature_model}, expected {expected_model}"
            )
        if not algorithm.ingest_transition_payload(payload, replay_buffer):
            raise ValueError(f"Compact episode {episode_id} was not accepted by rlt_chunk")
        accepted_episode_ids.add(episode_id)
        accepted += 1
        logging.info("[LEARNER] accepted compact episode=%s replay=%d", episode_id, len(replay_buffer))
    return accepted


def _process_interactions(interaction_queue: Queue, wandb_logger: WandBLogger | None):
    latest_release_id = None
    while not interaction_queue.empty():
        message = bytes_to_python_object(interaction_queue.get())
        control = parse_control_message(message)
        if control is not None:
            kind, handoff_id = control
            if kind == ACTOR_GPU_RELEASED:
                latest_release_id = handoff_id
                logging.info("[LEARNER] actor confirmed GPU release for handoff=%d", handoff_id)
            continue
        if wandb_logger and isinstance(message, dict) and "Interaction step" in message:
            wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")
    return latest_release_id


def _build_training_runtime(cfg: LearnerPipelineConfig) -> LearnerRuntime:
    """Construct the persistent head-only learner and restore it when requested."""
    training_device = torch.device(cfg.policy.device)
    needs_startup_cpu = cfg.gpu_handoff.enabled or cfg.offline_pretraining.enabled
    initial_device = torch.device("cpu") if needs_startup_cpu else training_device
    algorithm = HeadOnlyRLTChunkAlgorithm(cfg.policy, cfg.algorithm, device=initial_device)
    replay_buffer = ReplayBuffer(
        capacity=cfg.algorithm.online_buffer_capacity,
        device=str(training_device),
        state_keys=cfg.algorithm.state_keys,
        storage_device=cfg.algorithm.storage_device,
        optimize_memory=False,
        use_drq=False,
    )
    trainer = RLTrainer(
        algorithm=algorithm,
        data_mixer=OnlineReplayMixer(replay_buffer),
        batch_size=cfg.batch_size,
        preprocessor=None,
    )
    progress = _load_checkpoint(cfg, algorithm, replay_buffer)
    return LearnerRuntime(algorithm, replay_buffer, trainer, progress)


def _progress_postfix(values: dict[str, float]) -> dict[str, str]:
    """Select the useful RLT losses for a compact tqdm display."""
    aliases = {
        "critic": "loss_critic",
        "actor": "loss_actor",
        "bc": "bc_loss",
        "q": "q",
    }
    return {
        label: f"{float(values[key]):.4g}"
        for label, key in aliases.items()
        if key in values
    }


def _run_offline_pretraining(
    cfg: LearnerPipelineConfig,
    runtime: LearnerRuntime,
    wandb_logger: WandBLogger | None,
    shutdown_event: Any,
) -> bool:
    """Load compact features, train both RLT heads, and release startup CUDA state."""
    offline_cfg = cfg.offline_pretraining
    if not offline_cfg.enabled:
        return False
    progress = runtime.progress
    if progress.offline_complete:
        logging.info(
            "[LEARNER][OFFLINE] checkpoint already completed %d/%d startup updates",
            progress.offline_step,
            offline_cfg.steps,
        )
    else:
        logging.info(
            "[LEARNER][OFFLINE] loading pre-extracted compact episodes; steps=%d",
            offline_cfg.steps,
        )
        offline_replay = load_compact_replay(
            cfg,
            runtime.algorithm,
            shutdown_event=shutdown_event,
        )
        runtime.algorithm.to_device(cfg.policy.device)
        runtime.trainer.set_data_mixer(OnlineReplayMixer(offline_replay))
        bar = tqdm(
            total=offline_cfg.steps,
            initial=min(progress.offline_step, offline_cfg.steps),
            desc="Offline Actor/Critic training",
            unit="step",
            dynamic_ncols=True,
        )
        training_completed = False
        try:
            while progress.offline_step < offline_cfg.steps:
                if shutdown_event.is_set():
                    raise KeyboardInterrupt("offline pretraining interrupted")
                stats = runtime.trainer.training_step()
                progress.offline_step += 1
                progress.algorithm_step = runtime.algorithm.optimization_step
                values = stats.to_log_dict()
                bar.update(1)
                bar.set_postfix(_progress_postfix(values), refresh=False)
                if progress.offline_step % cfg.log_freq == 0 and wandb_logger:
                    wandb_logger.log_dict(
                        d={
                            **values,
                            "Offline pretraining step": progress.offline_step,
                            "Algorithm step": progress.algorithm_step,
                            "offline_replay_size": len(offline_replay),
                        },
                        mode="train",
                        custom_step_key="Algorithm step",
                    )
                if (
                    cfg.save_checkpoint
                    and cfg.save_freq > 0
                    and progress.offline_step % cfg.save_freq == 0
                ):
                    _save_checkpoint(cfg, runtime.algorithm, runtime.replay_buffer, progress=progress)
            training_completed = True
        finally:
            bar.close()
            runtime.trainer.set_data_mixer(OnlineReplayMixer(runtime.replay_buffer))
            del offline_replay
            gc.collect()
            if not training_completed:
                runtime.algorithm.to_device("cpu")
                _cuda_cleanup()

        progress.offline_complete = True
        runtime.trainer.reset_data_iterator()
        runtime.algorithm.to_device("cpu")
        _cuda_cleanup()
        logging.info(
            "[LEARNER][OFFLINE] completed Actor/Critic pretraining: steps=%d",
            progress.offline_step,
        )

    # This stable root artifact lets an Actor started after pretraining load the
    # initial weights before its gRPC receive thread has delivered a bundle.
    runtime.algorithm.save_pretrained(Path(cfg.output_dir) / ALGORITHM_DIR)
    if cfg.save_checkpoint:
        _save_checkpoint(cfg, runtime.algorithm, runtime.replay_buffer, progress=progress)
    return True


def _run_training_loop(
    cfg: LearnerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: Any,
    transition_queue: Queue,
    interaction_queue: Queue,
    parameters_queue: Queue,
    runtime: LearnerRuntime | None = None,
) -> None:
    handoff = cfg.gpu_handoff
    training_device = torch.device(cfg.policy.device)
    runtime = runtime or _build_training_runtime(cfg)
    algorithm = runtime.algorithm
    replay_buffer = runtime.replay_buffer
    trainer = runtime.trainer
    progress = runtime.progress
    if not handoff.enabled:
        algorithm.to_device(training_device)
    actor_release_id: int | None = None
    quota_ready_since: float | None = None
    threshold = handoff.update_quota_threshold

    logging.info(
        "[LEARNER] head-only runtime ready device=%s threshold=%d updates_per_episode=%d",
        algorithm._device,
        threshold,
        handoff.updates_per_episode,
    )
    while not shutdown_event.is_set() and progress.online_step < cfg.algorithm.online_steps:
        accepted = _ingest_compact_episodes(
            transition_queue, replay_buffer, algorithm, cfg, progress.accepted_episode_ids
        )
        if accepted:
            progress.pending_updates += accepted * handoff.updates_per_episode
            logging.info("[LEARNER] pending update quota=%d", progress.pending_updates)
        release_id = _process_interactions(interaction_queue, wandb_logger)
        if release_id is not None:
            actor_release_id = release_id

        ready = (
            progress.pending_updates >= threshold
            and len(replay_buffer) >= cfg.algorithm.online_step_before_learning
        )
        if not ready:
            quota_ready_since = None
            shutdown_event.wait(0.02)
            continue
        if handoff.enabled and actor_release_id is None:
            quota_ready_since = quota_ready_since or time.monotonic()
            if time.monotonic() - quota_ready_since > handoff.actor_release_timeout_s:
                raise TimeoutError("Actor did not confirm GPU release before learner timeout")
            shutdown_event.wait(0.02)
            continue

        active_handoff_id = actor_release_id or progress.completed_handoff_id + 1
        quota_ready_since = None
        if handoff.enabled:
            algorithm.to_device(training_device)
            logging.info("[LEARNER] acquired GPU for handoff=%d", active_handoff_id)
        burst_stats = None
        online_step_before_burst = progress.online_step
        for _ in range(threshold):
            if shutdown_event.is_set() or progress.online_step >= cfg.algorithm.online_steps:
                break
            burst_stats = trainer.training_step()
            progress.algorithm_step = algorithm.optimization_step
            progress.online_step += 1
            if progress.online_step % cfg.log_freq == 0:
                values = burst_stats.to_log_dict()
                values.update(
                    {
                        "Optimization step": progress.online_step,
                        "Algorithm step": progress.algorithm_step,
                        "replay_buffer_size": len(replay_buffer),
                    }
                )
                logging.info("[LEARNER] online_step=%d stats=%s", progress.online_step, values)
                if wandb_logger:
                    wandb_logger.log_dict(
                        d=values, mode="train", custom_step_key="Optimization step"
                    )

        progress.pending_updates -= threshold
        if handoff.enabled:
            # Closing the replay iterator also stops its CUDA prefetch worker;
            # otherwise queued batches keep allocations alive after the heads move to CPU.
            trainer.reset_data_iterator()
            gc.collect()
            algorithm.to_device("cpu")
            _cuda_cleanup()
            logging.info("[LEARNER] released GPU for handoff=%d", active_handoff_id)
        _push_weights(
            parameters_queue,
            algorithm,
            handoff_id=active_handoff_id,
            optimization_step=progress.algorithm_step,
        )
        progress.completed_handoff_id = active_handoff_id
        actor_release_id = None

        crossed_save_boundary = (
            cfg.save_freq > 0
            and progress.online_step // cfg.save_freq
            != online_step_before_burst // cfg.save_freq
        )
        if cfg.save_checkpoint and crossed_save_boundary:
            _save_checkpoint(cfg, algorithm, replay_buffer, progress=progress)


def _start_runtime(
    cfg: LearnerPipelineConfig, wandb_logger: WandBLogger | None, shutdown_event: Any
) -> None:
    transition_queue = Queue()
    interaction_queue = Queue()
    parameters_queue = Queue()
    server_worker = None
    try:
        runtime = _build_training_runtime(cfg)
        has_offline_initialization = _run_offline_pretraining(
            cfg,
            runtime,
            wandb_logger,
            shutdown_event,
        )
        if has_offline_initialization:
            _push_weights(
                parameters_queue,
                runtime.algorithm,
                handoff_id=0,
                optimization_step=runtime.progress.algorithm_step,
            )
            logging.info(
                "[LEARNER][OFFLINE] initial Actor weights ready; starting transport service"
            )
        if use_threads(cfg):
            from threading import Thread as Worker
        else:
            from torch.multiprocessing import Process as Worker
        server_worker = Worker(
            target=_start_server,
            args=(parameters_queue, transition_queue, interaction_queue, shutdown_event, cfg),
            daemon=True,
        )
        server_worker.start()
        _run_training_loop(
            cfg,
            wandb_logger,
            shutdown_event,
            transition_queue,
            interaction_queue,
            parameters_queue,
            runtime,
        )
    finally:
        shutdown_event.set()
        if server_worker is not None:
            server_worker.join(timeout=SHUTDOWN_TIMEOUT + 2)
        for queue in (transition_queue, interaction_queue, parameters_queue):
            queue.close()
            queue.cancel_join_thread()


def train(cfg: LearnerPipelineConfig, job_name: str | None = None) -> None:
    """Run the head-only EvoRL learner with optional compact offline pretraining."""
    _validate_startup_config(cfg)
    job_name = job_name or cfg.job_name
    if not job_name:
        raise ValueError("Learner job_name is required")
    log_dir = Path(cfg.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"learner_{job_name}.log"
    init_logging(log_file=str(log_file), display_pid=not use_threads(cfg))
    logging.info("[LEARNER] independent onlineRL_evoRL runtime; no generic learner delegation")
    logging.info(pformat(cfg.to_dict()))
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    shutdown_event = ProcessSignalHandler(
        use_threads(cfg), display_pid=not use_threads(cfg)
    ).shutdown_event
    _start_runtime(cfg, wandb_logger, shutdown_event)


@parser.wrap()
def train_cli(cfg: LearnerPipelineConfig):
    """Parse the EvoRL learner JSON and start its independent runtime."""
    require_package("grpcio", extra="hilserl", import_name="grpc")
    if not use_threads(cfg):
        ensure_multiprocessing_start_method(cfg.algorithm.concurrency.multiprocessing_context)
    return train(cfg, job_name=cfg.job_name)


if __name__ == "__main__":
    train_cli()
