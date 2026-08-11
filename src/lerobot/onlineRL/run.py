"""CLI entrypoint for dry-run local Piper RLT actor-critic execution."""

from __future__ import annotations

import argparse
import logging
import time

from .actor_runtime import ActorRuntime
from .config import load_config
from .hardware import make_hardware
from .keyboard import KeyboardController
from .learner_runtime import Learner
from .transport import LocalTransport
from .vla import load_feature_extractor


EPISODE_START_COUNTDOWN_S = 5


def _countdown_before_episode(delay_s: int = EPISODE_START_COUNTDOWN_S) -> None:
    logging.info("Next episode starts in %d seconds.", delay_s)
    for remaining in range(delay_s, 0, -1):
        logging.info("Episode starts in %d...", remaining)
        time.sleep(1.0)
    logging.info("Episode starts now.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--mode", choices=("local", "actor", "learner"), default=None)
    parser.add_argument("--actor_device", default=None)
    parser.add_argument("--learner_device", default=None)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--display_data", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug_every_steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to run; defaults to 1 for dry-run and continuous for real hardware.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config_path)
    for name in ("mode", "actor_device", "learner_device", "dry_run", "display_data", "debug_every_steps"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.runtime, name, value)
    cfg.validate()
    if cfg.runtime.mode == "learner":
        raise NotImplementedError("Standalone learner mode requires the reserved remote transport.")
    logging.info("onlineRL mode=%s dry_run=%s display_data=%s", cfg.runtime.mode, cfg.runtime.dry_run, cfg.runtime.display_data)
    logging.info(
        "onlineRL hardware control_hz=%.2f action_limit=%.2f top_camera=%s wrist_camera=%s",
        cfg.hardware.control_hz,
        cfg.hardware.action_limit,
        cfg.hardware.top_camera,
        cfg.hardware.wrist_camera,
    )
    logging.info(
        "onlineRL Stage1 model_path=%s norm_stats_path=%s task=%s",
        cfg.runtime.feature_extractor_kwargs.get("model_path"),
        cfg.runtime.feature_extractor_kwargs.get("norm_stats_path"),
        cfg.runtime.feature_extractor_kwargs.get("task_description"),
    )
    hardware = make_hardware(cfg.hardware, cfg.runtime.dry_run, cfg.runtime.display_data)
    extractor = load_feature_extractor(
        cfg.runtime.feature_extractor_factory,
        cfg.model,
        cfg.runtime.dry_run,
        cfg.runtime.feature_extractor_kwargs,
    )
    keyboard = KeyboardController()
    actor = ActorRuntime(cfg, hardware, extractor, keyboard)
    learner = Learner(cfg) if cfg.runtime.mode == "local" else None
    transport = LocalTransport() if learner is not None else None
    listener = None
    if not cfg.runtime.dry_run:
        listener = keyboard.start_pynput_listener()
    target_episodes = args.episodes if args.episodes is not None else (1 if cfg.runtime.dry_run else None)
    completed = 0
    try:
        while target_episodes is None or completed < target_episodes:
            if not cfg.runtime.dry_run:
                _countdown_before_episode()
            episode = actor.rollout_episode()
            if episode is None:
                logging.info("Episode was aborted with reset and intentionally not learned from.")
                continue
            outbox_path = actor.persist_outbox(episode)
            if learner is None or transport is None:
                completed += 1
                logging.info(
                    "Actor persisted %s episode with %d transitions to %s",
                    episode.outcome,
                    len(episode.transitions),
                    outbox_path,
                )
                continue
            transport.send_episode(episode)
            received = transport.receive_episode()
            if received is not None:
                weights = learner.ingest(received)
                actor.acknowledge_outbox(outbox_path)
                if weights is not None:
                    transport.publish_weights(weights)
            actor.apply_weights(transport.receive_weights())
            completed += 1
            logging.info(
                "Completed %s episode with %d transitions; replay episodes=%d",
                episode.outcome,
                len(episode.transitions),
                len(learner.replay),
            )
    finally:
        if listener is not None:
            listener.stop()
        actor.close()
        hardware.disconnect()


if __name__ == "__main__":
    main()
