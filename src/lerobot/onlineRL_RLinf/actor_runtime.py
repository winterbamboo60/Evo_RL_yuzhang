"""Execution actor: Piper control, episode construction and learner weight application."""

from __future__ import annotations

import uuid
import logging
import time
from collections import deque
from pathlib import Path

import torch

from .config import OnlineRLConfig
from .hardware import PiperHardware
from .keyboard import ControlState, KeyboardController
from .protocol import ActorWeights, Episode, Transition
from .rlt_model import RLTActorCritic
from .vla import FeatureExtractor, RLTFeatures, validate_features


ACTION_QUEUE_LOW_WATERMARK = 20
ACTION_QUEUE_POP_LIMIT = 20


class _TensorActionSmoother:
    def __init__(self, action_dim: int, window: int = 9):
        self.buffers = [deque(maxlen=window if window % 2 == 1 else window + 1) for _ in range(action_dim)]

    def reset(self) -> None:
        for buffer in self.buffers:
            buffer.clear()

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(action, dtype=torch.float32).reshape(-1).tolist()
        smoothed = []
        for value, buffer in zip(values, self.buffers, strict=True):
            buffer.append(float(value))
            smoothed.append(sum(buffer) / len(buffer))
        out = torch.tensor(smoothed, dtype=torch.float32)
        if out.numel() >= 7:
            out[-1] = _attenuate_gripper_value(float(out[-1]))
        return out


def _attenuate_gripper_value(value: float) -> float:
    raw = round(float(value) * 1000.0)
    ratio = min(max(raw, 0), 101100) / 101100
    if ratio >= 0.4:
        attenuated_raw = int(ratio * 101100)
    elif ratio >= 0.04:
        attenuated_raw = int(((ratio - 0.04) / 0.36) ** 3 * 0.4 * 101100)
    else:
        attenuated_raw = 0
    return attenuated_raw / 1000.0


class ActorRuntime:
    """Owns inference-only actor weights and never owns a critic or optimizer."""

    def __init__(self, cfg: OnlineRLConfig, hardware: PiperHardware, extractor: FeatureExtractor, keyboard: KeyboardController):
        self.cfg, self.hardware, self.extractor, self.keyboard = cfg, hardware, extractor, keyboard
        self.device = torch.device("cpu" if cfg.runtime.dry_run or not torch.cuda.is_available() else cfg.runtime.actor_device)
        self.actor = RLTActorCritic(cfg.model).to(self.device).eval()
        self.actor_version = 0
        self.initial_pose: torch.Tensor | None = None
        self._last_observation: dict | None = None
        self._action_queue: deque[tuple[torch.Tensor, RLTFeatures]] = deque()
        self._action_queue_source: str | None = None
        self._action_smoother = _TensorActionSmoother(cfg.model.action_dim)

    def apply_weights(self, weights: ActorWeights | None) -> None:
        if weights is None or weights.version <= self.actor_version:
            return
        self.actor.load_actor_state_dict(weights.state_dict)
        self.actor_version = weights.version

    def _extract_features(self, observation: dict) -> RLTFeatures:
        features = self.extractor(observation)
        validate_features(features, self.cfg.model)
        return features

    def _features(self, observation: dict | None = None) -> RLTFeatures:
        self._last_observation = self.hardware.observation() if observation is None else observation
        return self._extract_features(self._last_observation)

    def close(self) -> None:
        pass

    @staticmethod
    def _range(value: object) -> str:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        return f"shape={tuple(tensor.shape)} min={tensor.min().item():.4f} max={tensor.max().item():.4f} mean={tensor.mean().item():.4f}"

    @staticmethod
    def _values(value: object) -> str:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        return "[" + ", ".join(f"{item:.4f}" for item in tensor.tolist()) + "]"

    def _log_step(self, step: int, source: str, features: RLTFeatures, model_action: torch.Tensor, command: torch.Tensor, executed: torch.Tensor) -> None:
        if step % self.cfg.runtime.debug_every_steps != 0:
            return
        obs = self._last_observation or {}
        logging.info(
            "step=%d source=%s state[%s] top[%s] wrist[%s]",
            step,
            source,
            self._range(obs["state"]),
            self._range(obs["top_image"]),
            self._range(obs["wrist_image"]),
        )
        logging.info(
            "step=%d z_rl[%s] proprio[%s] ref_chunk[%s] command[%s] executed[%s]",
            step,
            self._range(features.z_rl),
            self._range(features.proprio),
            self._range(features.ref_chunk),
            self._range(command),
            self._range(executed),
        )
        logging.info(
            "step=%d model_action_values=%s command_values=%s executed_values=%s observed_state_values=%s",
            step,
            self._values(model_action),
            self._values(command),
            self._values(executed),
            self._values(obs["state"]),
        )

    def _actor_chunk(self, features: RLTFeatures) -> torch.Tensor:
        with torch.no_grad():
            return self.actor.act(features.z_rl.to(self.device), features.proprio.to(self.device), features.ref_chunk.to(self.device)).cpu()[0]

    def _clear_action_queue(self) -> None:
        self._action_queue.clear()
        self._action_queue_source = None

    def _fill_action_queue(self, source: str, features: RLTFeatures) -> None:
        queue_before = len(self._action_queue)
        chunk = features.ref_chunk[0, : self.cfg.model.num_action_chunks] if source == "vla" else self._actor_chunk(features)
        self._action_queue.extend((action.detach().cpu(), features) for action in chunk)
        self._action_queue_source = source
        mid = min(len(chunk) - 1, len(chunk) // 2)
        logging.info(
            "action_queue_fill source=%s before=%d added=%d after=%d chunk[%s] first=%s mid=%s last=%s ref_chunk[%s] z_rl[%s] proprio=%s",
            source,
            queue_before,
            len(chunk),
            len(self._action_queue),
            self._range(chunk),
            self._values(chunk[0]),
            self._values(chunk[mid]),
            self._values(chunk[-1]),
            self._range(features.ref_chunk),
            self._range(features.z_rl),
            self._values(features.proprio),
        )

    def _queue_next_features(self, fallback: RLTFeatures) -> RLTFeatures:
        return self._action_queue[0][1] if self._action_queue else fallback

    def _sleep_step(self, started_at: float) -> None:
        remaining = 1.0 / self.cfg.hardware.control_hz - (time.monotonic() - started_at)
        if remaining > 0:
            time.sleep(remaining)

    def rollout_episode(self) -> Episode | None:
        """Run one episode; R aborts and returns ``None`` without learner ingestion."""
        self.hardware.connect()
        if self.initial_pose is None:
            self.initial_pose = self.hardware.observation()["state"].detach().clone()
        state = ControlState()
        transitions: list[Transition] = []
        step = 0
        self._clear_action_queue()
        features = self._features()
        self._action_smoother.reset()
        while step < self.cfg.runtime.max_episode_steps:
            state = self.keyboard.poll(state)
            if state.reset_requested:
                self.hardware.set_manual_control(False)
                self.hardware.reset_to(self.initial_pose)
                return None
            self.hardware.set_manual_control(state.human_control)
            source = "human" if state.human_control else state.action_source
            if source == "human":
                self._clear_action_queue()
            elif self._action_queue_source not in {None, source}:
                self._clear_action_queue()
            if source != "human" and len(self._action_queue) < ACTION_QUEUE_LOW_WATERMARK:
                if self._action_queue_source is not None:
                    features = self._features(self._last_observation)
                self._fill_action_queue(source, features)
            executed_actions: list[torch.Tensor] = []
            transition_features: RLTFeatures | None = None
            terminal_event = None
            timeout = False
            steps_to_run = self.cfg.model.num_action_chunks if source == "human" else min(ACTION_QUEUE_POP_LIMIT, len(self._action_queue))
            for _ in range(steps_to_run):
                started_at = time.monotonic()
                state = self.keyboard.poll(state)
                if state.reset_requested:
                    self.hardware.set_manual_control(False)
                    self.hardware.reset_to(self.initial_pose)
                    return None
                if source != ("human" if state.human_control else state.action_source):
                    self._clear_action_queue()
                    break
                if state.human_control:
                    executed = self.hardware.send_human()
                    model_action = executed
                    command = executed
                    action_features = features
                else:
                    if not self._action_queue or self._action_queue_source != source:
                        break
                    model_action, action_features = self._action_queue.popleft()
                    command = self._action_smoother(model_action)
                    executed = self.hardware.send_automatic(command)
                    delta = (executed.detach().cpu() - command.detach().cpu()).abs()
                    if delta.max().item() > 1.0:
                        logging.warning(
                            "execution_mismatch step=%d source=%s max_abs_delta=%.4f command=%s executed=%s",
                            step + 1,
                            source,
                            delta.max().item(),
                            self._values(command),
                            self._values(executed),
                        )
                transition_features = transition_features or action_features
                executed_actions.append(executed.detach().cpu())
                step += 1
                self._last_observation = self.hardware.observation()
                self._log_step(step, source, action_features, model_action, command, executed)
                terminal_event = state.terminal_event
                timeout = step >= self.cfg.runtime.max_episode_steps
                state.terminal_event = None
                self._sleep_step(started_at)
                if terminal_event is not None or timeout:
                    break
            if not executed_actions or transition_features is None:
                continue
            next_features = self._queue_next_features(transition_features)
            action_chunk = torch.stack(executed_actions).unsqueeze(0)
            if action_chunk.shape[1] < self.cfg.model.num_action_chunks:
                pad = action_chunk[:, -1:].repeat(1, self.cfg.model.num_action_chunks - action_chunk.shape[1], 1)
                action_chunk = torch.cat((action_chunk, pad), dim=1)
            terminal = terminal_event is not None or timeout
            transitions.append(Transition(
                z_rl=transition_features.z_rl.cpu(), proprio=transition_features.proprio.cpu(), ref_chunk=transition_features.ref_chunk.cpu(), action=action_chunk.cpu(),
                next_z_rl=next_features.z_rl.cpu(), next_proprio=next_features.proprio.cpu(), next_ref_chunk=next_features.ref_chunk.cpu(),
                reward=1.0 if terminal_event == "success" else 0.0,
                terminal=terminal, truncated=timeout, action_source=source, intervention=(source == "human"),
            ))
            features = next_features
            if terminal:
                outcome = "success" if terminal_event == "success" else ("timeout" if timeout else "failure")
                return Episode(str(uuid.uuid4()), transitions, outcome, self.actor_version)
        return Episode(str(uuid.uuid4()), transitions, "timeout", self.actor_version)

    def persist_outbox(self, episode: Episode) -> Path:
        path = Path(self.cfg.runtime.outbox_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{episode.episode_id}.pt"
        torch.save(episode, target)
        return target

    @staticmethod
    def acknowledge_outbox(path: Path) -> None:
        """Delete a persisted episode only after learner ingestion succeeded."""
        path.unlink(missing_ok=True)
