"""Episode-counted replay buffer and transition sampler."""

from __future__ import annotations

import random
from collections import deque

from .protocol import Episode, Transition


class EpisodeReplayBuffer:
    """FIFO replay whose capacity and readiness are defined in complete episodes."""

    def __init__(self, capacity: int, sample_window: int):
        self._episodes: deque[Episode] = deque(maxlen=capacity)
        self.sample_window = sample_window

    def __len__(self) -> int:
        return len(self._episodes)

    def add(self, episode: Episode) -> None:
        if not episode.transitions:
            raise ValueError("Cannot replay an empty episode.")
        self._episodes.append(episode)

    def sample(self, batch_size: int) -> list[Transition]:
        if not self._episodes:
            raise RuntimeError("Cannot sample empty replay.")
        candidates = list(self._episodes)[-self.sample_window :]
        return [random.choice(random.choice(candidates).transitions) for _ in range(batch_size)]

    @property
    def episode_ids(self) -> list[str]:
        return [episode.episode_id for episode in self._episodes]
