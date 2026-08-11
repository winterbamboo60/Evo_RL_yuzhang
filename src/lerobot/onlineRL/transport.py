"""Local queues and a deliberately narrow transport boundary for remote migration."""

from __future__ import annotations

from queue import Empty, Queue
from typing import Generic, TypeVar

from .protocol import ActorWeights, Episode

T = TypeVar("T")


class LocalTransport:
    """In-memory transport used by the default local deployment and tests."""

    def __init__(self) -> None:
        self.episodes: Queue[Episode] = Queue()
        self.weights: Queue[ActorWeights] = Queue()

    def send_episode(self, episode: Episode) -> None:
        self.episodes.put(episode)

    def receive_episode(self) -> Episode | None:
        try:
            return self.episodes.get_nowait()
        except Empty:
            return None

    def publish_weights(self, weights: ActorWeights) -> None:
        self.weights.put(weights)

    def receive_weights(self) -> ActorWeights | None:
        latest = None
        while True:
            try:
                latest = self.weights.get_nowait()
            except Empty:
                return latest


class GrpcTransport:
    """Reserved remote transport boundary.

    The message contract is intentionally limited to complete episodes and actor
    weights. A production gRPC service can be added here without changing actor
    control, replay, or learner logic.
    """

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port

    def unavailable(self) -> None:
        raise NotImplementedError("Remote gRPC transport is reserved; use LocalTransport in this initial implementation.")
