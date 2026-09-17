"""EvoRL-owned gRPC learner service over LeRobot's byte transport."""

from __future__ import annotations

import logging
import time
from multiprocessing import Event, Queue
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _grpc_available

from .queue import get_last_item_from_queue

if TYPE_CHECKING or _grpc_available:
    import grpc

    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import receive_bytes_in_chunks, send_bytes_in_chunks

    _ServicerBase = services_pb2_grpc.LearnerServiceServicer
else:
    grpc = None
    services_pb2 = None
    services_pb2_grpc = None
    receive_bytes_in_chunks = None
    send_bytes_in_chunks = None
    _ServicerBase = object

MAX_WORKERS = 3
SHUTDOWN_TIMEOUT = 10


class LearnerService(_ServicerBase):
    """Stream weights and receive compact episodes/control messages for EvoRL."""

    def __init__(
        self,
        shutdown_event: Event,
        parameters_queue: Queue,
        seconds_between_pushes: float,
        transition_queue: Queue,
        interaction_message_queue: Queue,
        queue_get_timeout: float = 0.001,
    ) -> None:
        """Store the queues used by the independent EvoRL runtime."""
        self.shutdown_event = shutdown_event
        self.parameters_queue = parameters_queue
        self.seconds_between_pushes = seconds_between_pushes
        self.transition_queue = transition_queue
        self.interaction_message_queue = interaction_message_queue
        self.queue_get_timeout = queue_get_timeout

    def StreamParameters(self, request: services_pb2.Empty, context: grpc.ServicerContext):  # noqa: N802
        """Stream the latest completed RLT actor update to the actor."""
        del request, context
        logging.info("[LEARNER] Actor connected to EvoRL parameter stream")
        last_push_time = 0.0
        while not self.shutdown_event.is_set():
            remaining = self.seconds_between_pushes - (time.time() - last_push_time)
            if remaining > 0:
                self.shutdown_event.wait(remaining)
                continue
            buffer = get_last_item_from_queue(
                self.parameters_queue, block=True, timeout=self.queue_get_timeout
            )
            if buffer is None:
                continue
            yield from send_bytes_in_chunks(
                buffer,
                services_pb2.Parameters,
                log_prefix="[LEARNER] Sending EvoRL weights",
                silent=True,
            )
            last_push_time = time.time()
        return services_pb2.Empty()

    def SendTransitions(self, request_iterator, _context: grpc.ServicerContext):  # noqa: N802
        """Receive compact episode byte streams."""
        receive_bytes_in_chunks(
            request_iterator,
            self.transition_queue,
            self.shutdown_event,
            log_prefix="[LEARNER] compact episodes",
        )
        return services_pb2.Empty()

    def SendInteractions(self, request_iterator, _context: grpc.ServicerContext):  # noqa: N802
        """Receive metrics and GPU handoff controls."""
        receive_bytes_in_chunks(
            request_iterator,
            self.interaction_message_queue,
            self.shutdown_event,
            log_prefix="[LEARNER] interactions/control",
        )
        return services_pb2.Empty()

    def Ready(self, request: services_pb2.Empty, context: grpc.ServicerContext):  # noqa: N802
        """Report that the independent learner service is ready."""
        del request, context
        return services_pb2.Empty()


__all__ = ["MAX_WORKERS", "SHUTDOWN_TIMEOUT", "LearnerService"]
