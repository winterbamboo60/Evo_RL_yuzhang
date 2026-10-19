# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Send live previews from a separate process without waiting for the viewer.

Only previews use this mailbox. Dataset frames and robot actions must keep using
their normal recording/control paths. A slow viewer sees the latest snapshot,
instead of a growing queue of old frames.
"""

import contextlib
import logging
import math
import multiprocessing
import pickle
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_MAILBOX_BYTES = 16 * 1024 * 1024
_START_TIMEOUT_S = 30.0
_CLOSE_TIMEOUT_S = 2.0
_TERMINATE_TIMEOUT_S = 1.0


def _load_backend():
    # Keep SDK imports and all viewer operations in the sender process.
    from . import visualization_utils

    return visualization_utils


def _send_status(connection, status: str, detail: str | None = None) -> None:
    # The pipe carries only a few small status messages, never images.
    with contextlib.suppress(BrokenPipeError, EOFError, OSError):
        connection.send((status, detail[:1000] if detail else None))


def _visualization_worker(
    buffer,
    length,
    mailbox_lock,
    stop,
    status_connection,
    config: dict,
    backend_factory: Callable,
) -> None:
    backend = None
    attempted_init = False
    buffer_view = memoryview(buffer).cast("B")
    try:
        backend = backend_factory()
        attempted_init = True
        backend.init_visualization(
            config["display_mode"],
            session_name=config["session_name"],
            ip=config["ip"],
            port=config["port"],
        )
        _send_status(status_connection, "ready")

        while not stop.value:
            with mailbox_lock:
                if length.value:
                    # Release the mailbox before unpickling or calling the SDK.
                    # The producer can overwrite the next pending preview while
                    # this snapshot is being displayed.
                    payload = bytes(buffer_view[: length.value])
                    length.value = 0
                else:
                    payload = None
            if payload is not None:
                observation, action = pickle.loads(payload)
                backend.log_visualization_data(
                    config["display_mode"],
                    observation=observation,
                    action=action,
                    compress_images=config["compress_images"],
                )
            else:
                time.sleep(0.01)
    except Exception as error:
        _send_status(status_connection, "error", f"{type(error).__name__}: {error}")
    finally:
        if backend is not None and attempted_init:
            try:
                backend.shutdown_visualization(config["display_mode"])
            except Exception as error:
                _send_status(status_connection, "error", f"Preview shutdown failed: {error}")
        buffer_view.release()
        status_connection.close()


class AsyncVisualization:
    """A rate-limited, fixed-capacity latest-preview mailbox.

    ``start()`` waits for SDK initialization and should run before connecting
    hardware. ``submit()`` never waits for a busy sender: it either copies a
    snapshot into shared memory or skips that preview. Serialization and the
    bounded memory copy still execute in the caller, at most ``max_fps`` times
    per second. SDK errors after startup disable previews without raising in
    the control loop.
    """

    def __init__(
        self,
        display_mode: str,
        *,
        session_name: str = "recording",
        ip: str | None = None,
        port: int | None = None,
        compress_images: bool = False,
        max_fps: float = 15.0,
        _backend_factory: Callable[[], Any] | None = None,
    ):
        if not math.isfinite(max_fps) or max_fps <= 0:
            raise ValueError("Preview max_fps must be a finite positive number.")
        self._config = {
            "display_mode": display_mode,
            "session_name": session_name,
            "ip": ip,
            "port": port,
            "compress_images": compress_images,
        }
        self._backend_factory = _backend_factory or _load_backend
        self._interval_s = 1.0 / max_fps
        self._last_submit_t = -math.inf
        self._context = multiprocessing.get_context("spawn")
        self._buffer = None
        self._length = None
        self._lock = None
        self._stop = None
        self._status = None
        self._process = None
        self._started = False
        self._closed = False
        self._disabled = False
        self._warned_oversize = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot restart a closed preview sender.")
        if self._started:
            return
        self._buffer = self._context.RawArray("B", _MAILBOX_BYTES)
        self._length = self._context.RawValue("I", 0)
        self._lock = self._context.Lock()
        # A raw byte lets both submit and close avoid Event/Condition locks,
        # even if the worker is paused or dies while holding a mailbox lock.
        self._stop = self._context.RawValue("B", 0)
        self._status, worker_status = self._context.Pipe(duplex=False)
        self._process = self._context.Process(
            target=_visualization_worker,
            args=(
                self._buffer,
                self._length,
                self._lock,
                self._stop,
                worker_status,
                self._config,
                self._backend_factory,
            ),
            name="lerobot-preview-sender",
            daemon=True,
        )
        try:
            self._process.start()
            worker_status.close()
            if not self._status.poll(_START_TIMEOUT_S):
                raise RuntimeError(f"Preview sender did not initialize within {_START_TIMEOUT_S:g} seconds.")
            try:
                status, detail = self._status.recv()
            except EOFError as error:
                raise RuntimeError("Preview sender exited during initialization.") from error
            if status != "ready":
                raise RuntimeError(f"Preview sender initialization failed: {detail}")
            self._started = True
        except BaseException:
            worker_status.close()
            self.close()
            raise

    def _disable(self, reason: str) -> None:
        if not self._disabled:
            self._disabled = True
            logger.warning("Live preview disabled: %s. Robot control and dataset recording continue.", reason)

    def _check_worker(self) -> bool:
        if self._disabled or self._process is None:
            return False
        if self._status is not None:
            try:
                while self._status.poll():
                    status, detail = self._status.recv()
                    if status == "error":
                        self._disable(detail or "sender failed")
                        return False
            except (EOFError, OSError):
                self._disable("sender exited")
                return False
        if not self._process.is_alive():
            self._disable(f"sender exited with code {self._process.exitcode}")
            return False
        return True

    def submit(self, *, observation=None, action=None) -> bool:
        if self._closed or not self._started or not self._check_worker():
            return False
        if observation is None and action is None:
            return False
        now = time.monotonic()
        if now - self._last_submit_t < self._interval_s:
            return False
        if not self._lock.acquire(False):
            return False
        try:
            # Limit attempts as well as accepted snapshots, so an oversized
            # payload cannot trigger expensive serialization on every tick.
            self._last_submit_t = now
            # Pickle copies arrays into the snapshot, so a later mutation in the
            # producer cannot change the preview already submitted.
            payload = pickle.dumps((observation, action), protocol=pickle.HIGHEST_PROTOCOL)
            if len(payload) > _MAILBOX_BYTES:
                if not self._warned_oversize:
                    self._warned_oversize = True
                    logger.warning(
                        "Live preview exceeds the %d MiB mailbox; skipping it. Dataset frames are unaffected.",
                        _MAILBOX_BYTES // (1024 * 1024),
                    )
                return False
            buffer_view = memoryview(self._buffer).cast("B")
            try:
                buffer_view[: len(payload)] = payload
            finally:
                buffer_view.release()
            self._length.value = len(payload)
            return True
        except Exception as error:
            self._disable(f"cannot submit preview: {type(error).__name__}: {error}")
            return False
        finally:
            self._lock.release()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None and process.pid is not None:
            if self._started:
                self._check_worker()
            self._stop.value = 1
            process.join(timeout=_CLOSE_TIMEOUT_S)
            if process.is_alive():
                logger.warning("Live preview sender did not stop in time; terminating it.")
                process.terminate()
                process.join(timeout=_TERMINATE_TIMEOUT_S)
            if process.is_alive():
                process.kill()
                process.join(timeout=_TERMINATE_TIMEOUT_S)
            if not process.is_alive():
                process.close()
        elif process is not None:
            process.close()
        if self._status is not None:
            self._status.close()
        self._process = None
        self._status = None
        self._buffer = None
        self._length = None
        self._lock = None
        self._stop = None
