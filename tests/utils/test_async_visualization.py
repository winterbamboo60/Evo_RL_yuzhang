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

"""Real spawn-process tests; no viewer, SDK, or robot is started."""

import multiprocessing
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import lerobot.utils.async_visualization as async_viz
from lerobot.utils.async_visualization import AsyncVisualization


@dataclass
class _BackendFactory:
    connection: Any
    entered: Any
    release: Any
    failure: str | None = None
    block_first: bool = False

    def __call__(self):
        return self

    def init_visualization(self, display_mode, *, session_name, ip, port):
        self.connection.send(("init", os.getpid(), display_mode, session_name, ip, port))
        if self.failure == "init":
            raise RuntimeError("injected initialization failure")

    def log_visualization_data(self, display_mode, *, observation, action, compress_images):
        image = observation.get("image")
        self.connection.send(
            (
                "log",
                os.getpid(),
                observation["sequence"],
                action["sequence"],
                int(image.flat[0]) if image is not None else None,
                compress_images,
            )
        )
        self.entered.set()
        if self.failure == "log":
            raise RuntimeError("injected logging failure")
        if self.block_first and observation["sequence"] == 0:
            self.release.wait()

    def shutdown_visualization(self, display_mode):
        self.connection.send(("shutdown", os.getpid()))


def _make_preview(*, max_fps=1000000.0, failure=None, block_first=False):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    entered = context.Event()
    release = context.Event()
    preview = AsyncVisualization(
        "rerun",
        session_name="test-preview",
        ip="127.0.0.1",
        port=9000,
        compress_images=True,
        max_fps=max_fps,
        _backend_factory=_BackendFactory(sender, entered, release, failure, block_first),
    )
    return preview, receiver, sender, entered, release


def _receive(connection):
    assert connection.poll(10.0), "preview worker did not respond"
    return connection.recv()


def _hold_mailbox_worker(buffer, length, mailbox_lock, stop, status, config, backend_factory):
    """Simulate a sender paused forever while owning the mailbox lock."""
    mailbox_lock.acquire()
    status.send(("ready", None))
    while True:
        time.sleep(1.0)


def test_blocked_viewer_keeps_submission_fast_and_displays_only_latest_snapshot():
    preview, receiver, sender, entered, release = _make_preview(block_first=True)
    try:
        preview.start()
        initialized = _receive(receiver)
        assert initialized == ("init", preview._process.pid, "rerun", "test-preview", "127.0.0.1", 9000)
        assert initialized[1] != os.getpid()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        observation = {"sequence": 0, "image": image, "wrist_left": image.copy(), "wrist_right": image.copy()}
        action = {"sequence": 0}
        assert preview.submit(observation=observation, action=action)
        assert _receive(receiver) == ("log", initialized[1], 0, 0, 0, True)
        assert entered.wait(timeout=5.0)

        start = time.perf_counter()
        for sequence in range(1, 21):
            image.fill(sequence)
            observation["sequence"] = sequence
            action["sequence"] = sequence
            assert preview.submit(observation=observation, action=action)
        # The backend is still blocked indefinitely. No submit waits for it,
        # even after enough image data to exceed a typical OS pipe buffer.
        assert time.perf_counter() - start < 1.0
        image.fill(255)
        observation["sequence"] = 999
        action["sequence"] = 999
        release.set()
        assert _receive(receiver) == ("log", initialized[1], 20, 20, 20, True)
        preview.close()
        assert _receive(receiver) == ("shutdown", initialized[1])
        assert not receiver.poll(0.05)
        assert preview._buffer is None
        assert not preview.submit(observation=observation, action=action)
    finally:
        release.set()
        preview.close()
        receiver.close()
        sender.close()


def test_busy_mailbox_skips_without_serializing_or_waiting():
    preview, receiver, sender, _, release = _make_preview()
    try:
        preview.start()
        _receive(receiver)
        with preview._lock:
            start = time.perf_counter()
            # This lambda is deliberately unpicklable. A busy mailbox must be
            # skipped before attempting serialization.
            assert not preview.submit(observation={"unpicklable": lambda: None})
            assert time.perf_counter() - start < 0.1
        assert not preview._disabled
    finally:
        release.set()
        preview.close()
        receiver.close()
        sender.close()


def test_preview_rate_limit_does_not_submit_every_control_tick(monkeypatch):
    preview, receiver, sender, _, release = _make_preview(max_fps=15.0)
    try:
        preview.start()
        _receive(receiver)
        clock = [10.0]
        # Replace this module's clock, not the global time module used by
        # multiprocessing to implement bounded joins.
        monkeypatch.setattr(async_viz, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        assert preview.submit(observation={"sequence": 1}, action={"sequence": 1})
        clock[0] += 1 / 30
        assert not preview.submit(observation={"sequence": 2}, action={"sequence": 2})
        clock[0] += 1 / 30 + 1e-6
        assert preview.submit(observation={"sequence": 3}, action={"sequence": 3})
    finally:
        release.set()
        preview.close()
        receiver.close()
        sender.close()


def test_fixed_mailbox_rejects_and_rate_limits_oversized_preview(caplog, monkeypatch):
    preview, receiver, sender, _, release = _make_preview(max_fps=15.0)
    try:
        preview.start()
        _receive(receiver)
        clock = [10.0]
        monkeypatch.setattr(async_viz, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        assert len(preview._buffer) == async_viz._MAILBOX_BYTES
        assert not preview.submit(observation={"image": bytes(async_viz._MAILBOX_BYTES)})
        assert "mailbox" in caplog.text
        assert not preview._disabled
        # The rejected serialization attempt must still consume its rate-limit
        # interval, rather than repeating a 16 MiB pickle every control tick.
        assert not preview.submit(observation={"sequence": 1}, action={"sequence": 1})
        clock[0] += 1 / 15 + 1e-6
        assert preview.submit(observation={"sequence": 1}, action={"sequence": 1})
        assert _receive(receiver)[2:4] == (1, 1)
    finally:
        release.set()
        preview.close()
        receiver.close()
        sender.close()


def test_initialization_failure_raises_before_recording_and_cleans_up():
    preview, receiver, sender, _, release = _make_preview(failure="init")
    try:
        with pytest.raises(RuntimeError, match="injected initialization failure"):
            preview.start()
        assert preview._process is None
        assert preview._buffer is None
        assert not preview.submit(observation={"sequence": 1})
    finally:
        release.set()
        preview.close()
        receiver.close()
        sender.close()


def test_logging_failure_disables_preview_without_raising_in_submit(caplog):
    preview, receiver, sender, entered, release = _make_preview(failure="log")
    try:
        preview.start()
        _receive(receiver)
        assert preview.submit(observation={"sequence": 1}, action={"sequence": 1})
        assert entered.wait(timeout=5.0)
        preview._process.join(timeout=5.0)
        assert not preview._process.is_alive()
        assert not preview.submit(observation={"sequence": 2}, action={"sequence": 2})
        assert "injected logging failure" in caplog.text
        assert "dataset recording continue" in caplog.text
        assert not preview.submit(observation={"sequence": 3}, action={"sequence": 3})
        assert caplog.text.count("Live preview disabled") == 1
    finally:
        release.set()
        preview.close()
        receiver.close()
        sender.close()


def test_close_terminates_a_stuck_viewer_within_a_bound(monkeypatch):
    preview, receiver, sender, entered, release = _make_preview(block_first=True)
    monkeypatch.setattr(async_viz, "_CLOSE_TIMEOUT_S", 0.2)
    try:
        preview.start()
        _receive(receiver)
        process = preview._process
        assert preview.submit(observation={"sequence": 0}, action={"sequence": 0})
        assert entered.wait(timeout=5.0)
        start = time.perf_counter()
        preview.close()
        assert time.perf_counter() - start < 3.0
        assert process._closed
        assert preview._process is None
        assert preview._buffer is None
        preview.close()
    finally:
        # A killed waiter cannot acknowledge multiprocessing.Condition's
        # notify_all handshake. Do not set its fake backend Event afterward.
        preview.close()
        receiver.close()
        sender.close()


@pytest.mark.parametrize("kill_before_close", [False, True])
def test_worker_holding_or_abandoning_mailbox_cannot_block_submit_or_close(monkeypatch, kill_before_close):
    monkeypatch.setattr(async_viz, "_visualization_worker", _hold_mailbox_worker)
    monkeypatch.setattr(async_viz, "_CLOSE_TIMEOUT_S", 0.2)
    preview, receiver, sender, _, release = _make_preview()
    try:
        preview.start()
        process = preview._process
        mailbox_lock = preview._lock
        if kill_before_close:
            process.terminate()
            process.join(timeout=5.0)
            assert not process.is_alive()
        start = time.perf_counter()
        assert not preview.submit(observation={"sequence": 1}, action={"sequence": 1})
        assert time.perf_counter() - start < 0.1
        start = time.perf_counter()
        preview.close()
        assert time.perf_counter() - start < 3.0
        assert process._closed
        # A killed sender did leave the lock owned. Close must not acquire it
        # or an Event/Condition lock shared with that sender.
        assert not mailbox_lock.acquire(False)
    finally:
        release.set()
        preview.close()
        receiver.close()
        sender.close()


@pytest.mark.parametrize("max_fps", [0.0, -1.0, float("inf"), float("nan")])
def test_preview_rate_must_be_finite_and_positive(max_fps):
    with pytest.raises(ValueError, match="finite positive"):
        AsyncVisualization("rerun", max_fps=max_fps)
