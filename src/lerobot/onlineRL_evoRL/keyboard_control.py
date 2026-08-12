#!/usr/bin/env python

"""Nonblocking keyboard controls for online human-in-the-loop RL."""

from __future__ import annotations

import logging
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from queue import Empty, Queue

EPISODE_SUCCESS = "success"
EPISODE_FAILURE = "failure"

HOTKEYS = "I: manual takeover, S: success, F: failure, LEFT: rerecord, R: reset+rerecord, ESC: stop"
INTERVENTION_TOGGLE_COOLDOWN_S = 0.35


@dataclass
class KeyboardState:
    toggle_intervention: bool = False
    episode_outcome: str | None = None
    rerecord_episode: bool = False
    reset_episode: bool = False
    exit_episode: bool = False
    stop: bool = False


class KeyboardController:
    def __init__(self) -> None:
        self._events: Queue[str] = Queue()
        self._last_intervention_time = 0.0

    def push(self, key: str) -> None:
        if key:
            self._events.put(key)

    def poll(self, state: KeyboardState) -> KeyboardState:
        state.toggle_intervention = False
        while True:
            try:
                key = self._events.get_nowait()
            except Empty:
                break

            normalized = key.lower() if len(key) == 1 else key
            if normalized == "ESC":
                state.stop = True
                state.exit_episode = True
            elif normalized == "LEFT":
                state.rerecord_episode = True
                state.exit_episode = True
            elif normalized == "i":
                now = time.monotonic()
                if now - self._last_intervention_time < INTERVENTION_TOGGLE_COOLDOWN_S:
                    continue
                self._last_intervention_time = now
                state.toggle_intervention = True
            elif normalized == "s":
                state.episode_outcome = EPISODE_SUCCESS
                state.exit_episode = True
            elif normalized == "f":
                state.episode_outcome = EPISODE_FAILURE
                state.exit_episode = True
            elif normalized == "r":
                state.reset_episode = True
                state.rerecord_episode = True
                state.exit_episode = True
        return state


class TTYKeyboardListener:
    """Read hotkeys from the current terminal for SSH/headless sessions."""

    def __init__(self, controller: KeyboardController) -> None:
        self.controller = controller
        self._fd = sys.stdin.fileno()
        self._old_attrs: list[object] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, name="online-rl-keyboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def _read_key(self) -> str | None:
        chunk = os.read(self._fd, 1)
        if not chunk:
            return None
        if chunk == b"\x1b":
            sequence = bytearray(chunk)
            deadline = time.monotonic() + 0.02
            while time.monotonic() < deadline:
                ready, _, _ = select.select([self._fd], [], [], max(deadline - time.monotonic(), 0.0))
                if not ready:
                    break
                sequence.extend(os.read(self._fd, 1))
                last_byte = bytes(sequence[-1:])
                if len(sequence) >= 3 and last_byte in {b"A", b"B", b"C", b"D", b"~"}:
                    break
            sequence_bytes = bytes(sequence)
            if sequence_bytes in {b"\x1b[D", b"\x1bOD"}:
                return "LEFT"
            if sequence_bytes == b"\x1b":
                return "ESC"
            return None
        return chunk.decode("utf-8", errors="ignore")

    def _run(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue
            key = self._read_key()
            if key:
                self.controller.push(key)


def start_keyboard_listener(controller: KeyboardController) -> object | None:
    if sys.stdin.isatty():
        try:
            listener = TTYKeyboardListener(controller)
            listener.start()
        except OSError as error:
            logging.warning("TTY keyboard hotkeys unavailable: %s", error)
        else:
            logging.info("Keyboard hotkeys enabled on current TTY: %s", HOTKEYS)
            return listener

    try:
        from pynput import keyboard
    except ImportError as error:
        logging.warning("Keyboard hotkeys disabled: %s", error)
        return None

    def on_press(key):
        if key == keyboard.Key.left:
            controller.push("LEFT")
        elif key == keyboard.Key.esc:
            controller.push("ESC")
        else:
            controller.push(getattr(key, "char", "") or "")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    logging.info("Keyboard hotkeys enabled through pynput: %s", HOTKEYS)
    return listener


def stop_keyboard_listener(listener: object | None) -> None:
    if listener is None:
        return
    stop = getattr(listener, "stop", None)
    if callable(stop):
        stop()
