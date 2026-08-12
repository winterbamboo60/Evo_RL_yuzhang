"""Nonblocking keyboard state machine."""

from __future__ import annotations

import logging
import os
import select
import sys
import termios
import threading
import tty
from dataclasses import dataclass
from queue import Empty, Queue

HOTKEYS = "B: VLA/actor, I: manual takeover, S: success, Q/F: failure, R: reset episode"


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
        self._thread = threading.Thread(target=self._run, name="online-rl-tty-keyboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def _run(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue
            key = os.read(self._fd, 1).decode("utf-8", errors="ignore")
            if key:
                self.controller.push(key)


@dataclass
class ControlState:
    action_source: str = "vla"
    human_control: bool = False
    terminal_event: str | None = None
    reset_requested: bool = False


class KeyboardController:
    def __init__(self) -> None:
        self._events: Queue[str] = Queue()

    def push(self, key: str) -> None:
        self._events.put(key.lower())

    def poll(self, state: ControlState) -> ControlState:
        state.terminal_event = None
        state.reset_requested = False
        while True:
            try:
                key = self._events.get_nowait()
            except Empty:
                break
            if key == "b":
                state.action_source = "actor" if state.action_source == "vla" else "vla"
            elif key == "i":
                state.human_control = not state.human_control
            elif key == "s":
                state.terminal_event = "success"
            elif key in {"q", "f"}:
                state.terminal_event = "failure"
            elif key == "r":
                state.reset_requested = True
        return state

    def start_pynput_listener(self) -> object | None:
        """Start a real key listener, preferring the current terminal over X11."""
        if sys.stdin.isatty():
            try:
                listener = TTYKeyboardListener(self)
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

        listener = keyboard.Listener(on_press=lambda key: self.push(getattr(key, "char", "") or ""))
        listener.start()
        logging.info("Keyboard hotkeys enabled through pynput: %s", HOTKEYS)
        return listener
