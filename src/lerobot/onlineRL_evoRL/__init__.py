"""EvoRL compatibility layer built on the current LeRobot RL runtime.

The package intentionally keeps the historical import path and compact episode
wire schema.  Model construction, preprocessing and ordinary RL execution are
delegated to the current :mod:`lerobot.policies`, :mod:`lerobot.rollout` and
:mod:`lerobot.rl` implementations.
"""

__all__: list[str] = []
