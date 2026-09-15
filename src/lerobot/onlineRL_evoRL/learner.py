"""Compatibility entry point for the current LeRobot learner.

The learner service and transport are intentionally not forked.  Algorithm and
checkpoint behavior therefore stays aligned with the canonical LeRobot RL
runtime while historical launch paths continue to work.
"""

from lerobot.rl.learner import *  # noqa: F403
from lerobot.rl.learner import train_cli

if __name__ == "__main__":
    train_cli()
