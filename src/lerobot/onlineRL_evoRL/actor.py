"""Backward-compatible actor entry point.

Unlike the 0901 file, this module does not import and subclass itself.
"""

from .actor_new import *  # noqa: F403
from .actor_new import actor_cli

if __name__ == "__main__":
    actor_cli()
