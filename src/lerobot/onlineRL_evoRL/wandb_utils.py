"""Compatibility export of the canonical experiment logger."""

from lerobot.common.wandb_utils import *  # noqa: F403
from lerobot.common.wandb_utils import WandBLogger, cfg_to_group

__all__ = ["WandBLogger", "cfg_to_group"]
