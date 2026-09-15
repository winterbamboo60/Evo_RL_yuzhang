#!/usr/bin/env python

"""Train Pistar06 with LeRobot's standard offline training pipeline.

All runtime, optimizer, scheduler, distributed and checkpoint parameters are the
same as ``lerobot-train``. Only the value-target configuration is additional.
"""

import logging

from lerobot.configs import parser
from lerobot.configs.value_train import ValueTrainPipelineConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins


@parser.wrap()
def value_train(cfg: ValueTrainPipelineConfig):
    # ``parser.wrap`` deliberately recognizes only the exact annotated config
    # type. ``ValueTrainPipelineConfig`` subclasses ``TrainPipelineConfig``, so
    # calling the decorated entry point would parse the CLI a second time and
    # then pass two configs to the underlying function. ``functools.wraps``
    # exposes the undecorated training function for this in-process reuse.
    return train.__wrapped__(cfg)


def main() -> None:
    register_third_party_plugins()
    # Importing ValueTrainPipelineConfig registers Pistar06 before draccus parses
    # ``--policy.type=pistar06``.
    value_train()
    logging.shutdown()


if __name__ == "__main__":
    main()
