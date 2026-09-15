#!/usr/bin/env python

"""Value-training configuration built on LeRobot's standard training schema.

The dedicated ``lerobot-value-train`` entry point intentionally accepts the same
``--policy.*``, accelerator, parallelism, dataloader, optimizer, scheduler and
checkpoint options as ``lerobot-train``. This subclass only adds the
episode-outcome target definition used by Pistar06.
"""

from dataclasses import dataclass, field

from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.utils.recording_annotations import normalize_episode_success_label
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config

VALUE_TRAIN_CONFIG_NAME = TRAIN_CONFIG_NAME


@dataclass
class ValueTargetsConfig:
    success_field: str = "episode_success"
    default_success: str = "failure"
    c_fail_coef: float = 1.0
    target_field: str = "observation.value_target"

    def validate(self) -> None:
        normalized_default = normalize_episode_success_label(self.default_success)
        if normalized_default is None:
            raise ValueError("'targets.default_success' must be either 'success' or 'failure'.")
        self.default_success = normalized_default
        if not self.success_field:
            raise ValueError("'targets.success_field' must be non-empty.")
        if self.c_fail_coef < 0:
            raise ValueError("'targets.c_fail_coef' must be non-negative.")
        if not self.target_field.startswith("observation."):
            raise ValueError("'targets.target_field' must start with 'observation.'.")


@dataclass
class ValueTrainPipelineConfig(TrainPipelineConfig):
    """Standard LeRobot train config plus Pistar06 target construction."""

    targets: ValueTargetsConfig = field(default_factory=ValueTargetsConfig)

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.policy, Pistar06Config):
            policy_type = None if self.policy is None else self.policy.type
            raise ValueError(
                "lerobot-value-train requires '--policy.type=pistar06' or a Pistar06 "
                f"checkpoint, got {policy_type!r}."
            )
        self.targets.validate()
        self.policy.target_key = self.targets.target_field
        self.policy.success_field = self.targets.success_field
        self.policy.default_success = self.targets.default_success
        self.policy.c_fail_coef = self.targets.c_fail_coef
