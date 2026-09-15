from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, PolicyAction, TransitionKey
from lerobot.policies.pi05.processor_pi05 import (
    Pi05PrepareStateTokenizerProcessorStep,
    make_pi05_pre_post_processors,
)
from lerobot.processor import (
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.utils.recording_annotations import ACP_INDICATOR_FIELD_ALIASES

from .configuration_pi05_rlt import PI05RLTConfig


@ProcessorStepRegistry.register(name="pi05_rlt_acp_task_processor")
@dataclass
class PI05RLTACPTaskProcessorStep(ProcessorStep):
    enabled: bool = False
    indicator_field: str = "complementary_info.acp_indicator"
    dropout_prob: float = 0.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        result = transition.copy()
        complementary = dict(result.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        tasks = complementary.get("task")
        indicators = complementary.get(self.indicator_field)
        if indicators is None:
            indicators = next(
                (complementary[field] for field in ACP_INDICATOR_FIELD_ALIASES if field in complementary),
                None,
            )
        if tasks is None:
            raise KeyError("ACP requires the task field")
        if indicators is None:
            raise KeyError(f"ACP indicator field {self.indicator_field!r} is missing")
        if not isinstance(indicators, torch.Tensor) or indicators.ndim != 1:
            raise TypeError("ACP indicators must be a one-dimensional integer tensor")
        if indicators.dtype == torch.bool or indicators.dtype.is_floating_point:
            raise TypeError("ACP indicators must use an integer 0/1 dtype")
        values = indicators.detach().cpu().tolist()
        if len(values) != len(tasks) or any(value not in (0, 1) for value in values):
            raise ValueError("ACP indicators must contain one 0/1 value per task")
        conditioned = []
        for task, value in zip(tasks, values, strict=True):
            if self.dropout_prob and torch.rand(()).item() < self.dropout_prob:
                conditioned.append(task)
            else:
                conditioned.append(build_acp_tagged_task(task, is_positive=value == 1))
        complementary["task"] = conditioned
        result[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def reconcile_pi05_rlt_processors(
    config: PI05RLTConfig,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Install the active RLT/ACP task step into loaded PI0.5 processors.

    A ``pi05_rlt`` policy may be initialized from a plain current ``pi05``
    checkpoint. Its saved processor is otherwise valid, but it naturally does
    not contain this new step. Replacing an existing step also keeps resumed or
    fine-tuned RLT checkpoints aligned with the active config without creating
    duplicates.
    """
    steps = [step for step in preprocessor.steps if not isinstance(step, PI05RLTACPTaskProcessorStep)]
    tokenizer_index = next(
        index for index, step in enumerate(steps) if isinstance(step, Pi05PrepareStateTokenizerProcessorStep)
    )
    steps.insert(
        tokenizer_index,
        PI05RLTACPTaskProcessorStep(
            enabled=config.acp_enabled,
            indicator_field=config.acp_indicator_field,
            dropout_prob=config.acp_indicator_dropout_prob,
        ),
    )
    preprocessor.steps = steps
    return preprocessor, postprocessor


def make_pi05_rlt_pre_post_processors(
    config: PI05RLTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Use the current PI0.5 processors so RLT checkpoints remain native LeRobot policies."""
    del dataset_meta
    preprocessor, postprocessor = make_pi05_pre_post_processors(config, dataset_stats)
    return reconcile_pi05_rlt_processors(config, preprocessor, postprocessor)
