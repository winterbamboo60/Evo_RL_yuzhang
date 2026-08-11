"""Stage1 feature-extractor boundary for RLT execution."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from .config import ModelConfig


@dataclass
class RLTFeatures:
    z_rl: torch.Tensor
    proprio: torch.Tensor
    ref_chunk: torch.Tensor


class FeatureExtractor(Protocol):
    def __call__(self, observation: dict) -> RLTFeatures: ...


class FakeFeatureExtractor:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def __call__(self, observation: dict) -> RLTFeatures:
        proprio = torch.as_tensor(observation["state"], dtype=torch.float32).reshape(1, -1)
        z_rl = torch.zeros(1, self.cfg.z_dim)
        ref = proprio[:, None, :].repeat(1, self.cfg.ref_num_action_chunks, 1)
        return RLTFeatures(z_rl=z_rl, proprio=proprio, ref_chunk=ref)


def load_feature_extractor(
    factory_path: str, cfg: ModelConfig, dry_run: bool, kwargs: dict[str, Any] | None = None
) -> FeatureExtractor:
    if dry_run:
        return FakeFeatureExtractor(cfg)
    if not factory_path:
        raise ValueError("Real hardware requires runtime.feature_extractor_factory='module:function'.")
    module_name, separator, attr = factory_path.partition(":")
    if not separator:
        raise ValueError("feature_extractor_factory must use 'module:function'.")
    factory: Callable[..., FeatureExtractor] = getattr(importlib.import_module(module_name), attr)
    kwargs = dict(kwargs or {})
    signature = inspect.signature(factory)
    if len(signature.parameters) >= 2:
        return factory(cfg, kwargs)
    if kwargs:
        raise ValueError(f"feature_extractor_kwargs were provided but {factory_path} only accepts ModelConfig.")
    return factory(cfg)


def validate_features(features: RLTFeatures, cfg: ModelConfig) -> None:
    expected = (1, cfg.ref_num_action_chunks, cfg.action_dim)
    if tuple(features.z_rl.shape) != (1, cfg.z_dim) or tuple(features.proprio.shape) != (1, cfg.proprio_dim):
        raise ValueError("Stage1 RLT feature dimensions do not match onlineRL config.")
    if tuple(features.ref_chunk.shape) != expected:
        raise ValueError(f"Expected ref_chunk {expected}, got {tuple(features.ref_chunk.shape)}.")
