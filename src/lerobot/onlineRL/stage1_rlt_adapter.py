"""Local Stage1 OpenPI+RLT extractor migrated from RLinf's RLT path.

This module intentionally does not import ``rlinf``.  It reuses Evo/LeRobot's
local PI05 implementation plus the RLT token transformer migrated into
``lerobot.onlineRL``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from .config import ModelConfig
from .rlt_token_transformer import RLTTokenTransformer
from .vla import RLTFeatures


_PIPER_IMAGE_KEYS = (
    f"{OBS_IMAGES}.base_0_rgb",
    f"{OBS_IMAGES}.left_wrist_0_rgb",
    f"{OBS_IMAGES}.right_wrist_0_rgb",
)
_PIPER_TRAINING_ACTION_DIM = 32


def _require(options: dict[str, Any], key: str) -> Any:
    value = options.get(key)
    if value in (None, ""):
        raise ValueError(f"feature_extractor_kwargs.{key} is required for the local Stage1 RLT adapter.")
    return value


def _load_norm_stats(path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    payload = json.loads(Path(path).expanduser().read_text())
    stats = payload.get("norm_stats", payload)
    return {
        key: {stat_key: torch.as_tensor(stat_value, dtype=torch.float32) for stat_key, stat_value in value.items()}
        for key, value in stats.items()
    }


def _quantile_transform(tensor: torch.Tensor, stats: dict[str, torch.Tensor], *, inverse: bool) -> torch.Tensor:
    q01 = stats["q01"].to(device=tensor.device, dtype=tensor.dtype)
    q99 = stats["q99"].to(device=tensor.device, dtype=tensor.dtype)
    q01 = q01[: tensor.shape[-1]]
    q99 = q99[: tensor.shape[-1]]
    denom = torch.where(q99 == q01, torch.full_like(q99, 1e-8), q99 - q01)
    if inverse:
        return (tensor + 1.0) * denom / 2.0 + q01
    return 2.0 * (tensor - q01) / denom - 1.0


def _image_batch(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected image with shape HWC/CHW or batched image, got {tuple(tensor.shape)}.")
    tensor = tensor.to(dtype=torch.float32)
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.shape[1] not in (1, 3):
        raise ValueError(f"Expected image channels in dim 1 or -1, got {tuple(tensor.shape)}.")
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor.contiguous()


def _tensor_summary(value: Any) -> str:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    return f"shape={tuple(tensor.shape)} min={tensor.min().item():.4f} max={tensor.max().item():.4f} mean={tensor.mean().item():.4f}"


def _head_values(value: Any, count: int = 7) -> str:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)[:count]
    return "[" + ", ".join(f"{item:.4f}" for item in tensor.tolist()) + "]"


def _state_batch(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected state with shape D or [B,D], got {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _checkpoint_file(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser()
    candidates = [
        path / "model_state_dict" / "full_weights.pt",
        path / "actor" / "model_state_dict" / "full_weights.pt",
        path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find Stage1 full_weights.pt under {path}")


class LocalStage1RLTExtractor:
    """Frozen local PI05 + RLT token transformer feature extractor."""

    def __init__(self, cfg: ModelConfig, options: dict[str, Any]):
        self.cfg = cfg
        self.task_description = str(_require(options, "task_description"))
        self.device = torch.device(options.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.norm_stats = _load_norm_stats(_require(options, "norm_stats_path"))
        self.action_dim = int(options.get("action_dim", cfg.action_dim))
        self.model_action_dim = int(options.get("model_action_dim", _PIPER_TRAINING_ACTION_DIM))
        self.state_dim = int(options.get("state_dim", cfg.proprio_dim))
        self.dtype = str(options.get("dtype", "float32"))
        self.rlt_image_only = bool(options.get("rlt_image_only", False))
        self.rlt_use_mask = bool(options.get("rlt_use_mask", True))

        self.base_key, self.left_wrist_key, self.right_wrist_key = _PIPER_IMAGE_KEYS
        pi05_cfg = PI05Config(
            input_features={
                self.base_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                self.left_wrist_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                self.right_wrist_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,)),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.model_action_dim,))},
            device=str(self.device),
            dtype=self.dtype,
            chunk_size=int(options.get("action_chunk", cfg.ref_num_action_chunks)),
            n_action_steps=int(options.get("action_chunk", cfg.ref_num_action_chunks)),
            num_inference_steps=int(options.get("num_steps", 10)),
        )
        self.policy = PI05Policy(pi05_cfg).to(self.device).eval()
        self.rlt_module = RLTTokenTransformer(
            input_dim=int(options.get("rlt_input_dim", cfg.z_dim)),
            embed_dim=int(options.get("rlt_embed_dim", cfg.z_dim)),
            num_rl_tokens=int(options.get("rlt_num_rl_tokens", 1)),
            prefix_seq_len=int(options.get("rlt_prefix_seq_len", 1024)),
            num_layers=int(options.get("rlt_num_layers", 2)),
            num_heads=int(options.get("rlt_num_heads", 8)),
            mlp_ratio=float(options.get("rlt_mlp_ratio", 4.0)),
        ).to(device=self.device, dtype=getattr(torch, self.dtype)).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(options.get("tokenizer_name", pi05_cfg.tokenizer_name)),
            padding_side="right",
            local_files_only=bool(options.get("local_files_only", False)),
        )
        checkpoint = _checkpoint_file(_require(options, "model_path"))
        self._load_stage1_weights(checkpoint)
        logging.info(
            "Stage1 RLT adapter loaded checkpoint=%s device=%s dtype=%s image_keys=%s action_dim=%d model_action_dim=%d state_dim=%d rlt_prefix_seq_len=%d rlt_use_mask=%s",
            checkpoint,
            self.device,
            self.dtype,
            _PIPER_IMAGE_KEYS,
            self.action_dim,
            self.model_action_dim,
            self.state_dim,
            int(options.get("rlt_prefix_seq_len", 1024)),
            self.rlt_use_mask,
        )
        logging.info(
            "Stage1 norm stats state_q01=%s state_q99=%s action_q01=%s action_q99=%s",
            _head_values(self.norm_stats["state"]["q01"]),
            _head_values(self.norm_stats["state"]["q99"]),
            _head_values(self.norm_stats["actions"]["q01"]),
            _head_values(self.norm_stats["actions"]["q99"]),
        )

    def _load_stage1_weights(self, checkpoint: Path) -> None:
        state_dict = torch.load(checkpoint, map_location="cpu", mmap=True)
        rlt_state = {key.removeprefix("rlt_module."): value for key, value in state_dict.items() if key.startswith("rlt_module.")}
        base_state = {key: value for key, value in state_dict.items() if not key.startswith("rlt_module.")}
        fixed_base = self.policy._fix_pytorch_state_dict_keys(base_state, self.policy.config)
        remapped_base = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed_base.items()}
        base_missing, base_unexpected = self.policy.load_state_dict(remapped_base, strict=False)
        if base_missing or base_unexpected:
            logging.info(
                "Stage1 PI05 base loaded strict=False missing=%d unexpected=%d missing_head=%s unexpected_head=%s",
                len(base_missing),
                len(base_unexpected),
                list(base_missing)[:5],
                list(base_unexpected)[:5],
            )
        missing, unexpected = self.rlt_module.load_state_dict(rlt_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Failed to load RLT module cleanly: missing={missing}, unexpected={unexpected}")

    def _prompt(self, raw_state: torch.Tensor) -> str:
        norm_state = _quantile_transform(raw_state, self.norm_stats["state"], inverse=False).cpu()
        padded = torch.zeros(self.policy.config.max_state_dim, dtype=torch.float32)
        padded[: norm_state.shape[-1]] = norm_state[0, : self.policy.config.max_state_dim]
        discretized = np.digitize(padded.numpy(), bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_text = " ".join(map(str, discretized.tolist()))
        task = self.task_description.strip().replace("_", " ").replace("\n", " ")
        return f"Task: {task}, State: {state_text};\nAction: "

    def _batch(self, observation: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        raw_state = _state_batch(observation["state"])
        tokenized = self.tokenizer(
            [self._prompt(raw_state)],
            max_length=self.policy.config.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        norm_state = _quantile_transform(raw_state.to(self.device), self.norm_stats["state"], inverse=False)
        batch = {
            # The configured right-wrist slot is intentionally absent here;
            # PI05Policy marks missing configured cameras as mask=false, matching
            # RLinf PiperInputs zero right_wrist_0_rgb with image_mask=False.
            self.base_key: _image_batch(observation["top_image"]).to(self.device),
            self.left_wrist_key: _image_batch(observation["wrist_image"]).to(self.device),
            OBS_STATE: norm_state,
            OBS_LANGUAGE_TOKENS: tokenized["input_ids"].to(self.device),
            OBS_LANGUAGE_ATTENTION_MASK: tokenized["attention_mask"].to(self.device, dtype=torch.bool),
        }
        self._last_batch_debug = {
            "raw_state": raw_state,
            "norm_state": norm_state.detach().cpu(),
            "top_image": batch[self.base_key].detach().cpu(),
            "wrist_image": batch[self.left_wrist_key].detach().cpu(),
            "prompt_tokens": int(tokenized["attention_mask"].sum().item()),
        }
        return batch, raw_state

    @torch.no_grad()
    def _prefix_output(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        images, img_masks = self.policy._preprocess_images(batch)
        self._last_image_masks = [mask.detach().cpu() for mask in img_masks]
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(images, img_masks, tokens, masks)
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        att_4d = self.policy.model._prepare_attention_masks_4d(att_2d)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.policy.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (prefix_output, _), _ = self.policy.model.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        if self.rlt_image_only:
            num_image_tokens = prefix_output.shape[1] - tokens.shape[1]
            prefix_output = prefix_output[:, :num_image_tokens]
            prefix_pad_masks = prefix_pad_masks[:, :num_image_tokens]
        return prefix_output, prefix_pad_masks

    @torch.no_grad()
    def __call__(self, observation: dict) -> RLTFeatures:
        batch, raw_state = self._batch(observation)
        prefix_output, prefix_mask = self._prefix_output(batch)
        rlt_param = next(self.rlt_module.parameters())
        prefix_output = prefix_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
        mask = prefix_mask if self.rlt_use_mask else None
        z_rl = self.rlt_module.encode_flat(prefix_output, mask).to(dtype=torch.float32)
        actions = self.policy.predict_action_chunk(batch)[:, : self.cfg.ref_num_action_chunks, : self.model_action_dim]
        ref_chunk = _quantile_transform(actions, self.norm_stats["actions"], inverse=True)[:, :, : self.action_dim]
        debug = getattr(self, "_last_batch_debug", {})
        masks = [int(mask.sum().item()) for mask in getattr(self, "_last_image_masks", [])]
        logging.info(
            "Stage1 extract raw_state[%s] norm_state[%s] top[%s] wrist[%s] prompt_tokens=%s image_masks=%s prefix[%s] prefix_mask_true=%d z_rl[%s] action_norm[%s] ref_chunk[%s] first_ref=%s",
            _tensor_summary(debug.get("raw_state", raw_state)),
            _tensor_summary(debug.get("norm_state", batch[OBS_STATE])),
            _tensor_summary(debug.get("top_image", batch[self.base_key])),
            _tensor_summary(debug.get("wrist_image", batch[self.left_wrist_key])),
            debug.get("prompt_tokens", -1),
            masks,
            _tensor_summary(prefix_output),
            int(prefix_mask.sum().item()),
            _tensor_summary(z_rl),
            _tensor_summary(actions),
            _tensor_summary(ref_chunk),
            _head_values(ref_chunk[0, 0]),
        )
        return RLTFeatures(
            z_rl=z_rl.detach().to(dtype=torch.float32),
            proprio=raw_state.to(device=z_rl.device, dtype=torch.float32),
            ref_chunk=ref_chunk.detach().to(device=z_rl.device, dtype=torch.float32),
        )


def build_extractor(cfg: ModelConfig, options: dict[str, Any]) -> LocalStage1RLTExtractor:
    return LocalStage1RLTExtractor(cfg, options)
