"""RLinf-backed Stage1 OpenPI+RLT extractor for onlineRL.

Use this adapter when the Stage1 checkpoint was trained by RLinf and the runtime
is the RLinf/OpenPI container. Unlike ``stage1_rlt_adapter.py``, this path does
not reimplement PI05 locally; it calls RLinf's own ``extract_rlt_obs`` pipeline so
state, prompt, image transforms, RLT prefix extraction and action sampling match
Stage1 training.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .vla import RLTFeatures


class AttrDict(dict):
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _extend_lerobot_namespace_for_openpi() -> None:
    try:
        import lerobot
    except ImportError:
        return
    package_paths = getattr(lerobot, "__path__", None)
    if package_paths is None:
        return
    for entry in sys.path:
        candidate = Path(entry) / "lerobot"
        if (candidate / "common").is_dir() and str(candidate) not in package_paths:
            package_paths.append(str(candidate))


def _image_hwc(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Expected image HWC/CHW, got {array.shape}.")
    if array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return np.ascontiguousarray(array)


def _state_2d(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"Expected state D or [B,D], got {array.shape}.")
    return np.ascontiguousarray(array)


class RLinfStage1RLTExtractor:
    """Frozen RLinf OpenPI Stage1 feature extractor."""

    def __init__(self, cfg: ModelConfig, options: dict[str, Any]):
        self.cfg = cfg
        self.task_description = str(options.get("task_description", ""))
        self.device = torch.device(options.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.rlinf_path = Path(options.get("rlinf_path", "/home/yz/projects/RLinf_yuzhang"))
        _extend_lerobot_namespace_for_openpi()
        if str(self.rlinf_path) not in sys.path:
            sys.path.insert(0, str(self.rlinf_path))

        from rlinf.models.embodiment.openpi import get_model

        model_cfg = AttrDict(
            model_path=options["model_path"],
            openpi_data={
                "repo_id": options.get("repo_id"),
                "norm_stats_path": options.get("norm_stats_path"),
            },
            openpi=AttrDict(
                config_name=options.get("config_name", "pi05_piper_state"),
                num_images_in_input=options.get("num_images_in_input", 2),
                action_chunk=options.get("action_chunk", cfg.ref_num_action_chunks),
                num_steps=options.get("num_steps", 10),
                state_indices=options.get("state_indices", []),
                use_rlt=options.get("use_rlt", True),
                rlt_prefix_seq_len=options.get("rlt_prefix_seq_len", 1024),
                rlt_image_only=options.get("rlt_image_only", False),
                rlt_use_mask=options.get("rlt_use_mask", True),
                noise_method=options.get("noise_method", "flow_noise"),
                noise_params=options.get("noise_params", [0.16, 0.12, 200]),
                joint_logprob=options.get("joint_logprob", True),
                detach_critic_input=options.get("detach_critic_input", True),
            ),
        )
        self.model = get_model(model_cfg).to(self.device).eval()
        logging.info(
            "Loaded RLinf Stage1 extractor checkpoint=%s norm_stats=%s device=%s config_name=%s",
            options.get("model_path"),
            options.get("norm_stats_path"),
            self.device,
            model_cfg.openpi.config_name,
        )

    @torch.no_grad()
    def __call__(self, observation: dict) -> RLTFeatures:
        state = _state_2d(observation["state"])
        top = _image_hwc(observation["top_image"])[None, ...]
        wrist = _image_hwc(observation["wrist_image"])[None, ...]
        task = str(observation.get("prompt") or self.task_description)
        env_obs = {
            "states": state,
            "main_images": top,
            "wrist_images": wrist,
            "extra_view_images": None,
            "task_descriptions": [task],
        }
        out = self.model.extract_rlt_obs(env_obs)
        ref_chunk = out["ref_chunk"][:, : self.cfg.ref_num_action_chunks, : self.cfg.action_dim]
        return RLTFeatures(
            z_rl=out["z_rl"].detach().to(dtype=torch.float32),
            proprio=out["proprio"].detach().to(device=out["z_rl"].device, dtype=torch.float32),
            ref_chunk=ref_chunk.detach().to(device=out["z_rl"].device, dtype=torch.float32),
        )


def build_extractor(cfg: ModelConfig, options: dict[str, Any]) -> RLinfStage1RLTExtractor:
    return RLinfStage1RLTExtractor(cfg, options)
