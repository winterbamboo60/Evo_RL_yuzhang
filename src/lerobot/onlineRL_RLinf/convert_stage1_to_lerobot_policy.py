#!/usr/bin/env python3
"""Convert PI05 weights between onlineRL Stage1/RLT and LeRobot policy folders."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.onlineRL.rlt_token_transformer import RLTTokenTransformer
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stage1_dir", type=Path, help="RLT Stage1 checkpoint directory to export as LeRobot policy.")
    source.add_argument("--lerobot_policy_dir", type=Path, help="LeRobot PI05 policy directory to export for onlineRL.")
    parser.add_argument("--reference_policy_dir", type=Path, help="Reference LeRobot policy folder for --stage1_dir mode.")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--rlt_prefix_seq_len", default=1024, type=int)
    parser.add_argument("--rlt_input_dim", default=2048, type=int)
    parser.add_argument("--rlt_embed_dim", default=2048, type=int)
    parser.add_argument("--rlt_num_rl_tokens", default=1, type=int)
    parser.add_argument("--rlt_num_layers", default=2, type=int)
    parser.add_argument("--rlt_num_heads", default=8, type=int)
    parser.add_argument("--rlt_mlp_ratio", default=4.0, type=float)
    parser.add_argument("--template_stage1_dir", type=Path, help="Optional Stage1/RLT checkpoint used only for onlineRL key layout, RLT shapes/dtypes, and extra unused keys.")
    return parser.parse_args()


def _load_norm_stats(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload.get("norm_stats", payload)


def _checkpoint_file(stage1_dir: Path) -> Path:
    candidates = [
        stage1_dir / "model_state_dict" / "full_weights.pt",
        stage1_dir / "actor" / "model_state_dict" / "full_weights.pt",
        stage1_dir / "full_weights.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find full_weights.pt under {stage1_dir}")


def _tensor(values: list[float], dim: int) -> torch.Tensor:
    return torch.tensor(values[:dim], dtype=torch.float32)


def _copy_policy_json(ref_dir: Path, out_dir: Path, image_size: int) -> None:
    for name in ["policy_preprocessor.json", "policy_postprocessor.json", "train_config.json"]:
        shutil.copy2(ref_dir / name, out_dir / name)

    config_json = json.loads((ref_dir / "config.json").read_text())
    config_json["image_resolution"] = [image_size, image_size]
    for key in ["observation.images.wrist", "observation.images.top"]:
        if key in config_json.get("input_features", {}):
            config_json["input_features"][key]["shape"] = [3, 480, 640]
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=4) + "\n")


def _stage1_to_lerobot(args: argparse.Namespace) -> None:
    if args.reference_policy_dir is None:
        raise ValueError("--reference_policy_dir is required with --stage1_dir")
    stage1_dir = args.stage1_dir
    ref_dir = args.reference_policy_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy_policy_json(ref_dir, out_dir, args.image_size)

    config = PreTrainedConfig.from_pretrained(out_dir)
    policy = PI05Policy(config)
    target_state = policy.state_dict()

    raw_state = torch.load(_checkpoint_file(stage1_dir), map_location="cpu", mmap=True)
    base_state = {k: v for k, v in raw_state.items() if not k.startswith("rlt_module.")}
    fixed_state = policy._fix_pytorch_state_dict_keys(base_state, config)
    remapped_state = {k if k.startswith("model.") else f"model.{k}": v for k, v in fixed_state.items()}

    converted = {}
    skipped_extra = []
    skipped_shape = []
    for key, value in remapped_state.items():
        if key not in target_state:
            skipped_extra.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shape.append((key, tuple(value.shape), tuple(target_state[key].shape)))
            continue
        converted[key] = value.detach().cpu().contiguous()

    missing = [key for key in target_state if key not in converted]
    if missing or skipped_shape:
        raise RuntimeError(
            "Converted weights do not match target PI05 policy: "
            f"missing={missing[:20]} shape_mismatch={skipped_shape[:20]}"
        )
    save_file(converted, out_dir / "model.safetensors")

    stats = _load_norm_stats(stage1_dir / "norm_stats.json")
    action_dim = config.output_features["action"].shape[0]
    state_dim = config.input_features["observation.state"].shape[0]
    processor_state = {
        "observation.state.count": torch.tensor([0.0], dtype=torch.float32),
        "action.count": torch.tensor([0.0], dtype=torch.float32),
    }
    for name, values in stats["state"].items():
        processor_state[f"observation.state.{name}"] = _tensor(values, state_dim)
    for name, values in stats["actions"].items():
        processor_state[f"action.{name}"] = _tensor(values, action_dim)

    save_file(processor_state, out_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors")
    save_file(processor_state, out_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    shutil.copy2(stage1_dir / "norm_stats.json", out_dir / "norm_stats.json")

    print(f"saved {out_dir}")
    print(f"model tensors {len(converted)}")
    print(f"processor tensors {len(processor_state)}")
    print(f"skipped non-PI05 tensors {len(skipped_extra)}")


def _extract_lerobot_norm_stats(policy_dir: Path) -> dict:
    state_path = policy_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    if not state_path.is_file():
        state_path = policy_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not state_path.is_file():
        raise FileNotFoundError(f"Could not find policy normalizer safetensors under {policy_dir}")

    flat = load_file(state_path, device="cpu")
    mapping = {"observation.state": "state", "action": "actions"}
    stats = {"state": {}, "actions": {}}
    for prefix, target in mapping.items():
        for stat_name in ["mean", "std", "q01", "q99", "min", "max", "q10", "q50", "q90"]:
            key = f"{prefix}.{stat_name}"
            if key in flat:
                stats[target][stat_name] = flat[key].to(torch.float32).reshape(-1).tolist()
    for target in ["state", "actions"]:
        missing = {"q01", "q99"} - set(stats[target])
        if missing:
            raise RuntimeError(f"Missing {sorted(missing)} stats for {target} in {state_path}")
    return {"norm_stats": stats}


def _zero_rlt_state(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    if args.template_stage1_dir is not None:
        template = torch.load(_checkpoint_file(args.template_stage1_dir), map_location="cpu", mmap=True)
        return {key: torch.zeros_like(value, device="cpu") for key, value in template.items() if key.startswith("rlt_module.")}

    rlt = RLTTokenTransformer(
        input_dim=args.rlt_input_dim,
        embed_dim=args.rlt_embed_dim,
        num_rl_tokens=args.rlt_num_rl_tokens,
        prefix_seq_len=args.rlt_prefix_seq_len,
        num_layers=args.rlt_num_layers,
        num_heads=args.rlt_num_heads,
        mlp_ratio=args.rlt_mlp_ratio,
    )
    return {f"rlt_module.{key}": torch.zeros_like(value, device="cpu", dtype=torch.bfloat16) for key, value in rlt.state_dict().items()}


def _extra_template_base_state(args: argparse.Namespace, base_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    extras: dict[str, torch.Tensor] = {}
    lm_head = "paligemma_with_expert.paligemma.lm_head.weight"
    embed = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    if embed not in base_state and lm_head in base_state:
        extras[embed] = base_state[lm_head].detach().cpu().contiguous()

    if args.template_stage1_dir is not None:
        template = torch.load(_checkpoint_file(args.template_stage1_dir), map_location="cpu", mmap=True)
        for key, value in template.items():
            if key.startswith("noise_head.") and key not in base_state:
                extras[key] = torch.zeros_like(value, device="cpu")
    return extras


def _lerobot_to_online_rl(args: argparse.Namespace) -> None:
    policy_dir = args.lerobot_policy_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "model_state_dict"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model_path = policy_dir / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"Could not find model.safetensors under {policy_dir}")

    base_state = {
        key.removeprefix("model."): value.detach().cpu().contiguous()
        for key, value in load_file(model_path, device="cpu").items()
    }
    extra_state = _extra_template_base_state(args, base_state)
    rlt_state = _zero_rlt_state(args)
    full_state = {**base_state, **extra_state, **rlt_state}
    torch.save(full_state, weights_dir / "full_weights.pt")

    norm_stats = _extract_lerobot_norm_stats(policy_dir)
    (out_dir / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2) + "\n")
    shutil.copy2(policy_dir / "config.json", out_dir / "source_config.json")

    print(f"saved {out_dir}")
    print(f"base model tensors {len(base_state)}")
    print(f"extra template tensors {len(extra_state)}")
    print(f"zero rlt tensors {len(rlt_state)}")


def main() -> None:
    args = _parse_args()
    if args.lerobot_policy_dir is not None:
        _lerobot_to_online_rl(args)
    else:
        _stage1_to_lerobot(args)


if __name__ == "__main__":
    main()
