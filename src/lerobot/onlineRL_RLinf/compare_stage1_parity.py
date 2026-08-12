#!/usr/bin/env python3
"""Compare dataset actions with onlineRL and RLinf Stage1 VLA outputs.

The RLinf Stage1 training path for the default Piper package-sorting setup is:

docker run -d --gpus all --shm-size 120g --network host --name rlinf \
  -v /home/yz:/home/yz \
  docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero \
  tail -f /dev/null
docker exec -it rlinf /bin/bash
source switch_env openpi
cd /home/yz/projects/RLinf_yuzhang
bash examples/sft/run_vla_sft.sh realworld_rlt_stage1_sft_openpi_pi05

After Stage1 has produced a checkpoint, run this from
``/home/yz/projects/Evo-RL-loop-0810/src`` in the same OpenPI-capable
environment:

python -m lerobot.onlineRL.compare_stage1_parity --random_sample --seed 0 --with_rlinf
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.onlineRL.config import load_config


DEFAULT_ONLINE_CONFIG = Path("lerobot/onlineRL/configs/piper_rlt_ac_packageSorting.json")
DEFAULT_RLINF_PATH = Path("/home/yz/projects/RLinf_yuzhang")
DEFAULT_STAGE1_CONFIG = DEFAULT_RLINF_PATH / "examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml"
DEFAULT_STAGE1_EXPERIMENT = "realworld_rlt_stage1_sft_openpi_pi05"

STAGE1_LAUNCH = """\
docker run -d --gpus all \\
   --shm-size 120g \\
   --network host \\
   --name rlinf \\
   -v /home/yz:/home/yz \\
   docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero \\
   tail -f /dev/null

docker exec -it rlinf /bin/bash
source switch_env openpi
cd /home/yz/projects/RLinf_yuzhang
bash examples/sft/run_vla_sft.sh realworld_rlt_stage1_sft_openpi_pi05
"""


class AttrDict(dict):
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default=DEFAULT_ONLINE_CONFIG, type=Path)
    parser.add_argument("--dataset_path", default=None, type=Path)
    parser.add_argument("--rlinf_path", default=DEFAULT_RLINF_PATH, type=Path)
    parser.add_argument("--stage1_config", default=DEFAULT_STAGE1_CONFIG, type=Path)
    parser.add_argument(
        "--stage1_checkpoint",
        default=None,
        type=Path,
        help="Stage1 actor checkpoint dir. Defaults to onlineRL JSON model_path, then latest RLinf log checkpoint.",
    )
    parser.add_argument(
        "--no_stage1_defaults",
        action="store_true",
        help="Do not merge dataset/norm/OpenPI defaults from the RLinf Stage1 YAML.",
    )
    parser.add_argument("--print_stage1_launch", action="store_true", help="Print the RLinf Stage1 docker/openpi launch commands.")
    parser.add_argument("--episode", default=0, type=int)
    parser.add_argument("--frames", default="0,25,50,100", help="Comma-separated frame_index values inside the episode.")
    parser.add_argument("--num_samples", default=0, type=int, help="If >0, ignore --frames and sample evenly over the episode.")
    parser.add_argument("--random_sample", action="store_true", help="Randomly choose one episode/frame with a full future action chunk.")
    parser.add_argument("--seed", default=None, type=int, help="Seed for --random_sample.")
    parser.add_argument("--device", default=None, help="Override Stage1 device, e.g. cuda:0 or cpu.")
    parser.add_argument("--video_backend", default="torchcodec", help="LeRobot video backend: torchcodec, pyav, or video_reader.")
    parser.add_argument("--with_rlinf", action="store_true", help="Also run RLinf inference core if its environment is available.")
    parser.add_argument("--chunk", default=None, type=int, help="Action chunk length for error metrics; defaults to config ref_num_action_chunks.")
    parser.add_argument("--image_key", default="image", help="LeRobot video key for the main camera.")
    parser.add_argument("--wrist_image_key", default="wrist_image", help="LeRobot video key for the wrist camera.")
    parser.add_argument(
        "--no_dataset_prompt",
        action="store_true",
        help="Use runtime.feature_extractor_kwargs.task_description instead of the dataset prompt for onlineRL inference.",
    )
    parser.add_argument(
        "--no_inspect_inputs",
        action="store_true",
        help="Disable the step-by-step onlineRL/RLinf input-processing audit.",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read --stage1_config.") from exc
    with path.open() as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}.")
    return payload


def _get_nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _checkpoint_exists(path: str | Path | None) -> bool:
    if path in (None, ""):
        return False
    root = Path(path).expanduser()
    return any(
        candidate.is_file()
        for candidate in (
            root / "model_state_dict" / "full_weights.pt",
            root / "actor" / "model_state_dict" / "full_weights.pt",
            root,
        )
    )


def _latest_stage1_actor_checkpoint(rlinf_path: Path, experiment: str = DEFAULT_STAGE1_EXPERIMENT) -> Path | None:
    checkpoints = []
    pattern = f"*-{experiment}/{experiment}/checkpoints/global_step_*/actor/model_state_dict/full_weights.pt"
    for weights in (rlinf_path / "logs").glob(pattern):
        actor_dir = weights.parents[1]
        step_text = actor_dir.parent.name.removeprefix("global_step_")
        try:
            step = int(step_text)
        except ValueError:
            step = -1
        checkpoints.append((weights.stat().st_mtime, step, actor_dir))
    if not checkpoints:
        return None
    return max(checkpoints)[2]


def _apply_stage1_defaults(cfg, args: argparse.Namespace) -> Path:
    stage1 = {} if args.no_stage1_defaults else _read_yaml(args.stage1_config)
    kwargs = dict(cfg.runtime.feature_extractor_kwargs)

    dataset_path = args.dataset_path or _get_nested(stage1, "data", "train_data_paths")
    if dataset_path is None:
        raise ValueError("dataset_path is required when --no_stage1_defaults is set or the Stage1 YAML has no data.train_data_paths.")
    dataset_path = Path(str(dataset_path)).expanduser()

    actor_model = _get_nested(stage1, "actor", "model", default={}) or {}
    openpi_data = actor_model.get("openpi_data", {}) if isinstance(actor_model, dict) else {}
    openpi = actor_model.get("openpi", {}) if isinstance(actor_model, dict) else {}

    if args.stage1_checkpoint is not None:
        checkpoint = args.stage1_checkpoint.expanduser()
    elif _checkpoint_exists(kwargs.get("model_path")):
        checkpoint = Path(str(kwargs["model_path"])).expanduser()
    else:
        checkpoint = _latest_stage1_actor_checkpoint(args.rlinf_path)
    if checkpoint is None:
        raise FileNotFoundError(
            "Could not locate a Stage1 actor checkpoint. Pass --stage1_checkpoint after running "
            "bash examples/sft/run_vla_sft.sh realworld_rlt_stage1_sft_openpi_pi05."
        )
    kwargs["model_path"] = str(checkpoint)

    if isinstance(openpi_data, dict):
        for key in ("repo_id", "norm_stats_path"):
            if openpi_data.get(key) not in (None, ""):
                kwargs[key] = openpi_data[key]
    if isinstance(openpi, dict):
        mapping = {
            "config_name": "config_name",
            "num_images_in_input": "num_images_in_input",
            "num_steps": "num_steps",
            "state_indices": "state_indices",
            "use_rlt": "use_rlt",
            "rlt_prefix_seq_len": "rlt_prefix_seq_len",
            "rlt_image_only": "rlt_image_only",
            "rlt_use_mask": "rlt_use_mask",
            "noise_method": "noise_method",
            "noise_params": "noise_params",
            "joint_logprob": "joint_logprob",
            "detach_critic_input": "detach_critic_input",
        }
        for source_key, target_key in mapping.items():
            if openpi.get(source_key) not in (None, ""):
                kwargs[target_key] = openpi[source_key]
    if actor_model.get("num_action_chunks") not in (None, ""):
        chunks = int(actor_model["num_action_chunks"])
        kwargs["action_chunk"] = chunks
        cfg.model.ref_num_action_chunks = chunks
        cfg.model.num_action_chunks = chunks
    if actor_model.get("num_steps") not in (None, ""):
        kwargs["num_steps"] = int(actor_model["num_steps"])
    if actor_model.get("action_dim") not in (None, ""):
        action_dim = int(actor_model["action_dim"])
        kwargs["action_dim"] = action_dim
        cfg.model.action_dim = action_dim

    cfg.runtime.feature_extractor_kwargs = kwargs
    return dataset_path


def _episode_chunk(dataset_path: Path, episode: int) -> int:
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    return episode // int(info.get("chunks_size", 1000))


def _total_episodes(dataset_path: Path) -> int:
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    return int(info["total_episodes"])


def _read_episode_table(dataset_path: Path, episode: int):
    import pyarrow.parquet as pq

    chunk = _episode_chunk(dataset_path, episode)
    parquet_path = dataset_path / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    table = pq.read_table(parquet_path, columns=["state", "actions", "prompt", "frame_index"])
    return table.to_pydict()


def _read_random_sample(dataset_path: Path, horizon: int, rng: random.Random) -> tuple[int, dict[str, Any], list[int]]:
    episodes = list(range(_total_episodes(dataset_path)))
    rng.shuffle(episodes)
    for episode_index in episodes:
        episode = _read_episode_table(dataset_path, episode_index)
        max_row = len(episode["actions"]) - horizon
        if max_row >= 0:
            row = rng.randint(0, max_row)
            return episode_index, episode, [int(episode["frame_index"][row])]
    raise ValueError(f"No episode in {dataset_path} has at least {horizon} action frames.")


def _read_video_frame(
    dataset_path: Path, episode: int, video_key: str, frame_index: int, backend: str
) -> np.ndarray:
    from lerobot.datasets.video_utils import decode_video_frames

    chunk = _episode_chunk(dataset_path, episode)
    video_path = dataset_path / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode:06d}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    timestamp = frame_index / fps
    tolerance = max(1.0 / fps, 0.05)
    try:
        frame = decode_video_frames(video_path, [timestamp], tolerance_s=tolerance, backend=backend)[0]
    except TypeError as exc:
        if backend != "torchcodec" or "LocalFileOpener" not in str(exc):
            raise
        frame = decode_video_frames(video_path, [timestamp], tolerance_s=tolerance, backend="pyav")[0]
    array = frame.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    return array


def _select_frames(frame_indices: list[int], requested: str, num_samples: int) -> list[int]:
    if num_samples > 0:
        if num_samples == 1:
            return [frame_indices[0]]
        positions = np.linspace(0, len(frame_indices) - 1, num_samples).round().astype(int)
        return [int(frame_indices[pos]) for pos in positions]
    wanted = [int(item.strip()) for item in requested.split(",") if item.strip()]
    available = set(frame_indices)
    missing = [item for item in wanted if item not in available]
    if missing:
        raise ValueError(f"Requested frame_index values not in episode: {missing[:10]}")
    return wanted


def _format(values: Any) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{item:8.3f}" for item in arr.tolist()) + "]"


def _chunk_from_rows(actions: list[list[float]], row: int, horizon: int) -> np.ndarray:
    chunk = np.asarray(actions[row : row + horizon], dtype=np.float32)
    if len(chunk) == 0:
        raise ValueError("Empty action chunk.")
    if len(chunk) < horizon:
        pad = np.repeat(chunk[-1:], horizon - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk


def _max_abs(lhs: np.ndarray, rhs: np.ndarray) -> float:
    horizon = min(lhs.shape[0], rhs.shape[0])
    return float(np.max(np.abs(lhs[:horizon] - rhs[:horizon])))


def _build_online_extractor(cfg, args: argparse.Namespace):
    from lerobot.onlineRL.vla import load_feature_extractor

    kwargs = dict(cfg.runtime.feature_extractor_kwargs)
    if args.device:
        kwargs["device"] = args.device
    return load_feature_extractor(cfg.runtime.feature_extractor_factory, cfg.model, False, kwargs)


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


def _build_rlinf_model(cfg, args: argparse.Namespace):
    _extend_lerobot_namespace_for_openpi()
    sys.path.insert(0, str(args.rlinf_path))
    from rlinf.models.embodiment.openpi import get_model

    kwargs = dict(cfg.runtime.feature_extractor_kwargs)
    model_cfg = AttrDict(
        model_path=kwargs["model_path"],
        openpi_data={
            "repo_id": kwargs.get("repo_id"),
            "norm_stats_path": kwargs.get("norm_stats_path"),
        },
        openpi=AttrDict(
            config_name=kwargs.get("config_name", "pi05_piper_state"),
            num_images_in_input=kwargs.get("num_images_in_input", 2),
            action_chunk=kwargs.get("action_chunk", cfg.model.ref_num_action_chunks),
            num_steps=kwargs.get("num_steps", 10),
            state_indices=kwargs.get("state_indices", []),
            use_rlt=kwargs.get("use_rlt", True),
            rlt_prefix_seq_len=kwargs.get("rlt_prefix_seq_len", 1024),
            rlt_image_only=kwargs.get("rlt_image_only", False),
            rlt_use_mask=kwargs.get("rlt_use_mask", True),
            noise_method=kwargs.get("noise_method", "flow_noise"),
            noise_params=kwargs.get("noise_params", [0.16, 0.12, 200]),
            joint_logprob=kwargs.get("joint_logprob", True),
            detach_critic_input=kwargs.get("detach_critic_input", True),
        ),
    )
    model = get_model(model_cfg)
    device = torch.device(args.device or cfg.runtime.actor_device)
    model.to(device).eval()
    return model


def _rlinf_infer(model, state: np.ndarray, image: np.ndarray, wrist_image: np.ndarray, prompt: str) -> np.ndarray:
    env_obs = {
        "states": state[None, :].astype(np.float32),
        "main_images": image[None, ...],
        "wrist_images": wrist_image[None, ...],
        "extra_view_images": None,
        "task_descriptions": [prompt],
    }
    with torch.no_grad():
        out = model.extract_rlt_obs(env_obs)
    return out["ref_chunk"].detach().cpu().numpy()[0]


def _as_float_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _summary(name: str, value: Any, *, head: int = 8) -> str:
    arr = _as_float_array(value)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return f"{name}: shape={arr.shape} empty"
    head_values = ", ".join(f"{item:.4f}" for item in flat[:head].tolist())
    return (
        f"{name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={flat.min():.4f} max={flat.max():.4f} mean={flat.mean():.4f} "
        f"head=[{head_values}]"
    )


def _token_summary(name: str, tokens: Any, mask: Any | None = None, *, head: int = 32) -> str:
    token_arr = np.asarray(torch.as_tensor(tokens).detach().cpu()).reshape(-1)
    token_head = ", ".join(str(int(item)) for item in token_arr[:head].tolist())
    if mask is None:
        return f"{name}: shape={token_arr.shape} head=[{token_head}]"
    mask_arr = np.asarray(torch.as_tensor(mask).detach().cpu()).reshape(-1)
    return f"{name}: shape={token_arr.shape} mask_true={int(mask_arr.sum())} head=[{token_head}]"


def _safe_decode(tokenizer: Any, token_ids: Any, mask: Any | None = None) -> str:
    ids = torch.as_tensor(token_ids).detach().cpu().reshape(-1)
    if mask is not None:
        mask_tensor = torch.as_tensor(mask).detach().cpu().reshape(-1).bool()
        ids = ids[mask_tensor]
    try:
        return tokenizer.decode(ids.tolist(), skip_special_tokens=False)
    except Exception as exc:
        return f"<decode failed: {type(exc).__name__}: {exc}>"


def _print_lines(title: str, lines: list[str]) -> None:
    print(f"-- {title} --")
    for line in lines:
        print(line)


def _audit_online_inputs(online, state: np.ndarray, image: np.ndarray, wrist: np.ndarray, prompt: str, use_dataset_prompt: bool) -> None:
    if use_dataset_prompt and hasattr(online, "task_description"):
        online.task_description = prompt
    batch, raw_state = online._batch({"state": state, "top_image": image, "wrist_image": wrist})
    images, img_masks = online.policy._preprocess_images(batch)
    prompt_text = online._prompt(raw_state)
    token_ids = batch["observation.language.tokens"]
    token_mask = batch["observation.language.attention_mask"]
    lines = [
        f"task_description={online.task_description}",
        f"prompt_text={prompt_text!r}",
        f"decoded_prompt={_safe_decode(online.tokenizer, token_ids[0], token_mask[0])!r}",
        _token_summary("tokens", token_ids, token_mask),
        _summary("raw_state", raw_state),
        _summary("norm_state", batch["observation.state"]),
        _summary("input_top_image_CHW_0_1", batch[online.base_key]),
        _summary("input_wrist_image_CHW_0_1", batch[online.left_wrist_key]),
    ]
    for idx, (img, mask) in enumerate(zip(images, img_masks, strict=True)):
        lines.append(_summary(f"preprocessed_image[{idx}]", img))
        lines.append(_summary(f"image_mask[{idx}]", mask))
    _print_lines("onlineRL input pipeline", lines)


def _audit_rlinf_inputs(model, state: np.ndarray, image: np.ndarray, wrist: np.ndarray, prompt: str) -> None:
    from openpi.models import model as _model

    env_obs = {
        "states": state[None, :].astype(np.float32),
        "main_images": image[None, ...],
        "wrist_images": wrist[None, ...],
        "extra_view_images": None,
        "task_descriptions": [prompt],
    }
    to_process = model.obs_processor(env_obs)
    processed = model.input_transform(to_process, transpose=False)
    processed = model.precision_processor(processed)
    observation = _model.Observation.from_dict(processed)
    images, img_masks, lang_tokens, lang_masks, obs_state = model._preprocess_observation(observation, train=False)

    lines = [
        f"obs_processor_keys={sorted(to_process)}",
        _summary("env_states", env_obs["states"]),
        _summary("env_main_images_HWC", env_obs["main_images"]),
        _summary("env_wrist_images_HWC", env_obs["wrist_images"]),
        f"env_prompt={prompt!r}",
        _summary("to_process_observation/state", to_process.get("observation/state")),
        _summary("to_process_observation/image", to_process.get("observation/image")),
        _summary("to_process_observation/wrist_image", to_process.get("observation/wrist_image")),
        f"to_process_prompt={to_process.get('prompt')!r}",
        f"processed_keys={sorted(processed)}",
        _summary("processed_state", processed.get("state")),
        _summary("observation.state", observation.state),
        _summary("preprocess_state", obs_state),
    ]
    if lang_tokens is not None:
        lines.append(_token_summary("lang_tokens", lang_tokens, lang_masks))
    for idx, (img, mask) in enumerate(zip(images, img_masks, strict=True)):
        lines.append(_summary(f"preprocessed_image[{idx}]", img))
        lines.append(_summary(f"image_mask[{idx}]", mask))
    _print_lines("RLinf input pipeline", lines)


def _audit_pair(online, rlinf_model, state: np.ndarray, image: np.ndarray, wrist: np.ndarray, prompt: str, use_dataset_prompt: bool) -> None:
    _print_lines(
        "dataset frame input",
        [
            _summary("state", state),
            _summary("image_HWC", image),
            _summary("wrist_HWC", wrist),
            f"prompt={prompt!r}",
        ],
    )
    _audit_online_inputs(online, state, image, wrist, prompt, use_dataset_prompt)
    if rlinf_model is not None:
        _audit_rlinf_inputs(rlinf_model, state, image, wrist, prompt)


def _online_infer(online, state: np.ndarray, image: np.ndarray, wrist: np.ndarray, prompt: str, use_dataset_prompt: bool):
    if use_dataset_prompt and hasattr(online, "task_description"):
        online.task_description = prompt
    obs = {"state": state, "top_image": image, "wrist_image": wrist, "prompt": prompt}
    with torch.no_grad():
        return online(obs).ref_chunk.detach().cpu().numpy()[0]


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    if args.print_stage1_launch:
        print(STAGE1_LAUNCH)
        return
    cfg = load_config(args.config_path)
    dataset_path = _apply_stage1_defaults(cfg, args)
    horizon = int(args.chunk or cfg.model.ref_num_action_chunks)

    if args.random_sample:
        args.episode, episode, selected_frames = _read_random_sample(
            dataset_path, horizon, random.Random(args.seed)
        )
    else:
        episode = _read_episode_table(dataset_path, args.episode)
        selected_frames = _select_frames([int(item) for item in episode["frame_index"]], args.frames, args.num_samples)
    frame_indices = [int(item) for item in episode["frame_index"]]
    frame_to_row = {frame: row for row, frame in enumerate(frame_indices)}

    print(f"online_config={args.config_path}")
    print(f"stage1_config={args.stage1_config if not args.no_stage1_defaults else '<disabled>'}")
    print(f"dataset={dataset_path}")
    print(f"episode={args.episode} frames={selected_frames} horizon={horizon} random_sample={args.random_sample}")
    print(f"checkpoint={cfg.runtime.feature_extractor_kwargs.get('model_path')}")
    print(f"norm_stats={cfg.runtime.feature_extractor_kwargs.get('norm_stats_path')}")
    print(f"image_keys=({args.image_key}, {args.wrist_image_key}) dataset_prompt={not args.no_dataset_prompt}")
    print()

    online = _build_online_extractor(cfg, args)
    rlinf_model = _build_rlinf_model(cfg, args) if args.with_rlinf else None

    for frame in selected_frames:
        row = frame_to_row[frame]
        state = np.asarray(episode["state"][row], dtype=np.float32)
        prompt = str(episode["prompt"][row])
        image = _read_video_frame(dataset_path, args.episode, args.image_key, frame, args.video_backend)
        wrist = _read_video_frame(dataset_path, args.episode, args.wrist_image_key, frame, args.video_backend)
        actual_chunk = _chunk_from_rows(episode["actions"], row, horizon)

        if not args.no_inspect_inputs:
            _audit_pair(online, rlinf_model, state, image, wrist, prompt, not args.no_dataset_prompt)

        online_chunk = _online_infer(online, state, image, wrist, prompt, not args.no_dataset_prompt)
        rlinf_chunk = None if rlinf_model is None else _rlinf_infer(rlinf_model, state, image, wrist, prompt)

        print("=" * 100)
        print(f"frame_index={frame} row={row} prompt={prompt}")
        print(f"state            {_format(state)}")
        print(f"dataset action   {_format(actual_chunk[0])}")
        print(f"onlineRL action  {_format(online_chunk[0])}")
        if rlinf_chunk is not None:
            print(f"RLinf action     {_format(rlinf_chunk[0])}")
        print("-- chunk max_abs_error --")
        print(f"onlineRL vs dataset: {_max_abs(online_chunk, actual_chunk):.6f}")
        if rlinf_chunk is not None:
            print(f"RLinf    vs dataset: {_max_abs(rlinf_chunk, actual_chunk):.6f}")
            print(f"onlineRL vs RLinf  : {_max_abs(online_chunk, rlinf_chunk):.6f}")
        header = "-- first 5 actions: dataset | onlineRL" + (" | RLinf" if rlinf_chunk is not None else "") + " --"
        print(header)
        for idx in range(min(5, horizon)):
            parts = [f"t+{idx:02d}", _format(actual_chunk[idx]), _format(online_chunk[idx])]
            if rlinf_chunk is not None:
                parts.append(_format(rlinf_chunk[idx]))
            print(" | ".join(parts))


if __name__ == "__main__":
    main()
