#!/usr/bin/env python

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames


def _smooth_1d(arr: np.ndarray, window: int = 1) -> np.ndarray:
    """Savitzky-Golay smoothing for 1-D array. Use window=1 to disable."""
    if window <= 1 or arr.shape[0] < 5:
        return arr.copy()

    w = min(int(window), int(arr.shape[0]))
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return arr.copy()

    return savgol_filter(arr, window_length=w, polyorder=3).astype(arr.dtype)


def _select_video_key(camera_keys: list[str], requested_video_key: str | None) -> str:
    if len(camera_keys) == 0:
        raise ValueError("No camera key found in dataset.")
    if requested_video_key is not None:
        if requested_video_key not in camera_keys:
            raise ValueError(
                f"Unknown video_key '{requested_video_key}'. Available camera keys: {camera_keys}"
            )
        return requested_video_key

    for key in camera_keys:
        lower = key.lower()
        if ".front" in lower or "_front" in lower or lower.endswith("front"):
            return key
    return camera_keys[0]


def _select_video_keys(
    camera_keys: list[str],
    requested_video_keys: str | None,
    requested_video_key: str | None,
) -> list[str]:
    if len(camera_keys) == 0:
        raise ValueError("No camera key found in dataset.")

    if requested_video_keys is None or not requested_video_keys.strip():
        return [_select_video_key(camera_keys, requested_video_key)]

    resolved: list[str] = []
    for raw_key in requested_video_keys.split(","):
        key = raw_key.strip()
        if not key:
            continue
        if key not in camera_keys:
            raise ValueError(f"Unknown video key '{key}'. Available camera keys: {camera_keys}")
        if key not in resolved:
            resolved.append(key)

    if not resolved:
        raise ValueError("'viz.video_keys' is empty after parsing. Provide comma-separated camera keys.")

    return resolved


def _parse_episodes_arg(episodes_arg: str, total_episodes: int) -> list[int]:
    value = episodes_arg.strip().lower()
    if value == "all":
        return list(range(total_episodes))

    parsed: set[int] = set()
    for token in episodes_arg.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range '{part}'.")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(part))

    episodes = sorted(parsed)
    for ep in episodes:
        if ep < 0 or ep >= total_episodes:
            raise ValueError(f"Episode index out of range: {ep}, total_episodes={total_episodes}.")
    return episodes


def _to_1d_float(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.reshape(-1)


def _to_1d_int(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.reshape(-1)


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _curve_points(
    values: np.ndarray,
    current_step: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    n = len(values)
    if n == 0:
        return []

    denom_x = max(1, n - 1)
    denom_y = max(1e-6, y_max - y_min)

    points = []
    last_step = min(current_step, n - 1)
    for i in range(last_step + 1):
        x = int(round(x0 + width * (i / denom_x)))
        y_norm = np.clip((float(values[i]) - y_min) / denom_y, 0.0, 1.0)
        y = int(round(y0 + (1.0 - y_norm) * height))
        points.append((x, y))
    return points


def _draw_overlay(
    frame: Image.Image,
    values: np.ndarray,
    current_step: int,
    advantage_t: float,
    acp_t: int,
    highlight_current_point: bool,
    y_min: float,
    y_max: float,
    indicators: np.ndarray | None = None,
    show_zero_line: bool = False,
    targets: np.ndarray | None = None,
) -> Image.Image:
    # Keep signature/call-sites stable; style does not use these signals.
    _ = (advantage_t, acp_t, highlight_current_point, indicators)

    rgba = frame.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = rgba.size
    n = len(values)
    if n == 0:
        return Image.alpha_composite(rgba, overlay).convert("RGB")

    margin_x = max(12, width // 96)
    margin_top = max(8, height // 72)
    margin_bottom = max(24, height // 36)
    chart_x0 = margin_x
    chart_x1 = width - margin_x
    chart_y0 = margin_top
    chart_y1 = height - margin_bottom
    chart_w = max(1, chart_x1 - chart_x0)
    chart_h = max(1, chart_y1 - chart_y0)

    denom_y = max(1e-6, y_max - y_min)
    padding = denom_y * 0.05
    val_min = y_min - padding
    val_max = y_max + padding

    last = min(current_step, n - 1)
    denom_x = max(1, n - 1)

    # Light grid + current-frame vertical cursor.
    grid_color = (255, 255, 255, 30)
    for frac in (0.25, 0.5, 0.75):
        gy = int(round(chart_y0 + frac * chart_h))
        draw.line([(chart_x0, gy), (chart_x1, gy)], fill=grid_color, width=1)
    cx = int(round(chart_x0 + chart_w * (last / denom_x)))
    draw.line([(cx, chart_y0), (cx, chart_y1)], fill=(255, 255, 255, 40), width=1)

    # Zero-line: drawn only when range spans 0 (used for advantage curves).
    if show_zero_line and val_min < 0 < val_max:
        zero_frac = np.clip((0.0 - val_min) / (val_max - val_min), 0.0, 1.0)
        zy = int(round(chart_y0 + (1.0 - float(zero_frac)) * chart_h))
        draw.line([(chart_x0, zy), (chart_x1, zy)], fill=(255, 100, 100, 200), width=2)

    curve_width = max(2, width // 600)
    pred_color = (100, 200, 255, 220)   # blue  : model prediction V_pred
    target_color = (120, 255, 120, 220)  # green : ground-truth value target V_target
    has_targets = targets is not None and len(targets) == n

    # Target curve first so the prediction is drawn on top of it.
    if has_targets:
        target_points = _curve_points(
            values=targets,
            current_step=last,
            x0=chart_x0,
            y0=chart_y0,
            width=chart_w,
            height=chart_h,
            y_min=val_min,
            y_max=val_max,
        )
        if len(target_points) >= 2:
            draw.line(target_points, fill=target_color, width=curve_width)

    points = _curve_points(
        values=values,
        current_step=last,
        x0=chart_x0,
        y0=chart_y0,
        width=chart_w,
        height=chart_h,
        y_min=val_min,
        y_max=val_max,
    )

    if len(points) >= 2:
        draw.line(points, fill=pred_color, width=curve_width)

    if len(points) >= 1:
        px, py = points[-1]
        radius = max(4, curve_width + 2)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(255, 255, 100, 255))
        draw.ellipse(
            (px - radius - 2, py - radius - 2, px + radius + 2, py + radius + 2),
            outline=(255, 255, 100, 180),
            width=1,
        )

        value_font = _load_font(max(16, height // 22))
        if has_targets:
            value_text = f"pred {float(values[last]):.4f}  tgt {float(targets[last]):.4f}"
        else:
            value_text = f"{float(values[last]):.4f}"
        value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
        text_w = value_bbox[2] - value_bbox[0]
        text_h = value_bbox[3] - value_bbox[1]
        tx = min(px + 10, width - text_w - 10)
        ty = max(py - text_h - 12, chart_y0 + 2)
        draw.rectangle((tx - 4, ty - 2, tx + text_w + 4, ty + text_h + 2), fill=(0, 0, 0, 160))
        draw.text((tx, ty), value_text, fill=(255, 255, 100, 255), font=value_font)

    # Legend (top-left) distinguishing prediction vs target.
    if has_targets:
        legend_font = _load_font(max(13, height // 30))
        lx = chart_x0 + 6
        ly = chart_y0 + 4
        sw = max(10, height // 48)
        line_h = max(sw + 4, 18)
        for color, label in ((pred_color, "V_pred"), (target_color, "V_target")):
            tb = draw.textbbox((0, 0), label, font=legend_font)
            tw = tb[2] - tb[0]
            draw.rectangle((lx - 2, ly - 2, lx + sw + 6 + tw, ly + sw + 2), fill=(0, 0, 0, 150))
            draw.rectangle((lx, ly, lx + sw, ly + sw), fill=color)
            draw.text((lx + sw + 4, ly), label, fill=(235, 235, 235, 255), font=legend_font)
            ly += line_h

    small_font = _load_font(max(13, height // 30))
    frame_text = f"frame {last}/{n - 1}"
    frame_bbox = draw.textbbox((0, 0), frame_text, font=small_font)
    frame_w = frame_bbox[2] - frame_bbox[0]
    frame_h = frame_bbox[3] - frame_bbox[1]
    fx = width - frame_w - margin_x - 4
    fy = height - frame_h - 6
    draw.rectangle((fx - 3, fy - 1, fx + frame_w + 3, fy + frame_h + 1), fill=(0, 0, 0, 140))
    draw.text((fx, fy), frame_text, fill=(200, 200, 200, 220), font=small_font)

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _draw_indicator_overlay(
    frame: Image.Image,
    indicators: np.ndarray,
    current_step: int,
    advantages: np.ndarray | None = None,
) -> Image.Image:
    """Overlay the binarized advantage/disadvantage verdict on a single frame.

    Renders three things so the per-frame ACP indicator (1 = advantage, 0 =
    disadvantage) is unambiguous:
    * a verdict badge (top-center) — green ``ADVANTAGE`` when ``indicators[t]==1``
      else red ``DISADVANTAGE``, with the raw advantage value if available;
    * a full-episode timeline strip (bottom) coloring every frame green/red so the
      whole advantage/disadvantage segmentation is visible at a glance;
    * a white cursor on the strip marking the current frame, plus a count summary.
    """
    rgba = frame.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = rgba.size
    ind = np.asarray(indicators, dtype=np.int64).reshape(-1)
    n = int(ind.shape[0])
    if n == 0:
        return Image.alpha_composite(rgba, overlay).convert("RGB")

    last = min(current_step, n - 1)
    is_adv = int(ind[last]) == 1

    adv_color = (60, 200, 90, 255)     # green : advantage  (indicator == 1)
    disadv_color = (220, 70, 70, 255)  # red   : disadvantage (indicator == 0)
    badge_color = adv_color if is_adv else disadv_color
    mark = "✓" if is_adv else "✗"  # check / cross
    label = f"{mark} ADVANTAGE" if is_adv else f"{mark} DISADVANTAGE"
    if advantages is not None and last < len(advantages):
        label = f"{label}   A={float(advantages[last]):+.4f}"

    # Verdict badge (top-center).
    badge_font = _load_font(max(20, height // 18))
    tb = draw.textbbox((0, 0), label, font=badge_font)
    text_w = tb[2] - tb[0]
    text_h = tb[3] - tb[1]
    pad = max(8, height // 90)
    bx0 = max(0, (width - text_w) // 2 - pad)
    by0 = max(6, height // 60)
    bx1 = bx0 + text_w + 2 * pad
    by1 = by0 + text_h + 2 * pad
    draw.rectangle((bx0, by0, bx1, by1), fill=(0, 0, 0, 175), outline=badge_color, width=max(2, height // 240))
    draw.text((bx0 + pad, by0 + pad), label, fill=badge_color, font=badge_font)

    # Timeline strip (bottom): one column per frame, colored by its indicator.
    margin_x = max(12, width // 96)
    strip_h = max(14, height // 28)
    strip_x0 = margin_x
    strip_x1 = width - margin_x
    strip_w = max(1, strip_x1 - strip_x0)
    strip_y1 = height - max(20, height // 36)
    strip_y0 = strip_y1 - strip_h

    draw.rectangle((strip_x0 - 1, strip_y0 - 1, strip_x1 + 1, strip_y1 + 1), fill=(0, 0, 0, 150))
    for px in range(strip_w):
        frame_idx = min(n - 1, int(px * n / strip_w))
        col = adv_color if int(ind[frame_idx]) == 1 else disadv_color
        x = strip_x0 + px
        draw.line([(x, strip_y0), (x, strip_y1)], fill=(col[0], col[1], col[2], 235), width=1)

    # Current-frame cursor on the strip.
    cur_x = strip_x0 + int(round(strip_w * (last / max(1, n - 1))))
    draw.line(
        [(cur_x, strip_y0 - 3), (cur_x, strip_y1 + 3)],
        fill=(255, 255, 255, 255),
        width=max(1, width // 480),
    )

    # Summary text above the strip.
    adv_count = int(np.sum(ind == 1))
    summary = f"adv {adv_count}/{n} ({adv_count / float(n):.1%})   frame {last}/{n - 1}"
    small_font = _load_font(max(13, height // 32))
    sb = draw.textbbox((0, 0), summary, font=small_font)
    sw = sb[2] - sb[0]
    sh = sb[3] - sb[1]
    sx = strip_x0
    sy = strip_y0 - sh - 6
    draw.rectangle((sx - 3, sy - 2, sx + sw + 3, sy + sh + 2), fill=(0, 0, 0, 150))
    draw.text((sx, sy), summary, fill=(230, 230, 230, 255), font=small_font)

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _success_tag(value) -> str:
    if value is None:
        return "unknown"
    s = str(value).strip().lower()
    if s == "success":
        return "success"
    if s in ("failure", "failed", "fail"):
        return "failed"
    return s.replace("/", "_").replace(" ", "_") or "unknown"


def _build_episode_success_map(dataset: LeRobotDataset) -> dict[int, str]:
    episodes = getattr(dataset.meta, "episodes", None)
    if episodes is None:
        return {}
    eps_raw = episodes.with_format(None)
    if "episode_index" not in eps_raw.column_names:
        return {}
    ep_indices = np.asarray(eps_raw["episode_index"], dtype=np.int64).reshape(-1)
    if "episode_success" in eps_raw.column_names:
        success_values = eps_raw["episode_success"]
    else:
        success_values = [None] * len(ep_indices)
    return {int(idx): _success_tag(val) for idx, val in zip(ep_indices, success_values)}


def _build_output_video_path(
    output_dir: Path, repo_id: str, video_key: str, episode_index: int, success_tag: str = "unknown"
) -> Path:
    repo_tag = repo_id.replace("/", "_")
    key_tag = video_key.replace(".", "_")
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{success_tag}_{key_tag}.mp4"


def _build_output_video_path_multiview(
    output_dir: Path,
    repo_id: str,
    video_keys: list[str],
    episode_index: int,
    success_tag: str = "unknown",
) -> Path:
    repo_tag = repo_id.replace("/", "_")
    keys_tag = "__".join(key.replace(".", "_") for key in video_keys)
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{success_tag}_{keys_tag}_multiview.mp4"


def _decode_frames_at_timestamps(
    video_file: Path,
    timestamps_s: np.ndarray,
    tolerance_s: float,
    backend: str | None,
) -> list[Image.Image]:
    if timestamps_s.size == 0:
        return []
    frames = decode_video_frames(
        video_path=video_file,
        timestamps=timestamps_s.tolist(),
        tolerance_s=tolerance_s,
        backend=backend,
    )
    np_frames = frames.detach().cpu().numpy()
    if np_frames.ndim != 4:
        raise ValueError(f"Unexpected decoded frame tensor shape: {np_frames.shape}")
    if np_frames.shape[1] in (1, 3):
        np_frames = np.transpose(np_frames, (0, 2, 3, 1))
    if np_frames.dtype != np.uint8:
        if np.issubdtype(np_frames.dtype, np.floating):
            max_val = float(np.max(np_frames)) if np_frames.size > 0 else 1.0
            if max_val <= 1.0 + 1e-6:
                np_frames = np.clip(np_frames, 0.0, 1.0) * 255.0
            else:
                np_frames = np.clip(np_frames, 0.0, 255.0)
        else:
            np_frames = np.clip(np_frames, 0, 255)
        np_frames = np_frames.astype(np.uint8)
    return [Image.fromarray(np_frames[i]) for i in range(np_frames.shape[0])]


def _get_episode_video_time_bounds(
    dataset: LeRobotDataset,
    episode_index: int,
    video_key: str,
) -> tuple[float, float | None]:
    episodes = getattr(dataset.meta, "episodes", None)
    if episodes is None:
        return 0.0, None
    episodes_ds = episodes.with_format(None)
    if "episode_index" not in episodes_ds.column_names:
        return 0.0, None

    episode_indices = np.asarray(episodes_ds["episode_index"], dtype=np.int64).reshape(-1)
    matched = np.flatnonzero(episode_indices == episode_index)
    if matched.size == 0:
        return 0.0, None
    row = int(matched[0])

    from_col = f"videos/{video_key}/from_timestamp"
    to_col = f"videos/{video_key}/to_timestamp"
    from_ts = 0.0
    to_ts: float | None = None
    if from_col in episodes_ds.column_names:
        from_ts = float(episodes_ds[from_col][row])
    if to_col in episodes_ds.column_names:
        to_ts = float(episodes_ds[to_col][row])
    return from_ts, to_ts


def _get_video_encode_options(vcodec: str) -> tuple[dict[str, str], str]:
    if vcodec == "libsvtav1":
        return {"g": "2", "crf": "30", "preset": "12"}, "yuv420p"
    if vcodec == "h264_nvenc":
        return {
            "preset": "p4",
            "rc": "vbr",
            "cq": "28",
            "b": "0",
            "g": "60",
        }, "yuv420p"
    return {"g": "2", "crf": "30"}, "yuv420p"


def _get_episode_value_bounds(ep_values: np.ndarray) -> tuple[float, float]:
    if ep_values.shape[0] == 0:
        raise ValueError("Cannot determine value bounds from an empty episode.")
    return float(np.min(ep_values)), float(np.max(ep_values))


def _encode_pil_to_video(
    frames: list[Image.Image],
    video_path: Path,
    fps: int,
    vcodec: str,
) -> None:
    """Encode PIL frames directly to video without writing intermediate PNG files."""
    import av as _av

    video_options, pix_fmt = _get_video_encode_options(vcodec)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with _av.open(str(video_path), "w") as output:
        out_stream = output.add_stream(vcodec, fps, options=video_options)
        out_stream.pix_fmt = pix_fmt
        out_stream.width = frames[0].width
        out_stream.height = frames[0].height
        for pil_img in frames:
            av_frame = _av.VideoFrame.from_image(pil_img.convert("RGB"))
            for packet in out_stream.encode(av_frame):
                output.mux(packet)
        for packet in out_stream.encode():
            output.mux(packet)


def _export_single_episode(
    src_video_path: Path,
    dst_video_path: Path,
    ep_values: np.ndarray,
    ep_advantages: np.ndarray,
    ep_indicators: np.ndarray,
    episode_timestamps_s: np.ndarray,
    fps: int,
    vcodec: str,
    tolerance_s: float,
    video_backend: str | None,
    frame_storage_mode: str = "memory",
    temp_dir_root: Path | None = None,
    smooth_window: int = 1,
    show_zero_line: bool = False,
    fixed_bounds: tuple[float, float] | None = None,
    ep_targets: np.ndarray | None = None,
    indicator_mode: bool = False,
) -> Path:
    # ``indicator_mode`` swaps the value/advantage curve for the binarized
    # advantage/disadvantage verdict overlay; indicators stay un-smoothed (binary).
    ep_advantages_raw = np.asarray(ep_advantages, dtype=np.float32).reshape(-1)
    ep_values = _smooth_1d(ep_values, smooth_window)
    ep_advantages = _smooth_1d(ep_advantages, smooth_window)
    if ep_targets is not None:
        ep_targets = _smooth_1d(ep_targets, smooth_window)
    if not indicator_mode:
        if fixed_bounds is not None:
            y_min, y_max = float(fixed_bounds[0]), float(fixed_bounds[1])
        else:
            y_min, y_max = _get_episode_value_bounds(ep_values)
            if ep_targets is not None and len(ep_targets) > 0:
                y_min = min(y_min, float(np.min(ep_targets)))
                y_max = max(y_max, float(np.max(ep_targets)))
        if show_zero_line:
            y_min = min(y_min, -1e-6)
            y_max = max(y_max, 1e-6)
    decoded_frames = _decode_frames_at_timestamps(
        video_file=src_video_path,
        timestamps_s=episode_timestamps_s,
        tolerance_s=tolerance_s,
        backend=video_backend,
    )
    n_frames = min(len(decoded_frames), len(ep_indicators) if indicator_mode else len(ep_values))
    if n_frames == 0:
        raise ValueError(f"No decoded frames for video: {src_video_path}")

    def _compose(i: int) -> Image.Image:
        if indicator_mode:
            return _draw_indicator_overlay(
                frame=decoded_frames[i],
                indicators=ep_indicators,
                current_step=i,
                advantages=ep_advantages_raw,
            )
        return _draw_overlay(
            frame=decoded_frames[i],
            values=ep_values,
            current_step=i,
            advantage_t=float(ep_advantages[i]),
            acp_t=int(ep_indicators[i]),
            highlight_current_point=(int(ep_indicators[i]) == 1),
            y_min=y_min,
            y_max=y_max,
            indicators=ep_indicators,
            show_zero_line=show_zero_line,
            targets=ep_targets,
        )

    if frame_storage_mode == "disk":
        with tempfile.TemporaryDirectory(
            dir=str(temp_dir_root) if temp_dir_root is not None else None,
            prefix=f"{dst_video_path.stem}-frames-",
        ) as temp_dir:
            temp_path = Path(temp_dir)
            for i in range(n_frames):
                _compose(i).save(temp_path / f"frame-{i:06d}.png")

            encode_video_frames(
                imgs_dir=temp_path,
                video_path=dst_video_path,
                fps=fps,
                vcodec=vcodec,
                overwrite=True,
            )
        return dst_video_path

    composed_frames: list[Image.Image] = [_compose(i) for i in range(n_frames)]
    _encode_pil_to_video(composed_frames, dst_video_path, fps, vcodec)
    return dst_video_path


def _export_single_episode_multiview(
    src_video_paths: list[Path],
    camera_labels: list[str],
    dst_video_path: Path,
    ep_values: np.ndarray,
    ep_advantages: np.ndarray,
    ep_indicators: np.ndarray,
    episode_timestamps_per_cam: list[np.ndarray],
    fps: int,
    vcodec: str,
    tolerance_s: float,
    video_backend: str | None,
    frame_storage_mode: str = "memory",
    temp_dir_root: Path | None = None,
    smooth_window: int = 1,
    show_zero_line: bool = False,
    fixed_bounds: tuple[float, float] | None = None,
    ep_targets: np.ndarray | None = None,
    indicator_mode: bool = False,
) -> Path:
    ep_advantages_raw = np.asarray(ep_advantages, dtype=np.float32).reshape(-1)
    ep_values = _smooth_1d(ep_values, smooth_window)
    ep_advantages = _smooth_1d(ep_advantages, smooth_window)
    if ep_targets is not None:
        ep_targets = _smooth_1d(ep_targets, smooth_window)
    if not indicator_mode:
        if fixed_bounds is not None:
            y_min, y_max = float(fixed_bounds[0]), float(fixed_bounds[1])
        else:
            y_min, y_max = _get_episode_value_bounds(ep_values)
            if ep_targets is not None and len(ep_targets) > 0:
                y_min = min(y_min, float(np.min(ep_targets)))
                y_max = max(y_max, float(np.max(ep_targets)))
        if show_zero_line:
            y_min = min(y_min, -1e-6)
            y_max = max(y_max, 1e-6)

    all_cam_frames: list[list[Image.Image]] = []
    for src_path, ts in zip(src_video_paths, episode_timestamps_per_cam, strict=True):
        frames = _decode_frames_at_timestamps(
            video_file=src_path,
            timestamps_s=ts,
            tolerance_s=tolerance_s,
            backend=video_backend,
        )
        all_cam_frames.append(frames)

    n_frames = min(len(f) for f in all_cam_frames)
    n_frames = min(n_frames, len(ep_values))
    if n_frames == 0:
        raise ValueError(f"No decoded frames for multiview video: {dst_video_path}")

    single_w = all_cam_frames[0][0].width
    single_h = all_cam_frames[0][0].height
    n_cams = len(all_cam_frames)
    total_w = single_w * n_cams

    label_font_size = max(14, single_h // 30)
    label_font = _load_font(label_font_size)

    def _compose_frame(i: int) -> Image.Image:
        wide_frame = Image.new("RGB", (total_w, single_h))
        for cam_idx, cam_frames in enumerate(all_cam_frames):
            wide_frame.paste(cam_frames[i].resize((single_w, single_h)), (cam_idx * single_w, 0))

        label_draw = ImageDraw.Draw(wide_frame)
        for cam_idx, label in enumerate(camera_labels):
            short_label = label.split(".")[-1]
            lx = cam_idx * single_w + 8
            ly = 4
            bbox = label_draw.textbbox((lx, ly), short_label, font=label_font)
            label_draw.rectangle(
                (bbox[0] - 2, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
                fill=(0, 0, 0, 180),
            )
            label_draw.text((lx, ly), short_label, fill=(255, 255, 200), font=label_font)

        if indicator_mode:
            return _draw_indicator_overlay(
                frame=wide_frame,
                indicators=ep_indicators,
                current_step=i,
                advantages=ep_advantages_raw,
            )
        return _draw_overlay(
            frame=wide_frame,
            values=ep_values,
            current_step=i,
            advantage_t=float(ep_advantages[i]) if i < len(ep_advantages) else 0.0,
            acp_t=int(ep_indicators[i]) if i < len(ep_indicators) else 0,
            highlight_current_point=False,
            y_min=y_min,
            y_max=y_max,
            indicators=ep_indicators,
            show_zero_line=show_zero_line,
            targets=ep_targets,
        )

    if frame_storage_mode == "disk":
        with tempfile.TemporaryDirectory(
            dir=str(temp_dir_root) if temp_dir_root is not None else None,
            prefix=f"{dst_video_path.stem}-frames-",
        ) as temp_dir:
            temp_path = Path(temp_dir)
            for i in range(n_frames):
                _compose_frame(i).save(temp_path / f"frame-{i:06d}.png")

            encode_video_frames(
                imgs_dir=temp_path,
                video_path=dst_video_path,
                fps=fps,
                vcodec=vcodec,
                overwrite=True,
            )
        return dst_video_path

    composed_frames: list[Image.Image] = []
    for i in range(n_frames):
        composed_frames.append(_compose_frame(i))

    _encode_pil_to_video(composed_frames, dst_video_path, fps, vcodec)
    return dst_video_path


def _export_overlay_videos(
    dataset: LeRobotDataset,
    value_field: str,
    advantage_field: str,
    indicator_field: str,
    viz_episodes: str,
    video_key: str | None,
    video_keys: str | None,
    output_dir: Path,
    overwrite: bool,
    vcodec: str,
    frame_storage_mode: str = "memory",
    smooth_window: int = 1,
    values_override: np.ndarray | None = None,
    advantages_override: np.ndarray | None = None,
    indicators_override: np.ndarray | None = None,
    targets_override: np.ndarray | None = None,
    value_axis_range: tuple[float, float] | None = (-1.0, 0.0),
) -> list[Path]:
    """Export value-overlay videos.

    The value curve is plotted on a **fixed** y-axis given by ``value_axis_range``
    (default ``(-1.0, 0.0)``, the value-target scale) so the model's predictions are
    shown at their true magnitude rather than per-episode auto-stretched. Pass
    ``value_axis_range=None`` to restore the old per-episode min/max auto-scaling.

    The value/advantage/indicator series can come from two sources:
    * **In-memory arrays** – pass ``*_override`` (positionally aligned with the
      rows of ``dataset.hf_dataset``). Used by the inference pipeline so the viz
      reflects the freshly computed predictions even when several ACP fields
      share one column name (which would otherwise clobber each other on disk).
    * **Dataset columns** – when an override is ``None`` the corresponding
      ``*_field`` column is read from the dataset instead.
    """
    selected_video_keys = _select_video_keys(
        camera_keys=list(dataset.meta.camera_keys),
        requested_video_keys=video_keys,
        requested_video_key=video_key,
    )
    multiview_mode = len(selected_video_keys) > 1

    raw_dataset = dataset.hf_dataset.with_format(None)
    column_names = set(raw_dataset.column_names)

    if values_override is not None:
        values_all = np.asarray(values_override, dtype=np.float32).reshape(-1)
    else:
        if value_field not in column_names:
            raise KeyError(f"Missing value field '{value_field}' in dataset.")
        values_all = _to_1d_float(raw_dataset[value_field])

    if advantages_override is not None:
        advantages_all = np.asarray(advantages_override, dtype=np.float32).reshape(-1)
    elif advantage_field in column_names:
        advantages_all = _to_1d_float(raw_dataset[advantage_field])
    else:
        advantages_all = np.zeros_like(values_all, dtype=np.float32)

    if indicators_override is not None:
        indicators_all = np.asarray(indicators_override, dtype=np.int64).reshape(-1)
    elif indicator_field in column_names:
        indicators_all = _to_1d_int(raw_dataset[indicator_field])
    else:
        indicators_all = np.zeros(values_all.shape[0], dtype=np.int64)

    # Optional ground-truth value targets, drawn as a second curve for comparison.
    if targets_override is not None:
        targets_all = np.asarray(targets_override, dtype=np.float32).reshape(-1)
    else:
        targets_all = None

    episode_indices_all = np.asarray(raw_dataset["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices_all = np.asarray(raw_dataset["frame_index"], dtype=np.int64).reshape(-1)
    if "timestamp" in column_names:
        timestamps_all = np.asarray(raw_dataset["timestamp"], dtype=np.float64).reshape(-1)
    else:
        timestamps_all = frame_indices_all.astype(np.float64) / float(dataset.fps)

    if dataset.episodes is not None:
        available_episodes = sorted(dataset.episodes)
    else:
        available_episodes = list(range(dataset.meta.total_episodes))

    if viz_episodes.strip().lower() == "all":
        episodes = available_episodes
    else:
        requested = _parse_episodes_arg(viz_episodes, dataset.meta.total_episodes)
        episodes = [ep for ep in requested if ep in set(available_episodes)]

    output_dir.mkdir(parents=True, exist_ok=True)

    success_map = _build_episode_success_map(dataset)
    fps = int(dataset.fps)
    tolerance_s = float(getattr(dataset, "tolerance_s", 1e-4))
    video_backend = getattr(dataset, "video_backend", None)
    written_paths: list[Path] = []

    if multiview_mode:
        tasks = []
        for ep in episodes:
            ep_positions = np.flatnonzero(episode_indices_all == ep)
            if ep_positions.shape[0] == 0:
                continue

            ep_frame_indices = frame_indices_all[ep_positions]
            if bool(np.any(np.diff(ep_frame_indices) < 0)):
                ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

            ep_values = values_all[ep_positions]
            if ep_values.shape[0] == 0:
                continue

            ep_timestamps = timestamps_all[ep_positions]
            src_paths: list[Path] = []
            ts_per_cam: list[np.ndarray] = []
            for cam_key in selected_video_keys:
                src_path = Path(dataset.root) / dataset.meta.get_video_file_path(ep, cam_key)
                src_paths.append(src_path)
                from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, cam_key)
                cam_ts = from_ts + ep_timestamps
                if to_ts is not None:
                    cam_ts = np.minimum(cam_ts, to_ts)
                ts_per_cam.append(cam_ts)

            dst_path = _build_output_video_path_multiview(
                output_dir=output_dir,
                repo_id=dataset.repo_id,
                video_keys=selected_video_keys,
                episode_index=ep,
                success_tag=success_map.get(ep, "unknown"),
            )
            if dst_path.exists() and not overwrite:
                continue

            tasks.append(
                (
                    src_paths,
                    dst_path,
                    ep_values,
                    advantages_all[ep_positions],
                    indicators_all[ep_positions],
                    ts_per_cam,
                    targets_all[ep_positions] if targets_all is not None else None,
                )
            )

        desc_keys = ",".join(key.split(".")[-1] for key in selected_video_keys)
        for srcs, dst, vals, advs, inds, ts_per_cam, tgts in tqdm(
            tasks,
            total=len(tasks),
            desc=f"Export overlay multiview [{desc_keys}]",
            leave=False,
        ):
            written_paths.append(
                _export_single_episode_multiview(
                    src_video_paths=srcs,
                    camera_labels=selected_video_keys,
                    dst_video_path=dst,
                    ep_values=vals,
                    ep_advantages=advs,
                    ep_indicators=inds,
                    episode_timestamps_per_cam=ts_per_cam,
                    fps=fps,
                    vcodec=vcodec,
                    tolerance_s=tolerance_s,
                    video_backend=video_backend,
                    frame_storage_mode=frame_storage_mode,
                    temp_dir_root=output_dir,
                    smooth_window=smooth_window,
                    fixed_bounds=value_axis_range,
                    ep_targets=tgts,
                )
            )
    else:
        selected_video_key = selected_video_keys[0]
        tasks = []
        for ep in episodes:
            ep_positions = np.flatnonzero(episode_indices_all == ep)
            if ep_positions.shape[0] > 1:
                ep_frame_indices = frame_indices_all[ep_positions]
                if bool(np.any(np.diff(ep_frame_indices) < 0)):
                    ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

            ep_values = values_all[ep_positions]
            if ep_values.shape[0] == 0:
                continue

            ep_timestamps = timestamps_all[ep_positions]
            from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, selected_video_key)
            ep_video_timestamps = from_ts + ep_timestamps
            if to_ts is not None:
                ep_video_timestamps = np.minimum(ep_video_timestamps, to_ts)

            dst_path = _build_output_video_path(
                output_dir, dataset.repo_id, selected_video_key, ep,
                success_tag=success_map.get(ep, "unknown"),
            )
            if dst_path.exists() and not overwrite:
                continue

            src_path = Path(dataset.root) / dataset.meta.get_video_file_path(ep, selected_video_key)
            tasks.append(
                (
                    src_path,
                    dst_path,
                    ep_values,
                    advantages_all[ep_positions],
                    indicators_all[ep_positions],
                    ep_video_timestamps,
                    targets_all[ep_positions] if targets_all is not None else None,
                )
            )

        for src, dst, vals, advs, inds, ts, tgts in tqdm(
            tasks,
            total=len(tasks),
            desc=f"Export overlay [{selected_video_key}]",
            leave=False,
        ):
            written_paths.append(
                _export_single_episode(
                    src,
                    dst,
                    vals,
                    advs,
                    inds,
                    ts,
                    fps,
                    vcodec,
                    tolerance_s,
                    video_backend,
                    frame_storage_mode,
                    output_dir,
                    smooth_window=smooth_window,
                    fixed_bounds=value_axis_range,
                    ep_targets=tgts,
                )
            )

    return written_paths


# ── Advantage overlay export ─────────────────────────────────────────────────────

def _build_output_advantage_video_path(
    output_dir: Path, repo_id: str, video_key: str, episode_index: int, success_tag: str = "unknown"
) -> Path:
    repo_tag = repo_id.replace("/", "_")
    key_tag = video_key.replace(".", "_")
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{success_tag}_{key_tag}_adv.mp4"


def _build_output_advantage_video_path_multiview(
    output_dir: Path,
    repo_id: str,
    video_keys: list[str],
    episode_index: int,
    success_tag: str = "unknown",
) -> Path:
    repo_tag = repo_id.replace("/", "_")
    keys_tag = "__".join(key.replace(".", "_") for key in video_keys)
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{success_tag}_{keys_tag}_multiview_adv.mp4"


def _compute_soft_advantages_from_values(
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
) -> np.ndarray:
    """Soft advantage A_t = V_{t+n_step} - V_t within each episode.

    Positive when the value model expects the situation to improve over the next
    *n_step* frames; negative when it expects deterioration.  Terminal frames
    (no t+n_step in the same episode) get advantage 0.
    """
    n = values.shape[0]
    advantages = np.zeros(n, dtype=np.float32)
    for i in range(n):
        j = i + n_step
        if (
            j < n
            and episode_indices[j] == episode_indices[i]
            and frame_indices[j] == frame_indices[i] + n_step
        ):
            advantages[i] = float(values[j]) - float(values[i])
    return advantages


def export_advantage_overlay_videos(
    dataset: LeRobotDataset,
    raw_advantages_all: np.ndarray | None = None,
    value_field: str | None = None,
    n_step: int = 50,
    viz_episodes: str = "all",
    video_key: str | None = None,
    video_keys: str | None = None,
    output_dir: Path = Path("ad_viz"),
    overwrite: bool = False,
    vcodec: str = "libsvtav1",
    frame_storage_mode: str = "memory",
    smooth_window: int = 1,
) -> list[Path]:
    """Export per-episode soft-advantage overlay videos to *output_dir* (typically ``ad_viz/``).

    Two usage modes (exactly one must be supplied):

    * **Pass-through** – provide *raw_advantages_all* (1-D float array, dataset row order).
      The caller has already computed the n-step advantage externally (e.g. inside
      ``lerobot-value-infer``) and hands it in directly.

    * **Dataset field** – provide *value_field* (column name in the dataset parquet, e.g.
      ``"complementary_info.value"``).  The function reads the stored predicted values and
      computes the soft advantage as ``A_t = V_{t+n_step} - V_t`` within each episode.
      This mode lets you generate advantage videos from any dataset that already has a value
      column, without re-running inference.

    In both modes the curve is positive where the frame has a relative advantage and negative
    where it is at a disadvantage; a red zero-line separates the two regions.
    """
    if raw_advantages_all is None and value_field is None:
        raise ValueError("Provide either 'raw_advantages_all' or 'value_field'.")
    if raw_advantages_all is not None and value_field is not None:
        raise ValueError("Provide only one of 'raw_advantages_all' or 'value_field', not both.")

    selected_video_keys = _select_video_keys(
        camera_keys=list(dataset.meta.camera_keys),
        requested_video_keys=video_keys,
        requested_video_key=video_key,
    )
    multiview_mode = len(selected_video_keys) > 1

    raw_dataset = dataset.hf_dataset.with_format(None)
    episode_indices_all = np.asarray(raw_dataset["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices_all = np.asarray(raw_dataset["frame_index"], dtype=np.int64).reshape(-1)
    if "timestamp" in raw_dataset.column_names:
        timestamps_all = np.asarray(raw_dataset["timestamp"], dtype=np.float64).reshape(-1)
    else:
        timestamps_all = frame_indices_all.astype(np.float64) / float(dataset.fps)

    if raw_advantages_all is not None:
        adv_all = _to_1d_float(raw_advantages_all)
    else:
        if value_field not in raw_dataset.column_names:
            raise KeyError(
                f"Value field '{value_field}' not found in dataset. "
                f"Available columns: {raw_dataset.column_names}"
            )
        stored_values = _to_1d_float(raw_dataset[value_field])
        adv_all = _compute_soft_advantages_from_values(
            stored_values, episode_indices_all, frame_indices_all, n_step
        )

    dummy_zero = np.zeros_like(adv_all)

    success_map = _build_episode_success_map(dataset)

    if dataset.episodes is not None:
        available_episodes = sorted(dataset.episodes)
    else:
        available_episodes = list(range(dataset.meta.total_episodes))

    if viz_episodes.strip().lower() == "all":
        episodes = available_episodes
    else:
        requested = _parse_episodes_arg(viz_episodes, dataset.meta.total_episodes)
        episodes = [ep for ep in requested if ep in set(available_episodes)]

    output_dir.mkdir(parents=True, exist_ok=True)

    fps = int(dataset.fps)
    tolerance_s = float(getattr(dataset, "tolerance_s", 1e-4))
    video_backend = getattr(dataset, "video_backend", None)
    written_paths: list[Path] = []

    if multiview_mode:
        tasks = []
        for ep in episodes:
            ep_positions = np.flatnonzero(episode_indices_all == ep)
            if ep_positions.shape[0] == 0:
                continue
            ep_frame_indices = frame_indices_all[ep_positions]
            if bool(np.any(np.diff(ep_frame_indices) < 0)):
                ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

            ep_adv = adv_all[ep_positions]
            if ep_adv.shape[0] == 0:
                continue

            ep_timestamps = timestamps_all[ep_positions]
            src_paths: list[Path] = []
            ts_per_cam: list[np.ndarray] = []
            for cam_key in selected_video_keys:
                src_path = Path(dataset.root) / dataset.meta.get_video_file_path(ep, cam_key)
                src_paths.append(src_path)
                from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, cam_key)
                cam_ts = from_ts + ep_timestamps
                if to_ts is not None:
                    cam_ts = np.minimum(cam_ts, to_ts)
                ts_per_cam.append(cam_ts)

            dst_path = _build_output_advantage_video_path_multiview(
                output_dir=output_dir,
                repo_id=dataset.repo_id,
                video_keys=selected_video_keys,
                episode_index=ep,
                success_tag=success_map.get(ep, "unknown"),
            )
            if dst_path.exists() and not overwrite:
                continue

            tasks.append((src_paths, dst_path, ep_adv, dummy_zero[ep_positions], dummy_zero[ep_positions], ts_per_cam))

        desc_keys = ",".join(key.split(".")[-1] for key in selected_video_keys)
        for srcs, dst, adv, adv2, inds, ts_per_cam in tqdm(
            tasks, total=len(tasks), desc=f"Export adv overlay multiview [{desc_keys}]", leave=False
        ):
            written_paths.append(
                _export_single_episode_multiview(
                    src_video_paths=srcs,
                    camera_labels=selected_video_keys,
                    dst_video_path=dst,
                    ep_values=adv,
                    ep_advantages=adv2,
                    ep_indicators=inds,
                    episode_timestamps_per_cam=ts_per_cam,
                    fps=fps,
                    vcodec=vcodec,
                    tolerance_s=tolerance_s,
                    video_backend=video_backend,
                    frame_storage_mode=frame_storage_mode,
                    temp_dir_root=output_dir,
                    smooth_window=smooth_window,
                    show_zero_line=True,
                )
            )
    else:
        selected_video_key = selected_video_keys[0]
        tasks = []
        for ep in episodes:
            ep_positions = np.flatnonzero(episode_indices_all == ep)
            if ep_positions.shape[0] > 1:
                ep_frame_indices = frame_indices_all[ep_positions]
                if bool(np.any(np.diff(ep_frame_indices) < 0)):
                    ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

            ep_adv = adv_all[ep_positions]
            if ep_adv.shape[0] == 0:
                continue

            ep_timestamps = timestamps_all[ep_positions]
            from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, selected_video_key)
            ep_video_timestamps = from_ts + ep_timestamps
            if to_ts is not None:
                ep_video_timestamps = np.minimum(ep_video_timestamps, to_ts)

            dst_path = _build_output_advantage_video_path(
                output_dir, dataset.repo_id, selected_video_key, ep,
                success_tag=success_map.get(ep, "unknown"),
            )
            if dst_path.exists() and not overwrite:
                continue

            src_path = Path(dataset.root) / dataset.meta.get_video_file_path(ep, selected_video_key)
            tasks.append((src_path, dst_path, ep_adv, dummy_zero[ep_positions], dummy_zero[ep_positions], ep_video_timestamps))

        for src, dst, adv, adv2, inds, ts in tqdm(
            tasks, total=len(tasks), desc=f"Export adv overlay [{selected_video_key}]", leave=False
        ):
            written_paths.append(
                _export_single_episode(
                    src,
                    dst,
                    adv,
                    adv2,
                    inds,
                    ts,
                    fps,
                    vcodec,
                    tolerance_s,
                    video_backend,
                    frame_storage_mode,
                    output_dir,
                    smooth_window=smooth_window,
                    show_zero_line=True,
                )
            )

    return written_paths


# ── Binarized advantage/disadvantage indicator overlay export ───────────────────

def _build_output_indicator_video_path(
    output_dir: Path, repo_id: str, video_key: str, episode_index: int, success_tag: str = "unknown"
) -> Path:
    repo_tag = repo_id.replace("/", "_")
    key_tag = video_key.replace(".", "_")
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{success_tag}_{key_tag}_ind.mp4"


def _build_output_indicator_video_path_multiview(
    output_dir: Path,
    repo_id: str,
    video_keys: list[str],
    episode_index: int,
    success_tag: str = "unknown",
) -> Path:
    repo_tag = repo_id.replace("/", "_")
    keys_tag = "__".join(key.replace(".", "_") for key in video_keys)
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{success_tag}_{keys_tag}_multiview_ind.mp4"


def export_indicator_overlay_videos(
    dataset: LeRobotDataset,
    raw_indicators_all: np.ndarray | None = None,
    raw_advantages_all: np.ndarray | None = None,
    indicator_field: str | None = None,
    advantage_field: str | None = None,
    viz_episodes: str = "all",
    video_key: str | None = None,
    video_keys: str | None = None,
    output_dir: Path = Path("indicator_viz"),
    overwrite: bool = False,
    vcodec: str = "libsvtav1",
    frame_storage_mode: str = "memory",
    smooth_window: int = 1,
) -> list[Path]:
    """Export per-episode overlay videos showing the binarized ACP verdict per frame.

    For every frame the value pipeline emits an integer indicator (``1`` = advantage,
    ``0`` = disadvantage); this renders that decision directly onto the footage via a
    green/red verdict badge and a full-episode advantage/disadvantage timeline strip
    (see :func:`_draw_indicator_overlay`), so a reviewer can see at a glance which frames
    were classified advantageous.

    The indicator series can come from two sources (exactly one of ``raw_indicators_all``
    / ``indicator_field`` must be supplied):

    * **Pass-through** – provide ``raw_indicators_all`` (1-D int array in dataset row order),
      as done by ``lerobot-value-infer`` with the freshly computed indicators.
    * **Dataset field** – provide ``indicator_field`` (column name, e.g.
      ``"complementary_info.acp_indicator"``) to read the stored indicators from parquet.

    ``raw_advantages_all`` / ``advantage_field`` are optional and, when given, annotate the
    badge with the raw advantage value behind each decision.
    """
    if (raw_indicators_all is None) == (indicator_field is None):
        raise ValueError("Provide exactly one of 'raw_indicators_all' or 'indicator_field'.")

    selected_video_keys = _select_video_keys(
        camera_keys=list(dataset.meta.camera_keys),
        requested_video_keys=video_keys,
        requested_video_key=video_key,
    )
    multiview_mode = len(selected_video_keys) > 1

    raw_dataset = dataset.hf_dataset.with_format(None)
    episode_indices_all = np.asarray(raw_dataset["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices_all = np.asarray(raw_dataset["frame_index"], dtype=np.int64).reshape(-1)
    if "timestamp" in raw_dataset.column_names:
        timestamps_all = np.asarray(raw_dataset["timestamp"], dtype=np.float64).reshape(-1)
    else:
        timestamps_all = frame_indices_all.astype(np.float64) / float(dataset.fps)

    if raw_indicators_all is not None:
        ind_all = _to_1d_int(raw_indicators_all)
    else:
        if indicator_field not in raw_dataset.column_names:
            raise KeyError(
                f"Indicator field '{indicator_field}' not found in dataset. "
                f"Available columns: {raw_dataset.column_names}"
            )
        ind_all = _to_1d_int(raw_dataset[indicator_field])

    if raw_advantages_all is not None:
        adv_all = _to_1d_float(raw_advantages_all)
    elif advantage_field is not None and advantage_field in raw_dataset.column_names:
        adv_all = _to_1d_float(raw_dataset[advantage_field])
    else:
        adv_all = np.zeros_like(ind_all, dtype=np.float32)

    success_map = _build_episode_success_map(dataset)

    if dataset.episodes is not None:
        available_episodes = sorted(dataset.episodes)
    else:
        available_episodes = list(range(dataset.meta.total_episodes))

    if viz_episodes.strip().lower() == "all":
        episodes = available_episodes
    else:
        requested = _parse_episodes_arg(viz_episodes, dataset.meta.total_episodes)
        episodes = [ep for ep in requested if ep in set(available_episodes)]

    output_dir.mkdir(parents=True, exist_ok=True)

    fps = int(dataset.fps)
    tolerance_s = float(getattr(dataset, "tolerance_s", 1e-4))
    video_backend = getattr(dataset, "video_backend", None)
    written_paths: list[Path] = []

    if multiview_mode:
        tasks = []
        for ep in episodes:
            ep_positions = np.flatnonzero(episode_indices_all == ep)
            if ep_positions.shape[0] == 0:
                continue
            ep_frame_indices = frame_indices_all[ep_positions]
            if bool(np.any(np.diff(ep_frame_indices) < 0)):
                ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

            ep_ind = ind_all[ep_positions]
            if ep_ind.shape[0] == 0:
                continue

            ep_timestamps = timestamps_all[ep_positions]
            src_paths: list[Path] = []
            ts_per_cam: list[np.ndarray] = []
            for cam_key in selected_video_keys:
                src_path = Path(dataset.root) / dataset.meta.get_video_file_path(ep, cam_key)
                src_paths.append(src_path)
                from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, cam_key)
                cam_ts = from_ts + ep_timestamps
                if to_ts is not None:
                    cam_ts = np.minimum(cam_ts, to_ts)
                ts_per_cam.append(cam_ts)

            dst_path = _build_output_indicator_video_path_multiview(
                output_dir=output_dir,
                repo_id=dataset.repo_id,
                video_keys=selected_video_keys,
                episode_index=ep,
                success_tag=success_map.get(ep, "unknown"),
            )
            if dst_path.exists() and not overwrite:
                continue

            tasks.append((src_paths, dst_path, adv_all[ep_positions], ep_ind, ts_per_cam))

        desc_keys = ",".join(key.split(".")[-1] for key in selected_video_keys)
        for srcs, dst, adv, inds, ts_per_cam in tqdm(
            tasks, total=len(tasks), desc=f"Export indicator overlay multiview [{desc_keys}]", leave=False
        ):
            written_paths.append(
                _export_single_episode_multiview(
                    src_video_paths=srcs,
                    camera_labels=selected_video_keys,
                    dst_video_path=dst,
                    ep_values=adv,
                    ep_advantages=adv,
                    ep_indicators=inds,
                    episode_timestamps_per_cam=ts_per_cam,
                    fps=fps,
                    vcodec=vcodec,
                    tolerance_s=tolerance_s,
                    video_backend=video_backend,
                    frame_storage_mode=frame_storage_mode,
                    temp_dir_root=output_dir,
                    smooth_window=smooth_window,
                    indicator_mode=True,
                )
            )
    else:
        selected_video_key = selected_video_keys[0]
        tasks = []
        for ep in episodes:
            ep_positions = np.flatnonzero(episode_indices_all == ep)
            if ep_positions.shape[0] > 1:
                ep_frame_indices = frame_indices_all[ep_positions]
                if bool(np.any(np.diff(ep_frame_indices) < 0)):
                    ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

            ep_ind = ind_all[ep_positions]
            if ep_ind.shape[0] == 0:
                continue

            ep_timestamps = timestamps_all[ep_positions]
            from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, selected_video_key)
            ep_video_timestamps = from_ts + ep_timestamps
            if to_ts is not None:
                ep_video_timestamps = np.minimum(ep_video_timestamps, to_ts)

            dst_path = _build_output_indicator_video_path(
                output_dir, dataset.repo_id, selected_video_key, ep,
                success_tag=success_map.get(ep, "unknown"),
            )
            if dst_path.exists() and not overwrite:
                continue

            src_path = Path(dataset.root) / dataset.meta.get_video_file_path(ep, selected_video_key)
            tasks.append((src_path, dst_path, adv_all[ep_positions], ep_ind, ep_video_timestamps))

        for src, dst, adv, inds, ts in tqdm(
            tasks, total=len(tasks), desc=f"Export indicator overlay [{selected_video_key}]", leave=False
        ):
            written_paths.append(
                _export_single_episode(
                    src,
                    dst,
                    adv,
                    adv,
                    inds,
                    ts,
                    fps,
                    vcodec,
                    tolerance_s,
                    video_backend,
                    frame_storage_mode,
                    output_dir,
                    smooth_window=smooth_window,
                    indicator_mode=True,
                )
            )

    return written_paths
