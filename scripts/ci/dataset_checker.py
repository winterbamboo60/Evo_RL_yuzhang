#!/usr/bin/env python3
"""
LeRobot v3.0 dataset episode checker (web UI).

启动后在浏览器中:
  * 左侧列出每个 episode, 颜色表示标注状态 (绿=合格 / 红=不合格 / 灰=未标注);
  * 主区域播放该 episode 的预览视频 (所有相机画面横向拼接, 叠加元信息, 参考
    check_topWristVedio_index_label.py 的做法), 并展示元数据;
  * 点击 "合格 / 不合格" 按钮进行标注, 结果实时写入
        <dataset>/check/annotations.json
    重新打开时自动读取该文件并把已标注的状态回显到界面;
  * 可为每个 episode 设置保留区间的起点帧和长度, 导出时裁掉头尾帧;
  * "导出" 按钮根据标注结果, 从原数据集中剔除被标记为 "不合格" 的 episode,
    应用 episode 裁剪并另存为新的 LeRobot v3.0 数据集, 目录后缀默认 "_checked"。

用法:
    python dataset_checker.py /home/yuzhang/projects/VLA/datasets/package_scan_v6
    python dataset_checker.py <dataset_dir> --port 5000 --host 0.0.0.0

依赖: flask, pandas, pyarrow, numpy, 以及系统 ffmpeg/ffprobe。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

# 解析后的 ffmpeg 路径与 av1 软解码器 (在 init_dataset 中确定)
FFMPEG = "ffmpeg"
AV1_DECODER: str | None = None


def _resolve_ffmpeg(preferred: str | None = None):
    """挑选一个能用的 ffmpeg, 优先选具备 av1 软解码器 (libdav1d/libaom-av1) 的, 因为
    数据集视频多为 av1, 而部分 ffmpeg 构建只带需要硬件加速的原生 av1 解码器, 无法软解。"""
    cands = [preferred, "/usr/bin/ffmpeg", shutil.which("ffmpeg"), "ffmpeg"]
    working: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for c in cands:
        if not c or c in seen:
            continue
        seen.add(c)
        try:
            out = subprocess.run([c, "-hide_banner", "-decoders"],
                                 capture_output=True, text=True, timeout=20)
        except Exception:  # noqa: BLE001
            continue
        if out.returncode != 0:
            continue
        av1 = next((d for d in ("libdav1d", "libaom-av1") if d in out.stdout), None)
        working.append((c, av1))
    for c, av1 in working:
        if av1:
            return c, av1
    if working:
        return working[0]
    return "ffmpeg", None

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from flask import Flask, Response, abort, jsonify, request, send_file

# ─────────────────────────────────────────────────────────────────────────────
# 标注取值
QUALIFIED = "qualified"      # 合格
UNQUALIFIED = "unqualified"  # 不合格

FONT_CANDIDATES = [
    Path("/home/yuzhang/projects/VLA/datasets/fonts/ARIAL.TTF"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]

app = Flask(__name__)

# 全局数据集状态 (单数据集服务)
DS: dict = {}
_clip_lock = threading.Lock()
_clip_locks: dict[int, threading.Lock] = {}
_state_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# 数据集加载
# ─────────────────────────────────────────────────────────────────────────────
def _col(row: pd.Series, prefix: str, field: str):
    """读取 (可能是嵌套的) parquet 列的标量值, 兼容扁平/嵌套两种存储。"""
    key = f"{prefix}/{field}"
    val = row[key] if key in row.index else row[prefix][field]
    if hasattr(val, "__len__") and not isinstance(val, str):
        val = val[0]
    return val


def _scalar(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return v[0] if len(v) else None
    return v


def _camera_sort_key(key: str) -> tuple[int, str]:
    """Keep bimanual views in the operator-friendly left / top / right order."""
    short = key.rsplit(".", 1)[-1].lower()
    priority = {
        "left_wrist": 0,
        "left_top": 1,
        "top": 1,
        "right_top": 1,
        "right_wrist": 2,
    }
    return priority.get(short, 100), key


def _joint_group(name: str) -> str:
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    return "other"


def init_dataset(dataset_dir: Path) -> None:
    dataset_dir = dataset_dir.resolve()
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"[ERROR] 不是合法的 LeRobot 数据集 (缺少 {info_path})")

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    # 收集并按操作者视角排列所有视频 key。双臂三相机顺序为：左腕、顶视、右腕。
    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    video_keys.sort(key=_camera_sort_key)

    state_spec = features.get("observation.state")
    action_spec = features.get("action")
    if not state_spec or not action_spec:
        raise SystemExit("[ERROR] 数据集缺少 observation.state 或 action feature")
    state_shape = state_spec.get("shape")
    action_shape = action_spec.get("shape")
    joint_names = state_spec.get("names")
    if (
        not state_shape
        or len(state_shape) != 1
        or state_shape != action_shape
        or not joint_names
        or joint_names != action_spec.get("names")
        or len(joint_names) != int(state_shape[0])
    ):
        raise SystemExit("[ERROR] observation.state 与 action 必须是同名的一维关节向量")

    robot_type = str(info.get("robot_type", "unknown"))
    joint_groups = [_joint_group(name) for name in joint_names]
    if robot_type.startswith("bi_piper") and not {"left", "right"}.issubset(joint_groups):
        raise SystemExit("[ERROR] bi_piper 数据集的关节名称必须同时包含 left_ 与 right_ 前缀")

    # 读取 episodes 元数据 (一行一个 episode)
    ep_files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    episodes_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)

    DS.clear()
    DS.update(
        dir=dataset_dir,
        name=dataset_dir.name,
        info=info,
        robot_type=robot_type,
        fps=float(info["fps"]),
        video_keys=video_keys,
        joint_names=joint_names,
        joint_groups=joint_groups,
        joint_dim=int(state_shape[0]),
        episodes_df=episodes_df,
        check_dir=dataset_dir / "check",
        clip_dir=dataset_dir / "check" / "clips",
        ann_path=dataset_dir / "check" / "annotations.json",
    )
    DS["check_dir"].mkdir(parents=True, exist_ok=True)
    DS["clip_dir"].mkdir(parents=True, exist_ok=True)

    print(
        f"[OK] 数据集: {DS['name']}  robot={robot_type}  episodes={len(episodes_df)}  "
        f"fps={DS['fps']}  joints={DS['joint_dim']}  cameras={len(video_keys)}"
    )
    print(f"[OK] video_keys={video_keys}")
    print(f"[OK] ffmpeg={FFMPEG}  av1_decoder={AV1_DECODER or '(无, av1 视频可能无法预览)'}")


def _episode_row(ep_idx: int) -> pd.Series:
    df = DS["episodes_df"]
    sub = df[df["episode_index"] == ep_idx]
    if sub.empty:
        abort(404, f"episode {ep_idx} 不存在")
    return sub.iloc[0]


def episode_meta(ep_idx: int) -> dict:
    ep = _episode_row(ep_idx)
    tasks = ep.get("tasks")
    if isinstance(tasks, (list, tuple, np.ndarray)):
        task = ", ".join(str(t) for t in tasks)
    else:
        task = str(tasks)
    return {
        "episode_index": int(ep["episode_index"]),
        "length": int(ep["length"]),
        "success": str(_scalar(ep.get("episode_success", "unknown"))),
        "task": task,
    }


def _episode_joint_data(ep_idx: int) -> dict:
    """读取单个 episode 的 observation/action 逐帧关节数据。"""
    ep = _episode_row(ep_idx)
    features = DS["info"]["features"]
    state_spec = features.get("observation.state")
    action_spec = features.get("action")
    shape = state_spec.get("shape") if state_spec else None
    names = state_spec.get("names") if state_spec else None
    if (not action_spec or not shape or len(shape) != 1
            or shape != action_spec.get("shape") or not names
            or names != action_spec.get("names") or len(names) != int(shape[0])):
        abort(500, "observation.state 与 action 必须是同名的一维关节向量")

    chunk = int(_col(ep, "data", "chunk_index"))
    file_idx = int(_col(ep, "data", "file_index"))
    pattern = DS["info"].get(
        "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    path = DS["dir"] / pattern.format(chunk_index=chunk, file_index=file_idx)
    if not path.is_file():
        abort(500, f"缺少逐帧数据文件: {path}")

    columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    try:
        table = pq.read_table(
            path, columns=columns, filters=[("episode_index", "=", ep_idx)]
        ).sort_by([("frame_index", "ascending")])
    except Exception as e:  # noqa: BLE001
        abort(500, f"读取关节数据失败: {e}")

    if table.num_rows != int(ep["length"]):
        abort(500, f"episode {ep_idx} 帧数不一致: meta={ep['length']} data={table.num_rows}")

    data = table.to_pydict()
    frame_index = np.asarray(data["frame_index"], dtype=np.int64)
    timestamps = np.asarray(data["timestamp"], dtype=float)
    observations = np.asarray(data["observation.state"], dtype=float)
    actions = np.asarray(data["action"], dtype=float)
    dims = (table.num_rows, int(shape[0]))
    if (observations.shape != dims or actions.shape != dims
            or not np.array_equal(frame_index, np.arange(table.num_rows))
            or not np.isfinite(timestamps).all()
            or not np.isfinite(observations).all() or not np.isfinite(actions).all()):
        abort(500, f"episode {ep_idx} 的关节数据维度、帧序或数值无效")

    low = np.minimum(observations.min(axis=0), actions.min(axis=0))
    high = np.maximum(observations.max(axis=0), actions.max(axis=0))
    flat = low == high
    pad = np.maximum(np.abs(low) * 0.01, 1e-3)
    low[flat] -= pad[flat]
    high[flat] += pad[flat]
    return {
        "fps": DS["fps"],
        "names": names,
        "groups": [_joint_group(name) for name in names],
        "frame_index": frame_index.tolist(),
        "timestamp": timestamps.tolist(),
        "observation": observations.tolist(),
        "action": actions.tolist(),
        "ranges": np.column_stack((low, high)).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 检查状态读写 (标注 + episode 裁剪)
# ─────────────────────────────────────────────────────────────────────────────
def _read_check_state() -> dict:
    p: Path = DS["ann_path"]
    if not p.is_file():
        return {"labels": {}, "trims": {}}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        labels = data.get("labels", {})
        trims = data.get("trims", {})
        if not isinstance(labels, dict) or not isinstance(trims, dict):
            raise ValueError("labels 和 trims 必须是对象")
        return {
            "labels": {str(k): v for k, v in labels.items()},
            "trims": {str(k): v for k, v in trims.items()},
        }
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 读取标注失败 {p}: {e}")
        return {"labels": {}, "trims": {}}


def load_check_state() -> dict:
    with _state_lock:
        return _read_check_state()


def _write_check_state(state: dict) -> None:
    p: Path = DS["ann_path"]
    payload = {
        "dataset": DS["name"],
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "labels": {
            str(k): v for k, v in state.get("labels", {}).items()
            if v in (QUALIFIED, UNQUALIFIED)
        },
        "trims": {str(k): v for k, v in state.get("trims", {}).items()},
    }
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(p)


def load_annotations() -> dict:
    return load_check_state()["labels"]


def save_annotations(labels: dict) -> None:
    with _state_lock:
        state = _read_check_state()
        state["labels"] = labels
        _write_check_state(state)


def _strict_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} 必须是整数") from e
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} 必须是整数")
    if isinstance(value, str) and str(result) != value.strip():
        raise ValueError(f"{name} 必须是整数")
    return result


def episode_crop(ep_idx: int, trims: dict | None = None) -> dict:
    original_length = int(_episode_row(ep_idx)["length"])
    raw = (trims or {}).get(str(ep_idx))
    if raw is None:
        start_frame, length = 0, original_length
    else:
        if not isinstance(raw, dict):
            raise ValueError(f"episode {ep_idx} 的裁剪配置无效")
        start_frame = _strict_int(raw.get("start_frame"), "start_frame")
        length = _strict_int(raw.get("length"), "length")
    if start_frame < 0:
        raise ValueError("start_frame 不能小于 0")
    if length < 1:
        raise ValueError("length 必须至少为 1")
    if start_frame + length > original_length:
        raise ValueError(
            f"episode {ep_idx} 裁剪越界: start_frame={start_frame}, "
            f"length={length}, original_length={original_length}"
        )
    end_frame = start_frame + length
    return {
        "start_frame": start_frame,
        "length": length,
        "end_frame": end_frame,
        "head_trim": start_frame,
        "tail_trim": original_length - end_frame,
        "trimmed": start_frame != 0 or length != original_length,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 预览视频拼接 (ffmpeg)
# ─────────────────────────────────────────────────────────────────────────────
def _esc_text(text: str) -> str:
    text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
    return text


def _esc_path(path: str) -> str:
    return path.replace("\\", "/").replace(":", "\\:")


def _font_opt() -> str:
    for fp in FONT_CANDIDATES:
        if fp.exists():
            return f":fontfile='{_esc_path(str(fp))}'"
    return ""


def _video_segment(ep: pd.Series, key: str):
    chunk = int(_col(ep, f"videos/{key}", "chunk_index"))
    file_idx = int(_col(ep, f"videos/{key}", "file_index"))
    frm = float(_col(ep, f"videos/{key}", "from_timestamp"))
    to = float(_col(ep, f"videos/{key}", "to_timestamp"))
    path = (DS["dir"] / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4")
    return path, frm, to


def ensure_clip(ep_idx: int) -> Path:
    """生成 (并缓存) 某个 episode 的横向拼接预览 mp4 (libx264, 浏览器可播放)。"""
    out = DS["clip_dir"] / f"ep{ep_idx:05d}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out

    with _clip_lock:
        lock = _clip_locks.setdefault(ep_idx, threading.Lock())
    with lock:
        if out.exists() and out.stat().st_size > 0:
            return out

        ep = _episode_row(ep_idx)
        meta = episode_meta(ep_idx)
        keys = DS["video_keys"]
        font = _font_opt()

        info = DS["info"]
        inputs: list[str] = []
        for key in keys:
            path, frm, to = _video_segment(ep, key)
            if not path.exists():
                abort(500, f"缺少视频文件: {path}")
            dur = max(to - frm, 1.0 / DS["fps"])
            codec = info["features"][key].get("info", {}).get("video.codec")
            dec = ["-c:v", AV1_DECODER] if (codec == "av1" and AV1_DECODER) else []
            inputs += [*dec, "-ss", f"{frm:.6f}", "-t", f"{dur:.6f}", "-i", str(path)]

        n = len(keys)
        chains = []
        if n > 1:
            stack_in = "".join(f"[{i}:v]" for i in range(n))
            chains.append(f"{stack_in}hstack=inputs={n}[s]")
            last = "[s]"
        else:
            last = "[0:v]"

        label = (
            f"ep={meta['episode_index']}  len={meta['length']}f  "
            f"robot={DS['robot_type']}  cams={len(keys)}  "
            f"success={meta['success']}  task={meta['task']}"
        )
        chains.append(
            f"{last}drawtext=text='{_esc_text(label)}'{font}"
            f":fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6:x=8:y=8[t0]"
        )
        last = "[t0]"
        for i, key in enumerate(keys):
            width = int(info["features"][key]["shape"][1])
            short = key.split(".")[-1].upper()
            chains.append(
                f"{last}drawtext=text='{_esc_text(short)}'{font}"
                f":fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.5"
                f":x={i * width}+8:y=H-30[t{i + 1}]"
            )
            last = f"[t{i + 1}]"

        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            *inputs,
            "-filter_complex", ";".join(chains),
            "-map", last,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast",
            "-movflags", "+faststart",
            "-r", str(int(DS["fps"])),
            str(out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if res.returncode != 0 or not out.exists():
            if out.exists():
                out.unlink(missing_ok=True)
            abort(500, f"ffmpeg 生成预览失败 (ep{ep_idx}):\n{res.stderr[-800:]}")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 导出 (剔除不合格 episode, 裁剪帧范围, 生成新的合法 v3.0 数据集)
# ─────────────────────────────────────────────────────────────────────────────
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def _episode_stats(table: pa.Table, features: dict, old_row: dict) -> dict:
    from lerobot.datasets.compute_stats import compute_episode_stats

    values = {}
    for key, feature in features.items():
        if key in table.column_names and feature["dtype"] not in ("string", "image", "video"):
            values[key] = np.asarray(table[key].to_pylist())
    stats = compute_episode_stats(values, features)
    for key, feature in features.items():
        if feature["dtype"] in ("image", "video"):
            stats[key] = {
                stat: np.asarray(old_row[f"stats/{key}/{stat}"]) for stat in STAT_NAMES
            }
    return stats


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_tables(root: Path, relative_dir: str) -> pa.Table:
    files = sorted((root / relative_dir).rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root / relative_dir} 下没有 parquet 文件")
    return pa.concat_tables([pq.read_table(path) for path in files])


def _verify_export(root: Path) -> None:
    with open(root / "meta" / "info.json", encoding="utf-8") as f:
        info = json.load(f)
    data = _read_tables(root, "data")
    episodes = _read_tables(root, "meta/episodes")
    lengths = np.asarray(episodes["length"], dtype=np.int64)
    meta_episode_index = np.asarray(episodes["episode_index"], dtype=np.int64)
    data_episode_index = np.asarray(data["episode_index"], dtype=np.int64)

    if not np.array_equal(meta_episode_index, np.arange(len(lengths))):
        raise RuntimeError("导出后的 meta episode_index 不连续")
    if data.num_rows != int(lengths.sum()):
        raise RuntimeError("导出后的 episode 长度与 data 行数不一致")
    if np.any(data_episode_index < 0) or np.any(data_episode_index >= len(lengths)):
        raise RuntimeError("导出后的 data episode_index 越界")
    counts = np.bincount(data_episode_index, minlength=len(lengths))
    if not np.array_equal(counts, lengths):
        raise RuntimeError("导出后的各 episode 行数与 length 不一致")
    if not np.array_equal(np.asarray(data["index"]), np.arange(data.num_rows)):
        raise RuntimeError("导出后的全局 index 不连续")

    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    ends = np.cumsum(lengths)
    if not np.array_equal(np.asarray(episodes["dataset_from_index"]), starts):
        raise RuntimeError("导出后的 dataset_from_index 不正确")
    if not np.array_equal(np.asarray(episodes["dataset_to_index"]), ends):
        raise RuntimeError("导出后的 dataset_to_index 不正确")

    frame_index = np.asarray(data["frame_index"], dtype=np.int64)
    timestamps = np.asarray(data["timestamp"], dtype=float)
    fps = float(info["fps"])
    for ep_idx, length in enumerate(lengths):
        mask = data_episode_index == ep_idx
        expected_frames = np.arange(length)
        if not np.array_equal(frame_index[mask], expected_frames):
            raise RuntimeError(f"导出后的 episode {ep_idx} frame_index 不连续")
        if not np.allclose(timestamps[mask], expected_frames / fps, rtol=0, atol=1e-6):
            raise RuntimeError(f"导出后的 episode {ep_idx} timestamp 不正确")

    if int(info["total_episodes"]) != len(lengths):
        raise RuntimeError("导出后的 info.json total_episodes 不正确")
    if int(info["total_frames"]) != data.num_rows:
        raise RuntimeError("导出后的 info.json total_frames 不正确")
    for key, feature in info["features"].items():
        if feature.get("dtype") != "video":
            continue
        start = np.asarray(episodes[f"videos/{key}/from_timestamp"], dtype=float)
        end = np.asarray(episodes[f"videos/{key}/to_timestamp"], dtype=float)
        if np.any(end - start + 1e-6 < np.maximum(0, lengths - 1) / fps):
            raise RuntimeError(f"导出后的视频时间范围过短: {key}")


def _set_col(tbl: pa.Table, name: str, values) -> pa.Table:
    field = tbl.schema.field(name)
    pos = tbl.schema.get_field_index(name)
    values = [value.tolist() if isinstance(value, np.ndarray) else value for value in values]
    return tbl.set_column(pos, field, pa.array(values, type=field.type))


def do_export(suffix: str, only_qualified: bool) -> dict:
    try:
        from lerobot.datasets.compute_stats import aggregate_stats
    except ImportError as e:
        raise RuntimeError("导出裁剪数据集需要可导入 lerobot") from e

    src: Path = DS["dir"]
    state = load_check_state()
    labels = state["labels"]
    trims = state["trims"]

    all_eps = [int(x) for x in DS["episodes_df"]["episode_index"].tolist()]
    if only_qualified:
        kept = [e for e in all_eps if labels.get(str(e)) == QUALIFIED]
    else:
        kept = [e for e in all_eps if labels.get(str(e)) != UNQUALIFIED]
    dropped = [e for e in all_eps if e not in set(kept)]

    if not kept:
        return {"ok": False, "error": "没有可导出的 episode (全部被剔除)。"}

    suffix = suffix or "_checked"
    dst = src.parent / f"{src.name}{suffix}"
    if dst.exists():
        return {"ok": False, "error": f"目标目录已存在, 请换一个后缀: {dst}"}

    ep_map = {old: new for new, old in enumerate(kept)}  # 旧 -> 新 (连续)
    crops = {ep_idx: episode_crop(ep_idx, trims) for ep_idx in kept}
    trimmed = [ep_idx for ep_idx in kept if crops[ep_idx]["trimmed"]]
    trimmed_frames = sum(crops[ep_idx]["head_trim"] + crops[ep_idx]["tail_trim"] for ep_idx in kept)

    print(
        f"[EXPORT] {src.name} -> {dst.name}  保留 {len(kept)} / 丢弃 {len(dropped)}  "
        f"裁剪 {len(trimmed)} episodes / {trimmed_frames} frames"
    )

    tmp = Path(tempfile.mkdtemp(prefix=f".{dst.name}.tmp-", dir=dst.parent))
    try:
        # 1) 整库视频原样复制。导出数据只通过 episode 时间指针引用保留区间。
        (tmp / "meta").mkdir(parents=True, exist_ok=True)
        if (src / "videos").exists():
            shutil.copytree(src / "videos", tmp / "videos")

        # 2) 按 episode 裁剪逐帧数据，并重排所有局部/全局索引。
        data = _read_tables(src, "data").sort_by(
            [("episode_index", "ascending"), ("frame_index", "ascending")]
        )
        data_episode_index = np.asarray(data["episode_index"], dtype=np.int64)
        unique_eps, group_starts, group_lengths = np.unique(
            data_episode_index, return_index=True, return_counts=True
        )
        data_groups = {
            int(ep_idx): (int(start), int(length))
            for ep_idx, start, length in zip(unique_eps, group_starts, group_lengths, strict=True)
        }

        meta = _read_tables(src, "meta/episodes")
        meta_episode_index = np.asarray(meta["episode_index"], dtype=np.int64)
        if len(set(meta_episode_index.tolist())) != len(meta_episode_index):
            raise ValueError("episode metadata 中存在重复 episode_index")
        meta_positions = {int(ep_idx): i for i, ep_idx in enumerate(meta_episode_index)}
        missing_meta = [ep_idx for ep_idx in kept if ep_idx not in meta_positions]
        if missing_meta:
            raise ValueError(f"episode metadata 缺少: {missing_meta[:10]}")
        meta = meta.take(pa.array([meta_positions[ep_idx] for ep_idx in kept], type=pa.int64()))
        old_meta_rows = meta.to_pylist()

        parts = []
        all_stats = []
        lengths = []
        from_idx = []
        to_idx = []
        global_index = 0
        for old_ep_idx, old_row in zip(kept, old_meta_rows, strict=True):
            if old_ep_idx not in data_groups:
                raise ValueError(f"data 中缺少 episode {old_ep_idx}")
            group_start, group_length = data_groups[old_ep_idx]
            original_length = int(old_row["length"])
            if group_length != original_length:
                raise ValueError(
                    f"episode {old_ep_idx} 帧数不一致: meta={original_length}, data={group_length}"
                )
            episode_data = data.slice(group_start, group_length)
            if not np.array_equal(
                np.asarray(episode_data["frame_index"], dtype=np.int64),
                np.arange(original_length),
            ):
                raise ValueError(f"episode {old_ep_idx} 的 frame_index 不是从 0 开始连续递增")

            crop = crops[old_ep_idx]
            part = episode_data.slice(crop["start_frame"], crop["length"])
            new_ep_idx = ep_map[old_ep_idx]
            length = part.num_rows
            part = _set_col(part, "episode_index", [new_ep_idx] * length)
            part = _set_col(part, "frame_index", np.arange(length))
            part = _set_col(part, "timestamp", np.arange(length) / DS["fps"])
            part = _set_col(part, "index", np.arange(global_index, global_index + length))
            parts.append(part)
            lengths.append(length)
            from_idx.append(global_index)
            global_index += length
            to_idx.append(global_index)
            all_stats.append(_episode_stats(part, DS["info"]["features"], old_row))

        new_data = pa.concat_tables(parts)
        (tmp / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(new_data, tmp / "data" / "chunk-000" / "file-000.parquet")
        total_frames = new_data.num_rows

        # 3) 更新 episode 长度、数据指针、视频时间窗和裁剪后的统计。
        new_meta = _set_col(meta, "episode_index", list(range(len(kept))))
        new_meta = _set_col(new_meta, "length", lengths)
        new_meta = _set_col(new_meta, "dataset_from_index", from_idx)
        new_meta = _set_col(new_meta, "dataset_to_index", to_idx)
        for name in (
            "data/chunk_index", "data/file_index",
            "meta/episodes/chunk_index", "meta/episodes/file_index",
        ):
            if name in new_meta.schema.names:
                new_meta = _set_col(new_meta, name, [0] * len(kept))

        for key, feature in DS["info"]["features"].items():
            if feature.get("dtype") != "video":
                continue
            prefix = f"videos/{key}"
            old_start = np.asarray(meta[f"{prefix}/from_timestamp"], dtype=float)
            old_end = np.asarray(meta[f"{prefix}/to_timestamp"], dtype=float)
            new_start = [
                float(value) + crops[ep_idx]["head_trim"] / DS["fps"]
                for value, ep_idx in zip(old_start, kept, strict=True)
            ]
            new_end = [
                float(value) - crops[ep_idx]["tail_trim"] / DS["fps"]
                for value, ep_idx in zip(old_end, kept, strict=True)
            ]
            new_meta = _set_col(new_meta, f"{prefix}/from_timestamp", new_start)
            new_meta = _set_col(new_meta, f"{prefix}/to_timestamp", new_end)

        for column in new_meta.column_names:
            if not column.startswith("stats/"):
                continue
            _, feature, stat = column.split("/", 2)
            if stat in STAT_NAMES and all(feature in item for item in all_stats):
                new_meta = _set_col(new_meta, column, [item[feature][stat] for item in all_stats])

        (tmp / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(new_meta, tmp / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

        # 4) tasks 不随 episode 帧裁剪变化。
        if (src / "meta" / "tasks.parquet").exists():
            shutil.copy2(src / "meta" / "tasks.parquet", tmp / "meta" / "tasks.parquet")

        # 5) 用裁剪后的逐帧数据重算整库统计。
        with open(tmp / "meta" / "stats.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(aggregate_stats(all_stats)), f, ensure_ascii=False, indent=2)

        # 6) 更新数据集总量，验证完整性后原子发布目标目录。
        info = dict(DS["info"])
        info["total_episodes"] = len(kept)
        info["total_frames"] = total_frames
        info["splits"] = {"train": f"0:{len(kept)}"}
        with open(tmp / "meta" / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
        _verify_export(tmp)
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    print(f"[EXPORT] 完成 -> {dst}")
    return {
        "ok": True,
        "dst": str(dst),
        "kept": len(kept),
        "dropped": len(dropped),
        "trimmed": len(trimmed),
        "trimmed_frames": trimmed_frames,
        "total_frames": total_frames,
        "note": "视频文件保持原样，仅更新 episode 的视频时间引用；逐帧数据已按裁剪范围重建。",
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTTP 路由
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return Response(PAGE_HTML, mimetype="text/html")


@app.route("/api/info")
def api_info():
    return jsonify({
        "name": DS["name"],
        "dir": str(DS["dir"]),
        "robot_type": DS["robot_type"],
        "fps": DS["fps"],
        "video_keys": DS["video_keys"],
        "camera_count": len(DS["video_keys"]),
        "joint_dim": DS["joint_dim"],
        "joint_names": DS["joint_names"],
        "total_episodes": len(DS["episodes_df"]),
    })


@app.route("/api/episodes")
def api_episodes():
    state = load_check_state()
    labels = state["labels"]
    trims = state["trims"]
    items = []
    for ep_idx in DS["episodes_df"]["episode_index"].tolist():
        ep_idx = int(ep_idx)
        m = episode_meta(ep_idx)
        m["label"] = labels.get(str(ep_idx))  # qualified / unqualified / None
        m["crop"] = episode_crop(ep_idx, trims)
        items.append(m)
    counts = {
        "total": len(items),
        "qualified": sum(1 for i in items if i["label"] == QUALIFIED),
        "unqualified": sum(1 for i in items if i["label"] == UNQUALIFIED),
        "unlabeled": sum(1 for i in items if not i["label"]),
    }
    return jsonify({"episodes": items, "counts": counts})


@app.route("/api/clip/<int:ep_idx>")
def api_clip(ep_idx: int):
    path = ensure_clip(ep_idx)
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.route("/api/joints/<int:ep_idx>")
def api_joints(ep_idx: int):
    return jsonify(_episode_joint_data(ep_idx))


@app.route("/api/label", methods=["POST"])
def api_label():
    data = request.get_json(force=True)
    ep_idx = int(data["episode_index"])
    _episode_row(ep_idx)
    key = str(ep_idx)
    label = data.get("label")  # qualified / unqualified / null(清除)
    with _state_lock:
        state = _read_check_state()
        labels = state["labels"]
        if label in (QUALIFIED, UNQUALIFIED):
            labels[key] = label
        else:
            labels.pop(key, None)
        _write_check_state(state)
    return jsonify({"ok": True, "episode_index": ep_idx, "label": labels.get(key)})


@app.route("/api/trim", methods=["POST"])
def api_trim():
    data = request.get_json(force=True) or {}
    try:
        ep_idx = _strict_int(data.get("episode_index"), "episode_index")
        start_frame = _strict_int(data.get("start_frame"), "start_frame")
        length = _strict_int(data.get("length"), "length")
        crop = episode_crop(
            ep_idx,
            {str(ep_idx): {"start_frame": start_frame, "length": length}},
        )
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    with _state_lock:
        state = _read_check_state()
        if crop["trimmed"]:
            state["trims"][str(ep_idx)] = {
                "start_frame": crop["start_frame"],
                "length": crop["length"],
            }
        else:
            state["trims"].pop(str(ep_idx), None)
        _write_check_state(state)
    return jsonify({"ok": True, "episode_index": ep_idx, "crop": crop})


@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.get_json(force=True) or {}
    suffix = (data.get("suffix") or "_checked").strip()
    only_qualified = bool(data.get("only_qualified", False))
    try:
        result = do_export(suffix, only_qualified)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500
    code = 200 if result.get("ok") else 400
    return jsonify(result), code


# ─────────────────────────────────────────────────────────────────────────────
# 前端页面 (单文件内联)
# ─────────────────────────────────────────────────────────────────────────────
PAGE_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Checker</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif;
         background:#1d2025; color:#e6e6e6; height:100vh; display:flex; flex-direction:column; }
  header { padding:10px 16px; background:#15171b; border-bottom:1px solid #333;
           display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  .counts span { margin-right:12px; font-size:13px; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; vertical-align:middle;}
  .c-qual{background:#3fb950;} .c-unqual{background:#f85149;} .c-none{background:#6e7681;}
  .spacer{ flex:1; }
  button { cursor:pointer; border:none; border-radius:6px; padding:7px 14px; font-size:13px; color:#fff; }
  .btn-export{ background:#2f81f7; }
  .btn-export:hover{ background:#4a93ff; }
  main { flex:1; display:flex; min-height:0; }
  /* sidebar */
  #sidebar { width:280px; background:#15171b; border-right:1px solid #333; overflow-y:auto; }
  .filter { display:flex; gap:6px; padding:8px; position:sticky; top:0; background:#15171b; border-bottom:1px solid #2a2d33;}
  .filter button{ flex:1; background:#262a31; font-size:12px; padding:5px 0;}
  .filter button.active{ background:#3a4150; }
  .ep { padding:8px 12px; border-bottom:1px solid #21242a; cursor:pointer; font-size:13px;
        display:flex; align-items:center; gap:8px; border-left:4px solid transparent;}
  .ep:hover{ background:#21252b; }
  .ep.sel{ background:#2b3038; border-left-color:#2f81f7; }
  .ep .badge{ width:9px; height:9px; border-radius:50%; flex:none;}
  .ep .meta{ color:#8b949e; font-size:11px; margin-left:auto; }
  .ep.qualified .badge{ background:#3fb950;}
  .ep.unqualified .badge{ background:#f85149;}
  .ep.unlabeled .badge{ background:#6e7681;}
  /* main panel */
  #content { flex:1; min-width:0; padding:16px; overflow-y:auto; }
  #videoWrap{ background:#000; border-radius:8px; display:flex; justify-content:center; align-items:center;
              min-height:280px; overflow:hidden; }
  video{ display:block; max-width:100%; max-height:62vh; border-radius:8px; }
  #jointPanel{ margin-top:12px; padding:10px 12px; background:#15171b; border:1px solid #333;
               border-radius:8px; max-height:32vh; overflow:auto; flex:none; }
  .joint-head{ display:flex; align-items:center; gap:14px; margin-bottom:6px; font-size:13px; }
  .joint-head .legend{ color:#8b949e; } #jointFrame{ margin-left:auto; color:#8b949e; }
  .joint-row{ min-width:720px; display:grid; grid-template-columns:150px minmax(180px,1fr) minmax(180px,1fr) 82px;
              align-items:center; gap:10px; padding:6px 0; border-top:1px solid #262a31; }
  .joint-group{ position:sticky; top:0; z-index:1; margin:8px -2px 0; padding:5px 8px;
                background:#20252c; color:#f0f3f6; font-size:12px; font-weight:700;
                border-left:3px solid #6e7681; }
  .joint-group.left{ border-left-color:#58a6ff; }
  .joint-group.right{ border-left-color:#f0883e; }
  .joint-name{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12px; }
  .joint-reading{ display:grid; grid-template-columns:32px minmax(80px,1fr) 68px; align-items:center; gap:6px; }
  .joint-tag,.joint-value,.joint-delta{ font-size:12px; font-variant-numeric:tabular-nums; }
  .joint-tag.obs{ color:#58a6ff; } .joint-tag.act{ color:#f0883e; }
  .joint-value,.joint-delta{ text-align:right; color:#c9d1d9; }
  meter{ width:100%; height:14px; }
  meter::-webkit-meter-bar{ background:#30363d; border:0; border-radius:6px; }
  .obs-meter::-webkit-meter-optimum-value{ background:#58a6ff; border-radius:6px; }
  .act-meter::-webkit-meter-optimum-value{ background:#f0883e; border-radius:6px; }
  .obs-meter::-moz-meter-bar{ background:#58a6ff; } .act-meter::-moz-meter-bar{ background:#f0883e; }
  .joint-empty{ color:#8b949e; font-size:12px; padding:8px 0; }
  #info{ margin:12px 0; font-size:13px; color:#c9d1d9; line-height:1.7; }
  #info b{ color:#fff; }
  #trimPanel{ padding:10px 12px; background:#15171b; border:1px solid #333; border-radius:8px; }
  .trim-head{ display:flex; align-items:center; gap:10px; margin-bottom:8px; font-size:13px; }
  .trim-head span{ color:#8b949e; font-size:12px; }
  .trim-row{ display:flex; align-items:end; gap:8px; flex-wrap:wrap; }
  .trim-field{ display:flex; flex-direction:column; gap:4px; color:#8b949e; font-size:11px; }
  .trim-field input{ width:110px; padding:7px 8px; border-radius:6px; border:1px solid #444;
                     background:#0d1117; color:#fff; font-variant-numeric:tabular-nums; }
  .b-trim-save{ background:#2f81f7; }
  .b-trim-mark{ background:#6e40c9; }
  .b-trim-reset{ background:#444c56; }
  #trimStatus{ margin-top:8px; color:#8b949e; font-size:12px; }
  #trimStatus.changed{ color:#d2a8ff; }
  #trimStatus.error{ color:#ff7b72; }
  .label-btns{ display:flex; gap:12px; margin-top:8px; }
  .label-btns button{ flex:1; padding:14px; font-size:15px; font-weight:600; }
  .b-qual{ background:#238636; } .b-qual.on{ outline:3px solid #56d364; }
  .b-unqual{ background:#da3633; } .b-unqual.on{ outline:3px solid #ff7b72; }
  .b-clear{ background:#444c56; flex:none !important; }
  .hint{ color:#6e7681; font-size:12px; margin-top:10px; }
  /* modal */
  .modal-bg{ position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; align-items:center; justify-content:center; }
  .modal{ background:#22262d; padding:22px; border-radius:10px; width:420px; max-width:92vw; }
  .modal h3{ margin:0 0 14px; }
  .modal label{ display:block; font-size:13px; margin:10px 0 4px; color:#c9d1d9; }
  .modal input[type=text]{ width:100%; padding:8px; border-radius:6px; border:1px solid #444; background:#15171b; color:#fff;}
  .modal .row{ display:flex; align-items:center; gap:8px; margin-top:12px; font-size:13px;}
  .modal .actions{ display:flex; gap:10px; justify-content:flex-end; margin-top:18px; }
  .btn-ghost{ background:#373e47; }
  #exportLog{ white-space:pre-wrap; font-size:12px; color:#8b949e; margin-top:12px; max-height:200px; overflow:auto;}
</style>
</head>
<body>
<header>
  <h1 id="dsName">…</h1>
  <div class="counts" id="counts"></div>
  <div class="spacer"></div>
  <button class="btn-export" id="btnExport">导出已检查数据集</button>
</header>
<main>
  <div id="sidebar">
    <div class="filter">
      <button data-f="all" class="active">全部</button>
      <button data-f="qualified">合格</button>
      <button data-f="unqualified">不合格</button>
      <button data-f="unlabeled">未标注</button>
    </div>
    <div id="epList"></div>
  </div>
  <div id="content">
    <div id="videoWrap"><video id="player" controls autoplay loop muted></video></div>
    <div id="jointPanel">
      <div class="joint-head"><b>关节状态</b><span class="legend">观测 / 动作</span><span id="jointFrame"></span></div>
      <div id="jointRows" class="joint-empty">请选择左侧的 episode。</div>
    </div>
    <div id="info">请选择左侧的 episode。</div>
    <div id="trimPanel">
      <div class="trim-head"><b>Episode 裁剪</b><span>只影响导出结果，不修改原视频</span></div>
      <div class="trim-row">
        <label class="trim-field">起点帧（含）<input type="number" id="trimStart" min="0" step="1"></label>
        <label class="trim-field">保留帧数<input type="number" id="trimLength" min="1" step="1"></label>
        <button class="b-trim-save" id="bTrimSave">保存</button>
        <button class="b-trim-mark" id="bTrimStart">当前帧设为起点 ([)</button>
        <button class="b-trim-mark" id="bTrimEnd">当前帧设为终点 (])</button>
        <button class="b-trim-reset" id="bTrimReset">重置</button>
      </div>
      <div id="trimStatus">请选择左侧的 episode。</div>
    </div>
    <div class="label-btns">
      <button class="b-qual" id="bQual">合格 (A)</button>
      <button class="b-unqual" id="bUnqual">不合格 (D)</button>
      <button class="b-clear" id="bClear">清除</button>
    </div>
    <div class="hint">快捷键: ↑/↓ 或 J/K 切换 episode &nbsp;·&nbsp; [ 当前帧设为起点 &nbsp;·&nbsp; ] 当前帧设为终点 &nbsp;·&nbsp; A 合格 &nbsp;·&nbsp; D 不合格 &nbsp;·&nbsp; W 清除 &nbsp;·&nbsp; 空格 播放/暂停</div>
  </div>
</main>

<div class="modal-bg" id="modalBg">
  <div class="modal">
    <h3>导出已检查数据集</h3>
    <label>新数据集目录后缀</label>
    <input type="text" id="suffix" value="_checked">
    <div class="row">
      <input type="checkbox" id="onlyQual">
      <label for="onlyQual" style="margin:0;">仅导出"合格"的 episode (同时剔除未标注的)</label>
    </div>
    <div class="hint" id="exportInfo"></div>
    <div id="exportLog"></div>
    <div class="actions">
      <button class="btn-ghost" id="btnCancel">取消</button>
      <button class="btn-export" id="btnDoExport">开始导出</button>
    </div>
  </div>
</div>

<script>
let INFO = {}, EPS = [], COUNTS = {}, CUR = null, FILTER = "all";
let JOINTS = null, JOINT_ROWS = [], JOINT_FRAME = -1, JOINT_REQUEST = 0, VIDEO_FRAME_CALLBACK = null;

async function loadInfo(){
  const r = await fetch("/api/info"); const j = await r.json();
  INFO = j;
  document.getElementById("dsName").textContent =
    "📁 " + j.name + " · " + j.robot_type + " · " + j.joint_dim + "D · " +
    j.camera_count + " cameras · " + j.total_episodes + " episodes";
}
async function loadEpisodes(){
  const r = await fetch("/api/episodes"); const j = await r.json();
  EPS = j.episodes; COUNTS = j.counts; renderCounts(); renderList();
}
function renderCounts(){
  document.getElementById("counts").innerHTML =
    `<span><span class="dot c-qual"></span>合格 ${COUNTS.qualified}</span>` +
    `<span><span class="dot c-unqual"></span>不合格 ${COUNTS.unqualified}</span>` +
    `<span><span class="dot c-none"></span>未标注 ${COUNTS.unlabeled}</span>` +
    `<span>共 ${COUNTS.total}</span>`;
}
function cls(label){ return label==="qualified"?"qualified":label==="unqualified"?"unqualified":"unlabeled"; }
function visible(e){
  if(FILTER==="all") return true;
  if(FILTER==="unlabeled") return !e.label;
  return e.label===FILTER;
}
function renderList(){
  const box = document.getElementById("epList"); box.innerHTML="";
  EPS.filter(visible).forEach(e=>{
    const d = document.createElement("div");
    d.className = "ep " + cls(e.label) + (CUR===e.episode_index?" sel":"");
    d.dataset.idx = e.episode_index;
    d.innerHTML = `<span class="badge"></span><span>ep ${e.episode_index}</span>`+
                  `<span class="meta">${e.crop.trimmed?e.crop.length+"/":""}${e.length}f${e.crop.trimmed?" ✂":""} · ${e.success}</span>`;
    d.onclick = ()=>select(e.episode_index);
    box.appendChild(d);
  });
}
function epByIdx(i){ return EPS.find(e=>e.episode_index===i); }

function setTrimStatus(message, kind=""){
  const status = document.getElementById("trimStatus");
  status.textContent = message;
  status.className = kind;
}

function updateTrimPanel(){
  const e = epByIdx(CUR);
  if(!e) return;
  const start = document.getElementById("trimStart");
  const length = document.getElementById("trimLength");
  start.value = e.crop.start_frame;
  start.max = Math.max(0, e.length-1);
  length.value = e.crop.length;
  length.max = e.length-e.crop.start_frame;
  const last = e.crop.end_frame-1;
  setTrimStatus(
    `保留原始帧 ${e.crop.start_frame}–${last}，共 ${e.crop.length} 帧；头部删除 ${e.crop.head_trim} 帧，尾部删除 ${e.crop.tail_trim} 帧。`,
    e.crop.trimmed ? "changed" : ""
  );
}

function currentEpisodeFrame(){
  const e = epByIdx(CUR);
  const player = document.getElementById("player");
  const frame = Math.round((Number(player.currentTime)||0) * Number(INFO.fps));
  return e ? Math.min(e.length-1, Math.max(0, frame)) : 0;
}

async function saveTrim(startFrame, length){
  if(CUR===null) return;
  const episodeIndex = CUR;
  try{
    const r = await fetch("/api/trim", {method:"POST", headers:{"Content-Type":"application/json"},
              body:JSON.stringify({episode_index:episodeIndex, start_frame:startFrame, length})});
    const j = await r.json();
    if(!r.ok || !j.ok) throw new Error(j.error||("HTTP "+r.status));
    const e = epByIdx(j.episode_index);
    if(e) e.crop = j.crop;
    renderList();
    if(CUR===j.episode_index) updateTrimPanel();
  }catch(error){
    if(CUR===episodeIndex) setTrimStatus("保存失败: " + error.message, "error");
  }
}

function saveTrimInputs(){
  saveTrim(Number(document.getElementById("trimStart").value),
           Number(document.getElementById("trimLength").value));
}

function markTrimStart(){
  const e = epByIdx(CUR); if(!e) return;
  const start = currentEpisodeFrame();
  const end = start < e.crop.end_frame ? e.crop.end_frame : e.length;
  saveTrim(start, end-start);
}

function markTrimEnd(){
  const e = epByIdx(CUR); if(!e) return;
  const end = currentEpisodeFrame();
  if(end < e.crop.start_frame){
    setTrimStatus(`终点帧 ${end} 不能早于起点帧 ${e.crop.start_frame}。`, "error");
    return;
  }
  saveTrim(e.crop.start_frame, end-e.crop.start_frame+1);
}

function resetTrim(){
  const e = epByIdx(CUR); if(e) saveTrim(0, e.length);
}

async function loadJoints(idx){
  const ticket = ++JOINT_REQUEST;
  const box = document.getElementById("jointRows");
  JOINTS = null; JOINT_ROWS = []; JOINT_FRAME = -1;
  box.className = "joint-empty"; box.textContent = "读取关节数据中…";
  document.getElementById("jointFrame").textContent = "";
  try{
    const r = await fetch("/api/joints/" + idx);
    if(!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    if(ticket!==JOINT_REQUEST || CUR!==idx) return;
    if(!data.frame_index || !data.frame_index.length || data.names.length!==data.ranges.length)
      throw new Error("数据为空或维度不一致");
    JOINTS = data; buildJointRows();
  }catch(e){
    if(ticket!==JOINT_REQUEST || CUR!==idx) return;
    box.className = "joint-empty";
    box.textContent = "关节数据加载失败: " + e.message;
  }
}

function buildJointRows(){
  const box = document.getElementById("jointRows");
  box.className = ""; box.innerHTML = "";
  let previousGroup = null;
  JOINT_ROWS = JOINTS.names.map((name, j)=>{
    const group = (JOINTS.groups && JOINTS.groups[j]) || "other";
    if(group!==previousGroup){
      const heading = document.createElement("div");
      heading.className = "joint-group " + group;
      heading.textContent = group==="left" ? "左臂 (7D)" : group==="right" ? "右臂 (7D)" : "其他";
      box.appendChild(heading);
      previousGroup = group;
    }
    const row = document.createElement("div");
    row.className = "joint-row";
    row.innerHTML = `<span class="joint-name"></span>`+
      `<span class="joint-reading"><span class="joint-tag obs">观测</span><meter class="obs-meter"></meter><output class="joint-value obs-value"></output></span>`+
      `<span class="joint-reading"><span class="joint-tag act">动作</span><meter class="act-meter"></meter><output class="joint-value act-value"></output></span>`+
      `<output class="joint-delta"></output>`;
    row.querySelector(".joint-name").textContent = name;
    const obsMeter = row.querySelector(".obs-meter");
    const actMeter = row.querySelector(".act-meter");
    const [low, high] = JOINTS.ranges[j];
    for(const meter of [obsMeter, actMeter]){ meter.min=low; meter.max=high; }
    obsMeter.setAttribute("aria-label", name + " 观测值");
    actMeter.setAttribute("aria-label", name + " 动作值");
    box.appendChild(row);
    return {obsMeter, actMeter, obsValue:row.querySelector(".obs-value"),
            actValue:row.querySelector(".act-value"), delta:row.querySelector(".joint-delta")};
  });
  JOINT_FRAME = -1;
  renderJointFrame(document.getElementById("player").currentTime, true);
}

function renderJointFrame(time, force=false){
  if(!JOINTS || !JOINTS.frame_index.length) return;
  const i = Math.min(JOINTS.frame_index.length-1, Math.max(0, Math.round(time*JOINTS.fps)));
  if(i===JOINT_FRAME && !force) return;
  JOINT_FRAME = i;
  const ts = Number(JOINTS.timestamp[i]);
  document.getElementById("jointFrame").textContent =
    `Frame ${JOINTS.frame_index[i]} / ${JOINTS.frame_index.at(-1)} · ${Number.isFinite(ts)?ts.toFixed(3)+"s":"—"}`;
  JOINT_ROWS.forEach((el, j)=>{
    const obs = Number(JOINTS.observation[i][j]);
    const act = Number(JOINTS.action[i][j]);
    if(Number.isFinite(obs)){ el.obsMeter.value=obs; el.obsValue.textContent=obs.toFixed(3); }
    else{ el.obsMeter.removeAttribute("value"); el.obsValue.textContent="—"; }
    if(Number.isFinite(act)){ el.actMeter.value=act; el.actValue.textContent=act.toFixed(3); }
    else{ el.actMeter.removeAttribute("value"); el.actValue.textContent="—"; }
    const delta = act-obs;
    el.delta.textContent = Number.isFinite(delta) ? `Δ ${delta>=0?"+":""}${delta.toFixed(3)}` : "Δ —";
  });
}

function startJointSync(){
  const v = document.getElementById("player");
  if(typeof v.requestVideoFrameCallback!=="function") return;
  if(VIDEO_FRAME_CALLBACK!==null) v.cancelVideoFrameCallback(VIDEO_FRAME_CALLBACK);
  const tick = (_, meta)=>{
    renderJointFrame(meta.mediaTime);
    VIDEO_FRAME_CALLBACK = v.requestVideoFrameCallback(tick);
  };
  VIDEO_FRAME_CALLBACK = v.requestVideoFrameCallback(tick);
}

function select(idx){
  CUR = idx;
  const e = epByIdx(idx);
  const v = document.getElementById("player");
  v.src = "/api/clip/" + idx + "?t=" + Date.now();
  v.load();
  loadJoints(idx);
  document.getElementById("info").innerHTML =
     `<b>Episode ${e.episode_index}</b> &nbsp; 原始帧数: <b>${e.length}</b> &nbsp; 保留帧数: <b>${e.crop.length}</b> &nbsp; success: <b>${e.success}</b><br>`+
     `任务: ${e.task}`;
  updateTrimPanel();
  updateLabelButtons();
  // 同步侧栏选中态 & 滚动
  document.querySelectorAll(".ep").forEach(el=>{
    el.classList.toggle("sel", +el.dataset.idx===idx);
    if(+el.dataset.idx===idx) el.scrollIntoView({block:"nearest"});
  });
}
function updateLabelButtons(){
  const e = epByIdx(CUR);
  document.getElementById("bQual").classList.toggle("on", e && e.label==="qualified");
  document.getElementById("bUnqual").classList.toggle("on", e && e.label==="unqualified");
}
async function setLabel(label){
  if(CUR===null) return;
  const r = await fetch("/api/label", {method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({episode_index:CUR, label})});
  const j = await r.json();
  const e = epByIdx(CUR); e.label = j.label;
  recountAndRefresh();
  updateLabelButtons();
}
function recountAndRefresh(){
  COUNTS.qualified = EPS.filter(e=>e.label==="qualified").length;
  COUNTS.unqualified = EPS.filter(e=>e.label==="unqualified").length;
  COUNTS.unlabeled = EPS.filter(e=>!e.label).length;
  renderCounts(); renderList();
}
function move(delta){
  const vis = EPS.filter(visible);
  if(!vis.length) return;
  let i = vis.findIndex(e=>e.episode_index===CUR);
  i = i<0 ? 0 : Math.min(Math.max(i+delta,0), vis.length-1);
  select(vis[i].episode_index);
}

// 按钮
const player = document.getElementById("player");
player.addEventListener("loadeddata", ()=>{
  const e = epByIdx(CUR);
  if(e && INFO.fps) player.currentTime = e.crop.start_frame/INFO.fps;
  renderJointFrame(player.currentTime, true); startJointSync();
});
player.addEventListener("seeked", ()=>renderJointFrame(player.currentTime, true));
player.addEventListener("timeupdate", ()=>renderJointFrame(player.currentTime));
document.getElementById("bTrimSave").onclick = saveTrimInputs;
document.getElementById("bTrimStart").onclick = markTrimStart;
document.getElementById("bTrimEnd").onclick = markTrimEnd;
document.getElementById("bTrimReset").onclick = resetTrim;
document.getElementById("bQual").onclick   = ()=>setLabel("qualified");
document.getElementById("bUnqual").onclick = ()=>setLabel("unqualified");
document.getElementById("bClear").onclick  = ()=>setLabel(null);
document.querySelectorAll(".filter button").forEach(b=>{
  b.onclick = ()=>{ FILTER=b.dataset.f;
    document.querySelectorAll(".filter button").forEach(x=>x.classList.remove("active"));
    b.classList.add("active"); renderList(); };
});

// 键盘
document.addEventListener("keydown", ev=>{
  if(ev.target.tagName==="INPUT") return;
  const k = ev.key.toLowerCase();
  if(k==="arrowdown"||k==="j"){ move(1); ev.preventDefault(); }
  else if(k==="arrowup"||k==="k"){ move(-1); ev.preventDefault(); }
  else if(k==="["){ markTrimStart(); ev.preventDefault(); }
  else if(k==="]"){ markTrimEnd(); ev.preventDefault(); }
  else if(k==="a"){ setLabel("qualified"); }
  else if(k==="d"){ setLabel("unqualified"); }
  else if(k==="w"){ setLabel(null); }
  else if(k===" "){ const v=document.getElementById("player"); v.paused?v.play():v.pause(); ev.preventDefault(); }
});

// 导出
const modal = document.getElementById("modalBg");
document.getElementById("btnExport").onclick = ()=>{
  const trimmed = EPS.filter(e=>e.crop.trimmed);
  const removedFrames = trimmed.reduce((sum,e)=>sum+e.length-e.crop.length,0);
  document.getElementById("exportInfo").textContent =
     `将剔除被标记为"不合格"的 ${COUNTS.unqualified} 个 episode；裁剪 ${trimmed.length} 个 episode，共删除 ${removedFrames} 帧。`;
  document.getElementById("exportLog").textContent="";
  modal.style.display="flex";
};
document.getElementById("btnCancel").onclick = ()=> modal.style.display="none";
document.getElementById("btnDoExport").onclick = async ()=>{
  const suffix = document.getElementById("suffix").value || "_checked";
  const onlyQual = document.getElementById("onlyQual").checked;
  const log = document.getElementById("exportLog");
  log.textContent = "导出中, 请稍候 (复制视频可能需要一些时间)…";
  try{
    const r = await fetch("/api/export", {method:"POST", headers:{"Content-Type":"application/json"},
              body: JSON.stringify({suffix, only_qualified:onlyQual})});
    const j = await r.json();
    if(j.ok){
      log.textContent = `✅ 导出成功!\n目标: ${j.dst}\n保留 ${j.kept} 个 episode, 丢弃 ${j.dropped} 个；裁剪 ${j.trimmed} 个 episode，共删除 ${j.trimmed_frames} 帧；输出 ${j.total_frames} 帧。\n${j.note}`;
    }else{
      log.textContent = "❌ " + (j.error||"导出失败");
    }
  }catch(e){ log.textContent = "❌ " + e; }
};

(async ()=>{ await loadInfo(); await loadEpisodes(); if(EPS.length) select(EPS[0].episode_index); })();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="LeRobot 数据集 episode 网页检查/标注工具")
    ap.add_argument("dataset_pos", nargs="?", help="数据集目录（也可使用 --dataset）")
    ap.add_argument("--dataset", dest="dataset_opt", help="数据集目录")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--ffmpeg", default=None, help="指定 ffmpeg 路径 (默认自动挑选可软解 av1 的)")
    ap.add_argument("--check-joints", action="store_true", help="检查首个 episode 的关节数据后退出")
    args = ap.parse_args()

    global FFMPEG, AV1_DECODER
    FFMPEG, AV1_DECODER = _resolve_ffmpeg(args.ffmpeg)

    dataset_arg = args.dataset_opt or args.dataset_pos or "/home/lenovo/datasets/cube_catch_rollout_v4/0911_1"
    init_dataset(Path(dataset_arg))
    if args.check_joints:
        ep_idx = int(DS["episodes_df"].iloc[0]["episode_index"])
        with app.test_client() as client:
            response = client.get(f"/api/joints/{ep_idx}")
        assert response.status_code == 200, response.get_data(as_text=True)
        data = response.get_json()
        frames = data["frame_index"]
        assert frames == list(range(len(frames)))
        assert len(data["names"]) == len(data["ranges"]) == len(data["observation"][0]) == len(data["action"][0])
        assert all(low < high for low, high in data["ranges"])
        print(f"[OK] 关节数据检查通过: episode={ep_idx} frames={len(frames)} joints={len(data['names'])}")
        return
    print(f"\n  打开浏览器访问:  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
