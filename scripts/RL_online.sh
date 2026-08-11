#!/usr/bin/env bash
# # 环境准备
# source /home/yz/projects/env/package_sorting_env/bin/activate
# cd /home/yz/projects/Evo-RL-loop-0609
# # 启动learner
# bash scripts/RL_online.sh learner
# # 启动actor
# bash scripts/RL_online.sh actor
# # 同机启动learner+actor，并指定摄像头和奖励模型
# bash scripts/RL_online.sh both \
#   --wrist_camera.index_or_path 10 \
#   --top_camera.index_or_path 4 \
#   --reward_model.path /path/to/reward_classifier
# # 后台启动
# nohup bash scripts/RL_online.sh both > /home/yz/projects/outputs/logs/piper_online_rl_run.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG="${REPO_ROOT}/scripts/piper_online_rl.json"
CONFIG="${PIPER_ONLINE_RL_CONFIG:-$DEFAULT_CONFIG}"
MODE="${1:-both}"

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  cat <<EOF
Usage:
  bash scripts/RL_online.sh [learner|actor|both] [online options] [extra draccus overrides...]

Examples:
  bash scripts/RL_online.sh both
  bash scripts/RL_online.sh both \
    --wrist_camera.index_or_path 10 \
    --top_camera.index_or_path 4 \
    --reward_model.path /home/yz/models/reward_classifier
  bash scripts/RL_online.sh learner --policy.actor_learner_config.learner_port=50051
  bash scripts/RL_online.sh actor --policy.actor_learner_config.learner_host=192.168.1.10

Online options:
  --wrist_camera.index_or_path <id_or_path>   Wrist OpenCV camera index or path.
  --top_camera.index_or_path <id_or_path>     Top OpenCV camera index or path.
  --camera.width <int>                        Camera width, default 640.
  --camera.height <int>                       Camera height, default 480.
  --camera.fps <int>                          Camera FPS, default 30.
  --reward_model.path <path>                  Reward classifier model path.
  --reward_model.success_threshold <float>    Success threshold, default 0.5.
  --reward_model.success_reward <float>       Reward on success, default 1.0.

Config:
  ${CONFIG}

Notes:
  - Run learner before actor when they are started separately.
  - Override PIPER_ONLINE_RL_CONFIG to use another config file.
  - Extra arguments are passed through as draccus overrides.
EOF
  exit 0
fi

if [[ "$MODE" == learner || "$MODE" == actor || "$MODE" == both ]]; then
  shift || true
elif [[ "$MODE" == --* ]]; then
  MODE="both"
else
  echo "Unknown mode: ${MODE}. Expected learner, actor, or both." >&2
  exit 2
fi

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

LOG_DIR="/home/yz/projects/outputs/logs"
mkdir -p "$LOG_DIR"

WRIST_CAM=""
TOP_CAM=""
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30
REWARD_MODEL_PATH=""
REWARD_SUCCESS_THRESHOLD=0.5
REWARD_SUCCESS_REWARD=1.0
EXTRA_OVERRIDES=()

require_value() {
  local opt="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "Missing value for ${opt}" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wrist_camera.index_or_path)
      require_value "$1" "${2:-}"
      WRIST_CAM="$2"
      shift 2
      ;;
    --top_camera.index_or_path)
      require_value "$1" "${2:-}"
      TOP_CAM="$2"
      shift 2
      ;;
    --camera.width)
      require_value "$1" "${2:-}"
      CAMERA_WIDTH="$2"
      shift 2
      ;;
    --camera.height)
      require_value "$1" "${2:-}"
      CAMERA_HEIGHT="$2"
      shift 2
      ;;
    --camera.fps)
      require_value "$1" "${2:-}"
      CAMERA_FPS="$2"
      shift 2
      ;;
    --reward_model.path|--reward_classifier.path)
      require_value "$1" "${2:-}"
      REWARD_MODEL_PATH="$2"
      shift 2
      ;;
    --reward_model.success_threshold|--reward_classifier.success_threshold)
      require_value "$1" "${2:-}"
      REWARD_SUCCESS_THRESHOLD="$2"
      shift 2
      ;;
    --reward_model.success_reward|--reward_classifier.success_reward)
      require_value "$1" "${2:-}"
      REWARD_SUCCESS_REWARD="$2"
      shift 2
      ;;
    *)
      EXTRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$WRIST_CAM" || -n "$TOP_CAM" ]]; then
  if [[ -z "$WRIST_CAM" || -z "$TOP_CAM" ]]; then
    echo "Both --wrist_camera.index_or_path and --top_camera.index_or_path are required when overriding cameras." >&2
    exit 2
  fi
fi

CONFIG_TO_USE="$CONFIG"
TMP_CONFIG=""

cleanup_temp_config() {
  if [[ -n "${TMP_CONFIG:-}" && -f "$TMP_CONFIG" ]]; then
    rm -f "$TMP_CONFIG" >/dev/null 2>&1 || true
  fi
}

if [[ -n "$WRIST_CAM" || -n "$REWARD_MODEL_PATH" ]]; then
  TMP_CONFIG="$(mktemp "${LOG_DIR}/piper_online_rl_config.XXXXXX.json")"
  PIPER_BASE_CONFIG="$CONFIG"   PIPER_OUT_CONFIG="$TMP_CONFIG"   PIPER_WRIST_CAM="$WRIST_CAM"   PIPER_TOP_CAM="$TOP_CAM"   PIPER_CAMERA_WIDTH="$CAMERA_WIDTH"   PIPER_CAMERA_HEIGHT="$CAMERA_HEIGHT"   PIPER_CAMERA_FPS="$CAMERA_FPS"   PIPER_REWARD_MODEL_PATH="$REWARD_MODEL_PATH"   PIPER_REWARD_SUCCESS_THRESHOLD="$REWARD_SUCCESS_THRESHOLD"   PIPER_REWARD_SUCCESS_REWARD="$REWARD_SUCCESS_REWARD"   python - <<'PY_CONFIG'
import json
import os
from pathlib import Path


def scalar(value: str):
    try:
        return int(value)
    except ValueError:
        return value

base_path = Path(os.environ["PIPER_BASE_CONFIG"])
out_path = Path(os.environ["PIPER_OUT_CONFIG"])
with base_path.open() as f:
    cfg = json.load(f)

env = cfg.setdefault("env", {})
robot = env.setdefault("robot", {})
processor = env.setdefault("processor", {})

wrist_cam = os.environ.get("PIPER_WRIST_CAM", "")
top_cam = os.environ.get("PIPER_TOP_CAM", "")
if wrist_cam or top_cam:
    width = int(os.environ["PIPER_CAMERA_WIDTH"])
    height = int(os.environ["PIPER_CAMERA_HEIGHT"])
    fps = int(os.environ["PIPER_CAMERA_FPS"])
    robot["cameras"] = {
        "wrist": {
            "type": "opencv",
            "index_or_path": scalar(wrist_cam),
            "width": width,
            "height": height,
            "fps": fps,
        },
        "top": {
            "type": "opencv",
            "index_or_path": scalar(top_cam),
            "width": width,
            "height": height,
            "fps": fps,
        },
    }

reward_model_path = os.environ.get("PIPER_REWARD_MODEL_PATH", "")
if reward_model_path:
    processor["reward_classifier"] = {
        "pretrained_path": reward_model_path,
        "success_threshold": float(os.environ["PIPER_REWARD_SUCCESS_THRESHOLD"]),
        "success_reward": float(os.environ["PIPER_REWARD_SUCCESS_REWARD"]),
    }

with out_path.open("w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY_CONFIG
  CONFIG_TO_USE="$TMP_CONFIG"
  trap cleanup_temp_config EXIT
fi

COMMON_OVERRIDES=("${EXTRA_OVERRIDES[@]}")

check_python_deps() {
  python - <<'PY_CHECK'
missing = []
for module_name in ("draccus", "grpc", "torch"):
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        missing.append(module_name)
if missing:
    raise SystemExit(
        "Missing Python modules for online RL: "
        + ", ".join(missing)
        + ". Activate the Evo-RL environment or install the project with hilserl/grpc dependencies."
    )
PY_CHECK
}

run_learner() {
  python -m lerobot.rl.learner --config_path "$CONFIG_TO_USE" "${COMMON_OVERRIDES[@]}" "$@"
}

run_actor() {
  python -m lerobot.rl.actor --config_path "$CONFIG_TO_USE" "${COMMON_OVERRIDES[@]}" "$@"
}

check_python_deps

case "$MODE" in
  learner)
    run_learner "$@"
    ;;
  actor)
    run_actor "$@"
    ;;
  both)
    LEARNER_LOG="${LOG_DIR}/piper_online_rl_learner.log"
    ACTOR_LOG="${LOG_DIR}/piper_online_rl_actor.log"
    run_learner "$@" >"$LEARNER_LOG" 2>&1 &
    learner_pid=$!
    cleanup() {
      if kill -0 "$learner_pid" >/dev/null 2>&1; then
        kill "$learner_pid" >/dev/null 2>&1 || true
      fi
      cleanup_temp_config
    }
    trap cleanup EXIT INT TERM
    sleep 5
    run_actor "$@" >"$ACTOR_LOG" 2>&1
    ;;
esac
