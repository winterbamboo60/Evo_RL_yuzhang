#!/usr/bin/env bash
#
# 双 PiPER 主从臂 HIL 数据采集入口。仅组装并执行与 RL_data.sh 相同的
# lerobot-human-inloop-record 流程；校准文件需要在启动前准备。
#
# 默认 CAN 映射：
#   左主臂 can0 -> 左从臂 can1
#   右主臂 can2 -> 右从臂 can3
#
# 三路 Intel RealSense：
#   左臂子配置 wrist + top，右臂子配置 wrist
#   数据集视角键为 left_wrist、left_top、right_wrist。
#
# 纯人工遥操示例：
# bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data_bimanual.sh \
#   --dataset.root /home/lenovo/datasets/cube_catch_rollout_v4/0911_1 \
#   --dataset.single_task "Grab the cube" \
#   --top_camera.index_or_path 6 \
#   --left_wrist_camera.index_or_path 260422275773 \
#   --right_wrist_camera.index_or_path 260422275792

# VLA + 人工介入时额外传入：
#   --policy.path /path/to/checkpoint
#   --can0.control true
# --can0.control 是左右主臂的总开关：VLA 模式默认 false，不连接 can0/can2。
# 可继续复用 RL_data.sh 的：
#   --rtc.enabled true
#   --rtc.execution_horizon 25
#   --rtc_action_queue_threshold 32

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p -- "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/RL_data_bimanual_$(date '+%Y%m%d_%H%M%S')_$$.log"
export PYTHONUNBUFFERED=1
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[日志] ${LOG_FILE}"

DATASET_ROOT=""
SINGLE_TASK=""
TOP_CAM=""
LEFT_WRIST_CAM=""
RIGHT_WRIST_CAM=""
POLICY_PATH=""
VLA_CAN0_CONTROL="false"
SUCCESS_KEY="b"
FAILURE_KEY="f"
INTERVENTION_KEY="c"
RERECORD_KEY="a"
RESET_KEY="r"
RTC_ENABLED="false"
RTC_EXECUTION_HORIZON="25"
RTC_QUEUE_THRESHOLD="32"
LEFT_EXPOSURE="10000"
RIGHT_EXPOSURE="10000"
CAMERA_GAIN="16"
CAMERA_WHITE_BALANCE="3860"
LEFT_LEADER_PORT="can0"
LEFT_FOLLOWER_PORT="can1"
RIGHT_LEADER_PORT="can2"
RIGHT_FOLLOWER_PORT="can3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset.root) DATASET_ROOT="$2"; shift 2 ;;
        --dataset.single_task) SINGLE_TASK="$2"; shift 2 ;;
        --top_camera.index_or_path|--top_camera.index_or_path) TOP_CAM="$2"; shift 2 ;;
        --left_wrist_camera.index_or_path|--left_wrist_camera.index_or_path) LEFT_WRIST_CAM="$2"; shift 2 ;;
        --right_wrist_camera.index_or_path|--right_wrist_camera.index_or_path) RIGHT_WRIST_CAM="$2"; shift 2 ;;
        --left_wrist_camera.exposure|--left_wrist_camera.manual_exposure_us) LEFT_EXPOSURE="$2"; shift 2 ;;
        --right_wrist_camera.exposure|--right_wrist_camera.manual_exposure_us) RIGHT_EXPOSURE="$2"; shift 2 ;;
        --wrist_camera.gain|--wrist_camera.manual_gain) CAMERA_GAIN="$2"; shift 2 ;;
        --wrist_camera.white_balance|--wrist_camera.white_balance_kelvin) CAMERA_WHITE_BALANCE="$2"; shift 2 ;;
        --policy.path) POLICY_PATH="$2"; shift 2 ;;
        --can0.control|--vla.can0_control) VLA_CAN0_CONTROL="$2"; shift 2 ;;
        --can0.control=*|--vla.can0_control=*) VLA_CAN0_CONTROL="${1#*=}"; shift ;;
        --rtc.enabled) RTC_ENABLED="$2"; shift 2 ;;
        --rtc.execution_horizon) RTC_EXECUTION_HORIZON="$2"; shift 2 ;;
        --rtc_action_queue_threshold|--rtc.queue_threshold) RTC_QUEUE_THRESHOLD="$2"; shift 2 ;;
        --teleop.left_arm_config.port) LEFT_LEADER_PORT="$2"; shift 2 ;;
        --robot.left_arm_config.port) LEFT_FOLLOWER_PORT="$2"; shift 2 ;;
        --teleop.right_arm_config.port) RIGHT_LEADER_PORT="$2"; shift 2 ;;
        --robot.right_arm_config.port) RIGHT_FOLLOWER_PORT="$2"; shift 2 ;;
        *) echo "[错误] 未知参数：$1" >&2; exit 1 ;;
    esac
done

missing=()
[[ -z "$DATASET_ROOT" ]] && missing+=("--dataset.root")
[[ -z "$SINGLE_TASK" ]] && missing+=("--dataset.single_task")
[[ -z "$TOP_CAM" ]] && missing+=("--top_camera.index_or_path")
[[ -z "$LEFT_WRIST_CAM" ]] && missing+=("--left_wrist_camera.index_or_path")
[[ -z "$RIGHT_WRIST_CAM" ]] && missing+=("--right_wrist_camera.index_or_path")
if (( ${#missing[@]} )); then
    echo "[错误] 缺少必填参数：${missing[*]}" >&2
    exit 1
fi
if [[ "$RTC_ENABLED" != "true" && "$RTC_ENABLED" != "false" ]]; then
    echo "[错误] --rtc.enabled 只能是 true 或 false：$RTC_ENABLED" >&2
    exit 1
fi
if [[ "$VLA_CAN0_CONTROL" != "true" && "$VLA_CAN0_CONTROL" != "false" ]]; then
    echo "[错误] --can0.control 只能是 true 或 false：$VLA_CAN0_CONTROL" >&2
    exit 1
fi
if [[ "$RTC_ENABLED" == "true" && -z "$POLICY_PATH" ]]; then
    echo "[错误] --rtc.enabled=true 必须同时提供 --policy.path" >&2
    exit 1
fi
if [[ "$RTC_ENABLED" == "true" && ! "$RTC_EXECUTION_HORIZON" =~ ^[1-9][0-9]*$ ]]; then
    echo "[错误] --rtc.execution_horizon 必须是正整数：$RTC_EXECUTION_HORIZON" >&2
    exit 1
fi
if [[ "$RTC_ENABLED" == "true" && ! "$RTC_QUEUE_THRESHOLD" =~ ^[0-9]+$ ]]; then
    echo "[错误] --rtc_action_queue_threshold/--rtc.queue_threshold 必须是非负整数：$RTC_QUEUE_THRESHOLD" >&2
    exit 1
fi
if [[ -n "$POLICY_PATH" && ! -f "$POLICY_PATH/config.json" ]]; then
    echo "[错误] --policy.path 不是当前 LeRobot checkpoint：$POLICY_PATH" >&2
    exit 1
fi

USE_LEADERS="false"
if [[ -z "$POLICY_PATH" || "$VLA_CAN0_CONTROL" == "true" ]]; then
    USE_LEADERS="true"
fi

ACTIVE_PORTS=("$LEFT_FOLLOWER_PORT" "$RIGHT_FOLLOWER_PORT")
if [[ "$USE_LEADERS" == "true" ]]; then
    ACTIVE_PORTS=("$LEFT_LEADER_PORT" "$LEFT_FOLLOWER_PORT" "$RIGHT_LEADER_PORT" "$RIGHT_FOLLOWER_PORT")
fi

declare -A SEEN_PORTS=()
for port in "${ACTIVE_PORTS[@]}"; do
    if [[ -n "${SEEN_PORTS[$port]:-}" ]]; then
        echo "[错误] 启用的机械臂 CAN 端口必须互不相同，重复端口：$port" >&2
        exit 1
    fi
    SEEN_PORTS["$port"]=1
done

LEFT_CAMERAS="{
  wrist: {
    type: intelrealsense, serial_number_or_name: \"${LEFT_WRIST_CAM}\",
    width: 640, height: 480, fps: 30, use_rgb: true, use_depth: false, warmup_s: 2,
    exposure_mode: manual, exposure: ${LEFT_EXPOSURE}, gain: ${CAMERA_GAIN},
    white_balance: ${CAMERA_WHITE_BALANCE}
  },
  top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30}
}"
RIGHT_CAMERAS="{
  wrist: {
    type: intelrealsense, serial_number_or_name: \"${RIGHT_WRIST_CAM}\",
    width: 640, height: 480, fps: 30, use_rgb: true, use_depth: false, warmup_s: 2,
    exposure_mode: manual, exposure: ${RIGHT_EXPOSURE}, gain: ${CAMERA_GAIN},
    white_balance: ${CAMERA_WHITE_BALANCE}
  }
}"

COMMON=(
    --robot.type=bi_piper_follower
    --robot.id=my_bi_piper_follower
    "--robot.left_arm_config.port=${LEFT_FOLLOWER_PORT}"
    "--robot.right_arm_config.port=${RIGHT_FOLLOWER_PORT}"
    --robot.left_arm_config.speed_ratio=50
    --robot.right_arm_config.speed_ratio=50
    --robot.left_arm_config.require_calibration=true
    --robot.right_arm_config.require_calibration=true
    "--robot.left_arm_config.cameras=${LEFT_CAMERAS}"
    "--robot.right_arm_config.cameras=${RIGHT_CAMERAS}"
    --dataset.repo_id=rollout_evorl_bi_piper
    "--dataset.root=${DATASET_ROOT}"
    "--dataset.single_task=${SINGLE_TASK}"
    --dataset.no_stamp=true
    --dataset.num_episodes=110
    --dataset.episode_time_s=120000
    --dataset.reset_time_s=3
    --dataset.push_to_hub=false
    --dataset.streaming_encoding=true
    --dataset.encoder_threads=2
    --display_data=true
    --resume=false
)
TELEOP=(
    --teleop.type=bi_piper_leader
    --teleop.id=my_bi_piper_leader
    "--teleop.left_arm_config.port=${LEFT_LEADER_PORT}"
    "--teleop.right_arm_config.port=${RIGHT_LEADER_PORT}"
    --teleop.left_arm_config.command_speed_ratio=50
    --teleop.right_arm_config.command_speed_ratio=50
    --teleop.left_arm_config.require_calibration=true
    --teleop.right_arm_config.require_calibration=true
    --teleop.process_isolation=true
)

if [[ -z "$POLICY_PATH" ]]; then
    echo "[模式] 双臂纯人工录制；B=成功 F=失败 A=放弃并重录 C=人工模式（当前已是人工控制）R=归零重录 Esc=结束。"
    CMD=(
        lerobot-record
        "${COMMON[@]}"
        "${TELEOP[@]}"
        --teleop.left_arm_config.read_only_teaching_mode=true
        --teleop.right_arm_config.read_only_teaching_mode=true
        --enable_evorl_controls=true
        "--episode_success_key=${SUCCESS_KEY}"
        "--episode_failure_key=${FAILURE_KEY}"
        "--rerecord_episode_key=${RERECORD_KEY}"
        "--intervention_toggle_key=${INTERVENTION_KEY}"
        "--reset_episode_key=${RESET_KEY}"
    )
else
    CMD=(
        lerobot-rollout
        "${COMMON[@]}"
        --strategy.type=evorl_episodic
        "--strategy.success_key=${SUCCESS_KEY}"
        "--strategy.failure_key=${FAILURE_KEY}"
        "--strategy.rerecord_key=${RERECORD_KEY}"
        "--strategy.intervention_key=${INTERVENTION_KEY}"
        "--strategy.reset_key=${RESET_KEY}"
        "--policy.path=${POLICY_PATH}"
        --fps=30
        --return_to_initial_position=false
    )
    if [[ "$VLA_CAN0_CONTROL" == "true" ]]; then
        echo "[模式] 双臂 VLA + can0/can2 随动及人工介入；B=成功 F=失败 A=放弃并重录 C=立即人工接管 R=归零重录 Esc=结束。"
        CMD+=(
            "${TELEOP[@]}"
            --teleop.left_arm_config.manual_control=false
            --teleop.right_arm_config.manual_control=false
            --strategy.enable_intervention=true
        )
    else
        echo "[模式] 双臂 VLA 仅推理；can0/can2 不连接、不发送动作，C 已禁用；B=成功 F=失败 A=放弃并重录 R=归零重录 Esc=结束。"
        CMD+=(--strategy.enable_intervention=false)
    fi
    if [[ "$RTC_ENABLED" == "true" ]]; then
        echo "[RTC] 双臂 RTC 已启用：mode=guided execution_horizon=${RTC_EXECUTION_HORIZON} queue_threshold=${RTC_QUEUE_THRESHOLD}；C 接管时立即废弃旧 VLA 动作。"
        CMD+=(
            --inference.type=rtc
            --inference.rtc.mode=guided
            "--inference.rtc.execution_horizon=${RTC_EXECUTION_HORIZON}"
            "--inference.queue_threshold=${RTC_QUEUE_THRESHOLD}"
        )
    fi
fi

echo "dataset.root: ${DATASET_ROOT}"
if [[ "$USE_LEADERS" == "true" ]]; then
    echo "CAN: 左主=${LEFT_LEADER_PORT} 左从=${LEFT_FOLLOWER_PORT} 右主=${RIGHT_LEADER_PORT} 右从=${RIGHT_FOLLOWER_PORT}"
else
    echo "CAN: 左主=禁用 左从=${LEFT_FOLLOWER_PORT} 右主=禁用 右从=${RIGHT_FOLLOWER_PORT}"
fi
exec "${CMD[@]}"
