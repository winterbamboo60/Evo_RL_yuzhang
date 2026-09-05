#!/usr/bin/env bash
#
# 双 PiPER 主从臂 HIL 数据采集入口。仅组装并执行与 RL_data.sh 相同的
# lerobot-human-inloop-record 流程；校准文件需要在启动前准备。
#
# 默认 CAN 映射：
#   左主臂 can0 -> 左从臂 can1
#   右主臂 can2 -> 右从臂 can3c
#
# 三路 Intel RealSense：
#   左臂子配置 wrist + top，右臂子配置 wrist
#   数据集视角键为 left_wrist、left_top、right_wrist。
#
# 纯人工遥操示例：
# bash /home/lenovo/code/Evo-RL-loop-0901/scripts/RL_data_bimanual.sh \
#   --dataset.root /home/lenovo/datasets/bi_piper_test3 \
#   --dataset.single_task "test_task" \
#   --top_camera.serial_number_or_name 261822303677 \
#   --left_wrist_camera.serial_number_or_name 260422275773 \
#   --right_wrist_camera.serial_number_or_name 260422275792

# VLA + 人工介入时额外传入：
#   --policy.path /path/to/checkpoint
# 可继续复用 RL_data.sh 的：
#   --event.config.path /path/to/event_config.json
#   --rtc.enabled true
#   --rtc.execution_horizon 25
#   --rtc_action_queue_threshold 32

set -euo pipefail

DATASET_ROOT=""
SINGLE_TASK=""
TOP_CAM=""
LEFT_WRIST_CAM=""
RIGHT_WRIST_CAM=""
POLICY_PATH=""
EVENT_CONFIG_PATH=""
RTC_ENABLED="false"
RTC_EXECUTION_HORIZON="25"
RTC_ACTION_QUEUE_THRESHOLD="32"

LEFT_LEADER_PORT="can0"
LEFT_FOLLOWER_PORT="can1"
RIGHT_LEADER_PORT="can2"
RIGHT_FOLLOWER_PORT="can3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset.root)
            DATASET_ROOT="$2"; shift 2 ;;
        --dataset.single_task)
            SINGLE_TASK="$2"; shift 2 ;;
        --top_camera.serial_number_or_name|--top_camera.index_or_path)
            TOP_CAM="$2"; shift 2 ;;
        --left_wrist_camera.serial_number_or_name|--left_wrist_camera.index_or_path)
            LEFT_WRIST_CAM="$2"; shift 2 ;;
        --right_wrist_camera.serial_number_or_name|--right_wrist_camera.index_or_path)
            RIGHT_WRIST_CAM="$2"; shift 2 ;;
        --policy.path)
            POLICY_PATH="$2"; shift 2 ;;
        --event.config.path)
            EVENT_CONFIG_PATH="$2"; shift 2 ;;
        --rtc.enabled)
            RTC_ENABLED="$2"; shift 2 ;;
        --rtc.execution_horizon)
            RTC_EXECUTION_HORIZON="$2"; shift 2 ;;
        --rtc_action_queue_threshold)
            RTC_ACTION_QUEUE_THRESHOLD="$2"; shift 2 ;;
        --teleop.left_arm_config.port)
            LEFT_LEADER_PORT="$2"; shift 2 ;;
        --robot.left_arm_config.port)
            LEFT_FOLLOWER_PORT="$2"; shift 2 ;;
        --teleop.right_arm_config.port)
            RIGHT_LEADER_PORT="$2"; shift 2 ;;
        --robot.right_arm_config.port)
            RIGHT_FOLLOWER_PORT="$2"; shift 2 ;;
        *)
            echo "[错误] 未知参数：$1" >&2
            exit 1 ;;
    esac
done

MISSING=()
[[ -z "$DATASET_ROOT" ]] && MISSING+=("--dataset.root")
[[ -z "$SINGLE_TASK" ]] && MISSING+=("--dataset.single_task")
[[ -z "$TOP_CAM" ]] && MISSING+=("--top_camera.serial_number_or_name")
[[ -z "$LEFT_WRIST_CAM" ]] && MISSING+=("--left_wrist_camera.serial_number_or_name")
[[ -z "$RIGHT_WRIST_CAM" ]] && MISSING+=("--right_wrist_camera.serial_number_or_name")
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "[错误] 缺少必填参数：${MISSING[*]}" >&2
    exit 1
fi

if [[ "$RTC_ENABLED" != "true" && "$RTC_ENABLED" != "false" ]]; then
    echo "[错误] --rtc.enabled 只能是 true 或 false：$RTC_ENABLED" >&2
    exit 1
fi
if [[ "$RTC_ENABLED" == "true" && -z "$POLICY_PATH" ]]; then
    echo "[错误] --rtc.enabled=true 必须同时提供 --policy.path" >&2
    exit 1
fi

declare -A SEEN_PORTS=()
for PORT in "$LEFT_LEADER_PORT" "$LEFT_FOLLOWER_PORT" "$RIGHT_LEADER_PORT" "$RIGHT_FOLLOWER_PORT"; do
    if [[ -n "${SEEN_PORTS[$PORT]:-}" ]]; then
        echo "[错误] 四条机械臂 CAN 端口必须互不相同，重复端口：$PORT" >&2
        exit 1
    fi
    SEEN_PORTS["$PORT"]=1
done

if [[ -n "$POLICY_PATH" ]]; then
    if [[ ! -d "$POLICY_PATH" ]]; then
        echo "[错误] --policy.path 必须是可读取的 LeRobot policy 目录：$POLICY_PATH" >&2
        exit 1
    fi
    if [[ ! -f "$POLICY_PATH/config.json" ]]; then
        echo "[错误] --policy.path 缺少 config.json：$POLICY_PATH" >&2
        exit 1
    fi
    if [[ ! -f "$POLICY_PATH/model.safetensors" ]]; then
        echo "[错误] --policy.path 缺少 model.safetensors：$POLICY_PATH" >&2
        exit 1
    fi
    echo "policy.path: $POLICY_PATH"
    echo "[提示] VLA checkpoint 必须匹配双臂 14 维动作及 left_wrist/left_top/right_wrist 三路图像特征。"
fi

# LEFT_CAMERAS="{wrist: {type: intelrealsense, serial_number_or_name: \"$LEFT_WRIST_CAM\", width: 640, height: 480, fps: 30, use_depth: false}, top: {type: intelrealsense, serial_number_or_name: \"$TOP_CAM\", width: 640, height: 480, fps: 30, use_depth: false}}"
# RIGHT_CAMERAS="{wrist: {type: intelrealsense, serial_number_or_name: \"$RIGHT_WRIST_CAM\", width: 640, height: 480, fps: 30, use_depth: false}}"


LEFT_CAMERAS="{
    wrist: {
      type: intelrealsense,
      serial_number_or_name: \"${LEFT_WRIST_CAM}\",
      width: 640,
      height: 480,
      fps: 30,
      use_depth: false,
      warmup_s: 2,
      exposure_mode: manual,
      manual_exposure_us: 10000,
      manual_gain: 16,
      white_balance_kelvin: 3860
    },
    top: {
      type: intelrealsense,
      serial_number_or_name: \"${TOP_CAM}\",
      width: 640,
      height: 480,
      fps: 30,
      use_depth: false,
      warmup_s: 2,
      exposure_mode: manual,
      manual_exposure_us: 8200,
      manual_gain: 76,
      white_balance_kelvin: 3820
    }
  }"

RIGHT_CAMERAS="{
    wrist: {
      type: intelrealsense,
      serial_number_or_name: \"${RIGHT_WRIST_CAM}\",
      width: 640,
      height: 480,
      fps: 30,
      use_depth: false,
      warmup_s: 2,
      exposure_mode: manual,
      manual_exposure_us: 10000,
      manual_gain: 16,
      white_balance_kelvin: 3860
    }
}"

CMD=(
    lerobot-human-inloop-record
    --robot.type=bi_piper_follower
    --robot.id=my_bi_piper_follower
    "--robot.left_arm_config.port=$LEFT_FOLLOWER_PORT"
    "--robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT"
    --robot.left_arm_config.speed_ratio=50
    --robot.right_arm_config.speed_ratio=50
    --robot.left_arm_config.require_calibration=true
    --robot.right_arm_config.require_calibration=true
    "--robot.left_arm_config.cameras=$LEFT_CAMERAS"
    "--robot.right_arm_config.cameras=$RIGHT_CAMERAS"
    --teleop.type=bi_piper_leader
    --teleop.id=my_bi_piper_leader
    "--teleop.left_arm_config.port=$LEFT_LEADER_PORT"
    "--teleop.right_arm_config.port=$RIGHT_LEADER_PORT"
    --teleop.left_arm_config.command_speed_ratio=50
    --teleop.right_arm_config.command_speed_ratio=50
    --teleop.left_arm_config.require_calibration=true
    --teleop.right_arm_config.require_calibration=true
    --teleop.process_isolation=true
    --dataset.repo_id=local_data
    "--dataset.root=$DATASET_ROOT"
    "--dataset.single_task=$SINGLE_TASK"
    --dataset.num_episodes=110
    --dataset.episode_time_s=120000
    --dataset.reset_time_s=3
    --dataset.push_to_hub=False
    --display_data=true
    --resume=false
    --reset_on_timeout=false
)

if [[ -n "$EVENT_CONFIG_PATH" ]]; then
    CMD+=("--event_config_path=$EVENT_CONFIG_PATH")
fi
if [[ -n "$POLICY_PATH" ]]; then
    CMD+=("--policy.path=$POLICY_PATH")
    echo "[模式] VLA：启动时归位，左右主臂允许程序控制。"
else
    CMD+=(
        --teleop.left_arm_config.read_only_teaching_mode=true
        --teleop.right_arm_config.read_only_teaching_mode=true
    )
    echo "[模式] 人工示范：启动时不归位，左右主臂只读，允许硬件示教模式。"
fi
if [[ "$RTC_ENABLED" == "true" ]]; then
    CMD+=(
        --rtc.enabled=true
        "--rtc.execution_horizon=$RTC_EXECUTION_HORIZON"
        "--rtc_action_queue_threshold=$RTC_ACTION_QUEUE_THRESHOLD"
    )
fi

echo "dataset.root: $DATASET_ROOT"
echo "CAN: 左主=$LEFT_LEADER_PORT 左从=$LEFT_FOLLOWER_PORT 右主=$RIGHT_LEADER_PORT 右从=$RIGHT_FOLLOWER_PORT"
echo "RealSense: top=$TOP_CAM left_wrist=$LEFT_WRIST_CAM right_wrist=$RIGHT_WRIST_CAM"

exec "${CMD[@]}"
