#!/bin/bash
# 1 激活环境

# ```bash
# # mkdir -p ./package_sorting_env_raw && tar -xzf ./package_sorting.tar.gz -C ./package_sorting_env_raw
# source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
# /home/hpc/yuzhang/envs/package_sorting_env/bin/conda-unpack
# ```
# 用法：
#   ./RL_data.sh \
#     --dataset.root <数据存放目录> \
#     --dataset.single_task <任务指令> \
#     --wrist_camera.index_or_path <手腕摄像头设备索引或视频路径> \
#     --top_camera.index_or_path <顶部摄像头设备索引或视频路径> \
#     [--policy.path <策略模型路径>] \
#     [--policy.device cuda|cuda:N|cpu] \
#     [--can0.control true|false] \
# 只需要传入以下参数：
# DATASET_ROOT：采集的数据目标存放目录
# SINGLE_TASK：任务指令
# WRIST_CAM：手腕摄像头的设备索引或视频文件路径
# TOP_CAM：顶部摄像头的设备索引或视频文件路径
# POLICY_PATH：（可选）策略模型路径；若无则纯人工录制；若有则默认仅 VLA 推理、不连接 can0
# POLICY_DEVICE：VLA 加载及推理设备，默认 cuda；单进程可用 cuda:N 指定逻辑 GPU
# CAN0_CONTROL：仅 VLA 模式使用；true 恢复 can0 人工介入，false（默认）完全不连接或控制 can0
# 输出：
# 打印DATASET_ROOT所在位置


# Pick up the cup on the right
# Take the middle cup away
# Grab the left cupc
# clean 0828_left_row1: 28   0828_left_row3: 102 101c
# 0828_mid_row2:74   0828_mid_row3:53

# 一楼真机
# 纯人工示范（不传 policy.path）
# source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/cup_catch_v4/0828_left_row3 \
#   --dataset.single_task "Grab the left cup" \
#   --wrist_camera.index_or_path 260422275773 \
#   --top_camera.index_or_path 10

# 策略模型 + 人工介入
# smovla_cup_catch_v2_0819_40k
# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/smovla_cup_catch_v2_0819_40k_test1 \
#   --dataset.single_task "Pick up the cup on the right" \
#   --wrist_camera.index_or_path 6 \
#   --top_camera.index_or_path 12 \
#   --policy.path /home/hpc/yuzhang/outputs/smovla_cup_catch_v2_0819_40k \
#   --can0.control true


# 北京5080
# 纯人工示范（不传 policy.path）
# source /home/lenovo/code/envs/evo_0911/bin/activate
# cd /home/lenovo/code/Evo-RL-loop-0911
# bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data.sh \
#   --dataset.root /home/lenovo/datasets/cube_catch_rollout_v3/0911_1 \
#   --dataset.single_task "Grab the cube" \
#   --wrist_camera.index_or_path 260422275792 \
#   --top_camera.index_or_path 12

# 策略模型 + 人工介入
# smovla_cup_catch_v2_0819_40k
# source /home/lenovo/code/envs/evo_0911/bin/activate
# cd /home/lenovo/code/Evo-RL-loop-0911
# bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data.sh \
#   --dataset.root /home/lenovo/datasets/0909_pi05_sft_cube_catch_belt_50k_test1 \
#   --dataset.single_task "Grab the moving blocks on the conveyor belt" \
#   --wrist_camera.index_or_path 260422275792 \
#   --top_camera.index_or_path 12 \
#   --policy.path /home/lenovo/outputs/0909_pi05_sft_cube_catch_belt_50k \
#   --can0.control true

# source /home/lenovo/code/envs/evo_0911/bin/activate
# cd /home/lenovo/code/Evo-RL-loop-0911
# bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data.sh \
#   --dataset.root /home/lenovo/datasets/smovla_cup_catch_v2_0819_40k_test1 \
#   --dataset.single_task "Pick up the cup on the right" \
#   --wrist_camera.index_or_path 260422275773 \
#   --top_camera.index_or_path 261822303677 \
#   --policy.path /home/lenovo/outputs/0909_pi05_sft_cube_catch_belt_50k \
#   --can0.control true


# 合并数据集
# source /home/lenovo/code/envs/evo_0911/bin/activate
# cd /home/lenovo/code/Evo-RL-loop-0911
# python -m lerobot.scripts.lerobot_edit_dataset \
#     --repo_id /home/lenovo/datasets/v9_task123_0728_merged \
#     --operation.type merge \
#     --operation.repo_ids "['/home/lenovo/datasets/v9_task2_0728/v9_task2_0728_merged', '/home/lenovo/datasets/task0_grab_the_package_and_place_it_on_the_pal', '/home/lenovo/datasets/task2_grab_the_package_and_place_it_into_the_b']"

# source /home/lenovo/code/envs/evo_0911/bin/activate
# cd /home/lenovo/code/Evo-RL-loop-0911
# python -m lerobot.scripts.lerobot_edit_dataset \
#     --repo_id /home/lenovo/datasets/cube_catch_rollout_v3-1_merged \
#     --operation.type merge \
#     --operation.source_dir /home/lenovo/datasets/cube_catch_rollout_v3

# # 数据集字段规范化（非覆盖复制）
# # 已经包含 0901 complementary_info 字段的旧数据可直接训练，不需要转换。
# # 仅对缺少字段或使用临时 intervention bool 字段的数据执行：
# lerobot-migrate-evorl-dataset \
#     --source-root SOURCE_DATASET \
#     --destination-root DESTINATION_DATASET \
#     --destination-repo-id local/canonical_dataset \
#     --direction to-canonical \
#     --collector-policy-id pi05-checkpoint
# # destination-root 是新副本的磁盘目录；destination-repo-id 是写入 metadata 的数据集标识。
# # 存储仍为当前 LeRobot v3 Parquet/MP4，逐帧业务字段与 0901 对齐。


# cd /home/lenovo/outputs
# downloadyuzhang pi05_base_cup_catch_v2_0819_35k.tar

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p -- "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/RL_data_$(date '+%Y%m%d_%H%M%S')_$$.log"
export PYTHONUNBUFFERED=1
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[日志] ${LOG_FILE}"

DATASET_ROOT=""
SINGLE_TASK=""
WRIST_CAM=""
TOP_CAM=""
POLICY_PATH=""
POLICY_DEVICE="cuda"
VLA_CAN0_CONTROL="false"
SUCCESS_KEY="b"
FAILURE_KEY="f"
INTERVENTION_KEY="c"
RERECORD_KEY="a"
RESET_KEY="r"
RTC_ENABLED="false"
RTC_EXECUTION_HORIZON="25"
RTC_QUEUE_THRESHOLD="32"
WRIST_EXPOSURE="14000"
WRIST_GAIN="16"
WRIST_WHITE_BALANCE="3700"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset.root) DATASET_ROOT="$2"; shift 2 ;;
        --dataset.single_task) SINGLE_TASK="$2"; shift 2 ;;
        --wrist_camera.index_or_path|--wrist_camera.serial_number_or_name) WRIST_CAM="$2"; shift 2 ;;
        --top_camera.index_or_path) TOP_CAM="$2"; shift 2 ;;
        --wrist_camera.exposure|--wrist_camera.manual_exposure_us) WRIST_EXPOSURE="$2"; shift 2 ;;
        --wrist_camera.gain|--wrist_camera.manual_gain) WRIST_GAIN="$2"; shift 2 ;;
        --wrist_camera.white_balance|--wrist_camera.white_balance_kelvin) WRIST_WHITE_BALANCE="$2"; shift 2 ;;
        --policy.path) POLICY_PATH="$2"; shift 2 ;;
        --policy.device|--device) POLICY_DEVICE="$2"; shift 2 ;;
        --policy.device=*|--device=*) POLICY_DEVICE="${1#*=}"; shift ;;
        --can0.control|--vla.can0_control) VLA_CAN0_CONTROL="$2"; shift 2 ;;
        --can0.control=*|--vla.can0_control=*) VLA_CAN0_CONTROL="${1#*=}"; shift ;;
        --rtc.enabled) RTC_ENABLED="$2"; shift 2 ;;
        --rtc.execution_horizon) RTC_EXECUTION_HORIZON="$2"; shift 2 ;;
        --rtc_action_queue_threshold|--rtc.queue_threshold) RTC_QUEUE_THRESHOLD="$2"; shift 2 ;;
        *) echo "[错误] 未知参数：$1" >&2; exit 1 ;;
    esac
done

missing=()
[[ -z "$DATASET_ROOT" ]] && missing+=("--dataset.root")
[[ -z "$SINGLE_TASK" ]] && missing+=("--dataset.single_task")
[[ -z "$WRIST_CAM" ]] && missing+=("--wrist_camera.index_or_path")
[[ -z "$TOP_CAM" ]] && missing+=("--top_camera.index_or_path")
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
if [[ ! "$POLICY_DEVICE" =~ ^(cuda(:[0-9]+)?|cpu|mps|xpu)$ ]]; then
    echo "[错误] --policy.device/--device 必须是 cuda、cuda:N、cpu、mps 或 xpu：$POLICY_DEVICE" >&2
    exit 1
fi
if [[ -n "$POLICY_PATH" && "$POLICY_DEVICE" == cuda* ]]; then
    if ! PYTHON_BIN="$(command -v python)" || [[ ! -x "$PYTHON_BIN" ]]; then
        echo "[错误] 当前 PATH 中找不到可执行的 python；请先激活 LeRobot 环境" >&2
        exit 1
    fi
    if ! "$PYTHON_BIN" - "$POLICY_DEVICE" <<'PY'
import sys

import torch

requested = torch.device(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit(f"CUDA 不可用，无法在 {requested} 上加载 VLA")

device_count = torch.cuda.device_count()
device_index = requested.index if requested.index is not None else torch.cuda.current_device()
if device_index >= device_count:
    raise SystemExit(
        f"请求的逻辑设备 {requested} 不存在；当前 CUDA_VISIBLE_DEVICES 下仅有 {device_count} 张 GPU"
    )

print(f"[CUDA] VLA 将使用 {requested}（逻辑 GPU {device_index}: {torch.cuda.get_device_name(device_index)}）")
PY
    then
        echo "[错误] VLA CUDA 设备检查失败" >&2
        exit 1
    fi
fi

CAMERAS="{
  wrist: {
    type: intelrealsense,
    serial_number_or_name: \"${WRIST_CAM}\",
    width: 640, height: 480, fps: 30,
    use_rgb: true, use_depth: false, warmup_s: 2,
    exposure_mode: manual,
    exposure: ${WRIST_EXPOSURE}, gain: ${WRIST_GAIN}, white_balance: ${WRIST_WHITE_BALANCE}
  },
  top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30}
}"

COMMON=(
    --robot.type=piper_follower
    --robot.port=can1
    --robot.id=my_piper_follower
    --robot.speed_ratio=50
    "--robot.cameras=${CAMERAS}"
    --dataset.repo_id=rollout_evorl_piper
    "--dataset.root=${DATASET_ROOT}"
    "--dataset.single_task=${SINGLE_TASK}"
    --dataset.no_stamp=true
    --dataset.num_episodes=100
    --dataset.episode_time_s=120000
    --dataset.reset_time_s=3
    --dataset.push_to_hub=false
    --dataset.streaming_encoding=true
    --dataset.encoder_threads=2
    --display_data=true
    --resume=false
)
TELEOP=(
    --teleop.type=piper_leader
    --teleop.port=can0
    --teleop.id=my_piper_leader
    --teleop.command_speed_ratio=50
)


if [[ -z "$POLICY_PATH" ]]; then
    echo "[模式] 纯人工录制；B=成功 F=失败 A=放弃并重录 C=人工模式（当前已是人工控制）R=归零重录 Esc=结束。"
    CMD=(
        lerobot-record
        "${COMMON[@]}"
        "${TELEOP[@]}"
        --teleop.read_only_teaching_mode=true
        --enable_evorl_controls=true
        "--episode_success_key=${SUCCESS_KEY}"
        "--episode_failure_key=${FAILURE_KEY}"
        "--rerecord_episode_key=${RERECORD_KEY}"
        "--intervention_toggle_key=${INTERVENTION_KEY}"
        "--reset_episode_key=${RESET_KEY}"
    )
else
    echo "[模型] checkpoint=${POLICY_PATH} policy.device=${POLICY_DEVICE} rollout.device=${POLICY_DEVICE}"
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
        "--policy.device=${POLICY_DEVICE}"
        "--device=${POLICY_DEVICE}"
        --fps=30
        --return_to_initial_position=false
    )
    if [[ "$VLA_CAN0_CONTROL" == "true" ]]; then
        echo "[模式] VLA + can0 随动及人工介入；B=成功 F=失败 A=放弃并重录 C=立即人工接管 R=归零重录 Esc=结束。"
        CMD+=(
            "${TELEOP[@]}"
            --teleop.manual_control=false
            --strategy.enable_intervention=true
        )
    else
        echo "[模式] VLA 仅推理；can0 不连接、不发送动作，C 已禁用；B=成功 F=失败 A=放弃并重录 R=归零重录 Esc=结束。"
        CMD+=(--strategy.enable_intervention=false)
    fi
    if [[ "$RTC_ENABLED" == "true" ]]; then
        echo "[RTC] 已启用：mode=guided execution_horizon=${RTC_EXECUTION_HORIZON} queue_threshold=${RTC_QUEUE_THRESHOLD}；C 接管时立即废弃旧 VLA 动作。"
        CMD+=(
            --inference.type=rtc
            --inference.rtc.mode=guided
            "--inference.rtc.execution_horizon=${RTC_EXECUTION_HORIZON}"
            "--inference.queue_threshold=${RTC_QUEUE_THRESHOLD}"
        )
    fi
fi

echo "dataset.root: ${DATASET_ROOT}"
exec "${CMD[@]}"
