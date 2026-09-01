#!/bin/bash
# 1 激活环境

# ```bash
# # mkdir -p ./package_sorting_env_raw && tar -xzf ./package_sorting.tar.gz -C ./package_sorting_env_raw
# source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
# /home/hpc/yuzhang/envs/package_sorting_env/bin/conda-unpack
# ```
# 用法：
#   ./lerobot_record.sh \
#     --dataset.root <数据存放目录> \
#     --dataset.single_task <任务指令> \
#     --wrist_camera.index_or_path <手腕摄像头设备索引或视频路径> \
#     --top_camera.index_or_path <顶部摄像头设备索引或视频路径> \
#     [--policy.path <策略模型路径>] \
#     --event.config.path <事件配置文件路径>
# 只需要传入以下参数：
# DATASET_ROOT：采集的数据目标存放目录
# SINGLE_TASK：任务指令
# WRIST_CAM：手腕摄像头的设备索引或视频文件路径
# TOP_CAM：顶部摄像头的设备索引或视频文件路径
# POLICY_PATH：（可选）训练过程中使用的策略模型路径；若无则由纯人工示范生成；若由则优先由策略模型生成示范，同时允许人工介入
# EVENT_CONFIG_PATH：（可选）事件配置文件路径，用于定义质量事件及其热键；若无则仅有success/failed/record三种事件
# 输出：
# 打印DATASET_ROOT所在位置

# 纯人工示范（不传 policy.path）
# source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/cup_catch_v4/0828_left_row3 \
#   --dataset.single_task "Grab the left cup" \
#   --wrist_camera.index_or_path 260422275773 \
#   --top_camera.index_or_path 10

# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/cup_catch_v4/0828_mid_row3_add1 \
#   --dataset.single_task "Take the middle cup away" \
#   --wrist_camera.index_or_path 260422275773 \
#   --top_camera.index_or_path 10

# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/cup_catch_v4/0831_testNeedDelete \
#   --dataset.single_task "Pick up the cup on the right" \
#   --wrist_camera.index_or_path 260422275773 \
#   --top_camera.index_or_path 10

# Pick up the cup on the right
# Take the middle cup away
# Grab the left cupc
# clean 0828_left_row1: 28   0828_left_row3: 102 101c
# 0828_mid_row2:74   0828_mid_row3:53





# 策略模型 + 人工介入
# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/pi05_base_smovla_v3_0720_RLT_30K_test_1 \
#   --dataset.single_task "You are a parcel sorter. First, Grab the package and place it on the pallet. Then, Flip the package if the barcode is not facing up. Finally, Grab the scanned package and place it into the box." \
#   --wrist_camera.index_or_path 4 \
#   --top_camera.index_or_path 12 \
#   --policy.path /home/hpc/yuzhang/outputs/pi05_base_smovla_v3_0720_RLT_30K \
#   --event.config.path /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/event_config.json

# pi05_base_cup_catch_0813_tain0813_30K
# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/cup_catch/0814_right_1 \
#   --dataset.single_task "Pick up the cup on the right" \
#   --wrist_camera.index_or_path 4 \
#   --top_camera.index_or_path 12 \
#   --policy.path /home/hpc/yuzhang/outputs/pi05_base_cup_catch_0813_tain0813_30K \
#   --event.config.path /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/event_config.json

# pi05_base_cup_catch_v2_0819_25k
# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/pi05_base_cup_catch_v2_0819_25k_test1 \
#   --dataset.single_task "Grab the left cup" \
#   --wrist_camera.index_or_path 260422275773 \
#   --top_camera.index_or_path 12 \
#   --policy.path /home/hpc/yuzhang/outputs/pi05_base_cup_catch_v2_0819_25k \
#   --event.config.path /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/event_config.json

# smovla_cup_catch_v2_0819_40k
# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/smovla_cup_catch_v2_0819_40k_test1 \
#   --dataset.single_task "Pick up the cup on the right" \
#   --wrist_camera.index_or_path 6 \
#   --top_camera.index_or_path 12 \
#   --policy.path /home/hpc/yuzhang/outputs/smovla_cup_catch_v2_0819_40k \
#   --event.config.path /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/event_config.json

# pi05_base_cup_catch_v4_merged_train0829_30k
# source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
# cd /home/hpc/yuzhang/Evo-RL-loop-0817
# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/pi05_base_sft_cup_catch_v4_merged_train0831_30k_test_needDelete \
#   --dataset.single_task "Pick up the cup on the right" \
#   --wrist_camera.index_or_path 260422275773 \
#   --top_camera.index_or_path 261822305080 \
#   --policy.path /home/hpc/yuzhang/outputs/pi05_base_sft_cup_catch_v4_merged_train0831_30k \
#   --rtc.enabled true


# smovla_cup_catch_v4_merged_train0829_40k
# source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
# cd /home/hpc/yuzhang/Evo-RL-loop-0817
# bash /home/hpc/yuzhang/Evo-RL-loop-0817/scripts/RL_data.sh \
#   --dataset.root /home/hpc/yuzhang/datasets/smovla_cup_catch_v4_merged_train0829_40k_test_needDelete \
#   --dataset.single_task "Pick up the cup on the right" \
#   --wrist_camera.index_or_path 260422275773 \
#   --top_camera.index_or_path 12 \
#   --policy.path /home/hpc/yuzhang/outputs/smovla_cup_catch_v4_merged_train0829_40k


# 合并数据集
# cd /home/yz/projects/Evo-RL-loop-0609/src
# python -m lerobot.scripts.lerobot_edit_dataset \
#     --repo_id /home/yz/datasets/v9_task123_0728_merged \
#     --operation.type merge \
#     --operation.repo_ids "['/home/yz/datasets/v9_task2_0728/v9_task2_0728_merged', '/home/yz/datasets/task0_grab_the_package_and_place_it_on_the_pal', '/home/yz/datasets/task2_grab_the_package_and_place_it_into_the_b']"

# source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
# cd /home/hpc/yuzhang/Evo-RL-loop-0817/src
# python -m lerobot.scripts.lerobot_edit_dataset \
#     --repo_id /home/hpc/yuzhang/datasets/cup_catch_v4_merged \
#     --operation.type merge \
#     --operation.source_dir /home/hpc/yuzhang/datasets/cup_catch_v4

# cd /home/hpc/yuzhang/outputs
# downloadyuzhang pi05_base_cup_catch_v2_0819_35k.tar

set -e

# ---------- 解析参数 ----------
DATASET_ROOT=""
SINGLE_TASK=""
WRIST_CAM=""
TOP_CAM=""
POLICY_PATH=""
EVENT_CONFIG_PATH=""
RTC_ENABLED="false"
RTC_EXECUTION_HORIZON="25"
RTC_ACTION_QUEUE_THRESHOLD="32"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset.root)
            DATASET_ROOT="$2"; shift 2 ;;
        --dataset.single_task)
            SINGLE_TASK="$2"; shift 2 ;;
        --wrist_camera.index_or_path)
            WRIST_CAM="$2"; shift 2 ;;
        --top_camera.index_or_path)
            TOP_CAM="$2"; shift 2 ;;
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
        *)
            echo "[错误] 未知参数：$1" >&2
            exit 1 ;;
    esac
done

# ---------- 校验必填参数 ----------
MISSING=()
[[ -z "$DATASET_ROOT"  ]] && MISSING+=("--dataset.root")
[[ -z "$SINGLE_TASK"   ]] && MISSING+=("--dataset.single_task")
[[ -z "$WRIST_CAM"     ]] && MISSING+=("--wrist_camera.index_or_path")
[[ -z "$TOP_CAM"       ]] && MISSING+=("--top_camera.index_or_path")

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
    echo "policy.path: ${POLICY_PATH}"
    echo "[提示] PI05 模型加载时会忽略 checkpoint 中当前模型不需要的多余参数。"
fi

# ---------- 构建摄像头配置 ----------
# CAMERAS="{wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30}}"

# CAMERAS="{wrist: {type: intelrealsense, serial_number_or_name: \"${WRIST_CAM}\", width: 640, height: 480, fps: 30, use_depth: false}, top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps:
#   30}}"

CAMERAS="{wrist: {type: intelrealsense, serial_number_or_name: \"${WRIST_CAM}\", width: 640, height: 480, fps: 30, use_depth: false}, top: {type: intelrealsense, serial_number_or_name: ${TOP_CAM}, width: 640, height: 480, fps:
  30}}"


# ---------- 构建命令 ----------
CMD=(
    lerobot-human-inloop-record
    --robot.type=piper_follower
    --robot.port=can1
    --robot.id=my_piper_follower
    --robot.speed_ratio=50
    "--robot.cameras=${CAMERAS}"
    --teleop.type=piper_leader
    --teleop.port=can0
    --teleop.id=my_piper_leader
    --teleop.command_speed_ratio=50
    --dataset.repo_id=local_data
    "--dataset.root=${DATASET_ROOT}"
    "--dataset.single_task=${SINGLE_TASK}"
    --dataset.num_episodes=110
    --dataset.episode_time_s=120000
    --dataset.reset_time_s=3
    --dataset.push_to_hub=False
    --display_data=true
    --resume=false
    --reset_on_timeout=false
)

# 若提供了事件配置路径则追加
if [[ -n "$EVENT_CONFIG_PATH" ]]; then
    CMD+=("--event_config_path=${EVENT_CONFIG_PATH}")
fi

# 若提供了策略模型路径则追加（启用人机协同模式）
if [[ -n "$POLICY_PATH" ]]; then
    CMD+=("--policy.path=${POLICY_PATH}")
fi

# RTC 默认关闭；关闭时不向 Python 入口追加任何 RTC 参数，完整保留原执行路径。
if [[ "$RTC_ENABLED" == "true" ]]; then
    CMD+=(
        --rtc.enabled=true
        "--rtc.execution_horizon=${RTC_EXECUTION_HORIZON}"
        "--rtc_action_queue_threshold=${RTC_ACTION_QUEUE_THRESHOLD}"
    )
fi

# ---------- 执行 ----------
echo "dataset.root: ${DATASET_ROOT}"

exec "${CMD[@]}"
