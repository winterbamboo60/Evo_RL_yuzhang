#!/bin/bash
# 1 激活环境

# ```bash
# # mkdir -p ./package_sorting_env_raw && tar -xzf ./package_sorting.tar.gz -C ./package_sorting_env_raw
# source /home/yz/projects/env/package_sorting_env/bin/activate
# ./home/ghr/yuzhang/代码封装/package_sorting_env/bin/conda-unpack
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
# bash /home/ghr/yuzhang/Evo-RL-loop/scripts/RL_data.sh \
#   --dataset.root /home/ghr/datasets/output_001 \
#   --dataset.single_task "Flip the package if the barcode is not facing up. Advantage: positive" \
#   --wrist_camera.index_or_path 10 \
#   --top_camera.index_or_path 4

# 策略模型 + 人工介入
# bash /home/ghr/yuzhang/Evo-RL-loop/scripts/RL_data.sh \
#   --dataset.root /home/ghr/datasets/package_sorting_task2_loop_0612_1 \
#   --dataset.single_task "Flip the package if the barcode is not facing up. Advantage: positive" \
#   --wrist_camera.index_or_path 4 \
#   --top_camera.index_or_path 10 \
#   --policy.path /home/ghr/mm/package_scan_model_v6 \
#   --event.config.path /home/ghr/projects/VLA/Evo-RL-loop-0609/scripts/event_config.json

# 合并数据集
# cd /home/yz/projects/Evo-RL-loop-0609/src
# python -m lerobot.scripts.lerobot_edit_dataset \
#     --repo_id /home/yz/datasets/v9_task123_0728_merged \
#     --operation.type merge \
#     --operation.repo_ids "['/home/yz/datasets/v9_task2_0728/v9_task2_0728_merged', '/home/yz/datasets/task0_grab_the_package_and_place_it_on_the_pal', '/home/yz/datasets/task2_grab_the_package_and_place_it_into_the_b']"

# python -m lerobot.scripts.lerobot_edit_dataset \
#     --repo_id /home/yz/datasets/v9_task2_0728/v9_task2_0728_merged \
#     --operation.type merge \
#     --operation.source_dir /home/yz/datasets/v9_task2_0728

set -e

# ---------- 解析参数 ----------
DATASET_ROOT=""
SINGLE_TASK=""
WRIST_CAM=""
TOP_CAM=""
POLICY_PATH=""
EVENT_CONFIG_PATH=""
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

# ---------- 构建摄像头配置 ----------
CAMERAS="{wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30}}"

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
    --dataset.num_episodes=50
    --dataset.episode_time_s=120
    --dataset.reset_time_s=1
    --dataset.push_to_hub=False
    --display_data=true
    --resume=false
    "--event_config_path=${EVENT_CONFIG_PATH}"
    --reset_on_timeout=false
)

# 若提供了策略模型路径则追加（启用人机协同模式）
if [[ -n "$POLICY_PATH" ]]; then
    CMD+=("--policy.path=${POLICY_PATH}")
fi

# ---------- 执行 ----------
echo "dataset.root: ${DATASET_ROOT}"

exec "${CMD[@]}"