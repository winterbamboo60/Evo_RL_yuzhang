#!/bin/bash
# 用法：
#   ./lerobot_train.sh \
#     --DATASET_ROOT <数据存放目录> \
#     --ModelZoo <预训练模型路径> \
#     --OUTPUT_DIR <输出目录> \
#     [--history_pretrained_path <上一轮策略模型路径>]   # 可选：策略训练预训练权重，默认用基础模型
# 示例：
# conda activate evo-rl-0609
# source /home/yz/projects/env/package_sorting_env/bin/activate
# bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT /home/yz/datasets/package_sorting_task2_loop_0624_act_merged_122_checked \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR /home/yz/projects/outputs/V6_task2_0624_act_merged_122_onlyRL_Intervention_0626_H100 \
#   --history_pretrained_path /home/yz/package_scan_model_v6_H100

# bash /home/yuzhang/projects/VLA/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT /home/yuzhang/projects/VLA/datasets/package_sorting_task2_loop_0624_act_merged_122_checked \
#   --ModelZoo /home/yuzhang/projects/modelZoo \
#   --OUTPUT_DIR /home/yuzhang/projects/VLA/outputs/V6_task2_0624_act_merged_122_onlyRL_Intervention_0626 \
#   --history_pretrained_path /home/yuzhang/projects/modelZoo/package_scan_model_v6

# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT /home/yz/datasets/package_sorting_task2_V6_0624_act_merged_122_checked_merged_331 \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR /home/yz/projects/outputs/V6_task2_0624_act_merged_122_checked_merged_331_SFT+RL \
#   --history_pretrained_path /home/yz/package_scan_model_v6_H100 \
#   > /home/yz/projects/outputs/logs/train_0630_2.log 2>&1 &
# echo "PID: $!"

# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT /home/yz/datasets/package_sorting_task2_V6_0624_act_merged_122_checked_merged_331 \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR /home/yz/projects/outputs/V6_task2_0624_act_merged_122_checked_merged_331_SFT+RL_0626Checkpoint \
#   --history_pretrained_path /home/yz/package_scan_model_v6_H100 \
#   > /home/yz/projects/outputs/logs/train_0629_0626Checkpoint.log 2>&1 &
# echo "PID: $!"

# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT /home/yz/datasets/package_sorting_task2_V6_task2_0624_0630_merged_186 \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR /home/yz/projects/outputs/V6_task2_0624_0630_merged_186_0701_H100 \
#   --history_pretrained_path /home/yz/package_scan_model_v6_H100 \
#   > /home/yz/projects/outputs/logs/train_0701_onlyRL.log 2>&1 &
# echo "PID: $!"

# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT /home/yz/datasets/package_sorting_task2_V6_task2_0624_0630_merged_186 \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR /home/yz/projects/outputs/V6_task2_0624_0630_merged_186_noIncep_0701_H100 \
#   --history_pretrained_path /home/yz/package_scan_model_v6_H100 \
#   > /home/yz/projects/outputs/logs/train_0701_onlyRL_noIncep.log 2>&1 &
# echo "PID: $!"

# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT /home/yz/datasets/package_sorting_task2_V6_task2_0630_merged_64_SFT \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR /home/yz/projects/outputs/pi05_libero_base_task2_0630_merged_64_SFT_0702_onlySFT_H100 \
#   --history_pretrained_path /home/yz/modelZoo/pi05_libero_base \
#   > /home/yz/projects/outputs/logs/train_0702_onlySFT.log 2>&1 &
# echo "PID: $!"

# V6_task1&3_V7_task2_646
# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT "/home/yz/datasets/V6_task1&3_V7_task2_646" \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR "/home/yz/projects/outputs/V6_task1&3_V7_task2_646" \
#   --history_pretrained_path /home/yz/package_scan_model_v6_H100 \
#   > "/home/yz/projects/outputs/logs/train_0703_V6_task1&3_V7_task2_646.log" 2>&1 &
# echo "PID: $!"

# V8-1_0710
# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT "/home/yz/datasets/V8_0706/V8-1_task2_merged_391_rawTask" \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR "/home/yz/projects/outputs/V8-1_task2_merged_391" \
#   --history_pretrained_path /home/yz/modelZoo/V7_task2_20k \
#   > "/home/yz/projects/outputs/logs/V8-1_task2_merged_391_Train0713.log" 2>&1 &
# echo "PID: $!"

# V8_task123_0706
# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT "/home/yz/datasets/V8_0706/V8_task123_merged_952" \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR "/home/yz/projects/outputs/V8_task123_merged_952" \
#   --history_pretrained_path /home/yz/modelZoo/V7_task2_20k \
#   > "/home/yz/projects/outputs/logs/V8_task123_merged_952_Train0708.log" 2>&1 &
# echo "PID: $!"

# smovla_v2_0713
# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT "/home/yz/datasets/1F_0713/smovla_V2_0713" \
#   --ModelZoo /home/yz/modelZoo \
#   --policy_type "smolvla" \
#   --OUTPUT_DIR "/home/yz/projects/outputs/smovla_v2_0713" \
#   --history_pretrained_path /home/yz/modelZoo/smolvla_base \
#   > "/home/yz/projects/outputs/logs/smovla_v2_0713_Train0714.log" 2>&1 &
# echo "PID: $!"

# smovla_v3_0720
# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT "/home/yz/datasets/1F_0713/smovla_V3_0720_merged" \
#   --ModelZoo /home/yz/modelZoo \
#   --policy_type "smolvla" \
#   --OUTPUT_DIR "/home/yz/projects/outputs/smovla_v3_0720" \
#   --history_pretrained_path /home/yz/modelZoo/smolvla_base \
#   > "/home/yz/projects/outputs/logs/smovla_v3_0720_Train0720.log" 2>&1 &
# echo "PID: $!"

# pi05_lebero_smovla_v3_0720
# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT "/home/yz/datasets/1F_0713/smovla_V3_0720_merged" \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR "/home/yz/projects/outputs/pi05_smovla_v3_0720" \
#   > "/home/yz/projects/outputs/logs/pi05_smovla_v3_0720_Train0722.log" 2>&1 &
# echo "PID: $!"

# # pi05_base_smovla_v3_0720_size512
# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT "/home/yz/datasets/1F_0713/smovla_V3_0720_merged" \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR "/home/yz/projects/outputs/pi05_base_smovla_v3_0720_size512" \
#   > "/home/yz/projects/outputs/logs/pi05_base_smovla_v3_0720_size512_Train0727.log" 2>&1 &
# echo "PID: $!"

# pi05_base_v9_task123_0728
# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT "/home/yz/datasets/v9_task123_0728_merged" \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR "/home/yz/projects/outputs/pi05_base_v9_task123_0728" \
#   > "/home/yz/projects/outputs/logs/pi05_base_v9_task123_0728_Train0728.log" 2>&1 &
# echo "PID: $!"

# smovla_v9_task123_0728
# nohup bash /home/yz/projects/Evo-RL-loop-0609/scripts/RL_train.sh \
#   --DATASET_ROOT "/home/yz/datasets/v9_task123_0728_merged" \
#   --ModelZoo /home/yz/modelZoo \
#   --policy_type "smolvla" \
#   --OUTPUT_DIR "/home/yz/projects/outputs/smovla_v9_task123_0728" \
#   --history_pretrained_path /home/yz/modelZoo/smolvla_base \
#   > "/home/yz/projects/outputs/logs/smovla_v9_task123_0728_Train0728.log" 2>&1 &
# echo "PID: $!"

# pi05_base_smovla_v3_0720_RLT
# source /home/yz/projects/env/package_sorting_env/bin/activate
# nohup bash /home/yz/projects/Evo-RL-loop-0810/scripts/RL_train.sh \
#   --DATASET_ROOT "/home/yz/datasets/1F_0713/smovla_V3_0720_merged" \
#   --ModelZoo /home/yz/modelZoo \
#   --OUTPUT_DIR "/home/yz/projects/outputs/pi05_base_smovla_v3_0720_RLT" \
#   > "/home/yz/projects/outputs/logs/pi05_base_smovla_v3_0720_RLT_Train0811.log" 2>&1 &
# echo "PID: $!"

set -e

# ---------- 解析参数 ----------
DATASET_ROOT=""
MODELZOO_PATH=""
OUTPUT_DIR=""
HISTORY_PRETRAINED_PATH=""   # 可选：策略训练的预训练权重路径，未指定则用基础模型
policy_type="pi05"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --DATASET_ROOT)
            DATASET_ROOT="$2"; shift 2 ;;
        --ModelZoo)
            MODELZOO_PATH="$2"; shift 2 ;;
        --OUTPUT_DIR)
            OUTPUT_DIR="$2"; shift 2 ;;
        --history_pretrained_path)
            HISTORY_PRETRAINED_PATH="$2"; shift 2 ;;
        --policy_type)
            policy_type="$2"; shift 2 ;;
        *)
            echo "[错误] 未知参数：$1" >&2
            exit 1 ;;
    esac
done

# ---------- 校验必填参数 ----------
MISSING=()
[[ -z "$DATASET_ROOT"   ]] && MISSING+=("--DATASET_ROOT")
[[ -z "$MODELZOO_PATH" ]] && MISSING+=("--ModelZoo")
[[ -z "$OUTPUT_DIR"     ]] && MISSING+=("--OUTPUT_DIR")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "[错误] 缺少必填参数：${MISSING[*]}" >&2
    exit 1
fi

# if [ -d "${OUTPUT_DIR}" ]; then
#     rm -rf "${OUTPUT_DIR:?}"/*
#     echo "已清空 ${OUTPUT_DIR}"
# else
#     mkdir -p "${OUTPUT_DIR}"
#     echo "目录不存在，已创建 ${OUTPUT_DIR}"
# fi

#!/bin/bash

# ─────────────────────────────────────────────
#  wait_18h.sh
#  等待 18 小时后执行任务，每 30 分钟打印一次剩余时间
# ─────────────────────────────────────────────

# TOTAL_SECONDS=$((24 * 3600))   # 24 小时 = 86400 秒
# INTERVAL=$((30 * 60))          # 每 0.5 小时 = 1800 秒
# ELAPSED=0

# START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
# END_TIME=$(date -d "+18 hours" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
#            || date -v+18H '+%Y-%m-%d %H:%M:%S')   # 兼容 macOS

# echo "╔══════════════════════════════════════════╗"
# echo "║          ⏳  24 小时倒计时启动            ║"
# echo "╚══════════════════════════════════════════╝"
# echo "  开始时间：$START_TIME"
# echo "  预计执行：$END_TIME"
# echo "  打印间隔：每 30 分钟"
# echo "──────────────────────────────────────────"

# while [ "$ELAPSED" -lt "$TOTAL_SECONDS" ]; do
#     REMAINING=$((TOTAL_SECONDS - ELAPSED))
#     REM_H=$((REMAINING / 3600))
#     REM_M=$(( (REMAINING % 3600) / 60 ))

#     printf "[%s]  剩余时间：%2d 小时 %02d 分钟\n" \
#            "$(date '+%H:%M:%S')" "$REM_H" "$REM_M"

#     sleep "$INTERVAL"
#     ELAPSED=$((ELAPSED + INTERVAL))
# done

# echo "──────────────────────────────────────────"
# echo "✅  等待结束！开始执行任务..."
# echo "   执行时间：$(date '+%Y-%m-%d %H:%M:%S')"
# echo "══════════════════════════════════════════"


# ---------- Step 1: value 训练 ----------
# 推荐参数：batch_size * gradient_accumulation_steps =64, steps=2000
# echo "[Step 1/3] 开始 value 训练..."
# lerobot-value-train \
#     --dataset.repo_id=local_data \
#     "--dataset.root=${DATASET_ROOT}" \
#     --value.type=pistar06 \
#     --value.dtype=bfloat16 \
#     --value.push_to_hub=False \
#     --value.repo_id=local_value_model \
#     --value.vision_repo_id="${MODELZOO_PATH}/siglip-so400m-patch14-384" \
#     --value.language_repo_id="${MODELZOO_PATH}/gemma-3-270m" \
#     --value.device="cuda" \
#     --batch_size=16 \
#     --steps=8000 \
#     --save_freq=2000 \
#     --gradient_accumulation_steps=4 \
#     "--output_dir=${OUTPUT_DIR}/value_train" \
#     --job_name=value_train \
#     --wandb.enable=false \
#     # --value.freeze_vision_encoder=true \
#     # --value.freeze_language_model=true \
# echo "[Step 1/3] value 训练完成，模型已保存至 ${OUTPUT_DIR}/value_train"
# echo "[Step 2/3] 即将开始 value 推理..."

# # # ---------- Step 2: value 推理 ----------
# lerobot-value-infer \
#     --dataset.repo_id=local_data \
#     "--dataset.root=${DATASET_ROOT}" \
#     "--inference.checkpoint_path=${OUTPUT_DIR}/value_train" \
#     --runtime.device="cuda" \
#     --runtime.batch_size=64 \
#     --acp.enable=true \
#     --acp.n_step=50 \
#     --acp.positive_ratio=0.3 \
#     --acp.value_field=complementary_info.value_field \
#     --acp.advantage_field=complementary_info.advantage_field \
#     --acp.indicator_field=complementary_info.indicator_field\
#     --acp.force_intervention_positive=True \
#     "--output_dir=${OUTPUT_DIR}/value_infer" \
#     --job_name=value_infer \
#     --viz.enable=true \
#     --viz.episodes=all \
#     --viz.video_keys=observation.images.top,observation.images.wrist \
#     --viz.smooth_window=5 \
#     --viz.overwrite=false \
#     --viz.frame_storage_mode=memory \
#     --viz.indicator_only=false
# echo "[Step 2/3] value 推理完成，结果已保存至 ${OUTPUT_DIR}/value_infer"
# echo "[Step 3/3] 即将开始策略训练..."

# ---------- Step 3: 策略训练 ----------
# 推荐参数：batch_size=32，steps=30000，--use_8bit_optimizer=false
# 若显存低于34G，可以选择开启--use_8bit_optimizer=true
# 预训练权重：传了 --history_pretrained_path 就用它（从上一轮策略继续），否则用基础模型。
# --sft_train=true时，直接对给定数据集进行SFT训练
POLICY_PRETRAINED_PATH="${HISTORY_PRETRAINED_PATH:-${MODELZOO_PATH}/pi05base}"
echo "[Step 3/3] policy.pretrained_path = ${POLICY_PRETRAINED_PATH}"
lerobot-train \
    --use_8bit_optimizer=false \
    --dataset.repo_id=local_data \
    "--dataset.root=${DATASET_ROOT}" \
    --policy.type=${policy_type} \
    "--policy.pretrained_path=${POLICY_PRETRAINED_PATH}" \
    --policy.device="cuda" \
    --policy.train_expert_only=true \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --steps=50000 \
    --save_freq=10000 \
    --acp.enable=true \
    --acp.indicator_field=complementary_info.indicator_field \
    --acp.indicator_dropout_prob=0.3 \
    "--output_dir=${OUTPUT_DIR}/train" \
    --job_name=VLA_train \
    --wandb.enable=false \
    --policy.push_to_hub=False \
    --policy.repo_id=local_policy_model \
    --sft_train=true \
    --policy.dtype=bfloat16 \
    --policy.use_rlt=true \
    # --resume=true \
    # --config_path=/path/to/output_dir/checkpoints/last/pretrained_model/train_config.json
    # --policy.gradient_checkpointing=true \
echo "[Step 3/3] 策略训练完成"

# ---------- 输出策略模型保存路径 ----------
echo "OUTPUT_DIR: ${OUTPUT_DIR}/train/checkpoints/last/pretrained_model"