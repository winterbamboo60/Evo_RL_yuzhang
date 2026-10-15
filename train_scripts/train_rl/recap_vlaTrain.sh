#!/usr/bin/env bash
set -euo pipefail
RECAP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RECAP_SCRIPT_DIR/recap_common.sh"

NUM_GPUS=8
ROLLING_SAVE_FREQ=2000
recap_init "$@"
# Consume the new dataset produced by recap_valueInfer; never annotate the source.
DATASET_ROOT="$CFS_ROOT/craig/datasets/yida880/640/y480-merged_RLMarked"
DATASET_REPO_ID="local/$(basename "$DATASET_ROOT")"
POLICY_INIT="$CFS_ROOT/yuzhang/modelZoo/pi05base"
OUTPUT_DIR="$CFS_ROOT/yuzhang/outputs/1015_rl_vlaTrain_pi05_rlt_yida880-640-y480-merged_8gpu"
LOG_DIR="${OUTPUT_DIR}_logs"
STAGE=recap_vlaTrain

CMD=(
    bash "$EVORL_PROJECT_ROOT/scripts/RL_train.sh"
    --DATASET_ROOT "$DATASET_ROOT"
    --dataset.repo_id "$DATASET_REPO_ID"
    --history_pretrained_path "$POLICY_INIT"
    --OUTPUT_DIR "$OUTPUT_DIR"
    --policy_type pi05_rlt
    --num_gpus "$NUM_GPUS"
    --batch_size 8
    --gradient_accumulation_steps 1
    --steps 50000
    --save_freq 10000
    --rolling_save_freq "$ROLLING_SAVE_FREQ"
    --train_expert_only false
    --acp_enabled true
    --acp_indicator_field complementary_info.acp_indicator
    --acp_indicator_dropout_prob 0.3
    --rlt_alpha 1.0
    --tensorboard
    --
    '--rename_map={"observation.images.main":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'
    --policy.compile_model=false
    "${FORWARD_ARGS[@]}"
)
recap_run
