#!/usr/bin/env bash
set -euo pipefail
RECAP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RECAP_SCRIPT_DIR/recap_common.sh"

NUM_GPUS=8
ROLLING_SAVE_FREQ=2000
recap_init "$@"
DATASET_ROOT="$CFS_ROOT/craig/datasets/yida880/640/y480-merged"
DATASET_REPO_ID="local/$(basename "$DATASET_ROOT")"
# Must match recap_valueTrain.sh's output directory.
VALUE_CHECKPOINT="$CFS_ROOT/yuzhang/outputs/1015_rl_valueTrain_pistar06_yida880-640-y480-merged_8gpu"
OUTPUT_DIR="$CFS_ROOT/yuzhang/outputs/1015_rl_valueInfer_pistar06_yida880-640-y480-merged_8gpu"
LOG_DIR="${OUTPUT_DIR}_logs"
STAGE=recap_valueInfer

echo "标注数据集输出：${DATASET_ROOT}_RLMarked"
CMD=(
    "${LAUNCHER[@]}" -m lerobot.scripts.lerobot_value_infer
    "--dataset.repo_id=$DATASET_REPO_ID"
    "--dataset.root=$DATASET_ROOT"
    # Match the fallback used to construct value-training targets.
    --dataset.default_success=success
    "--inference.checkpoint_path=$VALUE_CHECKPOINT"
    --inference.checkpoint_ref=last
    --runtime.device=cuda
    --runtime.batch_size=64
    --runtime.num_workers=4
    --acp.enable=true
    --acp.n_step=50
    --acp.positive_ratio=0.3
    --acp.c_fail_coef=1.0
    --acp.force_intervention_positive=true
    --acp.value_field=complementary_info.value
    --acp.advantage_field=complementary_info.advantage
    --acp.indicator_field=complementary_info.acp_indicator
    --viz.enable=false
    "--output_dir=$OUTPUT_DIR"
    --job_name=recap_valueInfer_pistar06_yida880
    "${FORWARD_ARGS[@]}"
)
recap_run
