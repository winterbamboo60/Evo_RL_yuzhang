#!/usr/bin/env bash
set -euo pipefail
RECAP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RECAP_SCRIPT_DIR/recap_common.sh"

# Cloud job entry: change stage parameters here, without exporting paths beforehand.
NUM_GPUS=8
ROLLING_SAVE_FREQ=2000
recap_init "$@"
DATASET_ROOT="$CFS_ROOT/craig/datasets/yida880/640/y480-merged"
DATASET_REPO_ID="local/$(basename "$DATASET_ROOT")"
MODELZOO="$CFS_ROOT/yuzhang/modelZoo"
OUTPUT_DIR="$CFS_ROOT/yuzhang/outputs/1015_rl_valueTrain_pistar06_yida880-640-y480-merged_8gpu"
# Keep shell logs outside output_dir: the trainer requires a fresh directory on first launch.
LOG_DIR="${OUTPUT_DIR}_logs"
STAGE=recap_valueTrain

CMD=(
    "${LAUNCHER[@]}" -m lerobot.scripts.lerobot_value_train
    "--dataset.repo_id=$DATASET_REPO_ID"
    "--dataset.root=$DATASET_ROOT"
    --policy.type=pistar06
    --policy.device=cuda
    --policy.dtype=bfloat16
    "--policy.vision_repo_id=$MODELZOO/siglip-so400m-patch14-384"
    "--policy.language_repo_id=$MODELZOO/gemma-3-270m"
    --policy.push_to_hub=false
    --policy.repo_id=local/recap_value
    --targets.success_field=episode_success
    # Preserve explicit labels; treat episodes without a recognized label as successful.
    --targets.default_success=success
    --targets.c_fail_coef=1.0
    --batch_size=16
    --accelerator.gradient_accumulation.steps=1
    --accelerator.mixed_precision=bf16
    --steps=8000
    --save_freq=2000
    "--rolling_save_freq=$ROLLING_SAVE_FREQ"
    --num_workers=4
    --env_eval_freq=0
    "--output_dir=$OUTPUT_DIR"
    --job_name=recap_valueTrain_pistar06_yida880
    --wandb.enable=false
    --tensorboard.enable=true
    "--tensorboard.log_dir=$OUTPUT_DIR/tensorboard"
    "${FORWARD_ARGS[@]}"
)
recap_run
