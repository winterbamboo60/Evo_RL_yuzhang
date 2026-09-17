#!/usr/bin/env bash
# Standalone PI05-RLT feature extraction. One torchrun process owns one GPU.
#
# Required:
#   EVORL_EXTRACT_DATASET_ROOT=/path/to/lerobot_dataset
#   EVORL_EXTRACT_OUTPUT_DIR=/path/to/compact_output
#
# Optional:
#   EVORL_EXTRACT_GPUS=4 EVORL_EXTRACT_BATCH_SIZE=32 EVORL_EXTRACT_WORKERS=4
#   EVORL_EXTRACT_POLICY_PATH=/cloud/path/to/pretrained_model

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${EVORL_ENV_ROOT:-/home/lenovo/code/envs/evo_0911}"
PYTHON_BIN="${ENV_ROOT}/bin/python"
TORCHRUN_BIN="${ENV_ROOT}/bin/torchrun"
CONFIG_ROOT="${REPO_ROOT}/src/lerobot/onlineRL_evoRL/configs"
DEFAULT_CONFIG="${CONFIG_ROOT}/learner/Leanrer_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json"

CONFIG="${EVORL_EXTRACT_CONFIG:-$DEFAULT_CONFIG}"
DATASET_ROOT="${EVORL_EXTRACT_DATASET_ROOT:-}"
OUTPUT_DIR="${EVORL_EXTRACT_OUTPUT_DIR:-}"
DATASET_REPO_ID="${EVORL_EXTRACT_DATASET_REPO_ID:-local/offline_feature_extraction}"
NUM_GPUS="${EVORL_EXTRACT_GPUS:-1}"
BATCH_SIZE="${EVORL_EXTRACT_BATCH_SIZE:-8}"
NUM_WORKERS="${EVORL_EXTRACT_WORKERS:-4}"
STRIDE="${EVORL_EXTRACT_STRIDE:-2}"
STORAGE_DTYPE="${EVORL_EXTRACT_STORAGE_DTYPE:-float32}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,12p' "$0"
  "$PYTHON_BIN" -m lerobot.onlineRL_evoRL.extract_offline_features --help
  exit 0
fi
if [[ -z "$DATASET_ROOT" || -z "$OUTPUT_DIR" ]]; then
  echo "Set EVORL_EXTRACT_DATASET_ROOT and EVORL_EXTRACT_OUTPUT_DIR." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
args=(
  --config-path "$CONFIG"
  --dataset-root "$DATASET_ROOT"
  --dataset-repo-id "$DATASET_REPO_ID"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --stride "$STRIDE"
  --storage-dtype "$STORAGE_DTYPE"
)
if [[ -n "${EVORL_EXTRACT_POLICY_PATH:-}" ]]; then
  args+=(--policy-path "$EVORL_EXTRACT_POLICY_PATH")
fi
if [[ -n "${EVORL_EXTRACT_FEATURE_MODEL_PATH:-}" ]]; then
  args+=(--feature-model-path "$EVORL_EXTRACT_FEATURE_MODEL_PATH")
fi

if (( NUM_GPUS > 1 )); then
  if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "torchrun is missing: $TORCHRUN_BIN" >&2
    exit 2
  fi
  "$TORCHRUN_BIN" --standalone --nproc-per-node "$NUM_GPUS" \
    -m lerobot.onlineRL_evoRL.extract_offline_features "${args[@]}" "$@"
else
  "$PYTHON_BIN" -m lerobot.onlineRL_evoRL.extract_offline_features "${args[@]}" "$@"
fi
