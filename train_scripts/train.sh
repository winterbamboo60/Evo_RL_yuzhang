#!/usr/bin/env bash
set -e

ROLLING_SAVE_FREQ=2000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rolling_save_freq)
            if [[ $# -lt 2 || ! "$2" =~ ^(0|[1-9][0-9]*)$ ]]; then
                echo "[错误] --rolling_save_freq 必须是非负整数（0 表示关闭）" >&2
                exit 2
            fi
            ROLLING_SAVE_FREQ="$2"
            shift 2 ;;
        -h|--help)
            echo "Usage: bash train_scripts/train.sh [--rolling_save_freq N]"
            echo "每 N 步保存最新 checkpoint（默认 2000；0 关闭），保留每 10000 步 checkpoint。"
            exit 0 ;;
        *) echo "[错误] 未知参数：$1" >&2; exit 2 ;;
    esac
done

CFS_ROOT=""
for candidate in /mnt/cfs/popa97 /mnt/cfs/0z9lxh; do
    if [[ -d "$candidate/yuzhang/projects/Evo_RL_yuzhang" \
       && -f "$candidate/yuzhang/env/evo_0911_new/bin/activate" ]]; then
        CFS_ROOT="$candidate"
        break
    fi
done

if [[ -z "$CFS_ROOT" ]]; then
    echo "[错误] 找不到包含项目和训练环境的共享盘" >&2
    exit 2
fi

# 兼容模型配置、checkpoint 和环境中保存的开发机绝对路径。
LEGACY_CFS_ROOT=/mnt/cfs/0z9lxh
if [[ "$CFS_ROOT" != "$LEGACY_CFS_ROOT" ]]; then
    if [[ ! -e "$LEGACY_CFS_ROOT" && ! -L "$LEGACY_CFS_ROOT" ]]; then
        if ! ln -s "$CFS_ROOT" "$LEGACY_CFS_ROOT"; then
            echo "[错误] 无法创建路径别名：$LEGACY_CFS_ROOT -> $CFS_ROOT" >&2
            exit 2
        fi
    fi

    # 保留已有目录或链接，只接受指向同一共享盘的路径。
    if [[ ! "$LEGACY_CFS_ROOT/yuzhang" -ef "$CFS_ROOT/yuzhang" ]]; then
        echo "[错误] $LEGACY_CFS_ROOT 已存在，但未指向当前共享盘 $CFS_ROOT" >&2
        exit 2
    fi
    echo "路径兼容：$LEGACY_CFS_ROOT -> $CFS_ROOT"
fi

export EVORL_PROJECT_ROOT="${EVORL_PROJECT_ROOT:-$CFS_ROOT/yuzhang/projects/Evo_RL_yuzhang}"
export EVORL_ENV_ROOT="${EVORL_ENV_ROOT:-$CFS_ROOT/yuzhang/env/evo_0911_new}"

if [[ ! -d "$EVORL_PROJECT_ROOT" ]]; then
    echo "[错误] 训练项目目录不存在或不是目录：$EVORL_PROJECT_ROOT" >&2
    exit 2
fi

if [[ ! -d "$EVORL_ENV_ROOT" ]]; then
    echo "[错误] 训练环境目录不存在或不是目录：$EVORL_ENV_ROOT" >&2
    exit 2
fi

source "$EVORL_ENV_ROOT/bin/activate"

# activate 可能记录开发机上的旧路径，使用当前实际可访问的路径。
export VIRTUAL_ENV="$EVORL_ENV_ROOT"
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r

if ! "$VIRTUAL_ENV/bin/python" --version; then
    echo "[错误] 虚拟环境 Python 无法执行，请检查软链接目标和镜像：$VIRTUAL_ENV/bin/python" >&2
    exit 2
firolling_save_freq

echo "共享盘：$CFS_ROOT"
echo "训练项目：$EVORL_PROJECT_ROOT"
echo "训练环境：$EVORL_ENV_ROOT"
cd "$EVORL_PROJECT_ROOT"

# bash scripts/ci/env/reinstall_lerobot.sh \
#   --env-root "$EVORL_ENV_ROOT" \
#   --project-root "$EVORL_PROJECT_ROOT"

echo "Start RL training..."

command -v python

env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/RL_train.sh \
    --DATASET_ROOT "$CFS_ROOT/yuzhang/datasets/1012_pourWater_60ml_merge" \
    --history_pretrained_path "$CFS_ROOT/yuzhang/modelZoo/pi05base" \
    --OUTPUT_DIR "$CFS_ROOT/yuzhang/outputs/1012_pi05_rlt_sft_1012_pourWater_60ml_merge_8gpu" \
    --policy_type pi05 \
    --num_gpus 8 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --steps 50000 \
    --save_freq 10000 \
    --rolling_save_freq "$ROLLING_SAVE_FREQ" \
    --train_expert_only false \
    --tensorboard \
    -- \
    --rename_map='{"observation.images.left_top":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}' \
    --policy.compile_model=false
