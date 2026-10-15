#!/usr/bin/env bash
# Shared cloud environment setup. Stage-specific paths and parameters live in each entry script.

recap_init() {
    DRY_RUN=false
    FORWARD_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry_run) DRY_RUN=true; shift ;;
            --num_gpus)
                if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                    echo "[错误] --num_gpus 必须是正整数" >&2
                    return 2
                fi
                NUM_GPUS="$2"; shift 2 ;;
            --rolling_save_freq)
                if [[ $# -lt 2 || ! "$2" =~ ^(0|[1-9][0-9]*)$ ]]; then
                    echo "[错误] --rolling_save_freq 必须是非负整数" >&2
                    return 2
                fi
                ROLLING_SAVE_FREQ="$2"; shift 2 ;;
            --) shift; FORWARD_ARGS=("$@"); break ;;
            -h|--help)
                echo "Usage: bash $0 [--num_gpus N] [--rolling_save_freq N] [--dry_run] [-- CLI_OVERRIDES...]"
                echo "默认 8 GPU；路径和阶段参数直接在入口脚本中修改。"
                echo "--dry_run 仅显示路径和命令，不激活环境、不创建别名、不启动训练。"
                exit 0 ;;
            *) echo "[错误] 未知参数：$1；底层参数请放在 -- 后" >&2; return 2 ;;
        esac
    done

    # Prefer the mount containing these scripts; support arbitrary cloud mount names.
    local script_project candidate
    script_project="$(cd -- "$RECAP_SCRIPT_DIR/../.." && pwd -P)"
    if [[ -z "${CFS_ROOT:-}" ]]; then
        for candidate in "${script_project%/yuzhang/projects/Evo_RL_yuzhang}" \
            /mnt/cfs/popa97 /mnt/cfs/0z9lxh /mnt/cfs/*; do
            if [[ -d "$candidate/yuzhang/projects/Evo_RL_yuzhang" \
               && -f "$candidate/yuzhang/env/evo_0911_new/bin/activate" ]]; then
                CFS_ROOT="$candidate"
                break
            fi
        done
    fi
    if [[ -z "${CFS_ROOT:-}" || ! -d "$CFS_ROOT" ]]; then
        echo "[错误] 找不到共享盘上的项目和 evo_0911_new 环境" >&2
        return 2
    fi
    CFS_ROOT="${CFS_ROOT%/}"
    export EVORL_PROJECT_ROOT="$CFS_ROOT/yuzhang/projects/Evo_RL_yuzhang"
    export EVORL_ENV_ROOT="$CFS_ROOT/yuzhang/env/evo_0911_new"
    if [[ ! -d "$EVORL_PROJECT_ROOT/src/lerobot" \
       || ! -f "$EVORL_ENV_ROOT/bin/activate" ]]; then
        echo "[错误] 共享盘缺少项目源码或训练环境：$CFS_ROOT" >&2
        return 2
    fi
    PYTHON_BIN="$EVORL_ENV_ROOT/bin/python"
    LAUNCHER=("$PYTHON_BIN")
    if (( NUM_GPUS > 1 )); then
        LAUNCHER+=(-m torch.distributed.run --standalone "--nproc-per-node=$NUM_GPUS")
    fi
}

recap_run() {
    echo "共享盘：$CFS_ROOT"
    echo "项目：$EVORL_PROJECT_ROOT"
    echo "环境：$EVORL_ENV_ROOT"
    echo "GPU 数：$NUM_GPUS；CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-全部}"
    echo "数据集：$DATASET_ROOT"
    printf '[执行命令] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi

    # Same compatibility alias as existing cloud training scripts: saved model/tokenizer
    # configs and the packed virtualenv may still refer to the development mount.
    local legacy_root=/mnt/cfs/0z9lxh
    if [[ "$CFS_ROOT" != "$legacy_root" ]]; then
        if [[ ! -e "$legacy_root" && ! -L "$legacy_root" ]]; then
            ln -s -- "$CFS_ROOT" "$legacy_root"
        fi
        if [[ ! "$legacy_root/yuzhang" -ef "$CFS_ROOT/yuzhang" ]]; then
            echo "[错误] $legacy_root 已存在且未指向当前共享盘 $CFS_ROOT" >&2
            return 2
        fi
    fi

    # Some activate scripts use unset variables; keep nounset enabled for our own code.
    set +u
    source "$EVORL_ENV_ROOT/bin/activate"
    set -u
    export VIRTUAL_ENV="$EVORL_ENV_ROOT"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    export PYTHONPATH="$EVORL_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1
    hash -r
    "$PYTHON_BIN" --version
    cd "$EVORL_PROJECT_ROOT"
    mkdir -p "$LOG_DIR"
    "${CMD[@]}" 2>&1 | tee -a "$LOG_DIR/$STAGE.log"
}
