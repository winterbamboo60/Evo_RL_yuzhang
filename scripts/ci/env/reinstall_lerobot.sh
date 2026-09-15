#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

ENV_ROOT="${EVORL_ENV_ROOT:-/home/lenovo/code/envs/evo_0911}"
PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
REQUIRE_CUDA=false

usage() {
    cat <<'EOF'
Usage:
  bash scripts/ci/env/reinstall_lerobot.sh [options]

Options:
  --env-root PATH       Unpacked Evo environment.
  --project-root PATH   LeRobot checkout to install editable.
  --require-cuda        Fail verification when CUDA is unavailable.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-root) ENV_ROOT="${2:?--env-root requires a value}"; shift 2 ;;
        --project-root) PROJECT_ROOT="${2:?--project-root requires a value}"; shift 2 ;;
        --require-cuda) REQUIRE_CUDA=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ENV_ROOT="$(realpath -e -- "${ENV_ROOT}")"
PROJECT_ROOT="$(realpath -e -- "${PROJECT_ROOT}")"
PYTHON_BIN="${ENV_ROOT}/bin/python"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing environment Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${PROJECT_ROOT}/pyproject.toml" ]] || {
    echo "Missing pyproject.toml below project root: ${PROJECT_ROOT}" >&2
    exit 1
}
[[ -d "${PROJECT_ROOT}/src/lerobot" ]] || {
    echo "Missing src/lerobot below project root: ${PROJECT_ROOT}" >&2
    exit 1
}

echo "[lerobot] Installing editable source from ${PROJECT_ROOT}"
PIP_DISABLE_PIP_VERSION_CHECK=1 "${PYTHON_BIN}" -m pip install \
    --no-deps \
    --no-build-isolation \
    --force-reinstall \
    --editable "${PROJECT_ROOT}"

VERIFY_ARGS=(
    --env-root "${ENV_ROOT}"
    --expect-lerobot-root "${PROJECT_ROOT}/src"
)
if [[ "${REQUIRE_CUDA}" == true ]]; then
    VERIFY_ARGS+=(--require-cuda)
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_evo_0911.py" "${VERIFY_ARGS[@]}"

echo "[lerobot] Editable reinstall completed. Do not move this environment afterward; re-extract the archive instead."
