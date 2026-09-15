#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

ENV_ROOT="${EVORL_ENV_ROOT:-/home/lenovo/code/envs/evo_0911}"
PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
OUTPUT_DIR="${DEFAULT_PROJECT_ROOT}/artifacts/evo_0911"
ARCHIVE_PATH=""
KEEP_WORK=false
SKIP_SMOKE_TEST=false
WORK_DIR=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/ci/env/pack_evo_0911.sh [options]

Options:
  --env-root PATH       Source uv/venv environment.
  --project-root PATH   Current LeRobot source tree to bundle.
  --output-dir PATH     Artifact directory.
  --archive PATH        Exact .tar.gz output path; overrides --output-dir.
  --keep-work           Keep the temporary staging directory for diagnosis.
  --skip-smoke-test     Skip test extraction (not recommended).
  -h, --help            Show this help.

The source environment and project tree are never modified. venv-pack is run
through an existing command or an isolated `uvx --from venv-pack==0.2.0` tool.
EOF
}

cleanup() {
    if [[ -z "${WORK_DIR}" || ! -d "${WORK_DIR}" ]]; then
        return
    fi
    if [[ "${KEEP_WORK}" == true ]]; then
        echo "[pack] Keeping work directory: ${WORK_DIR}"
        return
    fi
    case "${WORK_DIR}" in
        */.evo_0911.pack.*) rm -rf -- "${WORK_DIR}" ;;
        *) echo "Refusing to clean unexpected work path: ${WORK_DIR}" >&2 ;;
    esac
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-root) ENV_ROOT="${2:?--env-root requires a value}"; shift 2 ;;
        --project-root) PROJECT_ROOT="${2:?--project-root requires a value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
        --archive) ARCHIVE_PATH="${2:?--archive requires a value}"; shift 2 ;;
        --keep-work) KEEP_WORK=true; shift ;;
        --skip-smoke-test) SKIP_SMOKE_TEST=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ENV_ROOT="$(realpath -e -- "${ENV_ROOT}")"
PROJECT_ROOT="$(realpath -e -- "${PROJECT_ROOT}")"
OUTPUT_DIR="$(realpath -m -- "${OUTPUT_DIR}")"
SOURCE_PYTHON="${ENV_ROOT}/bin/python"

[[ -f "${ENV_ROOT}/pyvenv.cfg" ]] || { echo "Not a venv: ${ENV_ROOT}" >&2; exit 1; }
[[ -x "${SOURCE_PYTHON}" ]] || { echo "Missing environment Python: ${SOURCE_PYTHON}" >&2; exit 1; }
[[ -f "${PROJECT_ROOT}/pyproject.toml" && -d "${PROJECT_ROOT}/src/lerobot" ]] || {
    echo "Not a LeRobot project: ${PROJECT_ROOT}" >&2
    exit 1
}
for command in cp git sha256sum tar; do
    command -v "${command}" >/dev/null || { echo "Missing command: ${command}" >&2; exit 1; }
done

PYTHON_VERSION="$(PYTHONDONTWRITEBYTECODE=1 "${SOURCE_PYTHON}" -c 'import platform; print(platform.python_version())')"
[[ "${PYTHON_VERSION}" == 3.12.* ]] || {
    echo "Expected Python 3.12, found ${PYTHON_VERSION}" >&2
    exit 1
}
echo "[pack] Checking source environment"
PYTHONDONTWRITEBYTECODE=1 "${SOURCE_PYTHON}" -m pip check

mkdir -p -- "${OUTPUT_DIR}"
if [[ -z "${ARCHIVE_PATH}" ]]; then
    GIT_REV="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo no-git)"
    if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null)" ]]; then
        GIT_REV="${GIT_REV}-dirty"
    fi
    ARCHIVE_PATH="${OUTPUT_DIR}/evo_0911-$(date '+%Y%m%d-%H%M%S')-${GIT_REV}-ubuntu24.04-x86_64-py312-cu128.tar.gz"
else
    ARCHIVE_PATH="$(realpath -m -- "${ARCHIVE_PATH}")"
    [[ "${ARCHIVE_PATH}" == *.tar.gz ]] || { echo "--archive must end in .tar.gz" >&2; exit 2; }
    mkdir -p -- "$(dirname -- "${ARCHIVE_PATH}")"
fi
[[ ! -e "${ARCHIVE_PATH}" ]] || { echo "Refusing to overwrite: ${ARCHIVE_PATH}" >&2; exit 1; }

WORK_PARENT="$(dirname -- "${ARCHIVE_PATH}")"
WORK_DIR="$(mktemp -d "${WORK_PARENT}/.evo_0911.pack.XXXXXX")"
STAGE_ENV="${WORK_DIR}/env"
SOURCE_COPY="${WORK_DIR}/source"
WHEEL_DIR="${WORK_DIR}/wheel"
SMOKE_ENV="${WORK_DIR}/smoke"
mkdir -p -- "${STAGE_ENV}" "${SOURCE_COPY}" "${WHEEL_DIR}"

ENV_KB="$(du -sk -- "${ENV_ROOT}" | awk '{print $1}')"
AVAILABLE_KB="$(df -Pk -- "${WORK_PARENT}" | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((ENV_KB * 2 + 1048576))"
if (( AVAILABLE_KB < REQUIRED_KB )); then
    echo "Insufficient free space: need about ${REQUIRED_KB} KiB, have ${AVAILABLE_KB} KiB" >&2
    exit 1
fi

echo "[pack] Copying ${ENV_ROOT} to isolated staging area"
cp -a -- "${ENV_ROOT}/." "${STAGE_ENV}/"

echo "[pack] Snapshotting the current project worktree"
tar -C "${PROJECT_ROOT}" \
    --exclude='./.git' \
    --exclude='./.venv' \
    --exclude='./.pytest_cache' \
    --exclude='./.ruff_cache' \
    --exclude='./artifacts' \
    --exclude='./output' \
    --exclude='./outputs' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    -cf - . | tar -C "${SOURCE_COPY}" -xf -

STAGE_PYTHON="${STAGE_ENV}/bin/python"
echo "[pack] Building a non-editable LeRobot wheel from the worktree snapshot"
PIP_DISABLE_PIP_VERSION_CHECK=1 "${STAGE_PYTHON}" -m pip wheel \
    --no-deps \
    --no-build-isolation \
    --wheel-dir "${WHEEL_DIR}" \
    "${SOURCE_COPY}"

mapfile -t LEROBOT_WHEELS < <(find "${WHEEL_DIR}" -maxdepth 1 -type f -name 'lerobot-*.whl' -print)
[[ ${#LEROBOT_WHEELS[@]} -eq 1 ]] || {
    echo "Expected exactly one LeRobot wheel, found ${#LEROBOT_WHEELS[@]}" >&2
    exit 1
}
LEROBOT_WHEEL="${LEROBOT_WHEELS[0]}"

echo "[pack] Replacing the editable LeRobot install only inside staging"
PIP_DISABLE_PIP_VERSION_CHECK=1 "${STAGE_PYTHON}" -m pip uninstall -y lerobot
PIP_DISABLE_PIP_VERSION_CHECK=1 "${STAGE_PYTHON}" -m pip install \
    --no-deps \
    --no-build-isolation \
    --force-reinstall \
    "${LEROBOT_WHEEL}"
"${STAGE_PYTHON}" -m pip check

# The copied entry points still contain ENV_ROOT in their shebangs. venv-pack
# rewrites only shebangs that point at the prefix it is currently packing, so
# first make those shebangs refer to STAGE_ENV. Unsupported non-POSIX uv
# activators are removed; venv-pack regenerates activate/csh/fish itself.
echo "[pack] Normalizing staged entry-point shebangs"
find "${STAGE_ENV}/bin" -maxdepth 1 -type f \
    \( -name '*.bat' -o -name '*.ps1' -o -name 'activate.nu' \
       -o -name 'activate.xsh' -o -name 'activate_this.py' \) -delete
PYTHONDONTWRITEBYTECODE=1 "${STAGE_PYTHON}" - \
    "${ENV_ROOT}" "${STAGE_ENV}" <<'PY'
import sys
from pathlib import Path

old_prefix = sys.argv[1].encode()
stage_prefix = sys.argv[2].encode()
generated_activators = {"activate", "activate.csh", "activate.fish"}

for path in (Path(sys.argv[2]) / "bin").iterdir():
    if path.name in generated_activators or path.is_symlink() or not path.is_file():
        continue
    data = path.read_bytes()
    if old_prefix not in data:
        continue
    expected_start = b"#!" + old_prefix + b"/bin/"
    if not data.startswith(expected_start) or data.count(old_prefix) != 1:
        raise SystemExit(f"old prefix is not a single shebang occurrence: {path}")
    path.write_bytes(data.replace(old_prefix, stage_prefix, 1))
PY

LEROBOT_FILE="$(PYTHONDONTWRITEBYTECODE=1 "${STAGE_PYTHON}" -c 'import pathlib, lerobot; print(pathlib.Path(lerobot.__file__).resolve())')"
case "${LEROBOT_FILE}" in
    "${STAGE_ENV}"/*) ;;
    *) echo "Bundled LeRobot still resolves outside staging: ${LEROBOT_FILE}" >&2; exit 1 ;;
esac

echo "[pack] Removing bytecode caches from staging"
find "${STAGE_ENV}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "${STAGE_ENV}" -depth -type d -name '__pycache__' -empty -delete

MANIFEST_DIR="${STAGE_ENV}/share/evo_0911"
mkdir -p -- "${MANIFEST_DIR}"
LEROBOT_VERSION="$(PYTHONDONTWRITEBYTECODE=1 "${STAGE_PYTHON}" -c 'import importlib.metadata; print(importlib.metadata.version("lerobot"))')"
PIP_DISABLE_PIP_VERSION_CHECK=1 "${STAGE_PYTHON}" -m pip freeze --all > "${MANIFEST_DIR}/pip-freeze.txt"
sed -i -E "s#^lerobot @ .*$#lerobot==${LEROBOT_VERSION}#" "${MANIFEST_DIR}/pip-freeze.txt"
git -C "${PROJECT_ROOT}" status --short > "${MANIFEST_DIR}/project-git-status.txt" 2>/dev/null || true
LEROBOT_WHEEL_SHA256="$(sha256sum "${LEROBOT_WHEEL}" | awk '{print $1}')"
printf '%s  %s\n' "${LEROBOT_WHEEL_SHA256}" "$(basename -- "${LEROBOT_WHEEL}")" \
    > "${MANIFEST_DIR}/lerobot-wheel.sha256"
{
    echo "BUNDLE_FORMAT_VERSION=1"
    echo "CREATED_AT_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "SOURCE_ENV=${ENV_ROOT}"
    echo "PROJECT_ROOT=${PROJECT_ROOT}"
    echo "PROJECT_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "PYTHON_VERSION=${PYTHON_VERSION}"
    echo "PLATFORM_ARCH=$(uname -m)"
    echo "OS_ID=$(sed -n 's/^ID=//p' /etc/os-release | tr -d '"' | head -n1)"
    echo "OS_VERSION_ID=$(sed -n 's/^VERSION_ID=//p' /etc/os-release | tr -d '"' | head -n1)"
    echo "GLIBC_VERSION=$(ldd --version | head -n1)"
    echo "BUNDLED_LEROBOT_VERSION=${LEROBOT_VERSION}"
    echo "BUNDLED_LEROBOT_WHEEL_SHA256=${LEROBOT_WHEEL_SHA256}"
    if command -v nvidia-smi >/dev/null; then
        echo "NVIDIA=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | paste -sd ';' -)"
    fi
} > "${MANIFEST_DIR}/environment.txt"

if command -v venv-pack >/dev/null; then
    VENV_PACK_CMD=(venv-pack)
elif command -v uvx >/dev/null; then
    VENV_PACK_CMD=(uvx --from venv-pack==0.2.0 venv-pack)
else
    echo "Neither venv-pack nor uvx is available" >&2
    exit 1
fi

PARTIAL_ARCHIVE="${WORK_DIR}/evo_0911.partial.tar.gz"
echo "[pack] Creating relocatable archive with venv-pack"
TMPDIR="${WORK_DIR}" "${VENV_PACK_CMD[@]}" -p "${STAGE_ENV}" -o "${PARTIAL_ARCHIVE}"

if [[ "${SKIP_SMOKE_TEST}" != true ]]; then
    echo "[pack] Smoke-testing extraction at a different path"
    mkdir -p -- "${SMOKE_ENV}"
    tar -xzf "${PARTIAL_ARCHIVE}" -C "${SMOKE_ENV}"
    for stale_prefix in "${ENV_ROOT}" "${STAGE_ENV}"; do
        if grep -RIlF -- "${stale_prefix}" "${SMOKE_ENV}/bin" >/dev/null 2>&1; then
            echo "Stale prefix remains in packed bin scripts: ${stale_prefix}" >&2
            exit 1
        fi
    done
    PATH="${SMOKE_ENV}/bin:${PATH}" "${SMOKE_ENV}/bin/python" \
        "${SCRIPT_DIR}/verify_evo_0911.py" \
        --env-root "${SMOKE_ENV}" \
        --expect-lerobot-root "${SMOKE_ENV}"
fi

mv -- "${PARTIAL_ARCHIVE}" "${ARCHIVE_PATH}"
chmod 0644 -- "${ARCHIVE_PATH}"
ARCHIVE_BASENAME="$(basename -- "${ARCHIVE_PATH}")"
(
    cd -- "$(dirname -- "${ARCHIVE_PATH}")"
    sha256sum "${ARCHIVE_BASENAME}" > "${ARCHIVE_BASENAME}.sha256"
)
MANIFEST_PATH="${ARCHIVE_PATH%.tar.gz}.manifest.txt"
cp -- "${MANIFEST_DIR}/environment.txt" "${MANIFEST_PATH}"
chmod 0644 -- "${ARCHIVE_PATH}.sha256" "${MANIFEST_PATH}"

ARCHIVE_SHA256="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
cat <<EOF
[pack] Completed
archive:  ${ARCHIVE_PATH}
sha256:   ${ARCHIVE_SHA256}
checksum: ${ARCHIVE_PATH}.sha256
manifest: ${MANIFEST_PATH}
EOF
