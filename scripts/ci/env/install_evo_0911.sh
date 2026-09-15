#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ARCHIVE=""
DEST=""
CHECKSUM=""
PROJECT_ROOT=""
INSTALL_EDITABLE=false
REQUIRE_CUDA=false
ALLOW_HOST_MISMATCH=false
TEMP_DEST=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/ci/env/install_evo_0911.sh --archive FILE --dest DIR [options]

Options:
  --checksum FILE       SHA256 sidecar; defaults to FILE.sha256 when present.
  --project-root PATH   LeRobot checkout used with --editable.
  --editable            Reinstall LeRobot editable after extraction.
  --require-cuda        Fail verification when CUDA is unavailable.
  --allow-host-mismatch Continue on a non-Ubuntu-24.04/x86_64 host (unsafe).
  -h, --help            Show this help.

The destination must not already exist. This script never runs conda-unpack.
EOF
}

cleanup() {
    if [[ -n "${TEMP_DEST}" && -d "${TEMP_DEST}" ]]; then
        case "${TEMP_DEST}" in
            */.evo_0911.install.*) rm -rf -- "${TEMP_DEST}" ;;
            *) echo "Refusing to clean unexpected temporary path: ${TEMP_DEST}" >&2 ;;
        esac
    fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive) ARCHIVE="${2:?--archive requires a value}"; shift 2 ;;
        --dest) DEST="${2:?--dest requires a value}"; shift 2 ;;
        --checksum) CHECKSUM="${2:?--checksum requires a value}"; shift 2 ;;
        --project-root) PROJECT_ROOT="${2:?--project-root requires a value}"; shift 2 ;;
        --editable) INSTALL_EDITABLE=true; shift ;;
        --require-cuda) REQUIRE_CUDA=true; shift ;;
        --allow-host-mismatch) ALLOW_HOST_MISMATCH=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${ARCHIVE}" ]] || { echo "--archive is required" >&2; exit 2; }
[[ -n "${DEST}" ]] || { echo "--dest is required" >&2; exit 2; }
ARCHIVE="$(realpath -e -- "${ARCHIVE}")"
DEST="$(realpath -m -- "${DEST}")"

[[ -f "${ARCHIVE}" ]] || { echo "Archive does not exist: ${ARCHIVE}" >&2; exit 1; }
[[ ! -e "${DEST}" ]] || { echo "Destination already exists: ${DEST}" >&2; exit 1; }

if [[ -z "${CHECKSUM}" && -f "${ARCHIVE}.sha256" ]]; then
    CHECKSUM="${ARCHIVE}.sha256"
fi
if [[ -n "${CHECKSUM}" ]]; then
    CHECKSUM="$(realpath -e -- "${CHECKSUM}")"
    EXPECTED_SHA256="$(awk 'NF {print $1; exit}' "${CHECKSUM}")"
    ACTUAL_SHA256="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
    [[ "${EXPECTED_SHA256}" == "${ACTUAL_SHA256}" ]] || {
        echo "SHA256 mismatch for ${ARCHIVE}" >&2
        exit 1
    }
    echo "[install] SHA256 verified: ${ACTUAL_SHA256}"
else
    echo "[install] Warning: no SHA256 sidecar was provided or found" >&2
fi

HOST_ARCH="$(uname -m)"
HOST_OS_ID="$(sed -n 's/^ID=//p' /etc/os-release 2>/dev/null | tr -d '"' | head -n1)"
HOST_OS_VERSION="$(sed -n 's/^VERSION_ID=//p' /etc/os-release 2>/dev/null | tr -d '"' | head -n1)"
if [[ "${HOST_ARCH}" != "x86_64" || "${HOST_OS_ID}" != "ubuntu" || "${HOST_OS_VERSION}" != "24.04" ]]; then
    if [[ "${ALLOW_HOST_MISMATCH}" != true ]]; then
        echo "Unsupported host: ${HOST_OS_ID} ${HOST_OS_VERSION} ${HOST_ARCH}" >&2
        echo "Expected Ubuntu 24.04 x86_64; pass --allow-host-mismatch only after ABI review." >&2
        exit 1
    fi
    echo "[install] Warning: continuing on host ${HOST_OS_ID} ${HOST_OS_VERSION} ${HOST_ARCH}" >&2
fi
[[ -x /usr/bin/python3.12 ]] || {
    echo "Required base interpreter is missing: /usr/bin/python3.12" >&2
    exit 1
}

/usr/bin/python3.12 - "${ARCHIVE}" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

archive = sys.argv[1]
with tarfile.open(archive, "r:gz") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            link = PurePosixPath(member.linkname)
            allowed_python_link = (
                member.name in {"bin/python", "bin/python3.12"}
                and member.linkname == "/usr/bin/python3.12"
            )
            if (link.is_absolute() or ".." in link.parts) and not allowed_python_link:
                raise SystemExit(
                    f"unsafe archive link: {member.name} -> {member.linkname}"
                )
PY

DEST_PARENT="$(dirname -- "${DEST}")"
mkdir -p -- "${DEST_PARENT}"
TEMP_DEST="$(mktemp -d "${DEST_PARENT}/.evo_0911.install.XXXXXX")"
echo "[install] Extracting to temporary directory: ${TEMP_DEST}"
tar -xzf "${ARCHIVE}" --no-same-owner --no-same-permissions -C "${TEMP_DEST}"

VERIFY_ARGS=(--env-root "${TEMP_DEST}" --expect-lerobot-root "${TEMP_DEST}")
if [[ "${REQUIRE_CUDA}" == true ]]; then
    VERIFY_ARGS+=(--require-cuda)
fi
PATH="${TEMP_DEST}/bin:${PATH}" "${TEMP_DEST}/bin/python" \
    "${SCRIPT_DIR}/verify_evo_0911.py" "${VERIFY_ARGS[@]}"

mv -- "${TEMP_DEST}" "${DEST}"
TEMP_DEST=""
echo "[install] Environment installed at ${DEST}"

if [[ "${INSTALL_EDITABLE}" == true ]]; then
    [[ -n "${PROJECT_ROOT}" ]] || {
        echo "--project-root is required with --editable" >&2
        exit 2
    }
    REINSTALL_ARGS=(--env-root "${DEST}" --project-root "${PROJECT_ROOT}")
    if [[ "${REQUIRE_CUDA}" == true ]]; then
        REINSTALL_ARGS+=(--require-cuda)
    fi
    bash "${SCRIPT_DIR}/reinstall_lerobot.sh" "${REINSTALL_ARGS[@]}"
fi

cat <<EOF
[install] Ready. Activate it in the current shell with:
export EVORL_ENV_ROOT="${DEST}"
source "\${EVORL_ENV_ROOT}/bin/activate"
EOF
