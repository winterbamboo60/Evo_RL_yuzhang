#!/usr/bin/env bash
# EvoRL PI05-RLT online actor/learner launcher.
#
# Usage:
#   PIPER_ONLINE_RL_LEARNER_CONFIG=/path/learner.json bash scripts/RL_online.sh learner
#   PIPER_ONLINE_RL_ACTOR_CONFIG=/path/actor.json bash scripts/RL_online.sh actor
#   bash scripts/RL_online.sh both

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${EVORL_ENV_ROOT:-/home/lenovo/code/envs/evo_0911}"
PYTHON_BIN="${ENV_ROOT}/bin/python"
MODE="${1:-both}"
CONFIG_ROOT="${REPO_ROOT}/src/lerobot/onlineRL_evoRL/configs"
DEFAULT_ACTOR_CONFIG="${CONFIG_ROOT}/actor/Actor_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json"
DEFAULT_LEARNER_CONFIG="${CONFIG_ROOT}/learner/Leanrer_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json"
SHARED_CONFIG="${PIPER_ONLINE_RL_CONFIG:-}"
ACTOR_CONFIG="${PIPER_ONLINE_RL_ACTOR_CONFIG:-${SHARED_CONFIG:-$DEFAULT_ACTOR_CONFIG}}"
LEARNER_CONFIG="${PIPER_ONLINE_RL_LEARNER_CONFIG:-${SHARED_CONFIG:-$DEFAULT_LEARNER_CONFIG}}"

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  echo "Usage: bash scripts/RL_online.sh [learner|actor|both] [actor draccus overrides...]"
  echo "Actor config:   PIPER_ONLINE_RL_ACTOR_CONFIG (default: ${DEFAULT_ACTOR_CONFIG})"
  echo "Learner config: PIPER_ONLINE_RL_LEARNER_CONFIG (default: ${DEFAULT_LEARNER_CONFIG})"
  echo "PIPER_ONLINE_RL_CONFIG remains a compatibility override for both paths."
  echo "In 'both' mode, trailing overrides are applied only to actor_new."
  echo "Both processes start from lerobot.onlineRL_evoRL; shared RL/transport primitives remain reusable."
  exit 0
fi

case "$MODE" in
  learner|actor|both) shift || true ;;
  --*) MODE="both" ;;
  *) echo "Unknown mode: ${MODE}. Expected learner, actor, or both." >&2; exit 2 ;;
esac

if [[ ! -f "$ACTOR_CONFIG" ]]; then
  echo "Online RL actor config does not exist: ${ACTOR_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "$LEARNER_CONFIG" ]]; then
  echo "Online RL learner config does not exist: ${LEARNER_CONFIG}" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Fused EvoRL Python is missing: ${PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
LOG_DIR="${EVORL_LOG_DIR:-${REPO_ROOT}/outputs/online_rl/logs}"
mkdir -p "$LOG_DIR"

check_python_deps() {
  "$PYTHON_BIN" - <<'PY_CHECK'
missing = []
for module_name in ("draccus", "grpc", "torch"):
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        missing.append(module_name)
if missing:
    raise SystemExit("Missing online RL dependencies in evo_0911: " + ", ".join(missing))
PY_CHECK
}

run_preflight() {
  "$PYTHON_BIN" -m lerobot.onlineRL_evoRL.preflight \
    --learner-config "$LEARNER_CONFIG" \
    --actor-config "$ACTOR_CONFIG" \
    --mode "$MODE"
}

run_learner() {
  "$PYTHON_BIN" -m lerobot.onlineRL_evoRL.learner --config_path "$LEARNER_CONFIG" "$@"
}

run_actor() {
  "$PYTHON_BIN" -m lerobot.onlineRL_evoRL.actor_new --config_path "$ACTOR_CONFIG" "$@"
}

check_python_deps
run_preflight
case "$MODE" in
  learner) run_learner "$@" ;;
  actor) run_actor "$@" ;;
  both)
    # Actor-only overrides are not forwarded to the learner's dedicated JSON.
    run_learner >"${LOG_DIR}/learner.log" 2>&1 &
    learner_pid=$!
    cleanup() {
      if kill -0 "$learner_pid" >/dev/null 2>&1; then
        kill "$learner_pid" >/dev/null 2>&1 || true
      fi
    }
    trap cleanup EXIT INT TERM
    # The optional offline feature/pretraining phase owns the shared GPU. Do
    # not construct the Actor's PI05 runtime until learner has released CUDA
    # and started its transport service.
    until grep -q "independent EvoRL gRPC server started" "${LOG_DIR}/learner.log" 2>/dev/null; do
      if ! kill -0 "$learner_pid" >/dev/null 2>&1; then
        echo "Learner exited during startup. Last log lines:" >&2
        tail -n 80 "${LOG_DIR}/learner.log" >&2 || true
        wait "$learner_pid"
      fi
      sleep 1
    done
    run_actor "$@" >"${LOG_DIR}/actor.log" 2>&1
    ;;
esac
