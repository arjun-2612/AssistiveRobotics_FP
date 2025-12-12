#!/usr/bin/env bash

set -euo pipefail

# ---------- args ----------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <robot_name> [run_machine]" >&2
  exit 1
fi
ROBOT_NAME="$1"
RUN_MACHINE="${2:-local}"

GITROOT="$(git rev-parse --show-toplevel)"
CFG="${GITROOT}/isaac/source/isaac_extension/simulation/config/${ROBOT_NAME}.yaml"

[[ -f "$CFG" ]] || { echo "Config not found: $CFG" >&2; exit 1; }

# ---------- helpers ----------
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1. Please install it."; exit 1; }; }
need yq; need git

# yq helpers (compatible with mikefarah/yq v4)
yq_val() { yq -r "$1 // \"\"" "$CFG"; }
# Expand $USER if present; default to "" if null
yq_expand_user() { yq -r "(${1} // \"\") | sub(\"\\$USER\"; env(USER))" "$CFG"; }

# ---------- read mode and build the python command (for local runs) ----------
MODE="$(yq_val '.mode')"
TASK="$(yq_val '.task.name')"
NUM_ENVS="$(yq_val '.task.num_envs')"
RESUME="$(yq_val '.task.resume')"
VIDEO="$(yq_val '.visuals.enable')"

build_local_cmd() {
  local -n _cmd=$1
  _cmd=( python )
  if [[ "$MODE" == "train" ]]; then
    _cmd+=( scripts/rsl_rl/train.py --task="$TASK" --num_envs "$NUM_ENVS" --headless )
    if [[ "$VIDEO" == "true" ]]; then
      local VLEN VINT
      VLEN="$(yq_val '.visuals.video_length')"
      VINT="$(yq_val '.visuals.video_interval')"
      _cmd+=( --video --video_length "$VLEN" --video_interval "$VINT" --enable_cameras )
    fi
    if [[ "$RESUME" == "true" ]]; then
      local RF CKPT
      RF="$(yq_val '.task.run_folder')"
      CKPT="$(yq_val '.task.checkpoint')"
      _cmd+=( --resume --load_run "$RF" --checkpoint "$CKPT" )
    fi
  elif [[ "$MODE" == "play" ]]; then
    local RUNF CKPT
    RUNF="$(yq_val '.play.folder')"
    CKPT="$(yq_val '.play.checkpoint')"
    _cmd+=( scripts/rsl_rl/play.py --task="$TASK" --num_envs "$NUM_ENVS" --load_run "$RUNF" --checkpoint "$CKPT" )
  else
    echo "Invalid mode in YAML: $MODE" >&2
    exit 1
  fi
}

# ---------- LOCAL PATH (native) ----------
build_local_cmd CMD
if [[ ${#CMD[@]} -eq 0 ]]; then
  echo "Nothing to run locally." >&2
  exit 1
fi
echo "Running locally (native): ${CMD[*]}"
exec "${CMD[@]}"