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
  elif [[ "$MODE" == "sim" ]]; then
    _cmd=()  # handled separately
  elif [[ "$MODE" == "convert" ]]; then
    local URDF_IN USD_OUT
    URDF_IN="$(yq_val '.convert.urdf_in')"
    USD_OUT="$(yq_val '.convert.usd_out')"
    _cmd+=( scripts/tools/convert_urdf.py "$GITROOT"/"$URDF_IN" "$GITROOT"/"$USD_OUT" )
  else
    echo "Invalid mode in YAML: $MODE" >&2
    exit 1
  fi
}

build_cluster_cmd() {
  local -n _cmd=$1
  _cmd=( python )
  if [[ "$MODE" == "train" ]]; then
    _cmd+=( /workspace/scripts/rsl_rl/train.py --task="$TASK" --num_envs "$NUM_ENVS" --headless )
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
    _cmd+=( /workspace/scripts/rsl_rl/play.py --task="$TASK" --num_envs "$NUM_ENVS" --load_run /workspace/logs/rsl_rl/"$ROBOT_NAME"_rl/"$RUNF" --checkpoint /workspace/logs/rsl_rl/"$ROBOT_NAME"_rl/"$RUNF"/"$CKPT" --headless )
  elif [[ "$MODE" == "sim" ]]; then
    _cmd=()  # handled separately
  elif [[ "$MODE" == "convert" ]]; then
    local URDF_IN USD_OUT
    URDF_IN="$(yq_val '.container.convert.urdf_in')"
    USD_OUT="$(yq_val '.container.convert.usd_out')"
    _cmd+=( /workspace/scripts/tools/convert_urdf.py "$GITROOT"/"$URDF_IN" "$GITROOT"/"$USD_OUT" )
  else
    echo "Invalid mode in YAML: $MODE" >&2
    exit 1
  fi
}

# ---------- LOCAL PATH (native) ----------
if [[ "$RUN_MACHINE" != "cluster" || ( "$MODE" != "train" && "$MODE" != "play" ) ]]; then
  # play/sim/convert always come here; train comes here when not cluster
  if [[ "$MODE" == "sim" ]]; then
    SIMPATH="$GIT_ROOT/isaac/source/isaac_extension/simulation/isaacsim_scripts"
    mapfile -t SCRIPT_FILES < <(yq -r '.sim.scripts[]' "$CFG")
    DESTPATH="$(yq_val '.sim.env_path')"
    for script in "${SCRIPT_FILES[@]}"; do
      SRC_PATH="$SIMPATH/$ROBOT_NAME/extensions/$script"
      cp "$SRC_PATH" "$DESTPATH"
    done
    cp "$SIMPATH/__init__.py" "$DESTPATH"
    echo "Successfully copied sim scripts to $DESTPATH"
    exit 0
  fi

  build_local_cmd CMD
  if [[ ${#CMD[@]} -eq 0 ]]; then
    echo "Nothing to run locally." >&2
    exit 1
  fi
  echo "Running locally (native): ${CMD[*]}"
  exec "${CMD[@]}"
fi

# ---------- CLUSTER PATH (containers + sbatch) — ONLY for MODE==train ----------
need docker

# container/hpc/slurm from YAML
DOCKER_USER="$(yq_val '.container.docker_user')"
IMAGE_NAME="$(yq_val '.container.image_name')"

HPC_USER="$(yq_expand_user '.hpc.user')"; [[ -z "$HPC_USER" ]] && HPC_USER="$USER"
HPC_HOST="$(yq_val '.hpc.host')"
HPC_CONTAINER_DIR="$(yq_val '.hpc.container_dir')"     # e.g. /projects/.../containers
HPC_PROJECT_DIR="$(yq_val '.hpc.project_dir')"       # e.g. /projects/sslab_isaaclab
HPC_LOGS_BASE="$(yq_val '.hpc.run_dir_base')"          # e.g. /projects/.../sslab_isaaclab/logs
APPT_CACHE="$(yq_val '.hpc.apptainer_cache')"          # e.g. /scratch/$USER/appt_cache
APPT_TMP="$(yq_val '.hpc.apptainer_tmp')"              # e.g. /scratch/$USER/appt_tmp

JOB_NAME="$(yq_val '.slurm.job_name')"
NODES="$(yq_val '.slurm.nodes')"
TASKS_PER_NODE="$(yq_val '.slurm.tasks_per_node')"
CPUS_PER_TASK="$(yq_val '.slurm.cpus_per_task')"
GRES="$(yq_val '.slurm.gres')"
MEM="$(yq_val '.slurm.mem')"
PARTITION="$(yq_val '.slurm.partition')"
TIME_LIMIT="$(yq_val '.slurm.time')"
OUT_FILE="$(yq_val '.slurm.output')"
ERR_FILE="$(yq_val '.slurm.error')"
MAIL_USER="$(yq_expand_user '.slurm.mail_user')"
MAIL_TYPE="$(yq_val '.slurm.mail_type')"

# # Build & push image locally only when training. Playing uses the image on cluster
if [[ $MODE == "train" ]]; then
  if [[ $RESUME == "true" ]]; then
    TAG="$(yq_val '.container.image_tag')"
    IMAGE_FQN="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"
  else
    # tag & names
    TAG="$(date +%Y%m%d-%H%M%S)"
    IMAGE_FQN="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"
    
    echo "==> Building Docker image"
    docker build -t "${IMAGE_NAME}":dev .

    docker tag "${IMAGE_NAME}":dev "${IMAGE_FQN}"

    echo "==> Pushing to Docker Hub"
    if [[ -n "${DOCKER_PASSWORD:-}" ]]; then
      echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USER" --password-stdin
    fi
    docker push "${IMAGE_FQN}"
  fi
elif [[ $MODE == "play" ]]; then
  TAG="$(yq_val '.container.image_tag')"
  IMAGE_FQN="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"
else
  echo "Cluster runs only supported for mode=train or mode=play" >&2
  exit 1
fi

SIF_PATH="${HPC_CONTAINER_DIR}/${HPC_USER}/${IMAGE_NAME}_${TAG}.sif"
HOST_LOGS_DIR="${HPC_LOGS_BASE}"

echo "== Cluster Job Info =="
echo "CFG (baked in image): /workspace/isaac/source/isaac_extension/simulation/config/${ROBOT_NAME}.yaml"
echo "Robot:          $ROBOT_NAME"
echo "Docker image:   $IMAGE_FQN"
echo "Explorer host:  ${HPC_USER}@${HPC_HOST}"
echo "SIF path:       $SIF_PATH"
echo "Logs on host:   $HOST_LOGS_DIR"

# write & submit sbatch
REMOTE_BASE="${REMOTE_BASE:-/projects/siliconsynapselab}" # your project root on cluster

# Where to store pulled SIF images on the cluster
SIF_DIR="${SIF_DIR:-${REMOTE_BASE}/containers}"

# Build command as an ARRAY, then join safely into a single string
declare -a CMD_ARR
build_cluster_cmd CMD_ARR

# Join with shell-quoting preserved
printf -v RUN_CMD '%q ' "${CMD_ARR[@]}"
RUN_CMD=${RUN_CMD% }   # trim final space
RUN_CMD="cd /workspace && ${RUN_CMD} \
  hydra.job.chdir=false \
  hydra.run.dir=/workspace/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S} \
  hydra.output_subdir=null"

# Prefer key-based auth (no password prompts)
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

echo "[LOCAL] Submitting to ${HPC_USER}@${HPC_HOST}"
echo "       Image: ${IMAGE_FQN}"
echo "       Logs bind: ${HOST_LOGS_DIR} -> /workspace/logs"

# Make a unique run folder for the job script on cluster
REMOTE_JOBDIR="${SIF_DIR}/jobs/${TAG}"
JOBFILE="job_${TAG}.sbatch"

# Name the SIF file we will pull to on the cluster
SIF_NAME="$(echo "${IMAGE_FQN}" | tr '/:' '_').sif"
SIF_PATH="${SIF_DIR}/${HPC_USER}/${SIF_NAME}"

# Base64-encode the command to avoid quoting issues
RUN_CMD_B64="$(printf '%s' "$RUN_CMD" | base64 -w0)"

mkdir -p /tmp/sbatch_jobs
LOCAL_JOB="/tmp/sbatch_jobs/${JOBFILE}"

# IMPORTANT: single-quoted heredoc so the host shell doesn't expand $VARS here
cat > "${LOCAL_JOB}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --partition=__PARTITION__
#SBATCH --time=__TIME_LIMIT__
#SBATCH --nodes=__NODES__
#SBATCH --ntasks-per-node=__TASKS_PER_NODE__
#SBATCH --cpus-per-task=__CPUS_PER_TASK__
#SBATCH --gres=__GRES__
#SBATCH --mem=__MEM__
#SBATCH --output=__OUT_FILE__
#SBATCH --error=__ERR_FILE__
#SBATCH --mail-user=__MAIL_USER__
#SBATCH --mail-type=__MAIL_TYPE__

set -euo pipefail
echo "[SLURM] Node: $(hostname)  Date: $(date)"

# Vars come from sbatch --export
: "${IMAGE_FQN:?}"; : "${SIF_PATH:?}"; : "${APPT_CACHE:?}"; : "${APPT_TMP:?}"; : "${HOST_LOGS_DIR:?}"; : "${RUN_CMD_B64:?}"

mkdir -p "${APPT_CACHE}" "${APPT_TMP}" "$(dirname "${SIF_PATH}")"

# Pull SIF if missing
if [[ ! -f "${SIF_PATH}" ]]; then
  echo "[Apptainer] Pulling docker://${IMAGE_FQN} -> ${SIF_PATH}"
  apptainer pull "${SIF_PATH}" "docker://${IMAGE_FQN}"
else
  echo "[Apptainer] Using existing SIF: ${SIF_PATH}"
fi

# Writable caches & user dirs (host-side)
RUNTIME_ROOT="${APPT_CACHE}/runtime"
KIT_CACHE="${RUNTIME_ROOT}/kit_cache"
KIT_DATA="${RUNTIME_ROOT}/kit_data"
OV_CACHE="${RUNTIME_ROOT}/ov_cache"
OV_DATA="${RUNTIME_ROOT}/ov_data"
OV_LOGS="${RUNTIME_ROOT}/ov_logs"
GL_CACHE="${RUNTIME_ROOT}/gl_cache"
NV_CACHE="${RUNTIME_ROOT}/nv_cache"
mkdir -p "${KIT_CACHE}" "${KIT_DATA}" "${OV_CACHE}" "${OV_DATA}" "${OV_LOGS}" "${GL_CACHE}" "${NV_CACHE}"

# Bind map (host -> container)
BIND_LIST="/scratch:/scratch,\
/projects/siliconsynapselab/sslab_isaaclab/outputs:/workspace/outputs,\
${KIT_CACHE}:/isaac-sim/kit/cache,\
${KIT_DATA}:/isaac-sim/kit/data,\
${OV_CACHE}:/root/.cache/ov,\
${OV_DATA}:/root/.local/share/ov/data,\
${OV_LOGS}:/root/.nvidia-omniverse/logs,\
${GL_CACHE}:/root/.cache/nvidia/GLCache,\
${NV_CACHE}:/root/.nv/ComputeCache,\
${HOST_LOGS_DIR}:/workspace/logs"

# Decode command and run
RUN_CMD="$(printf '%s' "${RUN_CMD_B64}" | base64 -d)"
echo "[JOB] Binds: ${BIND_LIST}"
echo "[JOB] CMD:   ${RUN_CMD}"

apptainer exec --nv -B "${BIND_LIST}" "${SIF_PATH}" bash -lc "${RUN_CMD}"
EOF

# Replace SBATCH placeholders with actual values (since heredoc was single-quoted)
sed -i \
  -e "s|__JOB_NAME__|${JOB_NAME}|g" \
  -e "s|__PARTITION__|${PARTITION}|g" \
  -e "s|__TIME_LIMIT__|${TIME_LIMIT}|g" \
  -e "s|__NODES__|${NODES}|g" \
  -e "s|__TASKS_PER_NODE__|${TASKS_PER_NODE}|g" \
  -e "s|__CPUS_PER_TASK__|${CPUS_PER_TASK}|g" \
  -e "s|__GRES__|${GRES}|g" \
  -e "s|__MEM__|${MEM}|g" \
  -e "s|__OUT_FILE__|${OUT_FILE}|g" \
  -e "s|__ERR_FILE__|${ERR_FILE}|g" \
  -e "s|__MAIL_USER__|${MAIL_USER}|g" \
  -e "s|__MAIL_TYPE__|${MAIL_TYPE}|g" \
  "${LOCAL_JOB}"

# Create remote dir and push job file
$SSH "${HPC_USER}@${HPC_HOST}" "mkdir -p '${REMOTE_JOBDIR}'"
scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${LOCAL_JOB}" "${HPC_USER}@${HPC_HOST}:${REMOTE_JOBDIR}/"

# Submit with env exported to the job
JOB_ID="$($SSH "${HPC_USER}@${HPC_HOST}" \
  "sbatch --parsable \
   --export=ALL,IMAGE_FQN='${IMAGE_FQN}',SIF_PATH='${SIF_PATH}',APPT_CACHE='${APPT_CACHE}',APPT_TMP='${APPT_TMP}',HOST_LOGS_DIR='${HOST_LOGS_DIR}',RUN_CMD_B64='${RUN_CMD_B64}' \
   '${REMOTE_JOBDIR}/${JOBFILE}'" \
  || true)"

if [[ -z "${JOB_ID}" ]]; then
  echo "[LOCAL] sbatch did not return a job id. Check your SSH key auth and cluster status."
  exit 2
fi

echo "[LOCAL] Submitted job: ${JOB_ID}"
echo "        stdout: $(echo "${OUT_FILE}" | sed "s/%j/${JOB_ID}/")"
echo "        stderr: $(echo "${ERR_FILE}" | sed "s/%j/${JOB_ID}/")"
