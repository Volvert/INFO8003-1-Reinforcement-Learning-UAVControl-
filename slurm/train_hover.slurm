#!/bin/bash
#
# Train PPO or SAC on QuadX-Hover for one (algo, flight_mode, seed) combo.
#
# Submit with:
#   sbatch --export=ALL,ALGO=ppo,FLIGHT_MODE=6,SEED=0 slurm/train_hover.slurm
#   sbatch --export=ALL,ALGO=sac,FLIGHT_MODE=6,SEED=1 slurm/train_hover.slurm
#
# Or use slurm/submit_all.sh to fire the whole sweep at once.
#
# ============================================================================
#SBATCH --job-name=hover

#SBATCH --output=logs/slurm/hover_%j.out
#SBATCH --error=logs/slurm/hover_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
##SBATCH --gres=gpu:1
#SBATCH --partition=all
# ============================================================================

set -euo pipefail

: "${ALGO:=ppo}"
: "${FLIGHT_MODE:=6}"
: "${SEED:=0}"
: "${EXTRA_ARGS:=}"

if [[ "${ALGO}" != "ppo" && "${ALGO}" != "sac" ]]; then
    echo "ERROR: ALGO must be 'ppo' or 'sac', got '${ALGO}'"
    exit 1
fi

source "${SLURM_SUBMIT_DIR}/slurm/_setup_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/slurm"

echo "[hover] algo=${ALGO}  flight_mode=${FLIGHT_MODE}  seed=${SEED}"
echo "[hover] extra args: ${EXTRA_ARGS}"

srun python -m training.train \
    --algo "${ALGO}" \
    --env hover \
    --flight-mode "${FLIGHT_MODE}" \
    --seed "${SEED}" \
    --device cpu \
    --wandb-project "${WANDB_PROJECT}" \
    ${EXTRA_ARGS}

echo "[hover] done at $(date -Is)"
