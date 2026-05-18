#!/bin/bash
#
# Train PPO or SAC on QuadX-Waypoints for one (algo, flight_mode, seed) combo.
#
# Submit with:
#   sbatch --export=ALL,ALGO=ppo,FLIGHT_MODE=6,SEED=0 slurm/train_waypoints.slurm
#   sbatch --export=ALL,ALGO=ppo,FLIGHT_MODE=0,SEED=2 slurm/train_waypoints.slurm
#
# Waypoints PPO is the longest non-dogfight run (3M steps in the YAML).
# With n_envs=8 on a decent GPU it should finish well inside 24h, but we
# ask for 48h to be safe — better than getting killed at the 23rd hour.
#
# ============================================================================
#SBATCH --job-name=waypts
#SBATCH --output=logs/slurm/waypts_%j.out
#SBATCH --error=logs/slurm/waypts_%j.err
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

echo "[waypoints] algo=${ALGO}  flight_mode=${FLIGHT_MODE}  seed=${SEED}"
echo "[waypoints] extra args: ${EXTRA_ARGS}"

srun python -m training.train \
    --algo "${ALGO}" \
    --env waypoints \
    --flight-mode "${FLIGHT_MODE}" \
    --seed "${SEED}" \
    --device cpu \
    --wandb-project "${WANDB_PROJECT}" \
    ${EXTRA_ARGS}

echo "[waypoints] done at $(date -Is)"
