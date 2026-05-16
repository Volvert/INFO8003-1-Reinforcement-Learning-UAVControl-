#!/bin/bash
#
# Train PPO + naive self-play on the Dogfight env for one seed.
#
# Submit with:
#   sbatch --export=ALL,SEED=0 slurm/train_dogfight.slurm
#   sbatch --export=ALL,SEED=0,SELFPLAY_INTERVAL=100000 slurm/train_dogfight.slurm
#
# The longest training run (5M steps in the YAML). Don't kill it just
# because the curve looks flat for the first ~500k steps — self-play is
# non-stationary and the policy needs time to find any reward signal at all.
#
# ============================================================================
#SBATCH --job-name=dogfight
#SBATCH --output=logs/slurm/dogfight_%j.out
#SBATCH --error=logs/slurm/dogfight_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=all
# ============================================================================

set -euo pipefail

: "${SEED:=0}"
: "${SELFPLAY_INTERVAL:=200000}"
: "${EXTRA_ARGS:=}"

source "${SLURM_SUBMIT_DIR}/slurm/_setup_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/slurm"

echo "[dogfight] seed=${SEED}  selfplay_interval=${SELFPLAY_INTERVAL}"
echo "[dogfight] extra args: ${EXTRA_ARGS}"

srun python -m training.train_dogfight \
    --seed "${SEED}" \
    --device cuda \
    --selfplay-interval "${SELFPLAY_INTERVAL}" \
    --wandb-project "${WANDB_PROJECT}" \
    ${EXTRA_ARGS}

echo "[dogfight] done at $(date -Is)"
