#!/bin/bash
#
# Quick pipeline smoke test (~5-10 min). Run this FIRST after creating the
# conda env, to verify GPU access + imports + a tiny training run before
# launching the real sweep.
#
#   sbatch slurm/smoke_test.slurm
#   tail -f logs/slurm/smoke_*.out
#
# ============================================================================
#SBATCH --job-name=smoke
#SBATCH --output=logs/slurm/smoke_%j.out
#SBATCH --error=logs/slurm/smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=all
# ============================================================================

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/_setup_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/slurm"

echo "[smoke] verifying imports + GPU..."
python - <<'PY'
import gymnasium, PyFlyt.gym_envs, torch, stable_baselines3, numpy
env = gymnasium.make("PyFlyt/QuadX-Hover-v4")
obs, info = env.reset(seed=0)
print(f"  numpy:             {numpy.__version__}")
print(f"  torch:             {torch.__version__}")
print(f"  stable_baselines3: {stable_baselines3.__version__}")
print(f"  PyFlyt obs shape:  {obs.shape}")
print(f"  CUDA available:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:               {torch.cuda.get_device_name(0)}")
else:
    print("  WARN: CUDA not visible — training will run on CPU (slow)")
env.close()
PY

echo "[smoke] running tiny training run (50k steps, no W&B)..."
srun python -m training.train \
    --algo ppo \
    --env hover \
    --flight-mode 6 \
    --seed 0 \
    --total-timesteps 50000 \
    --no-wandb

echo "[smoke] done at $(date -Is)"
