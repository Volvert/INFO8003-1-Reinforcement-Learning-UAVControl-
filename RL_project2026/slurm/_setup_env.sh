#!/bin/bash
# Common environment setup, sourced by every SLURM job script.
#
# Cluster: Montefiore Institute (Liege) "Alan" — conda-based, no Lmod.
# If anything below changes (conda location, env name, project path),
# adjust ONLY here — every job picks up the change automatically.

# ---------------------------------------------------------------------------
# Project layout — adjust if you move the project
# ---------------------------------------------------------------------------
export PROJECT_ROOT="/home/fvolvert/rl-project"
export CONDA_ENV="rl-project"

# ---------------------------------------------------------------------------
# Activate conda env
# ---------------------------------------------------------------------------
# In a non-interactive batch job, conda's shell function isn't defined yet,
# so we source its init script manually. Anaconda3 lives in $HOME on this
# cluster (verified via `conda env list`).
CONDA_INIT="${HOME}/anaconda3/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_INIT}" ]]; then
    echo "ERROR: conda init script not found at ${CONDA_INIT}"
    echo "Adjust CONDA_INIT in slurm/_setup_env.sh"
    exit 1
fi
source "${CONDA_INIT}"

# Deactivate any auto-activated env (e.g. (myenv) from .bashrc) before
# activating ours. Two deactivates handle nested cases.
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

conda activate "${CONDA_ENV}" || {
    echo "ERROR: cannot activate conda env '${CONDA_ENV}'"
    echo "Create it once with:"
    echo "    conda create -n ${CONDA_ENV} python=3.10 -y"
    echo "    conda activate ${CONDA_ENV}"
    echo "    cd ${PROJECT_ROOT}"
    echo "    pip install 'numpy<2.0'"
    echo "    grep -v '^numpy==' requirements.txt > requirements_local.txt"
    echo "    pip install -r requirements_local.txt"
    exit 1
}

# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------
# Run `wandb login` once on the login node; the key lands in ~/.netrc and
# every batch job picks it up automatically. No need to set WANDB_API_KEY.
export WANDB_PROJECT="info8003-rl"
export WANDB_ENTITY="florent-volvert"
export WANDB_SILENT="true"
# Avoid wandb spamming /tmp on shared nodes — direct it under our project
export WANDB_DIR="${PROJECT_ROOT}/logs/wandb"
mkdir -p "${WANDB_DIR}"

# ---------------------------------------------------------------------------
# Repro / threading
# ---------------------------------------------------------------------------
# Avoid OpenMP / MKL oversubscription. With n_envs=8 and an 8-CPU
# allocation, each PyBullet env should get ~1 thread.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Diagnostics — printed at job start, useful when reading logs later
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Job ${SLURM_JOB_ID:-<no-slurm>} on $(hostname)"
echo "Started: $(date -Is)"
echo "Partition: ${SLURM_JOB_PARTITION:-?}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-?}"
echo "Working dir: ${PROJECT_ROOT}"
echo "Conda env: ${CONDA_ENV} ($(python --version 2>&1))"
echo "Python: $(which python)"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES:-<none>}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "nvidia-smi not available"
echo "============================================================"

cd "${PROJECT_ROOT}"
