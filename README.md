# INFO8003-Project - UAV Control via Deep Reinforcement Learning

This project trains deep reinforcement learning agents to control drones
in three environments of increasing complexity:

- **Hover** - stabilize a quadrotor at a fixed position
- **Waypoints** - navigate through 4 randomly placed 3D waypoints
- **Dogfight** - 1v1 aerial combat between fixed-wing aircraft

We compare PPO and SAC across environments and flight modes, and submit
a trained agent to the class dogfight tournament. All environments use
[PyFlyt](https://arxiv.org/abs/2304.01305), a simulator built on PyBullet.

---

## Setup

```bash
conda create -n rl-project python=3.10 -y
conda activate rl-project
pip install -r requirements.txt
python scripts/test_env.py
```

On Linux, if `test_env.py` crashes with a PyBullet error:
```bash
sudo apt install libglib2.0-0 libgl1
```

---

## How the project works

Training a drone agent involves 3 steps: configure, train, analyze.

### 1. Configure

All hyperparameters (learning rate, batch size, network architecture, etc.)
are defined in `configs/algo_config.yaml`. Edit this file to change any
training setting. No need to touch the training scripts.

### 2. Train

**`training/train.py`** trains PPO or SAC on Hover or Waypoints.

```bash
python -m training.train --algo ppo --env hover --flight-mode 6 --seed 0
```

This script:
- Reads hyperparameters from `configs/algo_config.yaml`
- Creates `models/<run_id>/` containing:
  - `best_model.zip` - best checkpoint (highest eval reward)
  - `final_model.zip` - last checkpoint
  - `config.yaml` - exact hyperparameters used
- Creates `logs/<run_id>/` containing:
  - `eval/evaluations.npz` - learning curve data
  - `monitor/` - episode rewards and lengths during training

**`training/train_dogfight.py`** trains PPO with self-play on Dogfight.

```bash
python -m training.train_dogfight --seed 0
```

Same outputs as above, plus `models/<run_id>/checkpoints/` with
periodic snapshots used as opponents during self-play.

### 3. Analyze

**`analysis/compare_runs.py`** reads the `eval/evaluations.npz` files
and produces learning curves with bootstrap 95% CI and a final
performance table (mean, IQM, Welch t-test).

```bash
python -m analysis.compare_runs \
    --runs "logs/ppo_hover_fm6_seed*" "logs/sac_hover_fm6_seed*" \
    --labels PPO SAC \
    --output report/figures/hover_ppo_vs_sac.png
```

**`analysis/record_video.py`** loads a trained model and renders
episodes visually in the PyBullet GUI.

```bash
python -m analysis.record_video \
    --model models/<run_id>/best_model.zip \
    --env hover --flight-mode 6
```

**`scripts/evaluate.py`** runs the deterministic policy for N episodes
and prints detailed stats (mean reward, crash rate, waypoints reached).

```bash
python scripts/evaluate.py \
    --model models/<run_id>/best_model.zip \
    --env hover --n_episodes 20
```

**`scripts/watch_dogfight.py`** watches two agents fight in the GUI.

```bash
python scripts/watch_dogfight.py \
    --model models/<run_id>/final_model.zip \
    --opponent random
```

---

## Running on the Alan cluster

Instead of running scripts manually, SLURM jobs handle everything.

**`slurm/_setup_env.sh`** is sourced by every job - it activates the
conda environment, sets W&B variables, and moves to the project root.
You should not run it directly.

**`slurm/train_hover.slurm`**, **`train_waypoints.slurm`**,
**`train_dogfight.slurm`** are the actual job scripts. Each one
sources `_setup_env.sh` then calls `training/train.py` with the
right arguments. They are submitted by `submit_all.sh`.

**`slurm/submit_all.sh`** submits all 23 training jobs at once.

```bash
mkdir -p logs/slurm          # create log folder first
bash slurm/submit_all.sh     # submit all jobs

bash slurm/submit_all.sh hover     # hover only (6 jobs)
bash slurm/submit_all.sh waypts    # waypoints only (15 jobs)
bash slurm/submit_all.sh dogfight  # dogfight only (2 jobs)

squeue -u $USER              # check job status
tail -f logs/slurm/hover_<jobid>.out  # follow a running job
```

Each job writes its output to `logs/slurm/<name>_<jobid>.out`.

---

## Reproducibility

All scripts accept `--seed` to fix Python, NumPy, PyTorch and environment
random generators. Each run saves its exact config to `models/<run_id>/config.yaml`.
Full reproducibility is guaranteed up to PyBullet contact resolution
nondeterminism, which is hardware-dependent.

---

## Weights & Biases

Training curves and episode videos are logged to W&B automatically.
Use `--no-wandb` to disable. Project: ``.
