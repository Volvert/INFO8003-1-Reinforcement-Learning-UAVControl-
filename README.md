# INFO8003 — RL Project: UAV Control with PyFlyt

Train and evaluate deep RL agents (PPO and SAC) on PyFlyt drone environments
and compete in a 1v1 dogfight tournament.

**Course**: INFO8003-1 Reinforcement Learning — University of Liège  
**Authors**: Florent Volvert, Robin Kulczycki  
**Year**: 2025–2026

---

## Table of Contents

1. [Setup](#1-setup)
2. [Configuration](#2-configuration)
3. [Repo Layout](#3-repo-layout)
4. [Training](#4-training)
5. [Generating Figures and Statistics](#5-generating-figures-and-statistics)
6. [Evaluating a Model](#6-evaluating-a-model)
7. [Visualizing a Trained Policy](#7-visualizing-a-trained-policy)
8. [Watching a Dogfight](#8-watching-a-dogfight)
9. [Identifying the Best Models](#9-identifying-the-best-models)
10. [Tournament Submission](#10-tournament-submission)

---

## 1. Setup

```bash
# Python 3.10 venv recommended (matches the grading server)
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify everything imports and steps correctly before training
python scripts/test_env.py
```

If `test_env.py` crashes with a PyBullet error, you are missing a system
library. On Linux run:

```bash
sudo apt install libgl1 libegl1 libglib2.0-0
```

---

## 2. Configuration

### Weights & Biases

All training scripts log to W&B by default. Set your API key once:

```bash
wandb login
# paste your API key from https://wandb.ai/authorize
```

To change the W&B project name or entity, open `configs/algo_config.yaml`
and edit the top-level fields:

```yaml
# configs/algo_config.yaml
wandb_project: "info8003-rl"   # <-- your W&B project name
wandb_entity:  null            # <-- your W&B username (null = default entity)
```

To disable W&B for a single run:

```bash
python training/train.py --algo ppo --env hover --seed 0 --no-wandb
```

### Hyperparameters

All hyperparameters (learning rate, network architecture, entropy coefficient,
batch size, etc.) are centralized in `configs/algo_config.yaml`.
**Do not hardcode values in training scripts** — change them here only.

---

## 3. Repo Layout

```
configs/
  algo_config.yaml          # all hyperparameters (PPO, SAC, per-env)

training/
  train.py                  # main training entry point (hover + waypoints)
  train_dogfight.py         # PPO self-play on dogfight
  env_factory.py            # environment builders

analysis/
  compare_runs.py           # manual comparison (--runs --labels CLI)
  compare_all_runs.py       # auto-generate all report figures + txt
  record_video.py           # visualize a trained policy (human render)
  extract_metrics.py        # detailed per-model metrics (crash rate, winrate...)

scripts/                    # provided grading scripts — do not modify
  env_config.py
  wrappers.py
  dogfight_wrapper.py
  evaluate.py               # grading-style evaluation
  tournament.py             # round-robin Elo tournament
  submission_template.py
  test_env.py               # sanity check — run this first
  watch_dogfight.py         # watch two agents fight in real time

slurm/
  submit_all.sh             # submit all 23 training jobs to SLURM

models/                     # trained checkpoints
  <run_id>/
    best_model.zip          # best deterministic-eval checkpoint  <- use this
    final_model.zip         # last checkpoint (dogfight: use this)
    checkpoints/            # periodic snapshots
    config.yaml             # full hyperparams for this run

logs/                       # training logs
  <run_id>/
    eval/evaluations.npz    # evaluation curves (used by compare_runs.py)
    monitor/rank_*.csv      # episode-level reward and length

figures/                    # all generated PNG + TXT files (compare_all_runs.py)

submissions/                # tournament submission files
  groupXX_name.py
```

---

## 4. Training

### Single run

```bash
# PPO on Hover, mode 6, seed 0
python training/train.py --algo ppo --env hover --flight-mode 6 --seed 0

# SAC on Hover, mode 6, seed 0
python training/train.py --algo sac --env hover --flight-mode 6 --seed 0

# PPO on Waypoints, mode 6, seed 1
python training/train.py --algo ppo --env waypoints --flight-mode 6 --seed 1

# PPO on Waypoints, mode 0 (hard), seed 2
python training/train.py --algo ppo --env waypoints --flight-mode 0 --seed 2

# PPO self-play on Dogfight, seed 0
python training/train_dogfight.py --seed 0
```

### Full sweep (all runs for the report)

```bash
# Hover: PPO + SAC, 3 seeds, mode 6
for ALGO in ppo sac; do
  for SEED in 0 1 2; do
    python training/train.py --algo $ALGO --env hover --flight-mode 6 --seed $SEED
  done
done

# Waypoints: PPO, 4 modes, 3 seeds
for FM in 0 4 6 7; do
  for SEED in 0 1 2; do
    python training/train.py --algo ppo --env waypoints --flight-mode $FM --seed $SEED
  done
done

# Waypoints: SAC, mode 6, 3 seeds
for SEED in 0 1 2; do
  python training/train.py --algo sac --env waypoints --flight-mode 6 --seed $SEED
done

# Dogfight: PPO self-play, 2 seeds
for SEED in 0 1; do
  python training/train_dogfight.py --seed $SEED
done
```

### On a SLURM cluster

```bash
bash slurm/submit_all.sh            # submit all 23 jobs
bash slurm/submit_all.sh hover      # hover only
bash slurm/submit_all.sh waypoints  # waypoints only
bash slurm/submit_all.sh dogfight   # dogfight only
bash slurm/submit_all.sh --dry-run  # print commands without submitting
```

---

## 5. Generating Figures and Statistics

### Automatic (recommended)

Generates all PNG plots and TXT statistics files in one command.
Any comparison for which logs are missing is silently skipped.
Outputs go to `figures/`.

```bash
python analysis/compare_all_runs.py
# outputs: figures/hover_ppo_vs_sac.png   + figures/hover_ppo_vs_sac.txt
#          figures/waypoints_ppo_vs_sac.png + figures/waypoints_ppo_vs_sac.txt
#          figures/waypoints_modes.png      + figures/waypoints_modes.txt
```

The TXT files list each loaded run with its **exact directory name** and
final reward — use them to identify which `models/` folder is the best seed
for each configuration.

### Manual (single comparison)

```bash
# PPO vs SAC on Hover
python analysis/compare_runs.py \
  --runs "logs/ppo_hover_fm6_seed*" "logs/sac_hover_fm6_seed*" \
  --labels PPO SAC \
  --output figures/hover_ppo_vs_sac.png

# Flight mode comparison on Waypoints
python analysis/compare_runs.py \
  --runs "logs/ppo_waypoints_fm0_seed*" \
         "logs/ppo_waypoints_fm4_seed*" \
         "logs/ppo_waypoints_fm6_seed*" \
         "logs/ppo_waypoints_fm7_seed*" \
  --labels "mode 0" "mode 4" "mode 6" "mode 7" \
  --output figures/waypoints_modes.png
```

---

## 6. Evaluating a Model

### Detailed metrics (crash rate, episode length, waypoints reached, winrate)

```bash
# Single model — infers env and mode from the directory name
python scripts/extract_metrics.py \
  --model models/ppo_hover_fm6_seed0_<timestamp>/best_model.zip \
  --output-dir results/metrics

# All models at once
python scripts/extract_metrics.py \
  --models-dir models/ \
  --output-dir results/metrics

# Print a comparison table from all extracted JSON files
python scripts/extract_metrics.py --compare results/metrics/
```

> The script infers the environment and flight mode from the model directory
> name. Keep the default naming convention:
> `{ALGO}_{env}_fm{mode}_seed{N}_<timestamp>` (e.g. `ppo_hover_fm6_seed0_20260516-171930`).

### Grading script (matches what the professor uses)

```bash
python scripts/evaluate.py \
  --model models/ppo_hover_fm6_seed0_<timestamp>/best_model.zip \
  --env hover
```

---

## 7. Visualizing a Trained Policy

Opens a live PyBullet window and renders the agent playing. Requires a
display (use `--render` on a desktop or via VirtualGL on a cluster).

```bash
# Hover — watch 3 episodes
python analysis/record_video.py \
  --model models/ppo_hover_fm6_seed0_<timestamp>/best_model.zip \
  --env hover \
  --flight-mode 6 \
  --n-episodes 3 \
  --seed 42

# Waypoints — watch 2 episodes
python analysis/record_video.py \
  --model models/ppo_waypoints_fm6_seed0_<timestamp>/best_model.zip \
  --env waypoints \
  --flight-mode 6 \
  --n-episodes 2
```

The terminal prints the cumulative reward and step count for each episode.
Close the PyBullet window to advance to the next episode.

---

## 8. Watching a Dogfight

```bash
# Your agent (seed 1) vs a random opponent
python scripts/watch_dogfight.py \
  --model-a models/ppo_dogfight_seed1_<timestamp>/final_model.zip \
  --model-b random \
  --n-games 3

# Two trained agents vs each other
python scripts/watch_dogfight.py \
  --model-a models/ppo_dogfight_seed1_<timestamp>/final_model.zip \
  --model-b models/ppo_dogfight_seed0_<timestamp>/final_model.zip \
  --n-games 5
```

---

## 9. Identifying the Best Models

Run `compare_all_runs.py` then read the TXT files:

```bash
python analysis/compare_all_runs.py
cat figures/hover_ppo_vs_sac.txt     # exact run dir + final reward per seed
cat figures/waypoints_modes.txt
```

Copy the best checkpoints to a delivery folder:

```bash
mkdir -p deliverables/models

# Replace <run_dir> with the exact directory name shown in the TXT file
cp models/<run_dir>/best_model.zip  deliverables/models/ppo_hover_best.zip
cp models/<run_dir>/best_model.zip  deliverables/models/sac_hover_best.zip
cp models/<run_dir>/best_model.zip  deliverables/models/ppo_waypoints_best.zip
cp models/<run_dir>/best_model.zip  deliverables/models/sac_waypoints_best.zip
cp models/<run_dir>/final_model.zip deliverables/models/ppo_dogfight_best.zip
```

---

## 10. Tournament Submission

### 1. Create your submission file

```bash
cp scripts/submission_template.py submissions/groupXX_name.py
# edit submissions/groupXX_name.py and implement load_model()
```

```python
# submissions/groupXX_name.py
from stable_baselines3 import PPO

def load_model(path=None):
    return PPO.load(path or "models/ppo_dogfight_seed1_<timestamp>/final_model.zip")
```

### 2. Re-export from the pinned venv

The grading server uses exact versions from `requirements.txt`.
Re-export your model before submitting to avoid version mismatch errors:

```python
from stable_baselines3 import PPO
PPO.load("your_model.zip").save("your_model_pinned.zip")
```

### 3. Test locally

```bash
python scripts/tournament.py \
  submissions/groupXX_name.py \
  submissions/groupXX_name.py
```

### 4. Submit to the leaderboard

```
https://arthur-louette.com/leaderboard
```

---

## Reproducibility

Every training script accepts `--seed`. The same seed + the same
`requirements.txt` produces identical results up to nondeterminism in
PyBullet contact resolution (hardware-dependent and cannot be fully fixed).
All hyperparameters are saved automatically to `models/<run_id>/config.yaml`
at the start of each run.