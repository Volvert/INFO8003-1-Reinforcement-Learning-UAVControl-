# INFO8003 — RL Project: UAV Control with PyFlyt

Train and evaluate deep RL agents on PyFlyt drone environments and compete
in a 1v1 dogfight tournament.

## Setup

```bash
# Python 3.10 venv recommended (matches the grading server)
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify everything imports + steps before training
python scripts/test_env.py
```

If `test_env.py` crashes with a PyBullet error, you're missing a system
library (Linux: `libgl1`, `libegl1`). On Windows + WSL2 you may need
`sudo apt install libglib2.0-0 libgl1` inside WSL.

## Repo layout

```
scripts/
  common.py             # shared: env factories, algo builders, wandb helper
  test_env.py           # 30-second sanity check — run this first
  train_hover.py        # PPO/SAC on QuadX-Hover-v4
  train_waypoints.py    # PPO/SAC on QuadX-Waypoints-v4 (chunk 1)
  train_dogfight.py     # PPO self-play on MAFixedwingDogfightEnvV2 (chunk 2)
  analyze.py            # rliable-style aggregate stats + plots (chunk 3)
  env_config.py         # waypoint env-kwargs overrides (provided)
  wrappers.py           # FlattenWaypointEnv (provided)
  dogfight_wrapper.py   # DogfightSelfPlayEnv (provided)
  evaluate.py           # grading-style eval (provided)
  tournament.py         # round-robin Elo (provided)
  submission_template.py
results/
  models/               # trained checkpoints (final_* and best_*/best_model.zip)
  logs/                 # tensorboard event files
  eval/                 # JSON outputs from evaluate.py
submissions/
  groupXX_*.py          # tournament submissions (chunk 4)
```

## Training — what to actually run

The minimum you need for the report is **2 algorithms × 3 seeds × 3 envs**.
All seeds are CLI args; just loop a shell variable. On a Ryzen + GTX 1060
the time budgets below are realistic.

### Hover (cheapest — start here)

```bash
# ~10 min each (PPO with 4 envs); SAC ~30 min
for s in 0 1 2; do
  python scripts/train_hover.py --algo PPO --seed $s --total-timesteps 500000
  python scripts/train_hover.py --algo SAC --seed $s --total-timesteps 500000
done
```

### Waypoints (with the flight-mode comparison the rubric asks for)

```bash
# Mode 6 (easy) — both algos should solve this
for s in 0 1 2; do
  python scripts/train_waypoints.py --algo PPO --flight-mode 6 --seed $s
  python scripts/train_waypoints.py --algo SAC --flight-mode 6 --seed $s
done

# Mode 0 (hard) — one algo to demonstrate the difficulty gap
for s in 0 1 2; do
  python scripts/train_waypoints.py --algo PPO --flight-mode 0 --seed $s --total-timesteps 3000000
done
```

### Dogfight (chunk 2 — overnight run)

Self-play takes hours. Plan to start it once and let it run overnight.

## Evaluation

```bash
# Single model evaluation with detailed stats
python scripts/evaluate.py \
  --model results/models/best_PPO_Hover_seed0/best_model.zip \
  --env hover --n_episodes 30 --output results/eval/hover_PPO_s0.json

# Render a few episodes visually
python scripts/evaluate.py \
  --model results/models/best_PPO_Hover_seed0/best_model.zip \
  --env hover --n_episodes 3 --render
```

## Tournament submission

See `scripts/submission_template.py` for the interface contract. Chunk 4
will fill in `submissions/groupXX_main.py` once your dogfight model is trained.

## Wandb

By default every training run logs to wandb project `info8003-pyflyt`. Pass
`--no-wandb` to skip. Run name format: `{ALGO}_{ENV}_seed{N}[_modeM]`.

## Reproducibility

Every script accepts `--seed`. The same seed + the same `requirements.txt`
should produce the same wandb curves up to nondeterminism in PyBullet
contact resolution (which is hardware-dependent and can't fully be fixed).
For the report, log all hyperparameters and the git commit hash.

## Hardware notes

This setup was developed for a Ryzen CPU + GTX 1060 6GB. PyFlyt is
CPU-bound (PyBullet physics, small MLP nets), so adding GPU power won't
speed things up much. If you have more cores, increase `--n-envs` for PPO
and pass `--use-subproc`.
