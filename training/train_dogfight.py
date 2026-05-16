"""
Train PPO on the Dogfight environment with self-play.

Strategy
--------
We train PPO (the dogfight env reward is dense and on-policy methods handle
the long episodes well) against a rotating pool of frozen past-self
checkpoints. Every `selfplay_interval` env steps we:
  1. Snapshot the current policy (deep-copy).
  2. Set it as the opponent inside DogfightSelfPlayEnv.
This is a curriculum: the opponent grows in skill as we do, which prevents
both (a) overfitting to a static dummy and (b) policy collapse against a
too-strong opponent at init.

For the initial phase (until first snapshot), the opponent samples random
actions — `DogfightSelfPlayEnv` falls back to action_space.sample() when
opponent_policy is None. This gives the agent a curriculum from "easy"
random opponent to "harder" frozen-self.

Examples
--------
    python -m training.train_dogfight --seed 0
    python -m training.train_dogfight --seed 0 --total-timesteps 500000  # smoke test
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.env_factory import make_dogfight_env  # noqa: E402
from training.train import (                        # noqa: E402
    WandbCallback,
    load_hparams,
    save_config,
    set_global_seed,
)


# ---------------------------------------------------------------------------
# Self-play opponent updater
# ---------------------------------------------------------------------------
class SelfPlayCallback(BaseCallback):
    """
    Every `interval` env steps, snapshot the current policy and broadcast it
    as the opponent to every sub-env via `set_opponent_policy`.

    The snapshot is a deep copy on CPU — it lives independently of the
    learning policy and won't be updated by gradient steps. This is a
    standard naive-self-play setup; you can extend it to a population (PSRO,
    fictitious self-play) for the bonus innovation section.
    """

    def __init__(self, interval: int, verbose: int = 0):
        super().__init__(verbose)
        self.interval = interval
        self._last_snapshot_step = 0
        self._snapshot_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_snapshot_step < self.interval:
            return True
        self._last_snapshot_step = self.num_timesteps
        self._snapshot_count += 1

        # Deep-copy the policy and move to CPU. Inference for the opponent
        # happens during env.step on whatever device, but CPU copies avoid
        # GPU-memory bloat as the snapshot count grows.
        snapshot = copy.deepcopy(self.model.policy).to("cpu")
        snapshot.eval()

        # Wrap so it has the .predict(obs, deterministic=...) -> (action, info)
        # signature DogfightSelfPlayEnv expects.
        opponent = _PolicySnapshot(snapshot)

        # Push to every sub-env.
        self.training_env.env_method("set_opponent_policy", opponent)

        if self.verbose:
            print(f"[selfplay] step={self.num_timesteps:,} "
                  f"snapshot #{self._snapshot_count}")
        return True


class _PolicySnapshot:
    """Inference-only wrapper exposing the predict() interface."""
    def __init__(self, policy):
        self.policy = policy

    def predict(self, obs, deterministic: bool = True):
        with torch.no_grad():
            action, _ = self.policy.predict(obs, deterministic=deterministic)
        return action, {}


# ---------------------------------------------------------------------------
# Vec env builder for dogfight
# ---------------------------------------------------------------------------
def build_dogfight_vec_env(n_envs: int, seed: int, monitor_dir: Path):
    def _factory(rank: int):
        def _init():
            env = make_dogfight_env(seed=seed + rank)
            return Monitor(env, filename=str(monitor_dir / f"rank_{rank}"))
        return _init

    if n_envs == 1:
        vec = DummyVecEnv([_factory(0)])
    else:
        vec = SubprocVecEnv([_factory(i) for i in range(n_envs)])
    return VecMonitor(vec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=Path,
                   default=_PROJECT_ROOT / "configs" / "algo_config.yaml")
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--selfplay-interval", type=int, default=200_000,
                   help="Env steps between self-play opponent snapshots")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="info8003-rl")
    p.add_argument("--wandb-entity", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    hparams = load_hparams(args.config, "ppo", "dogfight")
    if args.total_timesteps is not None:
        hparams["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        hparams["n_envs"] = args.n_envs

    total_timesteps = int(hparams["total_timesteps"])
    n_envs = int(hparams["n_envs"])

    run_id = f"ppo_dogfight_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    model_dir = _PROJECT_ROOT / "models" / run_id
    log_dir = _PROJECT_ROOT / "logs" / run_id
    monitor_dir = log_dir / "monitor"
    for d in (model_dir, log_dir, monitor_dir):
        d.mkdir(parents=True, exist_ok=True)

    save_config(model_dir, args, hparams)

    print(f"[run] {run_id}")
    print(f"[run] total_timesteps={total_timesteps:,}  n_envs={n_envs}  "
          f"selfplay_interval={args.selfplay_interval:,}")

    # W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_id,
            config={**vars(args), **hparams},
            sync_tensorboard=True,
        )

    # Build env
    train_env = build_dogfight_vec_env(n_envs, args.seed, monitor_dir)

    # Build PPO
    excluded = {"total_timesteps", "n_envs", "eval_freq"}
    algo_kwargs = {k: v for k, v in hparams.items() if k not in excluded}
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=args.seed,
        verbose=1,
        device=args.device,
        tensorboard_log=str(log_dir),
        **algo_kwargs,
    )

    # Callbacks
    callbacks = [
        SelfPlayCallback(interval=args.selfplay_interval, verbose=1),
        CheckpointCallback(
            save_freq=max(args.selfplay_interval, 100_000),
            save_path=str(model_dir / "checkpoints"),
            name_prefix="ckpt",
        ),
    ]
    if use_wandb:
        callbacks.append(WandbCallback())

    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
    finally:
        final_path = model_dir / "final_model.zip"
        model.save(str(final_path))
        print(f"[run] saved final model -> {final_path}")
        train_env.close()
        if use_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
