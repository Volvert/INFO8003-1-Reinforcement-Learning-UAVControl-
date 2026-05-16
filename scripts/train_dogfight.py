"""Self-play PPO trainer for PyFlyt MAFixedwingDogfightEnvV2.

Self-play scheme: a FIFO pool of K most recent policy snapshots. Every
N total env-steps we (1) save the current policy as a snapshot, (2) evict
the oldest if the pool exceeds K, (3) sample a snapshot uniformly at random
and set it as the opponent in every parallel env.

Why a pool, not just "opponent = latest me"?
  - With latest-only, the policy chases its own current weakness. Easy to
    fall into rock-paper-scissors cycles where small updates flip which
    strategy "wins" and learning oscillates.
  - With a pool, the policy must stay robust against several recent
    versions of itself. This is fictitious self-play with a sliding window
    (full league play à la AlphaStar adds rated opponent selection — out
    of scope for this project).

Why update once per `snapshot_freq` rather than continuously?
  - The opponent must be stable for long enough that PPO can collect a
    meaningful advantage estimate against it. PPO's advantage relies on
    the env distribution being roughly stationary across an n_steps
    rollout; if the opponent changed every step, advantages would be noise.

Usage:
  python scripts/train_dogfight.py --seed 42 --total-timesteps 2000000
  python scripts/train_dogfight.py --seed 42 --total-timesteps 5000000 --no-progress

Output:
  results/models/final_PPO_Dogfight_seed42.zip       — final policy
  results/models/snapshots_PPO_Dogfight_seed42/      — last K snapshots
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from common import have_progress_bar_deps, init_wandb, make_algorithm, make_run_name
from dogfight_wrapper import DogfightSelfPlayEnv


ENV_SHORT = "Dogfight"


def _make_thunk(seed_offset):
    """Build one DogfightSelfPlayEnv with no opponent (random fallback).

    Opponent is set later via env_method('set_opponent_policy', ...) from
    the SelfPlayCallback. Tournament settings are baked in here because
    they're fixed by the project statement (60s, 30Hz).
    """
    def _thunk():
        env = DogfightSelfPlayEnv(
            team_size=1,
            opponent_policy=None,        # None => wrapper samples random actions
            flatten_observation=True,
            render_mode=None,
            max_duration_seconds=60.0,
            agent_hz=30,
        )
        env = Monitor(env)
        env.reset(seed=seed_offset)
        return env
    return _thunk


class SelfPlayCallback(BaseCallback):
    """Snapshot-pool self-play.

    Every snapshot_freq total env-steps:
      1. Save current model -> pool/snap_{step:09d}.zip
      2. Evict oldest snapshots until len(pool) <= pool_size
      3. Pick uniformly from the pool and set as opponent for all envs
    """

    def __init__(self, snapshot_freq, pool_size, snapshot_dir, algo_cls, verbose=1):
        super().__init__(verbose)
        self.snapshot_freq = snapshot_freq
        self.pool_size = pool_size
        self.snapshot_dir = snapshot_dir
        self.algo_cls = algo_cls
        self.snapshot_paths: list[str] = []
        self._next_snapshot_step = snapshot_freq
        os.makedirs(snapshot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # num_timesteps is the SB3 global step counter (already x n_envs).
        # Trigger when we cross a snapshot boundary; using >= instead of ==
        # because n_envs steps can skip past an exact boundary.
        if self.num_timesteps < self._next_snapshot_step:
            return True
        self._next_snapshot_step += self.snapshot_freq

        # 1. Snapshot. Use zero-padded step in filename so sort order =
        # chronological order — useful when inspecting snapshots later.
        path = os.path.join(self.snapshot_dir,
                            f"snap_{self.num_timesteps:09d}.zip")
        self.model.save(path)
        self.snapshot_paths.append(path)

        # 2. Evict
        while len(self.snapshot_paths) > self.pool_size:
            old = self.snapshot_paths.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass

        # 3. Sample new opponent. Load to CPU — opponent only does inference,
        # which is fast on CPU and avoids contention with the trainer's GPU.
        chosen = random.choice(self.snapshot_paths)
        opponent = self.algo_cls.load(chosen, device="cpu")
        self.training_env.env_method("set_opponent_policy", opponent)

        if self.verbose:
            print(f"\n[SelfPlay step={self.num_timesteps:>9,}] "
                  f"snapshot saved, pool_size={len(self.snapshot_paths)}, "
                  f"opponent={os.path.basename(chosen)}")

        # Logged to TB and wandb via sync_tensorboard
        self.logger.record("selfplay/pool_size", len(self.snapshot_paths))
        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--snapshot-freq", type=int, default=200_000,
                   help="Save+rotate opponent every N total env steps")
    p.add_argument("--pool-size", type=int, default=5,
                   help="Number of recent snapshots to keep in opponent pool")
    p.add_argument("--save-dir", default="results/models")
    p.add_argument("--log-dir", default="results/logs")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--wandb-project", default="info8003-pyflyt")
    return p.parse_args()


def main():
    args = parse_args()
    run_name = make_run_name("PPO", ENV_SHORT, args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    snapshot_dir = os.path.join(args.save_dir, f"snapshots_{run_name}")

    train_env = DummyVecEnv([
        _make_thunk(args.seed + 1000 * i) for i in range(args.n_envs)
    ])

    cfg = {**vars(args),
           "env_id": "MAFixedwingDogfightEnvV2",
           "agent_hz": 30, "max_duration_seconds": 60}
    wandb_run = init_wandb(args.wandb_project, run_name, cfg,
                           enabled=not args.no_wandb)

    tb_log = os.path.join(args.log_dir, run_name)

    # Two deviations from PPO defaults:
    #  - net_arch=[256, 256]: dogfight obs is 37-d and policy must encode
    #    relative geometry between two aircraft. 64-unit nets in our Hover
    #    config underfit here.
    #  - ent_coef=0.01: dogfight has multiple near-optimal turn directions
    #    (left vs right break is symmetric). Zero-entropy PPO tends to
    #    collapse to one and never explore the other; a small bonus keeps
    #    the policy diverse enough that self-play finds robust strategies.
    hp_overrides = dict(
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model = make_algorithm("PPO", train_env, args.seed,
                           tensorboard_log=tb_log, hp_overrides=hp_overrides)

    sp_cb = SelfPlayCallback(
        snapshot_freq=args.snapshot_freq,
        pool_size=args.pool_size,
        snapshot_dir=snapshot_dir,
        algo_cls=PPO,
        verbose=1,
    )
    callbacks = [sp_cb]
    if wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(verbose=2))

    print(f"\n[{run_name}] Training for {args.total_timesteps:,} timesteps")
    print(f"[{run_name}] Snapshot every {args.snapshot_freq:,} steps, "
          f"pool_size={args.pool_size}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=have_progress_bar_deps() and not args.no_progress,
    )

    final_path = os.path.join(args.save_dir, f"final_{run_name}.zip")
    model.save(final_path)
    print(f"\n[{run_name}] Final model: {final_path}")
    print(f"[{run_name}] Snapshots:    {snapshot_dir}/")

    train_env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
