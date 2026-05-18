"""Train PPO or SAC on PyFlyt/QuadX-Waypoints-v4 (navigation task).

Differences vs Hover trainer:
  - Pulls env_kwargs from env_config.WAYPOINT_ENV_KWARGS (less restrictive
    parameters than PyFlyt defaults — see env_config.py for justification).
  - Default flight_mode=6 (easy: ground-frame velocity control). For the
    flight-mode comparison required in the report, also run with mode=0.
  - Larger default timesteps budget — navigation needs more samples than
    pure stabilization.

Examples:
    # PPO with mode 6 (easy) — for the comparison
    python scripts/train_waypoints.py --algo PPO --flight-mode 6 --seed 42

    # PPO with mode 0 (hard) — same env, harder action space
    python scripts/train_waypoints.py --algo PPO --flight-mode 0 --seed 42 --total-timesteps 3000000

    # 3-seed × 2-mode × 2-algo sweep for the report (run sequentially)
    for s in 0 1 2; do
      for m in 0 6; do
        for a in PPO SAC; do
          python scripts/train_waypoints.py --algo $a --flight-mode $m --seed $s
        done
      done
    done
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3.common.callbacks import CallbackList

from common import (
    have_progress_bar_deps,
    init_wandb,
    make_algorithm,
    make_eval_callback,
    make_run_name,
    make_vec_env,
)
from env_config import get_env_kwargs


ENV_ID = "PyFlyt/QuadX-Waypoints-v4"
ENV_SHORT = "Waypoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["PPO", "SAC"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--flight-mode", type=int, default=6,
                   help="6=ground-frame velocity (easy), 0=angular vel+thrust (hard)")
    p.add_argument("--use-subproc", action="store_true")
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--save-dir", default="results/models")
    p.add_argument("--log-dir", default="results/logs")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable rich progress bar (cleaner output when "
                        "PyBullet logs interfere with the bar)")
    p.add_argument("--wandb-project", default="info8003-pyflyt")
    return p.parse_args()


def main():
    args = parse_args()
    # Suffix encodes flight_mode so different modes get distinct runs/artifacts.
    run_name = make_run_name(args.algo, ENV_SHORT, args.seed,
                             suffix=f"mode{args.flight_mode}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env_kwargs = get_env_kwargs("waypoints")
    n_envs = 1 if args.algo == "SAC" else args.n_envs

    train_env = make_vec_env(
        ENV_ID, n_envs=n_envs, flight_mode=args.flight_mode,
        seed=args.seed, env_kwargs=env_kwargs, use_subproc=args.use_subproc,
    )
    eval_env = make_vec_env(
        ENV_ID, n_envs=1, flight_mode=args.flight_mode,
        seed=args.seed + 9999, env_kwargs=env_kwargs,
    )

    cfg = {**vars(args), "env_id": ENV_ID, "n_envs_actual": n_envs,
           "env_kwargs": env_kwargs}
    wandb_run = init_wandb(args.wandb_project, run_name, cfg,
                           enabled=not args.no_wandb)

    tb_log = os.path.join(args.log_dir, run_name)
    model = make_algorithm(args.algo, train_env, args.seed, tensorboard_log=tb_log)

    eval_cb = make_eval_callback(
        eval_env=eval_env,
        save_path=os.path.join(args.save_dir, f"best_{run_name}"),
        log_path=os.path.join(args.log_dir, f"eval_{run_name}"),
        eval_freq=args.eval_freq,
        n_envs=n_envs,
        n_eval_episodes=args.n_eval_episodes,
    )
    callbacks = [eval_cb]
    if wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(verbose=2))

    print(f"\n[{run_name}] Training for {args.total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=have_progress_bar_deps() and not args.no_progress,
    )

    final_path = os.path.join(args.save_dir, f"final_{run_name}.zip")
    model.save(final_path)
    print(f"\n[{run_name}] Final model: {final_path}")
    print(f"[{run_name}] Best model:  {args.save_dir}/best_{run_name}/best_model.zip")

    train_env.close()
    eval_env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
