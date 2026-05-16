"""Train PPO or SAC on PyFlyt/QuadX-Hover-v4 (stabilization task).

Examples:
    # Quick PPO run (~10 min on Ryzen + 1060)
    python scripts/train_hover.py --algo PPO --total-timesteps 500000 --seed 42

    # SAC, longer run, no wandb
    python scripts/train_hover.py --algo SAC --total-timesteps 500000 --seed 42 --no-wandb

    # 3-seed sweep for the report (run sequentially)
    for s in 0 1 2; do
        python scripts/train_hover.py --algo PPO --seed $s
        python scripts/train_hover.py --algo SAC --seed $s
    done
"""

import argparse
import os
import sys

# Allow `from common import ...` etc. when running from anywhere
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


ENV_ID = "PyFlyt/QuadX-Hover-v4"
ENV_SHORT = "Hover"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["PPO", "SAC"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4,
                   help="Parallel envs (PPO only; SAC always uses 1)")
    p.add_argument("--flight-mode", type=int, default=0,
                   help="See project statement Table 1: 0=ang.vel+thrust, 6=easy")
    p.add_argument("--use-subproc", action="store_true",
                   help="Use SubprocVecEnv (faster, but harder to debug)")
    p.add_argument("--eval-freq", type=int, default=20_000,
                   help="Eval every N total env steps")
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
    run_name = make_run_name(args.algo, ENV_SHORT, args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # SAC is off-policy; vectorized rollouts don't help and the replay buffer
    # already provides sample efficiency. PPO is on-policy; more envs => more
    # diverse rollouts per update.
    n_envs = 1 if args.algo == "SAC" else args.n_envs

    train_env = make_vec_env(
        ENV_ID, n_envs=n_envs, flight_mode=args.flight_mode,
        seed=args.seed, use_subproc=args.use_subproc,
    )
    eval_env = make_vec_env(
        ENV_ID, n_envs=1, flight_mode=args.flight_mode,
        seed=args.seed + 9999,  # different seed range from train
    )

    # Wandb config: serialize all CLI args + env id for full reproducibility
    cfg = {**vars(args), "env_id": ENV_ID, "n_envs_actual": n_envs}
    wandb_run = init_wandb(args.wandb_project, run_name, cfg,
                           enabled=not args.no_wandb)

    tb_log = os.path.join(args.log_dir, run_name)
    model = make_algorithm(args.algo, train_env, args.seed, tensorboard_log=tb_log)

    # EvalCallback saves best_model.zip whenever deterministic eval reward
    # exceeds the previous best — this is what we evaluate at the end.
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

    # Save final model regardless of whether it's the best — useful to have
    # both for the report's "training stability" discussion.
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
