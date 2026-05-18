"""
Train PPO or SAC on QuadX-Hover or QuadX-Waypoints.

Examples
--------
    # Quick smoke test (overrides timesteps for debugging)
    python -m training.train --algo ppo --env hover --seed 0 --total-timesteps 50000

    # Real run — PPO on hover, mode 6 (easiest), seed 0, with W&B
    python -m training.train --algo ppo --env hover --flight-mode 6 --seed 0

    # SAC on waypoints, mode 4
    python -m training.train --algo sac --env waypoints --flight-mode 4 --seed 1

    # Compare flight modes (the report wants this)
    for fm in 0 4 6 7; do
        python -m training.train --algo ppo --env hover --flight-mode $fm --seed 0
    done

Outputs
-------
    models/<run_id>/best_model.zip       best deterministic-eval checkpoint
    models/<run_id>/final_model.zip      last checkpoint
    models/<run_id>/config.yaml          full hyperparams + CLI args (reproducibility)
    logs/<run_id>/                       tensorboard + Monitor csvs + eval npz

run_id = <algo>_<env>_fm<flight_mode>_seed<seed>_<timestamp>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

# Add project root to sys.path so package imports work when launched with
# either `python -m training.train` or `python training/train.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.env_factory import make_single_env  # noqa: E402

ALGO_CLS = {"ppo": PPO, "sac": SAC}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """
    Seed every RNG. SB3 also seeds internally when we pass `seed=`, but we
    set the globals so any code outside SB3 (env construction, callbacks,
    plotting in the same process) is also reproducible.

    Note: full determinism with CUDA is impossible without crippling speed
    (cuDNN benchmark / non-deterministic algorithms). We accept the small
    nondeterminism — it's standard practice and Agarwal et al. (2021)
    recommend reporting variance over multiple seeds anyway.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Vectorized env construction
# ---------------------------------------------------------------------------
def build_vec_env(
    env_name: str,
    n_envs: int,
    flight_mode: int,
    seed: int,
    monitor_dir: Path | None = None,
):
    """
    Build a vectorized env for training.

    PPO benefits a lot from many parallel envs (it's on-policy: more envs ->
    more diverse rollouts per gradient step). SAC is off-policy and gains
    less, so we typically run SAC with n_envs=1.

    SubprocVecEnv runs each env in its own process — significantly faster on
    PyBullet, which releases the GIL during physics steps.
    """
    def _factory(rank: int):
        def _init():
            env = make_single_env(env_name, flight_mode=flight_mode)
            env.reset(seed=seed + rank)
            log_path = str(monitor_dir / f"rank_{rank}") if monitor_dir else None
            return Monitor(env, filename=log_path)
        return _init

    if n_envs == 1:
        vec = DummyVecEnv([_factory(0)])
    else:
        vec = SubprocVecEnv([_factory(i) for i in range(n_envs)])
    return VecMonitor(vec)


# ---------------------------------------------------------------------------
# W&B integration
# ---------------------------------------------------------------------------
class WandbCallback(BaseCallback):
    """
    Lightweight W&B logger. We don't use the official `wandb.integration.sb3`
    callback because it sometimes lags behind SB3 versions; this one just
    mirrors what SB3 already pushes to its internal logger.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._wandb = None

    def _on_training_start(self) -> None:
        import wandb
        self._wandb = wandb

    def _on_rollout_end(self) -> None:
        if self._wandb is None or self._wandb.run is None:
            return
        # Push everything SB3 has logged this rollout. Logger's name_to_value
        # is the dict of latest scalar values.
        record = {k: v for k, v in self.logger.name_to_value.items()
                  if isinstance(v, (int, float))}
        record["global_step"] = self.num_timesteps
        self._wandb.log(record, step=self.num_timesteps)

    def _on_step(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Hyperparam loading
# ---------------------------------------------------------------------------
def load_hparams(config_path: Path, algo: str, env_name: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if algo not in cfg:
        raise KeyError(f"Algo {algo!r} not in {config_path}. Available: {list(cfg)}")
    if env_name not in cfg[algo]:
        raise KeyError(
            f"Env {env_name!r} not configured for {algo}. "
            f"Available: {list(cfg[algo])}"
        )
    return cfg[algo][env_name]


# ---------------------------------------------------------------------------
# Build the algorithm
# ---------------------------------------------------------------------------
def build_algorithm(
    algo: str,
    vec_env,
    hparams: dict,
    seed: int,
    tb_log_dir: Path,
    device: str,
):
    """Instantiate PPO or SAC with the given hyperparameters."""
    cls = ALGO_CLS[algo]

    # Strip non-algo keys from hparams; everything left is passed to the
    # algorithm constructor. This makes adding new hparams to the YAML safe.
    excluded = {"total_timesteps", "n_envs", "eval_freq"}
    algo_kwargs = {k: v for k, v in hparams.items() if k not in excluded}

    return cls(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        verbose=1,
        device=device,
        tensorboard_log=str(tb_log_dir),
        **algo_kwargs,
    )


# ---------------------------------------------------------------------------
# Run id + dirs
# ---------------------------------------------------------------------------
def make_run_id(algo: str, env_name: str, flight_mode: int, seed: int) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{algo}_{env_name}_fm{flight_mode}_seed{seed}_{ts}"


def save_config(out_dir: Path, args: argparse.Namespace, hparams: dict) -> None:
    """Persist exact run config so it's reproducible from the artifact alone."""

    def _safe(v):
        # Pass through plain YAML-friendly scalars; stringify everything else
        # (PosixPath, torch.TorchVersion, etc.).
        if isinstance(v, bool) or v is None:
            return v
        if type(v) in (int, float, str):
            return v
        if isinstance(v, dict):
            return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        return str(v)

    payload = {
        "args": _safe(vars(args)),
        "hparams": _safe(hparams),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=list(ALGO_CLS), required=True)
    p.add_argument("--env", choices=["hover", "waypoints"], required=True)
    p.add_argument("--flight-mode", type=int, default=6,
                   help="PyFlyt flight mode (0=hardest motor PWM, 7=easiest position PID). "
                        "Default 6.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=Path,
                   default=_PROJECT_ROOT / "configs" / "algo_config.yaml")
    p.add_argument("--total-timesteps", type=int, default=None,
                   help="Override total_timesteps from the YAML config")
    p.add_argument("--n-envs", type=int, default=None,
                   help="Override n_envs from the YAML config")
    p.add_argument("--device", default="auto",
                   help="auto / cpu / cuda / cuda:0 ...")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable W&B logging (TensorBoard only)")
    p.add_argument("--wandb-project", default="info8003-rl",
                   help="W&B project name")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--n-eval-episodes", type=int, default=10,
                   help="Episodes per evaluation pass")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    # Resolve hyperparams + apply CLI overrides
    hparams = load_hparams(args.config, args.algo, args.env)
    if args.total_timesteps is not None:
        hparams["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        hparams["n_envs"] = args.n_envs

    total_timesteps = int(hparams["total_timesteps"])
    n_envs = int(hparams["n_envs"])
    eval_freq = int(hparams.get("eval_freq", 25000))

    # Run dirs
    run_id = make_run_id(args.algo, args.env, args.flight_mode, args.seed)
    model_dir = _PROJECT_ROOT / "models" / run_id
    log_dir = _PROJECT_ROOT / "logs" / run_id
    monitor_dir = log_dir / "monitor"
    eval_dir = log_dir / "eval"
    for d in (model_dir, log_dir, monitor_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    save_config(model_dir, args, hparams)

    print(f"[run] {run_id}")
    print(f"[run] device={args.device}  algo={args.algo}  env={args.env}  "
          f"flight_mode={args.flight_mode}  seed={args.seed}")
    print(f"[run] total_timesteps={total_timesteps:,}  n_envs={n_envs}  "
          f"eval_freq={eval_freq:,} (per env)")

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
            save_code=True,
        )

    # Train + eval envs.
    # IMPORTANT: eval env uses seed+10000 so its episodes don't overlap with
    # training rollouts — biased eval otherwise.
    train_env = build_vec_env(
        args.env, n_envs=n_envs, flight_mode=args.flight_mode,
        seed=args.seed, monitor_dir=monitor_dir,
    )
    eval_env = build_vec_env(
        args.env, n_envs=1, flight_mode=args.flight_mode,
        seed=args.seed + 10_000,
    )

    # Algorithm
    model = build_algorithm(
        args.algo, train_env, hparams, args.seed, log_dir, args.device,
    )

    # Callbacks
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(eval_dir),
            eval_freq=eval_freq,             # per env -> total = eval_freq * n_envs
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=max(eval_freq * 2, 50_000),
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
        # Always save the final model + close envs, even if interrupted.
        final_path = model_dir / "final_model.zip"
        model.save(str(final_path))
        print(f"[run] saved final model -> {final_path}")
        train_env.close()
        eval_env.close()
        if use_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
