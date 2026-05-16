"""Shared utilities for RL training across PyFlyt environments.

Anything that's not specific to one environment lives here:
  - env factories (single + vectorized)
  - hyperparameter dicts for PPO / SAC
  - algorithm builder
  - wandb init helper
  - eval callback factory

Centralizing this keeps the per-env training scripts very short and means a
hyperparameter change for one algo applies everywhere.
"""

from __future__ import annotations

import os

import gymnasium
import PyFlyt.gym_envs  # noqa: F401  -- registers PyFlyt env IDs

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from wrappers import FlattenWaypointEnv


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def _make_thunk(env_id, flight_mode, seed_offset, env_kwargs, render_mode):
    """Closure that builds one PyFlyt env. Used to feed SubprocVecEnv/DummyVecEnv.

    Each thunk seeds with `base_seed + seed_offset` so vectorized envs have
    decorrelated trajectories. Monitor is applied so SB3 can log episodic
    rewards/lengths through the callback system.
    """
    def _thunk():
        kwargs = dict(env_kwargs or {})
        env = gymnasium.make(
            env_id, flight_mode=flight_mode,
            render_mode=render_mode, **kwargs,
        )
        # Waypoints env returns a Dict(attitude, target_deltas); flatten so
        # MlpPolicy can consume it. Hover returns Box, so this is a no-op.
        if isinstance(env.observation_space, gymnasium.spaces.Dict):
            env = FlattenWaypointEnv(env, max_waypoints=4)
        env = Monitor(env)
        env.reset(seed=seed_offset)  # seed the env at construction
        return env
    return _thunk


def make_vec_env(env_id, n_envs=4, flight_mode=0, seed=0, env_kwargs=None,
                 use_subproc=False, render_mode=None):
    """Build a vectorized PyFlyt env.

    Use SubprocVecEnv (`use_subproc=True`) when n_envs > 1 to actually run
    rollouts in parallel processes; on a Ryzen with 4 cores this gives ~3x
    speedup over DummyVecEnv for PyBullet-heavy envs. DummyVecEnv (the
    default) is single-process: simpler to debug and avoids PyBullet's
    occasional issues with fork/spawn.
    """
    thunks = [
        _make_thunk(env_id, flight_mode, seed + 1000 * i, env_kwargs, render_mode)
        for i in range(n_envs)
    ]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(thunks, start_method="spawn")
    return DummyVecEnv(thunks)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# These are SB3's defaults for continuous control, which are themselves the
# values from the original PPO/SAC papers. They work as a baseline for all
# three PyFlyt environments. We override only what's needed per-env (e.g.
# Dogfight uses a larger network).

PPO_DEFAULTS = dict(
    policy="MlpPolicy",
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[64, 64]),
)

SAC_DEFAULTS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",                 # automatic entropy tuning (Haarnoja 2018)
    policy_kwargs=dict(net_arch=[256, 256]),
)


def _tb_log_or_none(tensorboard_log):
    """Return tensorboard_log only if the tensorboard package is importable.

    SB3 raises ImportError at learn() time if tensorboard_log is set but
    tensorboard isn't installed. We'd rather log a warning and continue
    (wandb sync still captures metrics through SB3's stdout logger).
    """
    if tensorboard_log is None:
        return None
    try:
        import tensorboard  # noqa: F401
        return tensorboard_log
    except ImportError:
        print("[common] tensorboard not installed — TB logs disabled "
              "(install with: pip install tensorboard)")
        return None


def have_progress_bar_deps():
    """True iff SB3's progress_bar=True will work (needs tqdm + rich)."""
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401
        return True
    except ImportError:
        print("[common] tqdm or rich not installed — progress bar disabled "
              "(install with: pip install tqdm rich)")
        return False


def make_algorithm(algo, env, seed, tensorboard_log=None, hp_overrides=None):
    """Instantiate PPO or SAC with our defaults plus any overrides."""
    tensorboard_log = _tb_log_or_none(tensorboard_log)
    overrides = hp_overrides or {}
    if algo.upper() == "PPO":
        kwargs = {**PPO_DEFAULTS, **overrides,
                  "seed": seed, "tensorboard_log": tensorboard_log,
                  "verbose": 1}
        return PPO(env=env, **kwargs)
    if algo.upper() == "SAC":
        kwargs = {**SAC_DEFAULTS, **overrides,
                  "seed": seed, "tensorboard_log": tensorboard_log,
                  "verbose": 1}
        return SAC(env=env, **kwargs)
    raise ValueError(f"Unknown algo {algo!r} (expected 'PPO' or 'SAC')")


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------

def make_eval_callback(eval_env, save_path, log_path, eval_freq, n_envs,
                       n_eval_episodes=10):
    """EvalCallback that runs the deterministic policy every `eval_freq` steps.

    eval_freq is given in TOTAL env steps; SB3 interprets the callback's
    `eval_freq` arg in TRAIN-env steps per parallel env, hence the division.
    Saves the best model (by mean deterministic reward) to `save_path`.
    """
    return EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=max(eval_freq // max(n_envs, 1), 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )


# ---------------------------------------------------------------------------
# Wandb
# ---------------------------------------------------------------------------

def init_wandb(project, run_name, config, enabled=True):
    """Init a wandb run with tensorboard sync; returns the run or None."""
    if not enabled:
        return None
    import wandb
    return wandb.init(
        project=project,
        name=run_name,
        config=config,
        sync_tensorboard=True,   # mirrors SB3's TB scalars into wandb
        save_code=True,
        reinit=True,
    )


def make_run_name(algo, env_short, seed, suffix=""):
    """Canonical run name used for log dirs, model files, wandb."""
    s = f"{algo}_{env_short}_seed{seed}"
    return f"{s}_{suffix}" if suffix else s
