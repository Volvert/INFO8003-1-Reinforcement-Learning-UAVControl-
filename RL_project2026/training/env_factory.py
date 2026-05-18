"""
Training-side environment factory.

Single entry point for building training and evaluation envs. Wraps the
provided scripts/env_config.py and scripts/wrappers.py so callers don't have
to know about Dict observations or the dogfight self-play wrapper.

The grading script (scripts/evaluate.py) builds envs with the EXACT same
recipe (gymnasium.make + FlattenWaypointEnv + get_env_kwargs), so models
trained here will be evaluated on a matching observation space.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import gymnasium
import PyFlyt.gym_envs  # noqa: F401  -- registers PyFlyt envs with gymnasium

# Make the provided scripts/ directory importable regardless of where this
# module is imported from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from env_config import get_env_kwargs                          # noqa: E402
from wrappers import FlattenWaypointEnv                        # noqa: E402

ENV_IDS = {
    "hover": "PyFlyt/QuadX-Hover-v4",
    "waypoints": "PyFlyt/QuadX-Waypoints-v4",
}


def make_single_env(
    env_name: str,
    flight_mode: int = 6,
    render_mode: str | None = None,
    seed: int | None = None,
    **extra_kwargs: Any,
) -> gymnasium.Env:
    """
    Build a Gymnasium env for hover or waypoints.

    Mirrors scripts/evaluate.py:make_env exactly:
        gymnasium.make(env_id, flight_mode=..., render_mode=..., **get_env_kwargs(name))
        wrap with FlattenWaypointEnv if obs is Dict
    """
    if env_name not in ENV_IDS:
        raise ValueError(
            f"Unknown env_name {env_name!r}. Use 'hover' or 'waypoints', "
            f"or call make_dogfight_env() for dogfight."
        )

    env_kwargs = get_env_kwargs(env_name)
    env_kwargs.update(extra_kwargs)
    env = gymnasium.make(
        ENV_IDS[env_name],
        flight_mode=flight_mode,
        render_mode=render_mode,
        **env_kwargs,
    )

    if isinstance(env.observation_space, gymnasium.spaces.Dict):
        env = FlattenWaypointEnv(env, max_waypoints=4)

    if seed is not None:
        env.reset(seed=seed)
    return env


def make_dogfight_env(
    opponent_policy=None,
    render_mode: str | None = None,
    seed: int | None = None,
    **extra_kwargs: Any,
) -> gymnasium.Env:
    """
    Build the single-agent self-play wrapper around MAFixedwingDogfightEnvV2.

    Tournament kwargs are pinned by default to match what we are graded on
    (see scripts/tournament.py and the project statement). Override only if
    you have a specific reason — and document it in the report.
    """
    # Late import: keeps the dogfight dependency chain off the critical path
    # for hover/waypoints training.
    from dogfight_wrapper import DogfightSelfPlayEnv

    tournament_defaults = dict(
        team_size=1,
        flatten_observation=True,
        max_duration_seconds=60,
        agent_hz=30,
    )
    tournament_defaults.update(extra_kwargs)

    env = DogfightSelfPlayEnv(
        opponent_policy=opponent_policy,
        render_mode=render_mode,
        **tournament_defaults,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env
