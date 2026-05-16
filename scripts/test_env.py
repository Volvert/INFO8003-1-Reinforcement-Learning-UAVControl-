"""Smoke test: verify all three envs import + step + return expected shapes.

Run this FIRST after installing requirements, before any training. Catches
PyFlyt/PyBullet install issues, version mismatches, and shape surprises in
~30 seconds rather than 10 minutes into a training run.

    python scripts/test_env.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium
import numpy as np
import PyFlyt.gym_envs  # noqa: F401

from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv


def smoke_test_gym(env_id, flight_mode, env_kwargs=None, name=""):
    print(f"\n[{name}] env_id={env_id}, flight_mode={flight_mode}")
    env = gymnasium.make(env_id, flight_mode=flight_mode, **(env_kwargs or {}))
    if isinstance(env.observation_space, gymnasium.spaces.Dict):
        env = FlattenWaypointEnv(env, max_waypoints=4)
    print(f"  obs_space:    {env.observation_space}")
    print(f"  action_space: {env.action_space}")

    obs, info = env.reset(seed=0)
    print(f"  reset obs shape: {np.asarray(obs).shape}")

    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
    print(f"  50-step random total reward: {total_reward:.2f}")
    env.close()


def smoke_test_dogfight():
    print("\n[Dogfight] DogfightSelfPlayEnv (single-agent wrapper)")
    from dogfight_wrapper import DogfightSelfPlayEnv
    env = DogfightSelfPlayEnv(team_size=1, opponent_policy=None)
    print(f"  obs_space:    {env.observation_space}")
    print(f"  action_space: {env.action_space}")
    obs, info = env.reset(seed=0)
    print(f"  reset obs shape: {np.asarray(obs).shape}")
    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
    print(f"  50-step random total reward: {total_reward:.2f}")
    env.close()


def main():
    smoke_test_gym("PyFlyt/QuadX-Hover-v4", flight_mode=0, name="Hover")
    smoke_test_gym("PyFlyt/QuadX-Waypoints-v4", flight_mode=6,
                   env_kwargs=get_env_kwargs("waypoints"), name="Waypoints")
    smoke_test_dogfight()
    print("\nAll envs OK. You're ready to train.")


if __name__ == "__main__":
    main()
