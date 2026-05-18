"""
Roll out a trained policy with rendering. Useful for visualizing what a
trained agent actually does — embed in your report or W&B as a video.

Usage
-----
    python -m analysis.record_video --model models/ppo_hover_*/best_model.zip --env hover
    python -m analysis.record_video --model models/ppo_hover_*/best_model.zip --env waypoints --flight-mode 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.env_factory import make_single_env  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--env", choices=["hover", "waypoints"], required=True)
    p.add_argument("--flight-mode", type=int, default=6)
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from stable_baselines3 import PPO, SAC
    model = None
    for cls in [PPO, SAC]:
        try:
            model = cls.load(str(args.model))
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(f"Could not load {args.model} as PPO or SAC")

    env = make_single_env(args.env, flight_mode=args.flight_mode, render_mode="human")
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
        print(f"Episode {ep}: reward={total_reward:.2f}, steps={steps}")
    env.close()


if __name__ == "__main__":
    main()
