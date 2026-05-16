"""Visually watch a dogfight match between two checkpoints (PyBullet GUI).

By default plays a model against itself. Pass --opponent to specify a
different model — useful for seeing how much your final policy improved
over an early snapshot.

Usage:
  # Self-play: model vs same model
  python scripts/watch_dogfight.py --model results/models/final_PPO_Dogfight_seed42.zip

  # Compare current vs an early snapshot (visualize learning progress)
  python scripts/watch_dogfight.py ^
      --model results/models/final_PPO_Dogfight_seed42.zip ^
      --opponent results/models/snapshots_PPO_Dogfight_seed42/snap_000200000.zip

  # vs random (sanity check — your policy should crush this)
  python scripts/watch_dogfight.py --model PATH --opponent random

Controls (PyBullet GUI):
  - Mouse drag: rotate camera
  - Mouse wheel: zoom
  - Right-drag: pan
  - Esc: close window (will start next game)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class _RandomPolicy:
    """Stub policy with the SB3 .predict signature, samples uniformly."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=False):
        return self.action_space.sample(), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to .zip checkpoint controlling uav_0")
    parser.add_argument("--opponent", default=None,
                        help="Path to .zip for uav_1 (default: same as --model). "
                             "Use 'random' for a random-action opponent.")
    parser.add_argument("--n-games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-delay", type=float, default=0.0,
                        help="Seconds to sleep between sim steps (0=full speed)")
    args = parser.parse_args()

    # Imports here so PyBullet's stdout banner doesn't print on --help
    from stable_baselines3 import PPO
    from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2

    print(f"Loading model A (uav_0): {args.model}")
    model_a = PPO.load(args.model, device="cpu")

    if args.opponent is None:
        print("Model B (uav_1): same as A (self-play)")
        model_b = model_a
    elif args.opponent.lower() == "random":
        print("Model B (uav_1): random policy")
        model_b = "random"   # resolved per-env once we know action_space
    else:
        print(f"Model B (uav_1): {args.opponent}")
        model_b = PPO.load(args.opponent, device="cpu")

    wins_a = wins_b = draws = 0

    for game in range(args.n_games):
        print(f"\n=== Game {game + 1}/{args.n_games} (seed={args.seed + game}) ===")

        env = MAFixedwingDogfightEnvV2(
            team_size=1,
            assisted_flight=True,
            flatten_observation=True,
            render_mode="human",
            max_duration_seconds=60.0,
            agent_hz=30,
        )

        # Resolve random stub now that we have an action space
        if model_b == "random":
            model_b_real = _RandomPolicy(env.action_space("uav_1"))
        else:
            model_b_real = model_b

        observations, infos = env.reset(seed=args.seed + game)
        rewards_acc = {a: 0.0 for a in env.agents}
        max_steps = 60 * 30   # 60s × 30Hz = 1800 steps

        for step in range(max_steps):
            if not env.agents:
                break
            actions = {}
            for agent in env.agents:
                policy = model_a if agent == "uav_0" else model_b_real
                action, _ = policy.predict(observations[agent], deterministic=True)
                actions[agent] = action
            observations, rewards, terms, truncs, infos = env.step(actions)
            for a in rewards:
                rewards_acc[a] = rewards_acc.get(a, 0.0) + rewards[a]
            if args.step_delay > 0:
                time.sleep(args.step_delay)
            if len(env.agents) <= 1:
                break

        r_a = rewards_acc.get("uav_0", 0.0)
        r_b = rewards_acc.get("uav_1", 0.0)
        if r_a > r_b:
            winner = "A (uav_0)"; wins_a += 1
        elif r_b > r_a:
            winner = "B (uav_1)"; wins_b += 1
        else:
            winner = "draw"; draws += 1

        print(f"  Steps: {step + 1} ({(step + 1) / 30:.1f}s)")
        print(f"  uav_0 cumulative reward: {r_a:+.1f}")
        print(f"  uav_1 cumulative reward: {r_b:+.1f}")
        print(f"  Winner: {winner}")
        env.close()

    print(f"\n=== Final tally over {args.n_games} games ===")
    print(f"  A wins:  {wins_a}")
    print(f"  B wins:  {wins_b}")
    print(f"  Draws:   {draws}")


if __name__ == "__main__":
    main()
