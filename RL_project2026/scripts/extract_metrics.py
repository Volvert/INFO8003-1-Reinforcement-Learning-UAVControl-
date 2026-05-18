"""Extract a comprehensive metric report for one or more trained models.

For each (algo, env, seed) run, pulls together:

  A. Checkpoint metadata
       - SB3 / torch / numpy versions, training device
       - net architecture, total trainable params

  B. Behavioural eval (NEW rollout, deterministic policy, N episodes)
       - mean / median / std / IQR / min / max of episode return
       - mean / std of episode length
       - crash rate (fraction of episodes ending in big negative reward)
       - waypoints reached (Waypoints env only)
       - kill / death / draw rates (Dogfight env only, vs random opponent
         and vs self for sanity)

  C. Training-time metrics from tensorboard event files
       - final ep_rew_mean (last 100 rollouts)
       - peak ep_rew_mean
       - timestep at which 50% of peak was first reached (sample efficiency)
       - final policy entropy (PPO: -entropy_loss; SAC: log of ent_coef)
       - final explained_variance (PPO only, value-fn quality)
       - mean approx_kl over last 100 updates (PPO only, update size)

These together give you everything you need to populate the report's
results tables and to write defensible "why method A beat method B"
discussion paragraphs.

Usage:
  # Single checkpoint
  python scripts/extract_metrics.py --model results/models/final_PPO_Dogfight_seed42.zip

  # Every checkpoint in a directory (one summary JSON per run)
  python scripts/extract_metrics.py --models-dir results/models --output-dir results/metrics

  # Print a comparison table of all extracted metrics
  python scripts/extract_metrics.py --compare results/metrics
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Run-id parsing  (kept consistent with analyze.py / eval_all.py)
# ---------------------------------------------------------------------------
_PATTERN = re.compile(
    r"(final|best)_(?P<algo>PPO|SAC)_(?P<env>Hover|Waypoints|Dogfight)"
    r"_seed(?P<seed>\d+)(?:_mode(?P<mode>\d+))?"
)


def parse_run_id(model_path):
    base = os.path.basename(model_path).replace(".zip", "")
    if base in ("best_model", "model"):
        base = os.path.basename(os.path.dirname(model_path))
    m = _PATTERN.search(base)
    if not m:
        return None
    return {
        "algo": m.group("algo"),
        "env": m.group("env"),
        "seed": int(m.group("seed")),
        "mode": int(m.group("mode")) if m.group("mode") else None,
        "kind": m.group(1),     # "final" or "best"
    }


def run_name_from_info(info):
    """Reverse of parse_run_id — used to find matching TB log directory."""
    parts = [info["algo"], info["env"], f"seed{info['seed']}"]
    if info.get("mode") is not None:
        parts.append(f"mode{info['mode']}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# A. Checkpoint metadata
# ---------------------------------------------------------------------------

def checkpoint_metadata(model_path):
    """Read the SB3 zip without loading the model — fast and dependency-free.

    SB3 zips contain `system_info.txt` with the train-time environment, and
    `data` (JSON) with hyperparameters. We extract both.
    """
    import zipfile
    info = {}
    with zipfile.ZipFile(model_path, "r") as z:
        with z.open("system_info.txt") as f:
            info["system_info"] = f.read().decode("utf-8").strip()
        with z.open("_stable_baselines3_version") as f:
            info["sb3_version"] = f.read().decode("utf-8").strip()
        with z.open("data") as f:
            data = json.loads(f.read().decode("utf-8"))
        # data contains nested base64 blobs; pluck out human-readable bits.
        for k in ("learning_rate", "gamma", "gae_lambda", "n_steps",
                  "batch_size", "n_epochs", "ent_coef", "vf_coef",
                  "clip_range", "tau", "buffer_size", "policy_class"):
            if k in data:
                v = data[k]
                # Some entries are wrapped {":type:": ..., ":serialized:": ...}
                if isinstance(v, dict) and ":serialized:" in v:
                    v = v.get("py/object", v.get(":type:", "<serialized>"))
                info[f"hp_{k}"] = v if isinstance(v, (int, float, str, bool)) else str(v)[:100]

    # Total trainable params via torch.load — only thing that needs torch.
    try:
        import torch
        with zipfile.ZipFile(model_path, "r") as z:
            with z.open("policy.pth") as f:
                # weights_only=False because policy.pth contains optimizer state etc.
                state = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            n_params = sum(v.numel() for v in state.values()
                           if hasattr(v, "numel"))
            info["n_trainable_params"] = int(n_params)
    except Exception as e:
        info["n_trainable_params_error"] = str(e)
    return info


# ---------------------------------------------------------------------------
# B. Behavioural evaluation
# ---------------------------------------------------------------------------

def eval_quadx(model_path, env_id, n_episodes=20, flight_mode=0,
               env_kwargs=None, seed_base=10000):
    """Run deterministic policy in env, collect detailed stats."""
    import gymnasium
    import PyFlyt.gym_envs  # noqa: F401
    from stable_baselines3 import PPO, SAC
    from wrappers import FlattenWaypointEnv

    # Try PPO then SAC (one will succeed)
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception:
        model = SAC.load(model_path, device="cpu")

    env = gymnasium.make(env_id, flight_mode=flight_mode, **(env_kwargs or {}))
    if isinstance(env.observation_space, gymnasium.spaces.Dict):
        env = FlattenWaypointEnv(env, max_waypoints=4)

    rewards, lengths, crashes, waypoints = [], [], [], []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_base + i)
        ep_r, ep_len, last_r, last_info = 0.0, 0, 0.0, info
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r; ep_len += 1; last_r = r; last_info = info
            if term or trunc:
                break
        rewards.append(ep_r)
        lengths.append(ep_len)
        # Heuristic: large negative terminal reward => crash
        crashes.append(last_r <= -50)
        waypoints.append(last_info.get("num_targets_reached", 0))
    env.close()

    rewards = np.asarray(rewards); lengths = np.asarray(lengths)
    out = {
        "n_episodes":       int(n_episodes),
        "reward_mean":      float(rewards.mean()),
        "reward_std":       float(rewards.std(ddof=1)),
        "reward_median":    float(np.median(rewards)),
        "reward_iqm":       float(_iqm(rewards)),
        "reward_min":       float(rewards.min()),
        "reward_max":       float(rewards.max()),
        "length_mean":      float(lengths.mean()),
        "length_std":       float(lengths.std(ddof=1)),
        "crash_rate":       float(np.mean(crashes)),
        "all_rewards":      [float(x) for x in rewards],
    }
    if "Waypoints" in env_id:
        out["waypoints_mean"] = float(np.mean(waypoints))
        out["waypoints_max"]  = int(np.max(waypoints))
    return out


def eval_dogfight(model_path, n_episodes=10, seed_base=20000):
    """Dogfight eval: head-to-head against (a) self and (b) random."""
    from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")

    def play(opponent_kind):
        wins_a = wins_b = draws = 0
        rewards_a = []; lengths = []
        for i in range(n_episodes):
            env = MAFixedwingDogfightEnvV2(
                team_size=1, assisted_flight=True,
                flatten_observation=True, render_mode=None,
                max_duration_seconds=60.0, agent_hz=30,
            )
            obs, _ = env.reset(seed=seed_base + i)
            r_acc = {a: 0.0 for a in env.agents}
            steps = 0
            while env.agents and steps < 60 * 30:
                actions = {}
                for agent in env.agents:
                    if agent == "uav_0":
                        a, _ = model.predict(obs[agent], deterministic=True)
                    else:
                        if opponent_kind == "self":
                            a, _ = model.predict(obs[agent], deterministic=True)
                        else:    # random
                            a = env.action_space(agent).sample()
                    actions[agent] = a
                obs, r, term, trunc, _info = env.step(actions)
                for ag in r: r_acc[ag] = r_acc.get(ag, 0) + r[ag]
                steps += 1
                if len(env.agents) <= 1: break
            env.close()
            ra = r_acc.get("uav_0", 0.0); rb = r_acc.get("uav_1", 0.0)
            rewards_a.append(ra); lengths.append(steps)
            if ra > rb: wins_a += 1
            elif rb > ra: wins_b += 1
            else: draws += 1
        return {
            "n_games":      n_episodes,
            "wins_uav0":    wins_a,
            "wins_uav1":    wins_b,
            "draws":        draws,
            "winrate_uav0": wins_a / n_episodes,
            "reward_mean_uav0": float(np.mean(rewards_a)),
            "length_mean":  float(np.mean(lengths)),
        }

    return {
        "vs_self":   play("self"),
        "vs_random": play("random"),
    }


def _iqm(x):
    """Interquartile mean — what rliable / Agarwal et al. use."""
    x = np.sort(np.asarray(x))
    n = len(x); lo = n // 4; hi = n - lo
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


# ---------------------------------------------------------------------------
# C. Tensorboard log parsing
# ---------------------------------------------------------------------------

def tensorboard_metrics(log_dir):
    """Parse TB event files and pull the most useful scalar series.

    Note: this requires `tensorboard` (or `tbparse`) to read events. We try
    `tbparse` first because it's cleaner; fall back to tensorboard.
    """
    if not os.path.isdir(log_dir):
        return {"error": f"no such dir: {log_dir}"}
    try:
        from tbparse import SummaryReader
        reader = SummaryReader(log_dir, extra_columns={"dir_name"})
        df = reader.scalars
    except ImportError:
        # Fallback: use tensorboard's own EventAccumulator
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return {"error": "neither tbparse nor tensorboard installed; "
                             "pip install tbparse"}
        # Find the deepest dir containing event files
        event_dirs = []
        for root, _dirs, files in os.walk(log_dir):
            if any(f.startswith("events.out.tfevents") for f in files):
                event_dirs.append(root)
        if not event_dirs:
            return {"error": "no tfevents found"}
        # Use the newest one (training reuses log_dir but EvalCallback uses a sub-folder)
        best_dir = max(event_dirs, key=lambda d: os.path.getmtime(d))
        ea = EventAccumulator(best_dir, size_guidance={"scalars": 0})
        ea.Reload()
        rows = []
        for tag in ea.Tags()["scalars"]:
            for ev in ea.Scalars(tag):
                rows.append({"tag": tag, "step": ev.step, "value": ev.value})
        if not rows:
            return {"error": "no scalars found"}
        import pandas as pd
        df = pd.DataFrame(rows)

    if df.empty:
        return {"error": "empty TB log"}

    out = {}
    def last_n_mean(tag, n=100):
        sub = df[df["tag"] == tag].sort_values("step")
        return float(sub["value"].tail(n).mean()) if len(sub) else None

    def peak(tag):
        sub = df[df["tag"] == tag]
        return float(sub["value"].max()) if len(sub) else None

    def first_step_above(tag, threshold):
        sub = df[df["tag"] == tag].sort_values("step")
        hits = sub[sub["value"] >= threshold]
        return int(hits["step"].iloc[0]) if not hits.empty else None

    # Series we care about. SB3 names them 'rollout/...' and 'train/...'.
    out["final_ep_rew_mean"]    = last_n_mean("rollout/ep_rew_mean")
    out["peak_ep_rew_mean"]     = peak("rollout/ep_rew_mean")
    out["final_ep_len_mean"]    = last_n_mean("rollout/ep_len_mean")
    out["final_explained_var"]  = last_n_mean("train/explained_variance")
    out["final_approx_kl"]      = last_n_mean("train/approx_kl")
    out["final_clip_fraction"]  = last_n_mean("train/clip_fraction")
    out["final_entropy_loss"]   = last_n_mean("train/entropy_loss")
    out["final_value_loss"]     = last_n_mean("train/value_loss")
    out["final_policy_std"]     = last_n_mean("train/std")  # PPO Gaussian policy std

    # Sample-efficiency proxy: when did we first hit 50% of peak reward?
    if out["peak_ep_rew_mean"] is not None:
        out["timestep_at_50pct_peak"] = first_step_above(
            "rollout/ep_rew_mean", 0.5 * out["peak_ep_rew_mean"]
        )
    return out


# ---------------------------------------------------------------------------
# Per-model orchestration
# ---------------------------------------------------------------------------

def extract_one(model_path, logs_dir, n_eval_episodes, n_dogfight_games):
    info = parse_run_id(model_path)
    out = {"model_path": model_path, "run_info": info}

    # A. checkpoint metadata
    out["checkpoint"] = checkpoint_metadata(model_path)

    # B. behavioural eval
    if info is None:
        out["error"] = "couldn't parse run id"
        return out
    if info["env"] == "Hover":
        env_id = "PyFlyt/QuadX-Hover-v4"; env_kwargs = {}
    elif info["env"] == "Waypoints":
        env_id = "PyFlyt/QuadX-Waypoints-v4"
        from env_config import WAYPOINT_ENV_KWARGS
        env_kwargs = WAYPOINT_ENV_KWARGS.copy()
    else:
        env_id = None; env_kwargs = None

    if env_id is not None:
        try:
            out["behavioural"] = eval_quadx(
                model_path, env_id,
                n_episodes=n_eval_episodes,
                flight_mode=info.get("mode") or 0,
                env_kwargs=env_kwargs,
            )
        except Exception as e:
            out["behavioural_error"] = repr(e)
    else:
        try:
            out["behavioural"] = eval_dogfight(model_path, n_dogfight_games)
        except Exception as e:
            out["behavioural_error"] = repr(e)

    # C. tensorboard
    if logs_dir is not None and info is not None:
        run_log_dir = os.path.join(logs_dir, run_name_from_info(info))
        out["tensorboard"] = tensorboard_metrics(run_log_dir)
    return out


# ---------------------------------------------------------------------------
# Compare mode: print a table from extracted JSONs
# ---------------------------------------------------------------------------

def compare(metrics_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(metrics_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        info = data.get("run_info") or {}
        beh = data.get("behavioural", {}) or {}
        tb  = data.get("tensorboard", {}) or {}
        if "vs_self" in beh:    # dogfight
            row = {
                "run":        os.path.basename(path).replace(".json", ""),
                "env":        info.get("env"),
                "algo":       info.get("algo"),
                "seed":       info.get("seed"),
                "winrate_vs_random": beh["vs_random"]["winrate_uav0"],
                "rew_self":   beh["vs_self"]["reward_mean_uav0"],
                "final_rew":  tb.get("final_ep_rew_mean"),
                "peak_rew":   tb.get("peak_ep_rew_mean"),
            }
        else:
            row = {
                "run":         os.path.basename(path).replace(".json", ""),
                "env":         info.get("env"),
                "algo":        info.get("algo"),
                "seed":        info.get("seed"),
                "mode":        info.get("mode"),
                "rew_mean":    beh.get("reward_mean"),
                "rew_std":     beh.get("reward_std"),
                "rew_iqm":     beh.get("reward_iqm"),
                "len_mean":    beh.get("length_mean"),
                "crash_rate":  beh.get("crash_rate"),
                "waypoints":   beh.get("waypoints_mean"),
                "final_rew_tb":  tb.get("final_ep_rew_mean"),
                "peak_rew_tb":   tb.get("peak_ep_rew_mean"),
                "expl_var":    tb.get("final_explained_var"),
                "approx_kl":   tb.get("final_approx_kl"),
                "step50pct":   tb.get("timestep_at_50pct_peak"),
            }
        rows.append(row)
    if not rows:
        print("No metric JSONs found in", metrics_dir)
        return
    # Print as Markdown-ish table
    keys = list(rows[0].keys())
    widths = {k: max(len(k), max(len(_fmt(r.get(k))) for r in rows)) for k in keys}
    print(" | ".join(k.ljust(widths[k]) for k in keys))
    print("-+-".join("-" * widths[k] for k in keys))
    for r in rows:
        print(" | ".join(_fmt(r.get(k)).ljust(widths[k]) for k in keys))


def _fmt(x):
    if x is None: return "-"
    if isinstance(x, float): return f"{x:.3g}"
    return str(x)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Single .zip checkpoint")
    p.add_argument("--models-dir", help="Process every final_*.zip / best_*/ in dir")
    p.add_argument("--logs-dir", default="results/logs")
    p.add_argument("--output-dir", default="results/metrics")
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--n-dogfight-games", type=int, default=10)
    p.add_argument("--compare", help="Compare-mode: table from JSON dir")
    args = p.parse_args()

    if args.compare:
        compare(args.compare)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    targets = []
    if args.model:
        targets.append(args.model)
    if args.models_dir:
        targets.extend(sorted(glob.glob(os.path.join(args.models_dir, "final_*.zip"))))
        targets.extend(sorted(glob.glob(os.path.join(args.models_dir, "best_*", "best_model.zip"))))
    if not targets:
        print("Pass --model or --models-dir.")
        return

    for path in targets:
        info = parse_run_id(path)
        run_name = run_name_from_info(info) if info else os.path.basename(path)
        print(f"\n>>> {run_name}")
        result = extract_one(path, args.logs_dir, args.n_eval_episodes, args.n_dogfight_games)
        out_path = os.path.join(args.output_dir, f"{run_name}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    saved -> {out_path}")
        # Quick console summary
        beh = result.get("behavioural", {})
        if "reward_mean" in beh:
            print(f"    reward = {beh['reward_mean']:.2f} ± {beh['reward_std']:.2f}  "
                  f"crash_rate = {beh['crash_rate']*100:.0f}%")
        elif "vs_random" in beh:
            print(f"    vs_random winrate = {beh['vs_random']['winrate_uav0']*100:.0f}%, "
                  f"vs_self mean reward = {beh['vs_self']['reward_mean_uav0']:.1f}")

    print(f"\nDone. Compare with:  python scripts/extract_metrics.py --compare {args.output_dir}")


if __name__ == "__main__":
    main()
