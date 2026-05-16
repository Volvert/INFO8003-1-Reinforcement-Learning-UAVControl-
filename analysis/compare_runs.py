"""
Compare training runs and produce report-ready plots + statistics.

Reads SB3 EvalCallback outputs (logs/<run_id>/eval/evaluations.npz) and
Monitor.csv files, aggregates across seeds, and produces:
  - Mean +/- bootstrapped 95% CI of evaluation reward over time
  - Final-performance comparison with Welch's t-test between algorithms
  - Per-flight-mode comparison plots

Following Agarwal et al. (2021) "Deep RL at the Edge of the Statistical
Precipice", we use:
  - bootstrap CIs instead of Gaussian std error (robust to small N)
  - report median + IQM, not just mean (mean is sensitive to outliers)
  - report all individual seeds, not just aggregates

Usage
-----
    # Compare two algorithms on hover (after running multiple seeds of each)
    python -m analysis.compare_runs --runs logs/ppo_hover_*  logs/sac_hover_* \\
                                    --labels PPO SAC \\
                                    --output report/figures/hover_comparison.png

    # Compare flight modes for PPO on hover
    python -m analysis.compare_runs --runs logs/ppo_hover_fm0_* logs/ppo_hover_fm6_* \\
                                    --labels "mode 0" "mode 6"
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_eval_curve(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load the (timesteps, mean_reward_per_eval) curve produced by EvalCallback.

    Returns None if the eval log is missing (e.g. crashed run).
    """
    eval_path = run_dir / "eval" / "evaluations.npz"
    if not eval_path.exists():
        return None
    data = np.load(eval_path)
    timesteps = data["timesteps"]                # shape (n_evals,)
    results = data["results"]                    # shape (n_evals, n_eval_episodes)
    mean_per_eval = results.mean(axis=1)
    return timesteps, mean_per_eval


def load_runs(patterns: list[str]) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    """Expand glob patterns and load every matching run."""
    runs = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            run_dir = Path(p)
            curve = load_eval_curve(run_dir)
            if curve is None:
                print(f"  [skip] {run_dir.name}: no eval log")
                continue
            t, r = curve
            runs.append((run_dir, t, r))
            print(f"  [load] {run_dir.name}: {len(t)} eval points, "
                  f"final reward = {r[-1]:.2f}")
    return runs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000,
                 ci: float = 0.95, rng: np.random.Generator | None = None
                 ) -> tuple[float, float, float]:
    """
    Bootstrap mean and CI. Robust for small N (we typically have 3-5 seeds).

    Returns (mean, lower, upper).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return values.mean(), lo, hi


def interquartile_mean(values: np.ndarray) -> float:
    """
    IQM: mean of the middle 50% of values. Recommended by Agarwal et al.
    as more robust than the plain mean.
    """
    if len(values) < 4:
        return float(values.mean()) if len(values) else float("nan")
    q1, q3 = np.quantile(values, [0.25, 0.75])
    middle = values[(values >= q1) & (values <= q3)]
    return float(middle.mean())


def welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test (unequal variance). Returns (t, p)."""
    from scipy import stats
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


# ---------------------------------------------------------------------------
# Curve aggregation across seeds
# ---------------------------------------------------------------------------
def aggregate_curves(runs: list[tuple[Path, np.ndarray, np.ndarray]]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate eval curves across seeds.

    Different runs may have eval points at slightly different timesteps (eval
    fires after a complete rollout, which doesn't always land exactly on
    eval_freq * n_envs). We interpolate every run onto a common grid.

    Returns (grid, mean, lower_ci, upper_ci) — all shape (n_grid,).
    """
    all_t = np.concatenate([t for _, t, _ in runs])
    grid = np.linspace(all_t.min(), all_t.max(), 100)

    interpolated = np.stack([
        np.interp(grid, t, r) for _, t, r in runs
    ], axis=0)  # shape (n_seeds, n_grid)

    mean = interpolated.mean(axis=0)
    rng = np.random.default_rng(0)
    cis = np.array([
        bootstrap_ci(interpolated[:, i], rng=rng) for i in range(len(grid))
    ])
    return grid, mean, cis[:, 1], cis[:, 2]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(groups: dict[str, list[tuple[Path, np.ndarray, np.ndarray]]],
                    output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)

    colors = plt.cm.tab10.colors
    for i, (label, runs) in enumerate(groups.items()):
        if not runs:
            print(f"  [warn] no runs for {label!r}")
            continue
        grid, mean, lo, hi = aggregate_curves(runs)
        ax.plot(grid, mean, label=f"{label} (n={len(runs)})", color=colors[i % len(colors)])
        ax.fill_between(grid, lo, hi, color=colors[i % len(colors)], alpha=0.2)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean evaluation reward (deterministic)")
    ax.set_title("Training curves (mean +/- bootstrap 95% CI)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"[plot] saved -> {output}")


# ---------------------------------------------------------------------------
# Final-performance table
# ---------------------------------------------------------------------------
def final_performance_table(groups: dict[str, list[tuple[Path, np.ndarray, np.ndarray]]]
                            ) -> None:
    print("\nFinal performance (mean of last 5 eval points per seed):")
    print(f"  {'group':<20} {'n':>3}  {'mean':>10}  {'IQM':>10}  "
          f"{'95% CI':>22}")
    print("  " + "-" * 75)

    finals: dict[str, np.ndarray] = {}
    for label, runs in groups.items():
        if not runs:
            continue
        per_seed_final = np.array([r[-5:].mean() for _, _, r in runs])
        finals[label] = per_seed_final
        m, lo, hi = bootstrap_ci(per_seed_final)
        iqm = interquartile_mean(per_seed_final)
        print(f"  {label:<20} {len(per_seed_final):>3}  {m:>10.2f}  "
              f"{iqm:>10.2f}  [{lo:>8.2f}, {hi:>8.2f}]")

    # Pairwise t-tests
    labels = list(finals.keys())
    if len(labels) >= 2:
        print("\nPairwise Welch's t-tests on final performance:")
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                t, p = welch_t_test(finals[labels[i]], finals[labels[j]])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {labels[i]:<20} vs {labels[j]:<20}  "
                      f"t={t:>+6.2f}  p={p:.4f}  {sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="Glob pattern(s), one per group. Use spaces to separate groups.")
    p.add_argument("--labels", nargs="+", required=True,
                   help="One label per group, in matching order.")
    p.add_argument("--output", type=Path,
                   default=Path("report/figures/comparison.png"))
    args = p.parse_args()

    if len(args.runs) != len(args.labels):
        print("ERROR: --runs and --labels must have the same number of entries.")
        sys.exit(1)

    print("Loading runs:")
    groups = {}
    for label, pattern in zip(args.labels, args.runs):
        print(f"\n  group {label!r} from {pattern!r}")
        groups[label] = load_runs([pattern])

    plot_comparison(groups, args.output)
    final_performance_table(groups)


if __name__ == "__main__":
    main()
