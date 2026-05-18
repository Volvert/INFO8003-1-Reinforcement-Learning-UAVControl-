"""
compare_all_runs.py
-------------------
Automatically generates all comparison plots and .txt statistics files
for the report. Run from the project root:

    python analysis/compare_all_runs.py

For each comparison, the script checks whether valid logs exist before
running. If no logs are found for a group, it is silently skipped.
All outputs are saved to figures/.

Comparisons generated
---------------------
  hover_ppo_vs_sac     : PPO vs SAC on Hover (mode 6)
  waypoints_ppo_vs_sac : PPO vs SAC on Waypoints (mode 6)
  waypoints_modes      : PPO on Waypoints, modes 0 / 4 / 6 / 7
"""
from __future__ import annotations

import glob
import io
import os
import sys
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ANALYSIS_DIR.parent
sys.path.insert(0, str(_ANALYSIS_DIR))

import compare_runs as cr

COMPARISONS = [
    (
        "hover_ppo_vs_sac",
        [
            ("logs/ppo_hover_fm6_seed*", "PPO"),
            ("logs/sac_hover_fm6_seed*", "SAC"),
        ],
    ),
    (
        "waypoints_ppo_vs_sac",
        [
            ("logs/ppo_waypoints_fm6_seed*", "PPO"),
            ("logs/sac_waypoints_fm6_seed*", "SAC"),
        ],
    ),
    (
        "waypoints_modes",
        [
            ("logs/ppo_waypoints_fm0_seed*", "mode 0"),
            ("logs/ppo_waypoints_fm4_seed*", "mode 4"),
            ("logs/ppo_waypoints_fm6_seed*", "mode 6"),
            ("logs/ppo_waypoints_fm7_seed*", "mode 7"),
        ],
    ),
]

OUTPUT_DIR = _PROJECT_ROOT / "figures"


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()


def main():
    os.chdir(_PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for stem, groups_def in COMPARISONS:
        print(f"\n{'='*60}")
        print(f"  {stem}")
        print(f"{'='*60}")

        valid_groups = []
        for pattern, label in groups_def:
            valid = [
                m for m in glob.glob(pattern)
                if (Path(m) / "eval" / "evaluations.npz").exists()
            ]
            if valid:
                valid_groups.append((pattern, label))
                print(f"  [ok]   {label}: {len(valid)} run(s) found")
            else:
                print(f"  [skip] {label}: no valid logs -- skipped")

        if len(valid_groups) < 2:
            print(f"  -> Not enough valid groups, comparison skipped.")
            continue

        patterns = [p for p, _ in valid_groups]
        labels   = [l for _, l in valid_groups]
        output   = OUTPUT_DIR / f"{stem}.png"
        txt_out  = OUTPUT_DIR / f"{stem}.txt"

        buffer   = io.StringIO()
        original = sys.stdout
        sys.stdout = Tee(original, buffer)

        groups = {}
        print("Loading runs:")
        for label, pattern in zip(labels, patterns):
            print(f"\n  group {label!r} from {pattern!r}")
            groups[label] = cr.load_runs([pattern])

        cr.plot_comparison(groups, output)
        cr.final_performance_table(groups)

        sys.stdout = original

        with open(txt_out, "w") as f:
            f.write(buffer.getvalue())

        print(f"\n  -> PNG : {output}")
        print(f"  -> TXT : {txt_out}")

    print(f"\n{'='*60}")
    print("  Done. All files saved to figures/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()