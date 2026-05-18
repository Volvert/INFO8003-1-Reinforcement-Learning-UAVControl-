"""
Tournament submission. Rename this file to groupXX_<name>.py for submission.

The grader (scripts/tournament.py) imports this module and calls load_model().
Whatever load_model() returns must implement:

    action, info = model.predict(obs, deterministic=True)
        obs    : np.ndarray, shape (37,)        -- dogfight observation
        action : np.ndarray, shape (4,)         -- [roll, pitch, yaw, throttle] in [-1, 1]
        info   : any                            -- can be {} (unused)

Re-exporting the model
----------------------
Before submitting, re-export your final .zip from a venv that matches the
grading server's requirements.txt EXACTLY. Pickled SB3 checkpoints embed
version-specific module paths; mismatched numpy / torch / sb3 versions can
fail to load at grading time. Loading + saving is enough — no retraining.

    from stable_baselines3 import PPO
    PPO.load("models/<run_id>/best_model.zip").save(
        "submissions/groupXX_aggressive.zip"
    )
"""
from __future__ import annotations

import os
from pathlib import Path

# Path is relative to this .py file so it works regardless of CWD when the
# grader imports the module.
_HERE = Path(__file__).resolve().parent
_DEFAULT_CHECKPOINT = _HERE / "group56_aggressive.zip"


def load_model(path: str | os.PathLike | None = None):
    """
    Load and return the trained dogfight policy.

    The checkpoint must have been saved with a Box(37,) observation space
    and Box(4,) action space (matches the tournament environment's
    flatten_observation=True, team_size=1, agent_hz=30 setup).
    """
    from stable_baselines3 import PPO  # SAC also works if you used SAC

    checkpoint = Path(path) if path is not None else _DEFAULT_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Tournament checkpoint not found: {checkpoint}. "
            f"Place your re-exported .zip alongside this .py file or pass "
            f"the path explicitly."
        )
    return PPO.load(str(checkpoint))


if __name__ == "__main__":
    # Quick smoke test: load and run one predict() call to catch issues
    # before the grader runs.
    import numpy as np

    model = load_model()
    obs = np.zeros(37, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    print(f"OK — action shape {action.shape}, dtype {action.dtype}")
    print(f"action sample: {action}")
