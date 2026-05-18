#!/bin/bash
#
# Submit the full experiment sweep from PROJECT_PLAN.md:
#
#   1. Hover, PPO + SAC, 3 seeds, flight mode 6        (6 jobs)
#   2. Waypoints, PPO, 3 seeds, modes {0, 4, 6, 7}    (12 jobs)
#   3. Waypoints, SAC, 3 seeds, mode 6                 (3 jobs)
#   4. Dogfight, PPO with self-play, 2 seeds           (2 jobs)
#                                                     ----------
#                                                     23 jobs total
#
# Usage:
#   bash slurm/submit_all.sh           # submit everything
#   bash slurm/submit_all.sh hover     # only hover runs
#   bash slurm/submit_all.sh waypts    # only waypoints
#   bash slurm/submit_all.sh dogfight  # only dogfight
#   bash slurm/submit_all.sh --dry-run # just print sbatch commands
#
# Run from the project root:
#   cd /home/USER/runprojRL/rl-project
#   bash slurm/submit_all.sh
#

set -euo pipefail

mkdir -p "${PWD}/logs/slurm"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

DRY_RUN=0
WHICH="all"
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        hover|waypts|waypoints|dogfight|all) WHICH="$arg" ;;
        *) echo "Unknown arg: $arg" ; exit 1 ;;
    esac
done

submit() {
    # submit <script> <export-vars-string> <human-label>
    local script="$1" exports="$2" label="$3"
    local cmd=(sbatch --export="ALL,${exports}" --job-name="${label}" "${script}")
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[dry-run] ${cmd[*]}"
    else
        echo "[submit] ${label}"
        "${cmd[@]}"
    fi
}

# ---------------------------------------------------------------------------
# 1. Hover: PPO + SAC, 3 seeds, mode 6
# ---------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "hover" ]]; then
    echo "==> Hover sweep (6 jobs)"
    for ALGO in ppo sac; do
        for SEED in 0 1 2; do
            submit "${SCRIPT_DIR}/train_hover.slurm" \
                "ALGO=${ALGO},FLIGHT_MODE=6,SEED=${SEED}" \
                "h_${ALGO}_s${SEED}"
        done
    done
fi

# ---------------------------------------------------------------------------
# 2. Waypoints: PPO, 3 seeds, modes {0,4,6,7} -> flight-mode comparison
#    + Waypoints SAC, mode 6, 3 seeds
# ---------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "waypts" || "${WHICH}" == "waypoints" ]]; then
    echo "==> Waypoints PPO flight-mode sweep (12 jobs)"
    for FM in 0 4 6 7; do
        for SEED in 0 1 2; do
            submit "${SCRIPT_DIR}/train_waypoints.slurm" \
                "ALGO=ppo,FLIGHT_MODE=${FM},SEED=${SEED}" \
                "w_ppo_fm${FM}_s${SEED}"
        done
    done

    echo "==> Waypoints SAC, mode 6, 3 seeds (3 jobs)"
    for SEED in 0 1 2; do
        submit "${SCRIPT_DIR}/train_waypoints.slurm" \
            "ALGO=sac,FLIGHT_MODE=6,SEED=${SEED}" \
            "w_sac_s${SEED}"
    done
fi

# ---------------------------------------------------------------------------
# 3. Dogfight: PPO + self-play, 2 seeds
# ---------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "dogfight" ]]; then
    echo "==> Dogfight self-play (2 jobs)"
    for SEED in 0 1; do
        submit "${SCRIPT_DIR}/train_dogfight.slurm" \
            "SEED=${SEED}" \
            "df_s${SEED}"
    done
fi

echo
echo "Done. Useful commands:"
echo "  squeue -u \$USER                            # see your queue"
echo "  scancel -u \$USER                           # cancel all jobs"
echo "  tail -f logs/slurm/<jobname>_<jobid>.out   # watch a job's stdout"
