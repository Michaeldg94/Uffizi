"""End-to-end pipeline runner for the Uffizi RL course project.

Runs scripts 01 through 07 in order, with budgets controlled by the
--medium flag. Equivalent to running each numbered script by hand.

Quick mode (default): smoke test, runs in minutes on a laptop.
Medium mode (--medium): converged training, full sweeps, ~2 hours on
M3 Pro. Produces publication-quality figures and JSON artifacts in
outputs/.

Usage:
    uv run python run_project.py            # quick smoke run
    uv run python run_project.py --medium   # converged run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_script(script: str, args: list[str]) -> None:
    """Invoke a numbered script as a separate Python subprocess.

    Numbered scripts (01_, 02_, etc.) cannot be imported as Python
    modules because identifiers cannot start with a digit. We invoke
    them as scripts via the same Python interpreter that runs this
    pipeline, so dependencies and the working directory are shared.
    """

    print(f"\n{'=' * 60}")
    print(f"RUNNING: {script} {' '.join(args)}")
    print('=' * 60)

    cmd = [sys.executable, script] + args
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {result.returncode}")


def main() -> None:
    """Run all 7 numbered scripts in order."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medium", action="store_true",
                        help="Converged budgets (~2 hours). Default is smoke run.")
    args = parser.parse_args()

    if args.medium:
        q_episodes = "25000"
        ppo_steps = "500000"
        ppo_seeds = "3"
        sweep_resolution = "7"
        sweep_seeds = "3"
    else:
        q_episodes = "1000"
        ppo_steps = "1024"
        ppo_seeds = "1"
        sweep_resolution = "3"
        sweep_seeds = "1"

    pipeline = "uffizi_rl/pipeline"
    _run_script(f"{pipeline}/01_check_environment.py", [])
    _run_script(f"{pipeline}/02_train_q_learning.py", ["--episodes", q_episodes])
    _run_script(f"{pipeline}/03_train_deep_rl.py", ["--timesteps", ppo_steps, "--seeds", ppo_seeds])
    _run_script(f"{pipeline}/04_run_equilibrium.py", [])
    _run_script(f"{pipeline}/05_evaluate_interventions.py", [])
    _run_script(f"{pipeline}/06_run_sweeps.py", ["--resolution", sweep_resolution, "--seeds", sweep_seeds])
    _run_script(f"{pipeline}/07_make_figures.py", [])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Outputs: outputs/01_*.json through outputs/06_*.json")
    print("Figures: outputs/figures/")


if __name__ == "__main__":
    main()
