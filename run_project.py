"""End-to-end pipeline runner for the Uffizi RL course project.

Runs scripts 01 through 07 in order, with budgets controlled by the
--medium flag. Equivalent to running each numbered script by hand.

Quick mode (default): smoke test, runs in minutes on a laptop.
Medium mode (--medium): converged training, full sweeps, ~2 hours on
M3 Pro. Produces publication-quality figures and JSON artifacts in
outputs/.

Reseed mode (--reseed): reproduce the multi-seed reseed (n=5) FROM SCRATCH
via scripts/reseed_reproduce.py instead of the 01-07 pipeline. Trains seeds
1, 3, 7, 202606 on CPU (per-cell resumable), assembles each seed's booking
matrix and toy Q result, sanity-checks them, and prints the grand average.

Usage:
    uv run python run_project.py            # quick smoke run
    uv run python run_project.py --medium   # converged run
    uv run python run_project.py --reseed   # reproduce the n=5 reseed (hours; resumable)
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
    parser.add_argument("--reseed", action="store_true",
                        help="Reproduce the multi-seed reseed (n=5) from scratch via "
                             "scripts/reseed_reproduce.py, instead of the 01-07 pipeline.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="(with --reseed) training seeds. Default: 1 3 7 202606.")
    parser.add_argument("--workers", type=int, default=None,
                        help="(with --reseed) parallel CPU workers for the 120-cell training phase.")
    parser.add_argument("--skip-train", action="store_true",
                        help="(with --reseed) only assemble/sanity/average from cached models.")
    args = parser.parse_args()

    if args.reseed:
        reseed_args: list[str] = []
        if args.seeds:
            reseed_args += ["--seeds", *[str(s) for s in args.seeds]]
        if args.workers:
            reseed_args += ["--workers", str(args.workers)]
        if args.skip_train:
            reseed_args += ["--skip-train"]
        _run_script("scripts/reseed_reproduce.py", reseed_args)
        return

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
