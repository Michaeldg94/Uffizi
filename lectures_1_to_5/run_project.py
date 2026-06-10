"""End-to-end runner for the Uffizi RL course project.

Runs the pipeline scripts in order:
  01_check_environment.py  -- MDP formulation, graph validation, simulator calibration
  02_train_q_learning.py   -- Tabular Q-learning on the 12-room toy graph + baselines
  07_make_figures.py       -- Generate all figures from the saved outputs

This is a smoke run by default. Pass --episodes 25000 for a converged
Q-learning training (takes a few minutes).

Usage:
    uv run python run_project.py
    uv run python run_project.py --episodes 25000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_script(script: str, args: list[str]) -> None:
    """Invoke a numbered script as a separate Python subprocess.

    Numbered scripts (01_, 02_, etc.) cannot be imported as Python modules
    because identifiers cannot start with a digit. We invoke them as
    scripts via the same Python interpreter that runs this pipeline.
    """

    print(f"\n{'=' * 60}")
    print(f"RUNNING: {script} {' '.join(args)}")
    print('=' * 60)
    cmd = [sys.executable, script] + args
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {result.returncode}")


def main() -> None:
    """Run scripts 01 and 02, then 07 (figures)."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Q-learning episodes. 1000=smoke, 25000=converged.")
    args = parser.parse_args()

    pipeline = "uffizi_rl/pipeline"
    _run_script(f"{pipeline}/01_check_environment.py", [])
    _run_script(f"{pipeline}/02_train_q_learning.py", ["--episodes", str(args.episodes)])
    _run_script(f"{pipeline}/07_make_figures.py", [])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Outputs: outputs/01_environment.json, outputs/02_q_learning.json")
    print("Figures: outputs/figures/01_*.png + 02_*.png (11 figures total)")


if __name__ == "__main__":
    main()
