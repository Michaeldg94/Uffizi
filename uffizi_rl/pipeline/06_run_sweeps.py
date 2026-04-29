"""SCRIPT 06: Population sweeps and tipping-point analysis (Phase 7).

Sweeps three parameters of the population game:
  1. Type B fraction (0.0 to 1.0): how many visitors are guidebook tourists
  2. Daily volume (1200 to 12000 visitors): how busy the museum is
  3. Preference heterogeneity (0.4 to 1.8): how diverse Type A tastes are

Each sweep reveals the operating boundaries: at what parameter values
does the system tip from comfortable to congested? These curves are
the empirical equivalent of comparative statics in economics.

Run AFTER 05. Writes:
  outputs/06_sweeps.json

Usage:
    uv run python 06_run_sweeps.py                  # quick (3 points per sweep)
    uv run python uffizi_rl/pipeline/06_run_sweeps.py --resolution 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from uffizi_rl import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from uffizi_rl import config  # noqa: E402
from uffizi_rl.analysis.phase_transition import (  # noqa: E402
    infer_tipping_point,
    run_heterogeneity_sweep,
    run_type_b_ratio_sweep,
    run_volume_sweep,
)
from uffizi_rl.pipeline._paths import ensure_outputs_dir  # noqa: E402


def main() -> None:
    """Run the three population sweeps and detect tipping points."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=3,
                        help="Number of points per sweep. 3=quick, 7=medium, 11=full.")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of random seeds per point. 1=quick, 3=medium, 5=full.")
    args = parser.parse_args()

    output_dir = ensure_outputs_dir()

    seeds_list = tuple(range(1, args.seeds + 1))
    ratios = tuple(np.linspace(0.0, 1.0, args.resolution))
    if args.resolution <= 3:
        volumes = (1200, 5000, 9000)
    elif args.resolution <= 7:
        volumes = (1200, 3500, 5000, 7000, 10000)
    else:
        volumes = (1200, 2500, 4000, 5500, 8000, 10000, 12000)

    print(f"[1/3] Type B ratio sweep ({len(ratios)} points, {len(seeds_list)} seeds)")
    ratio_records = run_type_b_ratio_sweep(ratios=ratios, seeds=seeds_list)
    for rec in ratio_records:
        print(f"  type_b={rec['type_b_ratio']:.2f}: welfare={rec['total_welfare_mean']:.0f}")

    print(f"[2/3] Volume sweep ({len(volumes)} points)")
    volume_records = run_volume_sweep(volumes=volumes, seeds=seeds_list)
    for rec in volume_records:
        print(f"  volume={rec['daily_total']:.0f}: welfare={rec['total_welfare_mean']:.0f}")

    print("[3/3] Heterogeneity sweep")
    heter_records = run_heterogeneity_sweep(seeds=seeds_list)

    type_b_tip = infer_tipping_point(ratio_records, key="total_welfare_mean")
    volume_tip = infer_tipping_point(volume_records, key="total_welfare_mean")
    print(f"\nTipping points: type_b={type_b_tip}, volume={volume_tip}")

    summary = {
        "type_b_ratio_sweep": ratio_records,
        "volume_sweep": volume_records,
        "heterogeneity_sweep": heter_records,
        "type_b_tipping_point": type_b_tip,
        "volume_tipping_point": volume_tip,
    }
    out_path = output_dir / "06_sweeps.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
