"""SCRIPT 04: Population equilibrium and Price of Anarchy (Phase 5).

Approximates the population game equilibrium by iterated best response
(visitors gradually adjust their strategies based on observed congestion)
and computes the Price of Anarchy: the welfare ratio between the
selfish equilibrium and a coordinator-optimized outcome.

The headline result (PoA approx 1.004 in our simulator) tells us the
Uffizi's congestion problem is NOT a coordination failure but a
capacity problem at the Botticelli rooms.

Run AFTER 03. Writes:
  outputs/04_equilibrium.json

Usage:
    uv run python uffizi_rl/pipeline/04_run_equilibrium.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from uffizi_rl import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from uffizi_rl import config  # noqa: E402
from uffizi_rl.analysis.metrics import price_of_anarchy  # noqa: E402
from uffizi_rl.analysis.phase_transition import (  # noqa: E402
    population_rollout_equilibrium,
    simulate_day_metrics,
)
from uffizi_rl.pipeline._paths import ensure_outputs_dir  # noqa: E402


def main() -> None:
    """Run iterated best response and compute the Price of Anarchy."""

    output_dir = ensure_outputs_dir()

    daily_total = config.DAILY_VISITORS_NORMAL
    seed = config.DEFAULT_SEED + 30

    print("[1/2] Iterated best-response rollout (4 iterations)")
    records = population_rollout_equilibrium(
        iterations=4, seed=seed, daily_total=daily_total,
        type_b_fraction=config.TYPE_B_FRACTION_DEFAULT,
    )
    for i, rec in enumerate(records):
        print(f"  iter {i}: welfare={rec['total_welfare']:.0f}, "
              f"botticelli_overcrowded_frac={rec['botticelli_over80_frac']:.3f}")

    print("[2/2] Social-optimum proxy and PoA")
    social_opt = simulate_day_metrics(
        type_b_fraction=config.TYPE_B_FRACTION_DEFAULT,
        daily_total=daily_total,
        seed=seed + 99,
        heterogeneity_scale=1.7,       # high preference diversity
        trail_acceptance_prob=0.55,    # high trail adoption
    )
    poa = price_of_anarchy(
        social_optimum_welfare=social_opt["total_welfare"],
        equilibrium_welfare=records[-1]["total_welfare"],
    )
    print(f"  social_opt_welfare={social_opt['total_welfare']:.0f}")
    print(f"  Price of Anarchy = {poa:.4f}")

    summary = {
        "iterations": records,
        "social_optimum_proxy": social_opt,
        "price_of_anarchy": poa,
    }
    out_path = output_dir / "04_equilibrium.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
