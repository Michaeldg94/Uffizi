"""SCRIPT 05: Intervention evaluation and portfolio optimization (Phase 6).

Evaluates the curated Pigovian / common-pool-governance interventions
by activating them one at a time in the crowd simulator and measuring
welfare (= appreciation minus peak-masterpiece-congestion externality
cost), revenue, peak Botticelli density, and experience quality. Then
runs greedy forward selection + 1-opt local search to find the best
portfolio.

The catalog has been filtered to externality-targeting interventions
only: those that tax the congestion externality (group surcharge),
cap the behavior that creates it (group cap, slot cap, timed entry,
quiet hours), or redistribute demand away from the four masterpiece
rooms during peak hours. Non-Pigovian "be nice to locals" interventions
(Last-Hour Locals, Resident Annual Pass, Lunch Free Entry, etc.) have
been removed because they do not reduce peak masterpiece density.

Run AFTER 04. Writes:
  outputs/05_interventions.json

Usage:
    uv run python uffizi_rl/pipeline/05_evaluate_interventions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from uffizi_rl import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from uffizi_rl import config  # noqa: E402
from uffizi_rl.analysis.phase_transition import simulate_day_metrics  # noqa: E402
from uffizi_rl.analysis.portfolio import (  # noqa: E402
    combined_intervention_kwargs,
    curated_intervention_specs,
    optimize_portfolio,
    screen_interventions,
)
from uffizi_rl.pipeline._paths import ensure_outputs_dir  # noqa: E402


def main() -> None:
    """Evaluate each curated intervention and find the best portfolio."""

    output_dir = ensure_outputs_dir()

    daily_total = config.DAILY_VISITORS_NORMAL
    seed = config.DEFAULT_SEED + 40
    common = {
        "type_b_fraction": config.TYPE_B_FRACTION_DEFAULT,
        "daily_total": daily_total,
        "revenue_model": True,
    }

    print("[1/3] Evaluating each curated intervention individually")
    rows = [{"intervention": "Status Quo (baseline)", **{
        k: v for k, v in simulate_day_metrics(seed=seed, **common).items()
        if k in {"total_welfare", "revenue", "peak_botticelli_density", "occupancy_gini"}
    }}]
    for spec in curated_intervention_specs():
        m = simulate_day_metrics(seed=seed, **common, **spec.kwargs)
        rows.append({
            "intervention": spec.label,
            "total_welfare": m["total_welfare"],
            "revenue": m.get("revenue", 0.0),
            "peak_botticelli_density": m["peak_botticelli_density"],
            "occupancy_gini": m["occupancy_gini"],
        })
    rows.append({"intervention": "All Interventions Combined", **{
        k: v for k, v in simulate_day_metrics(seed=seed, **common, **combined_intervention_kwargs()).items()
        if k in {"total_welfare", "revenue", "peak_botticelli_density", "occupancy_gini"}
    }})

    baseline_w = rows[0]["total_welfare"]
    baseline_r = rows[0]["revenue"]
    print(f"  Baseline: welfare={baseline_w:.0f}, revenue={baseline_r:.0f}")
    print(f"  All combined: welfare={rows[-1]['total_welfare']:.0f} ({rows[-1]['total_welfare']-baseline_w:+.0f}), "
          f"revenue={rows[-1]['revenue']:.0f} ({rows[-1]['revenue']-baseline_r:+.0f})")

    print("[2/3] Greedy + local-search portfolio optimization")
    portfolio = optimize_portfolio(seed=seed, daily_total=daily_total, lambda_revenue=0.01)
    print(f"  active interventions: {portfolio.active_interventions}")
    print(f"  portfolio welfare: {portfolio.welfare:.0f} ({portfolio.welfare - baseline_w:+.0f})")
    print(f"  portfolio revenue: {portfolio.revenue:.0f} ({portfolio.revenue - baseline_r:+.0f})")

    print("[3/3] Full screen ranked by welfare delta")
    screen = screen_interventions(seed=seed, daily_total=daily_total)
    for row in screen[:5]:
        print(f"  {row['name']:30s} delta_w={row['delta_w']:+.1f}, delta_r={row['delta_r']:+.1f}")

    summary = {
        "baseline_welfare": baseline_w,
        "baseline_revenue": baseline_r,
        "table": rows,
        "portfolio": {
            "active": portfolio.active_interventions,
            "welfare": portfolio.welfare,
            "revenue": portfolio.revenue,
        },
        "ranked_screen": screen,
    }
    out_path = output_dir / "05_interventions.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
