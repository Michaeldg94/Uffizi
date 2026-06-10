"""Exhaustive grid-search over all 2^9 intervention combinations.

Each atomic intervention is on/off; we evaluate every combination on the
same simulator seed for fair comparison. Reports the top-K combos by
welfare, by revenue, and by a scalarized welfare+revenue objective.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uffizi_rl.analysis.portfolio import _evaluate
from uffizi_rl.config import DAILY_VISITORS_NORMAL, DEFAULT_SEED

ATOMIC_INTERVENTIONS = [
    ("tour_group_cap", {"tour_group_cap": 10}),
    ("timed_entry", {"timed_entry": True}),
    ("quiet_hours", {"quiet_hours": True}),
    ("group_surcharge", {"per_person_group_surcharge": 150.0}),
    ("rama", {"rama": True}),
    ("secondary_enrichment", {"secondary_attractor_enrichment": True}),
    ("extended_hours", {"extended_hours": True}),
    ("dynamic_pricing", {"dynamic_pricing": True}),
    ("annual_pass", {"resident_annual_pass": True}),
]

N = len(ATOMIC_INTERVENTIONS)
TOTAL = 1 << N

print(f"Sweeping {TOTAL} combinations of {N} interventions...")
t0 = time.time()

results = []
for bits in range(TOTAL):
    kwargs = {}
    active_names = []
    for i, (name, kw) in enumerate(ATOMIC_INTERVENTIONS):
        if bits & (1 << i):
            kwargs.update(kw)
            active_names.append(name)
    metrics = _evaluate(kwargs, seed=DEFAULT_SEED, daily_total=DAILY_VISITORS_NORMAL)
    results.append({
        "bits": bits,
        "active": active_names,
        "welfare": float(metrics["total_welfare"]),
        "revenue": float(metrics["revenue"]),
    })
    if (bits + 1) % 32 == 0:
        elapsed = time.time() - t0
        eta = elapsed * (TOTAL - bits - 1) / (bits + 1)
        print(f"  {bits + 1}/{TOTAL}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

# Baseline (no interventions)
baseline = next(r for r in results if not r["active"])
bw, br = baseline["welfare"], baseline["revenue"]

# Sort and print top-10 by various criteria.
def header(title):
    print(f"\n=== {title} ===")
    print(f"{'rank':>4} {'welfare':>8} {'Δw':>7} {'revenue':>9} {'Δr (EUR)':>10} active")


header("TOP 10 BY WELFARE")
for i, r in enumerate(sorted(results, key=lambda x: -x["welfare"])[:10], 1):
    print(f"{i:>4} {r['welfare']:>8.0f} {r['welfare']-bw:>+7.1f} {r['revenue']:>9.0f} {r['revenue']-br:>+10.0f} {','.join(r['active']) or '(baseline)'}")

header("TOP 10 BY REVENUE")
for i, r in enumerate(sorted(results, key=lambda x: -x["revenue"])[:10], 1):
    print(f"{i:>4} {r['welfare']:>8.0f} {r['welfare']-bw:>+7.1f} {r['revenue']:>9.0f} {r['revenue']-br:>+10.0f} {','.join(r['active']) or '(baseline)'}")

header("TOP 10 BY WELFARE+REVENUE (each Δ as %% of baseline)")
def score(r):
    return (r["welfare"] - bw) / bw + (r["revenue"] - br) / br
for i, r in enumerate(sorted(results, key=lambda x: -score(x))[:10], 1):
    print(f"{i:>4} {r['welfare']:>8.0f} {r['welfare']-bw:>+7.1f} {r['revenue']:>9.0f} {r['revenue']-br:>+10.0f} {','.join(r['active']) or '(baseline)'}")

header("TOP 10 PARETO FRONTIER (both welfare and revenue improve)")
pareto = [r for r in results if r["welfare"] > bw and r["revenue"] > br]
for i, r in enumerate(sorted(pareto, key=lambda x: -score(x))[:10], 1):
    print(f"{i:>4} {r['welfare']:>8.0f} {r['welfare']-bw:>+7.1f} {r['revenue']:>9.0f} {r['revenue']-br:>+10.0f} {','.join(r['active']) or '(baseline)'}")

# Write the full sweep to disk for the LaTeX writeup.
out_path = Path("outputs/exhaustive_sweep.json")
out_path.parent.mkdir(exist_ok=True, parents=True)
with out_path.open("w") as fh:
    json.dump({"baseline_welfare": bw, "baseline_revenue": br, "results": results}, fh, indent=2)

print(f"\nElapsed total: {time.time() - t0:.0f}s")
print(f"Written: {out_path}")
