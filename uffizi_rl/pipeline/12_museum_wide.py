"""PIPELINE 12: museum-wide results (the CROWD-SIMULATOR stream).

This is a SEPARATE stream from the RL. The agent-based crowd simulator runs the
whole population, the real visitor mix, base crowd vs intervened crowd (the
11-lever Pigovian bundle), at each crowding level. The RL booking agent
(pipelines 07-10) is a different stream that learns to move ON TOP of these
crowds; it does not produce these numbers.

Results are broken out by the THREE real segments (the Instagram tourists are the
majority and must be explicit):
    instagram  -- Instagram tourist (famous-room selfie)
    standard   -- normal tourist
    art_lover  -- connoisseur
For each: completed count, mean welfare per completed visitor, total welfare,
revenue. Plus museum-wide totals (welfare, revenue, peak masterpiece density).

Output: outputs/12_museum_wide.json   CPU-capped.
Usage:  python uffizi_rl/pipeline/12_museum_wide.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
torch.set_num_threads(2)

from uffizi_rl import config
from uffizi_rl.environment.crowd_simulator import CrowdSimulator
from uffizi_rl.analysis.portfolio import combined_intervention_kwargs

CROWDS = [500, 2500, 5000]
LABEL = {500: "500", 2500: "2500", 5000: "max"}
SEEDS = list(range(900000, 900003))               # CRN seeds (full population sim is heavy)
SEGMENTS = ("instagram", "standard", "art_lover")
SEG_LABEL = {"instagram": "Instagram tourist", "standard": "Normal tourist", "art_lover": "Art lover"}
TBF = config.TYPE_B_FRACTION_DEFAULT
IV = combined_intervention_kwargs()


def run_sim(kw, crowd, seed):
    sim = CrowdSimulator(daily_total=crowd, seed=seed, type_a_fraction=1.0 - TBF,
                         heterogeneity_scale=1.0, trail_acceptance_prob=0.30,
                         revenue_model=True, **kw)
    sim.run_day()
    return sim


def per_segment(sim):
    comp = list(sim.completed_visitors)
    out = {}
    for seg in SEGMENTS:
        vs = [v for v in comp if getattr(getattr(v, "profile", None), "segment", None) == seg]
        w = [float(v.experienced_welfare) for v in vs]
        out[seg] = {
            "completed": len(vs),
            "welfare_mean": float(np.mean(w)) if w else 0.0,
            "welfare_total": float(np.sum(w)) if w else 0.0,
            "revenue": float(sum(getattr(v, "ticket_price", 0.0) for v in vs)),
        }
    out["_all"] = {
        "completed": len(comp),
        "welfare_total": float(sum(v.experienced_welfare for v in comp)),
        "revenue": float(sum(getattr(v, "ticket_price", 0.0) for v in comp)),
        "peak_botticelli": None,
    }
    return out


def cell(kw, crowd):
    sims = [run_sim(kw, crowd, s) for s in SEEDS]
    segs = [per_segment(s) for s in sims]
    agg = {}
    for seg in list(SEGMENTS) + ["_all"]:
        keys = segs[0][seg].keys()
        agg[seg] = {k: float(np.mean([sg[seg][k] for sg in segs if sg[seg][k] is not None])) for k in keys if k != "peak_botticelli"}
    return agg


def main():
    print("=== PIPELINE 12: museum-wide crowd-simulator results (base vs intervened) ===", flush=True)
    print(f"  bundle = {len(IV)} levers | seeds = {SEEDS} | Type-B fraction = {TBF}", flush=True)
    out = {"seeds": SEEDS, "crowds": CROWDS, "type_b_fraction": TBF, "cells": {}}
    for crowd in CROWDS:
        print(f"\n  --- crowd = {LABEL[crowd]} ---", flush=True)
        print(f"    {'segment':>17} {'cond':>11} {'completed':>10} {'welfare/visitor':>16} {'welfare_total':>14} {'revenue':>9}", flush=True)
        for cond, kw in (("base", {}), ("intervened", IV)):
            agg = cell(kw, crowd)
            out["cells"][f"{cond}_{crowd}"] = agg
            for seg in SEGMENTS:
                s = agg[seg]
                print(f"    {SEG_LABEL[seg]:>17} {cond:>11} {s['completed']:>10.0f} {s['welfare_mean']:>16.2f} "
                      f"{s['welfare_total']:>14.0f} {s['revenue']:>9.0f}", flush=True)
    (Path("outputs") / "12_museum_wide.json").write_text(json.dumps(out, indent=2))
    print("\n  wrote outputs/12_museum_wide.json", flush=True)
    # per-segment welfare/visitor, base -> intervened
    print("\n  welfare per visitor (base -> intervened):", flush=True)
    for crowd in CROWDS:
        b = out["cells"][f"base_{crowd}"]; i = out["cells"][f"intervened_{crowd}"]
        parts = []
        for seg in SEGMENTS:
            bw, iw = b[seg]["welfare_mean"], i[seg]["welfare_mean"]
            d = 100 * (iw - bw) / abs(bw) if bw else float("nan")
            parts.append(f"{SEG_LABEL[seg]}: {bw:.1f}->{iw:.1f} ({d:+.0f}%)")
        print(f"    {LABEL[crowd]:>5}: " + " | ".join(parts), flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
