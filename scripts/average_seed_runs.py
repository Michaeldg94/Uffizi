"""GRAND AVERAGE across the reseeded runs (booking algorithm matrix + toy Q-learning).

Reads every per-seed file in outputs/seeds/ and computes, across training seeds, the
mean and standard deviation of each metric. Two kinds of file are understood:

  results_rl_booking_seed*.json   (from reseed_booking_grid.py): per cell
      {art, tourist} x {500, 2500, max}, the 5 algorithm arms
      (ppo_base, dqn_base, mppo_int, ppo_int, dqn_int). Reports each arm's points
      mean +/- std, plus the headline grid (mppo_int intervened vs ppo_base baseline,
      with the % lift recomputed from the cell means).

  q_learning_seed*.json           (a copy of outputs/02_q_learning.json per seed):
      averages the vanilla Q-learning and Double-Q eval mean_return across seeds, so
      the "does Double-Q help" comparison gets error bars too.

Optionally folds in the original committed single-seed booking run as one more seed:
  --include-canonical   also reads outputs/results_rl_booking.json (the seed-42 run)

Writes outputs/results_rl_booking_grandavg.json (+ a toy section) and prints the grids.
The "n" column is how many training seeds went into each cell, so a missing profile,
crowd, or arm is obvious.

  uv run python scripts/average_seed_runs.py
  uv run python scripts/average_seed_runs.py --include-canonical
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

OUTDIR = ROOT / "outputs" / "seeds"
CANONICAL = ROOT / "outputs" / "results_rl_booking.json"
PROFILES = ("art", "tourist")
CROWDS = ("500", "2500", "5000")
LABEL = {"500": "500", "2500": "2500", "5000": "max"}
ARMS = ("ppo_base", "dqn_base", "mppo_int", "ppo_int", "dqn_int")


def _mean_std(vals):
    arr = np.array(vals, dtype=float)
    if arr.size == 0:
        return None, None, 0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return round(float(arr.mean()), 1), round(std, 1), arr.size


def _arm_points(cell, arm):
    """Points for an arm in a cell, tolerating both the new per-arm schema and the
    old headline-only schema (baseline_points / intervened_points)."""
    arms = cell.get("arms", {})
    if arm in arms and arms[arm].get("points") is not None:
        return float(arms[arm]["points"])
    if arm == "ppo_base" and cell.get("baseline_points") is not None:
        return float(cell["baseline_points"])
    if arm == "mppo_int" and cell.get("intervened_points") is not None:
        return float(cell["intervened_points"])
    return None


def _checkins(cell):
    arms = cell.get("arms", {})
    if "mppo_int" in arms and arms["mppo_int"].get("checkins_of_3") is not None:
        return float(arms["mppo_int"]["checkins_of_3"])
    if cell.get("checkins_of_3") is not None:
        return float(cell["checkins_of_3"])
    return None


def average_matrix(files):
    runs = {}
    for f in files:
        data = json.loads(Path(f).read_text())
        seed = str(data.get("train_seed", Path(f).stem))
        cells = {}
        for profile, by_crowd in data.get("profiles", {}).items():
            for crowd, cell in by_crowd.items():
                cells[(profile, crowd)] = cell
        runs[seed] = cells
    seeds = sorted(runs)
    print(f"=== BOOKING MATRIX: grand average across {len(runs)} seed(s): {', '.join(seeds)} ===\n",
          flush=True)

    grand = {"description": ("Per-cell, per-arm mean +/- std across training seeds. Each seed = one "
                             "reseed_booking_grid.py run; eval = fixed CRN crowds (900000..905). "
                             "Headline % recomputed from the mppo_int and ppo_base means."),
             "n_seeds": len(runs), "seeds": seeds, "profiles": {}}

    for profile in PROFILES:
        print(f"{profile.upper()}", flush=True)
        print(f"  {'crowd':>5} " + " ".join(f"{a:>14}" for a in ARMS) + f" {'vs base':>9} {'n':>3}", flush=True)
        grand["profiles"][profile] = {}
        for crowd in CROWDS:
            agg = {"arms": {}}
            n_cell = 0
            for arm in ARMS:
                pts = [p for p in (_arm_points(runs[s].get((profile, crowd), {}), arm) for s in seeds)
                       if p is not None]
                m, sd, n = _mean_std(pts)
                if n:
                    agg["arms"][arm] = {"points_mean": m, "points_std": sd, "n_seeds": n}
                    n_cell = max(n_cell, n)
            ci = [c for c in (_checkins(runs[s].get((profile, crowd), {})) for s in seeds) if c is not None]
            ci_m, ci_sd, ci_n = _mean_std(ci)
            base_m = agg["arms"].get("ppo_base", {}).get("points_mean")
            int_m = agg["arms"].get("mppo_int", {}).get("points_mean")
            if base_m is not None and int_m is not None and base_m:
                agg["baseline_points_mean"] = base_m
                agg["intervened_points_mean"] = int_m
                agg["pct_vs_baseline_from_means"] = round(100 * (int_m - base_m) / abs(base_m), 1)
            if ci_m is not None:
                agg["checkins_of_3_mean"] = ci_m
                agg["checkins_of_3_std"] = ci_sd
            grand["profiles"][profile][crowd] = agg
            cells = " ".join(
                (f"{agg['arms'][a]['points_mean']:>7.0f}+/-{agg['arms'][a]['points_std']:<5.0f}"
                 if a in agg["arms"] else f"{'--':>14}")
                for a in ARMS)
            pct = agg.get("pct_vs_baseline_from_means")
            pct_s = f"{pct:>+8.0f}%" if pct is not None else f"{'--':>9}"
            print(f"  {LABEL[crowd]:>5} {cells} {pct_s} {n_cell:>3}", flush=True)
        print(flush=True)

    out = ROOT / "outputs" / "results_rl_booking_grandavg.json"
    out.write_text(json.dumps(grand, indent=2))
    print(f"  wrote {out}\n", flush=True)


def average_toy(files):
    if not files:
        return
    qs, dqs, seeds = [], [], []
    for f in files:
        data = json.loads(Path(f).read_text())
        seeds.append(str(data.get("seed", Path(f).stem)))
        if data.get("q_learning_eval", {}).get("mean_return") is not None:
            qs.append(float(data["q_learning_eval"]["mean_return"]))
        if data.get("double_q_learning_eval", {}).get("mean_return") is not None:
            dqs.append(float(data["double_q_learning_eval"]["mean_return"]))
    print(f"=== TOY Q-LEARNING: grand average across {len(files)} seed(s): {', '.join(sorted(seeds))} ===",
          flush=True)
    qm, qsd, qn = _mean_std(qs)
    dm, dsd, dn = _mean_std(dqs)
    print(f"  vanilla Q-learning : {qm} +/- {qsd}  (n={qn})", flush=True)
    print(f"  Double Q-learning  : {dm} +/- {dsd}  (n={dn})", flush=True)
    toy = {"n_seeds": len(files), "seeds": sorted(seeds),
           "q_learning": {"mean_return_mean": qm, "mean_return_std": qsd, "n": qn},
           "double_q_learning": {"mean_return_mean": dm, "mean_return_std": dsd, "n": dn}}
    out = ROOT / "outputs" / "results_toy_grandavg.json"
    out.write_text(json.dumps(toy, indent=2))
    print(f"  wrote {out}\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-canonical", action="store_true",
                    help="Also fold in outputs/results_rl_booking.json (the original seed-42 run).")
    args = ap.parse_args()

    matrix_files = sorted(OUTDIR.glob("results_rl_booking_seed*.json"))
    if args.include_canonical and CANONICAL.exists():
        matrix_files = [CANONICAL] + matrix_files
    toy_files = sorted(OUTDIR.glob("q_learning_seed*.json"))

    if not matrix_files and not toy_files:
        raise SystemExit(f"No per-seed files in {OUTDIR}. Run reseed_booking_grid.py first, "
                         f"or pass --include-canonical to use {CANONICAL.name}.")

    if matrix_files:
        average_matrix(matrix_files)
    if toy_files:
        average_toy(toy_files)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
