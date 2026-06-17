"""Reproduce the Uffizi RL multi-seed reseed FROM SCRATCH (portable, single entry point).

This is the reproducible driver behind `python run_project.py --reseed`. For the given
TRAINING seeds (default 1 3 7 202606), it runs end to end:

  A) toy tabular Q-learning per seed  -> outputs/seeds/q_learning_seed<s>.json
  B) the 120 booking cells (5 arms x 2 profiles x 3 crowds x N seeds), trained in a CPU
     process pool. Per-cell resumable: an existing seed-stamped model zip is reused, so a
     re-run is cheap and an interrupted run can simply be restarted.
  C) one sequential single-writer assembly pass per seed -> the authoritative
     outputs/seeds/results_rl_booking_seed<s>.json (reloads cached models, no retraining).
  D) sanity checks on every result file (valid JSON, no NaN/Inf, mppo_int check-ins in [0,3]).
  E) the grand average across ALL seed files in outputs/seeds/, including the committed
     seed 42 -> prints the two grids with mean +/- std and n (expected n = 5).

Everything runs on CPU (CUDA_VISIBLE_DEVICES="") so results are bit-deterministic and fast
for these tiny MLPs; this was validated once by the sequential-vs-parallel gate
(scripts/phaseA2_seq_vs_par.sh + scripts/phaseA2_compare.py). Nothing here changes any
environment, reward, capacity, hyperparameter, training budget, or the six fixed evaluation
crowd seeds 900000..900005 - only the training seed varies.

Usage:
    python run_project.py --reseed                      # full reseed (hours on first run)
    python scripts/reseed_reproduce.py --seeds 7        # add a single extra seed
    python scripts/reseed_reproduce.py --workers 8 --episodes 25000
    python scripts/reseed_reproduce.py --skip-train     # only re-assemble/sanity/average from cache
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = "scripts/reseed_booking_grid.py"
TOY = "uffizi_rl/pipeline/02_train_q_learning.py"
SANITY = "scripts/phaseD_sanity.py"
AVERAGER = "scripts/average_seed_runs.py"

ARMS = ["ppo_base", "dqn_base", "mppo_int", "ppo_int", "dqn_int"]
PROFILES = ["art", "tourist"]
CROWDS = [500, 2500, 5000]
DEFAULT_SEEDS = [1, 3, 7, 202606]


def cpu_env() -> dict:
    """Environment that forces CPU training (hide the GPU from torch/SB3)."""
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = ""
    return e


def py(script: str, *args: str, env: dict | None = None, capture: bool = False):
    """Run a repo script with the current interpreter, cwd = repo root."""
    cmd = [sys.executable, script, *map(str, args)]
    return subprocess.run(cmd, cwd=ROOT, env=env or cpu_env(),
                          capture_output=capture, text=True)


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def phase_a_toy(seeds, episodes) -> None:
    banner(f"PHASE A - toy tabular Q-learning ({episodes} episodes) for seeds {seeds}")
    seeds_dir = ROOT / "outputs" / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    canonical = ROOT / "outputs" / "02_q_learning.json"
    backup = canonical.with_suffix(".json.reseed_bak") if canonical.exists() else None
    if backup:
        shutil.copy2(canonical, backup)  # the loop overwrites the committed seed-42 file
    try:
        for s in seeds:
            r = py(TOY, "--seed", s, "--episodes", episodes)
            if r.returncode != 0:
                raise SystemExit(f"toy Q-learning failed for seed {s} (exit {r.returncode})")
            shutil.copy2(canonical, seeds_dir / f"q_learning_seed{s}.json")
            print(f"  wrote outputs/seeds/q_learning_seed{s}.json", flush=True)
    finally:
        if backup:
            shutil.move(backup, canonical)  # restore the committed seed-42 artifact
            print("  restored committed outputs/02_q_learning.json", flush=True)


def phase_b_train(seeds, workers) -> None:
    jobs = [(s, a, p, c) for s in seeds for a in ARMS for p in PROFILES for c in CROWDS]
    banner(f"PHASE B - train {len(jobs)} booking cells on CPU, {workers} workers (resumable)")
    t0 = time.time()
    failures = []

    def train(job):
        s, a, p, c = job
        r = py(DRIVER, "--seed", s, "--arms", a, "--profiles", p, "--crowds", c, capture=True)
        return job, r.returncode, r.stderr

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(train, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            job, rc, err = fut.result()
            tag = "ok" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{i:3d}/{len(jobs)}] seed={job[0]} {job[1]} {job[2]} {job[3]}: {tag}", flush=True)
            if rc != 0:
                failures.append((job, err.strip().splitlines()[-5:] if err else []))
    print(f"  Phase B wall-clock: {(time.time() - t0) / 60:.1f} min", flush=True)
    if failures:
        for job, tail in failures:
            print(f"  FAILED {job}:\n    " + "\n    ".join(tail), flush=True)
        raise SystemExit(f"Phase B: {len(failures)} cell(s) failed - stop and inspect.")


def phase_c_assemble(seeds) -> None:
    banner(f"PHASE C - single-writer assembly per seed {seeds} (reuse cache, no retrain)")
    for s in seeds:
        r = py(DRIVER, "--seed", s)  # all 30 cells, reuses cached zips, writes full JSON
        if r.returncode != 0:
            raise SystemExit(f"Phase C assembly failed for seed {s} (exit {r.returncode})")
        print(f"  assembled outputs/seeds/results_rl_booking_seed{s}.json", flush=True)


def phase_d_sanity() -> None:
    banner("PHASE D - sanity-check every result file")
    if py(SANITY).returncode != 0:
        raise SystemExit("Phase D sanity check failed - stop and inspect.")


def phase_e_average() -> None:
    banner("PHASE E - grand average across all seed files (expect n = 5)")
    if py(AVERAGER).returncode != 0:
        raise SystemExit("Phase E averaging failed.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help="Training seeds to (re)produce. Default: 1 3 7 202606. Never 42.")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) // 2)),
                    help="Parallel CPU workers for Phase B (each cell uses 2 threads).")
    ap.add_argument("--episodes", type=int, default=25000, help="Toy Q-learning episodes.")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip Phases A+B; only assemble/sanity/average from cached models.")
    args = ap.parse_args()

    if 42 in args.seeds:
        raise SystemExit("Refusing to reseed 42: it is the committed first seed.")

    banner(f"RESEED REPRODUCE | seeds={args.seeds} workers={args.workers} | CPU-forced | "
           f"eval CRN 900000..905 frozen")
    if not args.skip_train:
        phase_a_toy(args.seeds, args.episodes)
        phase_b_train(args.seeds, args.workers)
    phase_c_assemble(args.seeds)
    phase_d_sanity()
    phase_e_average()
    banner("RESEED REPRODUCE COMPLETE")
    print("Deliverables: outputs/seeds/{results_rl_booking,q_learning}_seed<s>.json", flush=True)
    print("Grand averages: outputs/results_rl_booking_grandavg.json, outputs/results_toy_grandavg.json",
          flush=True)


if __name__ == "__main__":
    main()
