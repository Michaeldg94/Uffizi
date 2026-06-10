"""PIPELINE 10: the deliverable evaluation (intervened vs matched baseline).

Everything evaluated the same way: deterministic, seeds 900000-5, from the fixed
entrance, both profiles, all crowds.
  art lover  intervened = RamaArtLoverEnv (ramabook_art_{c}) vs PlannedRouteEnv (opt_ArtWalk_{c})
  tourist    intervened = RamaTouristEnv  (ramabook_tourist_{c}) vs the tourist walk (opt_TouristWalk_{c})

Booking policies are evaluated DETERMINISTICALLY (the argmax books cleanly;
stochastic adds lead-choice noise that undersells). Prints the headline grid plus a
path-sanity summary (entry room, exit behaviour, and any non-adjacent teleports) so
the learned routes can be audited.

Run AFTER pipeline 07 (baselines) and 08 (booking grid).
Usage:  python uffizi_rl/pipeline/10_evaluate_booking.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
torch.set_num_threads(2)
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from uffizi_rl import config
from uffizi_rl.analysis.portfolio import combined_intervention_kwargs
from uffizi_rl.interventions.intervention_config import InterventionConfig
from uffizi_rl.environment.rama_art_lover_env import RamaArtLoverEnv, GROUPS
from uffizi_rl.environment.rama_tourist_env import RamaTouristEnv
from uffizi_rl.environment.planned_route_env import PlannedRouteEnv
from uffizi_rl.environment.visitor_profiles import sample_type_b_profile

IV = InterventionConfig.from_kwargs(**combined_intervention_kwargs())
SEED = config.DEFAULT_SEED
SEEDS = range(900000, 900006)
CROWDS = [500, 2500, 5000]
LABEL = {500: "500", 2500: "2500", 5000: "max"}
MAST = ("A11", "A35", "A38")
INTERV_CLS = {"art": RamaArtLoverEnv, "tourist": RamaTouristEnv}
BWALK = {"art": "opt_ArtWalk", "tourist": "opt_TouristWalk"}
M = Path("outputs/models/newenv")


def baseline_env(profile, seed, crowd):
    env = PlannedRouteEnv(seed=seed, episode_minutes=config.MUSEUM_OPEN_MINUTES,
                          daily_total=crowd, random_start=False)
    if profile == "tourist":
        base = np.array([config.ROOM_DATA[r]["importance"] for r in config.ROOM_IDS], dtype=float)
        recog = base.copy()
        for i, r in enumerate(config.ROOM_IDS):
            if r.startswith("A") or r in {"E4", "E5", "E7"}:
                continue
            recog[i] = 7.0 if r in {"C6", "C7", "C8", "C9", "C10", "C11"} else 2.0

        def _s(rng):
            p = sample_type_b_profile(rng); p.importance_vector = recog.copy(); return p
        env.inner.art_crowd_alpha = 1.0
        env.inner._profile_sampler = _s
        env.inner.agent_profile = _s(env.inner.rng)
        env.inner.dwell_per_importance = 1.4
        env.inner.dwell_importance_floor = 5.0
    return env


def eval_intervened(profile, crowd):
    EnvCls = INTERV_CLS[profile]
    model = MaskablePPO.load(M / f"ramabook_{profile}_{crowd}.zip")
    prior = list(EnvCls(seed=SEED, daily_total=crowd, interventions=IV, random_start=False)._learned_arrival)
    R = []; CI = []; LEADS = Counter(); entry = Counter(); ex = 0; teleports = 0
    for s in SEEDS:
        env = EnvCls(seed=s, daily_total=crowd, interventions=IV, random_start=False, compute_plan=False)
        env._learned_arrival = list(prior); env._arr_alpha = 0.0
        env.seed_value = s; env._episode_counter = 0
        obs, _ = env.reset(); tot = 0.0; d = t = False; lead = None
        seq = []
        while not (d or t):
            a, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
            obs, r, d, t, _ = env.step(int(a)); tot += r
            if env._phase == "walk":
                if lead is None:
                    lead = env._lead_days; entry[env.inner.current_room] += 1
                if not seq or seq[-1] != env.inner.current_room:
                    seq.append(env.inner.current_room)
        R.append(tot); LEADS[lead] += 1
        CI.append(sum(1 for g in GROUPS if env.inner._rama_checkin.get(g, False)))
        ex += int(env.inner.current_room == "EXIT")
        dist = env.inner._distances
        for u, v in zip(seq[:-1], seq[1:]):
            if "EXIT" in (u, v):
                continue
            if dist.get(u, {}).get(v, 99) != 1:
                teleports += 1
    return (float(np.mean(R)), float(np.mean(CI)), dict(LEADS),
            entry.most_common(1)[0][0] if entry else "?", ex, teleports)


def eval_baseline(profile, crowd):
    model = PPO.load(M / f"{BWALK[profile]}_{crowd}.zip")
    R = []
    for s in SEEDS:
        env = baseline_env(profile, s, crowd); env.seed_value = s; env._episode_counter = 0
        obs, _ = env.reset(); tot = 0.0; d = t = False
        while not (d or t):
            a, _ = model.predict(obs, deterministic=True); obs, r, d, t, _ = env.step(int(a)); tot += r
        R.append(tot)
    return float(np.mean(R))


def main():
    print("=== PIPELINE 10: intervened vs matched baseline (deterministic) ===", flush=True)
    import json
    results = {"description": ("RL booking grid (the optimal individual agent): intervened (RAMA) "
                               "vs matched no-intervention walk baseline, deterministic eval, seeds 900000-5"),
               "crowds": CROWDS, "profiles": {}}
    for profile in ("art", "tourist"):
        print(f"\n{profile.upper()}\n  {'crowd':>5} {'baseline':>9} {'intervened':>11} {'checkins':>9} "
              f"{'lead':>14} {'vs base':>9} {'entry':>6} {'exit':>5} {'teleport':>9}", flush=True)
        results["profiles"][profile] = {}
        for c in CROWDS:
            bs = eval_baseline(profile, c)
            iv, ci, leads, entry, ex, tp = eval_intervened(profile, c)
            pct = 100 * (iv - bs) / abs(bs) if bs else float("nan")
            print(f"  {LABEL[c]:>5} {bs:>9.0f} {iv:>11.0f} {ci:>7.1f}/3 {str(leads):>14} "
                  f"{pct:>+8.0f}% {entry:>6} {ex:>3}/6 {tp:>9}", flush=True)
            results["profiles"][profile][str(c)] = {
                "baseline_points": round(bs, 1), "intervened_points": round(iv, 1),
                "pct_vs_baseline": round(pct, 1), "checkins_of_3": round(ci, 2),
                "lead_days_chosen": leads, "entry_room": entry,
                "exit_count_of_6": ex, "teleports": tp}
    (Path("outputs") / "results_rl_booking.json").write_text(json.dumps(results, indent=2))
    print("\n  wrote outputs/results_rl_booking.json", flush=True)
    print("teleport=0 means every room change is a 1-hop neighbour (no map jumps).", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
