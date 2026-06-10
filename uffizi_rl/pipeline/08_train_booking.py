"""PIPELINE 08: the RAMA booking grid (the learned centerpiece).

Both visitor profiles learn the Reservation-Anchored Masterpiece Access booking
decision (lead time + per-masterpiece book) against a frozen one-shot plan, then
pace the visit to keep the appointments. Trained AND evaluated optimal-per-crowd
(500 / 2500 / max) inside the full intervention bundle. Crowd-dependent slot
availability means a busier day forces a longer lead, so "book early when busier"
emerges as the crowd rises.

  art lover -> RamaArtLoverEnv  (MaskablePPO)  -> ramabook_art_{c}
  tourist   -> RamaTouristEnv   (MaskablePPO)  -> ramabook_tourist_{c}

gamma=0.999 (the booking reward is delayed to check-in). Resumable: a saved model
is reused. CPU-capped. Set RAMA_TS to change the per-cell budget (default 100k).

Usage:  python uffizi_rl/pipeline/08_train_booking.py
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
from sb3_contrib.common.wrappers import ActionMasker

from uffizi_rl import config
from uffizi_rl.analysis.portfolio import combined_intervention_kwargs
from uffizi_rl.interventions.intervention_config import InterventionConfig
from uffizi_rl.environment.rama_art_lover_env import RamaArtLoverEnv, GROUPS
from uffizi_rl.environment.rama_tourist_env import RamaTouristEnv

IV = InterventionConfig.from_kwargs(**combined_intervention_kwargs())
SEED = config.DEFAULT_SEED
TS = int(os.environ.get("RAMA_TS", "100000"))
CROWDS = [500, 2500, 5000]
LABEL = {500: "500", 2500: "2500", 5000: "max"}
SEEDS = range(900000, 900006)
MAST = ("A11", "A35", "A38")
CLS = {"art": RamaArtLoverEnv, "tourist": RamaTouristEnv}
MODELS = Path("outputs/models/newenv")
MODELS.mkdir(parents=True, exist_ok=True)


def _mask(env):
    return env.get_action_mask()


def train_eval(profile: str, crowd: int):
    EnvCls = CLS[profile]
    path = MODELS / f"ramabook_{profile}_{crowd}.zip"
    raw = EnvCls(seed=SEED, daily_total=crowd, interventions=IV, random_start=False)
    if path.exists():
        print(f"  [{profile} {LABEL[crowd]}] reuse", flush=True)
        model = MaskablePPO.load(path)
    else:
        print(f"  [{profile} {LABEL[crowd]}] training {TS}...", flush=True)
        model = MaskablePPO("MlpPolicy", ActionMasker(raw, _mask), learning_rate=3e-4, n_steps=2048,
                            batch_size=256, n_epochs=10, gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
                            clip_range=0.2, seed=SEED, verbose=0, policy_kwargs={"net_arch": [128, 128]})
        model.learn(TS)
        model.save(path)
    prior = list(raw._learned_arrival)               # frozen one-shot plan, carried into eval
    R = []; CI = []; LEADS = Counter()
    for s in SEEDS:
        env = EnvCls(seed=s, daily_total=crowd, interventions=IV, random_start=False, compute_plan=False)
        env._learned_arrival = list(prior); env._arr_alpha = 0.0
        env.seed_value = s; env._episode_counter = 0
        obs, _ = env.reset(); d = t = False; lead = None
        while not (d or t):
            a, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
            obs, r, d, t, _ = env.step(int(a))
            if env._phase == "walk" and lead is None:
                lead = env._lead_days
        LEADS[lead] += 1
        CI.append(sum(1 for g in GROUPS if env.inner._rama_checkin.get(g, False)))
        R.append(sum(env.inner._extracted.get(mm, 0.0) for mm in MAST))
    return CI, dict(LEADS)


def main():
    print(f"=== PIPELINE 08: RAMA booking grid | TS={TS} ===", flush=True)
    for profile in ("art", "tourist"):
        print(f"\n{profile.upper()}\n  {'crowd':>5} {'checkins':>9} {'lead-days (book ahead)':>26}", flush=True)
        for c in CROWDS:
            ci, leads = train_eval(profile, c)
            print(f"  {LABEL[c]:>5} {np.mean(ci):>7.1f}/3 {str(leads):>26}", flush=True)
    print("\nBook-early should emerge: lead rises with crowd. Full eval vs baseline in pipeline 10.", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
