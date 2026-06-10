"""PIPELINE 07: matched no-intervention baselines (art lover + normal tourist).

The fair comparison point for the RAMA booking grid (pipeline 08): the SAME walk
structure and visitor character, with NO interventions (crowded masterpieces, no
room enrichment, no extended hours), trained AND evaluated per crowd from the
fixed entrance.

  art lover -> PlannedRouteEnv (default connoisseur profile)        -> opt_ArtWalk_{c}
  tourist   -> PlannedRouteEnv + recognition taste + short dwells   -> opt_TouristWalk_{c}

random_start=False is REQUIRED: the intervened agents start at the fixed entrance,
and a random-start-trained policy degenerates under deterministic eval from the
entrance. Resumable: a saved model is reused, not retrained. CPU-capped.

Usage:  python uffizi_rl/pipeline/07_train_baselines.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
torch.set_num_threads(2)
from stable_baselines3 import PPO

from uffizi_rl import config
from uffizi_rl.environment.planned_route_env import PlannedRouteEnv
from uffizi_rl.environment.visitor_profiles import sample_type_b_profile

SEED = config.DEFAULT_SEED
TS = 150_000
CROWDS = [500, 2500, 5000]
SEEDS = range(900000, 900006)
LABEL = {500: "500", 2500: "2500", 5000: "max"}
GAMMA = {"art": 0.995, "tourist": 0.997}
MODELS = Path("outputs/models/newenv")
MODELS.mkdir(parents=True, exist_ok=True)


def make_env(profile: str, seed: int, crowd: int):
    """PlannedRouteEnv walk, no interventions; tourist gets recognition taste."""
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


def train_eval(profile: str, crowd: int):
    stem = "opt_ArtWalk" if profile == "art" else "opt_TouristWalk"
    path = MODELS / f"{stem}_{crowd}.zip"
    if path.exists():
        print(f"  [{profile} {LABEL[crowd]}] reuse", flush=True)
        model = PPO.load(path)
    else:
        print(f"  [{profile} {LABEL[crowd]}] training {TS}...", flush=True)
        model = PPO("MlpPolicy", make_env(profile, SEED, crowd), learning_rate=3e-4, n_steps=2048,
                    batch_size=256, n_epochs=10, gamma=GAMMA[profile], gae_lambda=0.98,
                    ent_coef=0.01, clip_range=0.2, seed=SEED, verbose=0,
                    policy_kwargs={"net_arch": [128, 128]})
        model.learn(TS)
        model.save(path)
    R = []; VIS = []; ex = 0
    for s in SEEDS:
        env = make_env(profile, s, crowd); env.seed_value = s; env._episode_counter = 0
        obs, _ = env.reset(); tot = 0.0; d = t = False
        while not (d or t):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, t, _ = env.step(int(a)); tot += r
        R.append(tot); VIS.append(env.inner.time_elapsed)
        ex += int(env.inner.current_room == "EXIT")
    return float(np.mean(R)), float(np.mean(VIS)), ex


def main():
    print("=== PIPELINE 07: matched no-intervention baselines ===", flush=True)
    for profile in ("art", "tourist"):
        print(f"\n{profile.upper()}\n  {'crowd':>5} {'points':>8} {'visit':>6} {'exit':>5}", flush=True)
        for c in CROWDS:
            rw, vis, ex = train_eval(profile, c)
            print(f"  {LABEL[c]:>5} {rw:>8.0f} {vis:>6.0f} {ex:>3}/6", flush=True)
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
