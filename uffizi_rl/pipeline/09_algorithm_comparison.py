"""PIPELINE 09: algorithm comparison (PPO / MaskablePPO / DQN x baseline / intervened).

Completes the 30-cell matrix. The other arms already exist: MaskablePPO-intervened
is pipeline 08 (ramabook_*), PPO-baseline is pipeline 07 (opt_*Walk_*), and
MaskablePPO-baseline is omitted by design (== PPO-baseline; nothing to mask). This
stage fills the remaining 18 cells, ONE AT A TIME, CPU-capped + niced, resumable:

  PPO-intervened (unmasked)  -> ppo_book_{profile}_{crowd}
  DQN-intervened             -> dqn_book_{profile}_{crowd}
  DQN-baseline               -> dqn_base_{profile}_{crowd}

This is the heavy stage (~18 trainings). It is resumable, so it can be run in
several sittings; finished cells are reused. HEED THE LAPTOP: each cell is capped
to 2 cores. Hyperparameters are matched to what works on these envs (PPO-intervened
== the MaskablePPO config minus the mask; DQN sized for this budget/horizon, not the
old raw-navigation ablation).

Usage:  python uffizi_rl/pipeline/09_algorithm_comparison.py [only_kind only_profile only_crowd]
        (no args = run/refresh the whole matrix)
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
from sb3_contrib import MaskablePPO  # noqa: F401  (MaskablePPO-intervened lives in pipeline 08)
from stable_baselines3 import PPO, DQN

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
INTERV_CLS = {"art": RamaArtLoverEnv, "tourist": RamaTouristEnv}
BASE_GAMMA = {"art": 0.995, "tourist": 0.997}
MODELS = Path("outputs/models/newenv")
MODELS.mkdir(parents=True, exist_ok=True)


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


def make_model(kind, profile, crowd):
    if kind == "ppo_book":
        raw = INTERV_CLS[profile](seed=SEED, daily_total=crowd, interventions=IV, random_start=False)
        return PPO("MlpPolicy", raw, learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
                   gamma=0.999, gae_lambda=0.98, ent_coef=0.01, clip_range=0.2, seed=SEED,
                   verbose=0, policy_kwargs={"net_arch": [128, 128]})
    if kind == "dqn_book":
        raw = INTERV_CLS[profile](seed=SEED, daily_total=crowd, interventions=IV, random_start=False)
        gamma = 0.999
    else:  # dqn_base
        raw = baseline_env(profile, SEED, crowd); gamma = BASE_GAMMA[profile]
    return DQN("MlpPolicy", raw, learning_rate=5e-4, batch_size=128, buffer_size=100_000,
               learning_starts=2_000, gamma=gamma, target_update_interval=1000, train_freq=4,
               gradient_steps=1, exploration_fraction=0.3, exploration_final_eps=0.05,
               seed=SEED, verbose=0, policy_kwargs={"net_arch": [128, 128]})


def eval_cell(kind, profile, crowd, model):
    if kind == "dqn_base":
        R = []
        for s in SEEDS:
            env = baseline_env(profile, s, crowd); env.seed_value = s; env._episode_counter = 0
            obs, _ = env.reset(); tot = 0.0; d = t = False
            while not (d or t):
                a, _ = model.predict(obs, deterministic=True); obs, r, d, t, _ = env.step(int(a)); tot += r
            R.append(tot)
        return f"points={np.mean(R):.0f}"
    EnvCls = INTERV_CLS[profile]
    prior = list(EnvCls(seed=SEED, daily_total=crowd, interventions=IV, random_start=False)._learned_arrival)
    R = []; CI = []; LEADS = Counter()
    for s in SEEDS:
        env = EnvCls(seed=s, daily_total=crowd, interventions=IV, random_start=False, compute_plan=False)
        env._learned_arrival = list(prior); env._arr_alpha = 0.0
        env.seed_value = s; env._episode_counter = 0
        obs, _ = env.reset(); tot = 0.0; d = t = False; lead = None
        while not (d or t):
            a, _ = model.predict(obs, deterministic=True); obs, r, d, t, _ = env.step(int(a)); tot += r
            if env._phase == "walk" and lead is None:
                lead = env._lead_days
        R.append(tot); LEADS[lead] += 1
        CI.append(sum(1 for g in GROUPS if env.inner._rama_checkin.get(g, False)))
    return f"points={np.mean(R):.0f} checkins={np.mean(CI):.1f}/3 lead={dict(LEADS)}"


def main():
    sel = sys.argv[1:4] if len(sys.argv) >= 4 else None
    cells = []
    for p in ("art", "tourist"):
        for c in CROWDS:
            for kind, steps in (("ppo_book", 100_000), ("dqn_book", 100_000), ("dqn_base", 150_000)):
                if sel and (kind, p, str(c)) != tuple(sel):
                    continue
                cells.append((kind, p, c, steps))
    print(f"=== PIPELINE 09: algorithm comparison | {len(cells)} cells | resumable, capped ===", flush=True)
    loaders = {"ppo_book": PPO, "dqn_book": DQN, "dqn_base": DQN}
    for kind, p, c, steps in cells:
        path = MODELS / f"{kind}_{p}_{c}.zip"
        tag = f"{kind} {p} {LABEL[c]}"
        if path.exists():
            print(f"  [{tag}] reuse", flush=True)
            model = loaders[kind].load(path)
        else:
            print(f"  [{tag}] training {steps}...", flush=True)
            model = make_model(kind, p, c)
            model.learn(steps); model.save(path)
        print(f"    -> {tag}: {eval_cell(kind, p, c, model)}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
