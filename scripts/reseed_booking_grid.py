"""RESEED DRIVER: retrain + evaluate the FULL booking algorithm matrix under ONE seed.

Why this exists. The booking centerpiece and its algorithm comparison were trained with
a single training seed (config.DEFAULT_SEED = 42): pipeline 07 (PPO walk baselines),
08 (MaskablePPO RAMA), and 09 (the DQN / unmasked-PPO control arms). Evaluation already
uses six FIXED common-random-number crowd seeds (900000..900005); those are NOT training
seeds and they stay fixed. To put error bars on every cell we retrain under a DIFFERENT
training seed, evaluate each policy on the same fixed CRN crowds, and average across
training seeds with scripts/average_seed_runs.py.

The 30-cell matrix, for ONE --seed, faithfully reproducing pipelines 07 + 08 + 09 + 10:
  {art, tourist} x {500, 2500, max=5000} x 5 algorithm arms
    ppo_base : PPO walk, no interventions             (pipeline 07)   eval: plain
    dqn_base : DQN walk, no interventions             (pipeline 09)   eval: plain
    mppo_int : MaskablePPO RAMA (the centerpiece)     (pipeline 08)   eval: masked
    ppo_int  : unmasked PPO RAMA                      (pipeline 09)   eval: plain
    dqn_int  : DQN RAMA                               (pipeline 09)   eval: plain
  eval is deterministic, CRN seeds 900000..905, from the fixed entrance (pipeline 10).

The headline figure-15 grid is mppo_int vs ppo_base; the other arms are the algorithm
comparison. Models are saved seed-stamped under outputs/models/seeds/ so seeds never
collide with each other or with the committed seed-42 models in outputs/models/newenv/.
Per-cell resumable: an existing seed-stamped .zip is reused.

Result -> outputs/seeds/results_rl_booking_seed{SEED}.json (averaged by average_seed_runs.py).

CPU-capped to 2 threads (this is a laptop). The full matrix is HEAVY (~3.6M steps/seed);
it is per-cell resumable, so leave it overnight or split across mates with the flags.

  nice -n 10 uv run python scripts/reseed_booking_grid.py --seed 123                 # full matrix
  nice -n 10 uv run python scripts/reseed_booking_grid.py --seed 123 --profiles art  # split by profile
  nice -n 10 uv run python scripts/reseed_booking_grid.py --seed 123 --arms mppo_int ppo_base  # headline only

Fast smoke test (one cell, all arms, tiny budget, throwaway seed):
  nice -n 10 uv run python scripts/reseed_booking_grid.py --seed 999999 \
      --profiles art --crowds 500 --base-ts 2500 --ts 2500
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
torch.set_num_threads(2)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO, DQN

from uffizi_rl import config
from uffizi_rl.analysis.portfolio import combined_intervention_kwargs
from uffizi_rl.interventions.intervention_config import InterventionConfig
from uffizi_rl.environment.rama_art_lover_env import RamaArtLoverEnv, GROUPS
from uffizi_rl.environment.rama_tourist_env import RamaTouristEnv
from uffizi_rl.environment.planned_route_env import PlannedRouteEnv
from uffizi_rl.environment.visitor_profiles import sample_type_b_profile

IV = InterventionConfig.from_kwargs(**combined_intervention_kwargs())
EVAL_SEEDS = range(900000, 900006)          # FIXED common-random-number crowds; do NOT vary
MAST = ("A11", "A35", "A38")
LABEL = {500: "500", 2500: "2500", 5000: "max"}
INTERV_CLS = {"art": RamaArtLoverEnv, "tourist": RamaTouristEnv}
BASE_GAMMA = {"art": 0.995, "tourist": 0.997}
LOADERS = {"ppo_base": PPO, "dqn_base": DQN, "mppo_int": MaskablePPO, "ppo_int": PPO, "dqn_int": DQN}
KIND = {"ppo_base": "base", "dqn_base": "base", "mppo_int": "int", "ppo_int": "int", "dqn_int": "int"}
MASKED = {"mppo_int": True}                 # only MaskablePPO uses masks at eval
DEFAULT_BASE_TS = int(os.environ.get("RESEED_BASE_TS", "150000"))   # pipelines 07 + 09 dqn_base
DEFAULT_INT_TS = int(os.environ.get("RESEED_TS", "100000"))         # pipelines 08 + 09 int arms
MODELS = Path("outputs/models/seeds")
OUTDIR = Path("outputs/seeds")
ALL_ARMS = ["ppo_base", "dqn_base", "mppo_int", "ppo_int", "dqn_int"]


def baseline_env(profile: str, seed: int, crowd: int):
    """PlannedRouteEnv walk, no interventions; tourist gets recognition taste.

    Identical to pipeline 07 make_env / pipeline 09 + 10 baseline_env.
    """
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


def _mask(env):
    return env.get_action_mask()


def build_model(arm: str, profile: str, crowd: int, seed: int):
    """The exact training config of each arm, mirroring pipelines 07 / 08 / 09."""
    if arm == "ppo_base":
        return PPO("MlpPolicy", baseline_env(profile, seed, crowd), learning_rate=3e-4,
                   n_steps=2048, batch_size=256, n_epochs=10, gamma=BASE_GAMMA[profile],
                   gae_lambda=0.98, ent_coef=0.01, clip_range=0.2, seed=seed, verbose=0,
                   policy_kwargs={"net_arch": [128, 128]})
    if arm == "dqn_base":
        return DQN("MlpPolicy", baseline_env(profile, seed, crowd), learning_rate=5e-4,
                   batch_size=128, buffer_size=100_000, learning_starts=2_000,
                   gamma=BASE_GAMMA[profile], target_update_interval=1000, train_freq=4,
                   gradient_steps=1, exploration_fraction=0.3, exploration_final_eps=0.05,
                   seed=seed, verbose=0, policy_kwargs={"net_arch": [128, 128]})
    raw = INTERV_CLS[profile](seed=seed, daily_total=crowd, interventions=IV, random_start=False)
    if arm == "mppo_int":
        return MaskablePPO("MlpPolicy", ActionMasker(raw, _mask), learning_rate=3e-4,
                           n_steps=2048, batch_size=256, n_epochs=10, gamma=0.999,
                           gae_lambda=0.98, ent_coef=0.01, clip_range=0.2, seed=seed, verbose=0,
                           policy_kwargs={"net_arch": [128, 128]})
    if arm == "ppo_int":
        return PPO("MlpPolicy", raw, learning_rate=3e-4, n_steps=2048, batch_size=256,
                   n_epochs=10, gamma=0.999, gae_lambda=0.98, ent_coef=0.01, clip_range=0.2,
                   seed=seed, verbose=0, policy_kwargs={"net_arch": [128, 128]})
    # dqn_int
    return DQN("MlpPolicy", raw, learning_rate=5e-4, batch_size=128, buffer_size=100_000,
               learning_starts=2_000, gamma=0.999, target_update_interval=1000, train_freq=4,
               gradient_steps=1, exploration_fraction=0.3, exploration_final_eps=0.05,
               seed=seed, verbose=0, policy_kwargs={"net_arch": [128, 128]})


def train_arm(arm: str, profile: str, crowd: int, seed: int, ts: int):
    path = MODELS / f"{arm}_{profile}_{crowd}_seed{seed}.zip"
    if path.exists():
        print(f"  [{arm} {profile} {LABEL[crowd]}] reuse {path.name}", flush=True)
        return LOADERS[arm].load(path)
    print(f"  [{arm} {profile} {LABEL[crowd]}] training {ts} (seed {seed})...", flush=True)
    model = build_model(arm, profile, crowd, seed)
    model.learn(ts)
    model.save(path)
    return model


def eval_base(profile: str, crowd: int, model) -> dict:
    """Deterministic eval, no interventions (pipeline 10 eval_baseline)."""
    R = []
    for s in EVAL_SEEDS:
        env = baseline_env(profile, s, crowd); env.seed_value = s; env._episode_counter = 0
        obs, _ = env.reset(); tot = 0.0; d = t = False
        while not (d or t):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, t, _ = env.step(int(a)); tot += r
        R.append(tot)
    return {"points": round(float(np.mean(R)), 1)}


def eval_int(profile: str, crowd: int, model, masked: bool) -> dict:
    """Deterministic eval inside the bundle (pipeline 10 eval_intervened / pipeline 09)."""
    EnvCls = INTERV_CLS[profile]
    prior = list(EnvCls(seed=config.DEFAULT_SEED, daily_total=crowd,
                        interventions=IV, random_start=False)._learned_arrival)
    R = []; CI = []; LEADS = Counter(); entry = Counter(); ex = 0; teleports = 0
    for s in EVAL_SEEDS:
        env = EnvCls(seed=s, daily_total=crowd, interventions=IV, random_start=False, compute_plan=False)
        env._learned_arrival = list(prior); env._arr_alpha = 0.0
        env.seed_value = s; env._episode_counter = 0
        obs, _ = env.reset(); tot = 0.0; d = t = False; lead = None
        seq = []
        while not (d or t):
            if masked:
                a, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
            else:
                a, _ = model.predict(obs, deterministic=True)
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
    return {"points": round(float(np.mean(R)), 1), "checkins_of_3": round(float(np.mean(CI)), 2),
            "lead_days_chosen": dict(LEADS),
            "entry_room": entry.most_common(1)[0][0] if entry else "?",
            "exit_count_of_6": ex, "teleports": teleports}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED,
                    help="TRAINING seed for this run. Give each teammate a different one (not 42).")
    ap.add_argument("--profiles", nargs="+", default=["art", "tourist"], choices=["art", "tourist"])
    ap.add_argument("--crowds", nargs="+", type=int, default=[500, 2500, 5000], choices=[500, 2500, 5000])
    ap.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS,
                    help="Which algorithm arms to run. Default: all 5.")
    ap.add_argument("--ts", type=int, default=DEFAULT_INT_TS,
                    help="Intervened-arm budget (mppo_int/ppo_int/dqn_int). Default 100000 = pipelines 08/09.")
    ap.add_argument("--base-ts", type=int, default=DEFAULT_BASE_TS,
                    help="Baseline-arm budget (ppo_base/dqn_base). Default 150000 = pipelines 07/09.")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    arms = [a for a in ALL_ARMS if a in args.arms]      # keep canonical order

    print(f"=== RESEED matrix | train_seed={args.seed} | eval CRN 900000..905 "
          f"| arms={arms} | base_ts={args.base_ts} ts={args.ts} ===", flush=True)

    results = {
        "description": ("Full booking algorithm matrix retrained under one training seed and evaluated "
                        "deterministically on the fixed CRN crowds (900000..905). One file per training "
                        "seed; average across files with scripts/average_seed_runs.py. Headline grid = "
                        "mppo_int (intervened) vs ppo_base (baseline)."),
        "train_seed": args.seed,
        "eval_crn_seeds": list(EVAL_SEEDS),
        "crowds": args.crowds,
        "arms": arms,
        "profiles": {},
    }
    for profile in args.profiles:
        print(f"\n{profile.upper()}", flush=True)
        results["profiles"][profile] = {}
        for c in args.crowds:
            cell = {"arms": {}}
            for arm in arms:
                ts = args.base_ts if KIND[arm] == "base" else args.ts
                model = train_arm(arm, profile, c, args.seed, ts)
                m = (eval_base(profile, c, model) if KIND[arm] == "base"
                     else eval_int(profile, c, model, MASKED.get(arm, False)))
                cell["arms"][arm] = m
                extra = f" checkins={m['checkins_of_3']}/3 lead={m['lead_days_chosen']}" if KIND[arm] == "int" else ""
                print(f"    {arm:>9} {LABEL[c]:>4}: points={m['points']:>8.0f}{extra}", flush=True)
            # headline convenience: MaskablePPO intervened vs PPO baseline walk (figure 15)
            if "ppo_base" in cell["arms"] and "mppo_int" in cell["arms"]:
                bs = cell["arms"]["ppo_base"]["points"]; iv = cell["arms"]["mppo_int"]["points"]
                cell["baseline_points"] = bs
                cell["intervened_points"] = iv
                cell["pct_vs_baseline"] = round(100 * (iv - bs) / abs(bs), 1) if bs else float("nan")
                cell["checkins_of_3"] = cell["arms"]["mppo_int"]["checkins_of_3"]
            results["profiles"][profile][str(c)] = cell

    out = OUTDIR / f"results_rl_booking_seed{args.seed}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {out}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
