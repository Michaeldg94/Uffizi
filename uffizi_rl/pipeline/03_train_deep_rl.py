"""SCRIPT 03: Deep RL on the full 98-room museum (Phase 3).

Trains MaskablePPO (sb3-contrib) on the full Uffizi environment with
action masking and a 3-stage curriculum. Also runs ablations: PPO and
DQN without action masking, used as controls to demonstrate that
masking is essential for graph-structured environments.

Demonstrates Lectures 6-7 (Policy-based methods) and Lecture 8 (Deep RL).

Run AFTER 02. Writes:
  outputs/03_deep_rl.json (PPO mean return, ablation comparison)

Usage:
    uv run python uffizi_rl/pipeline/03_train_deep_rl.py
    uv run python uffizi_rl/pipeline/03_train_deep_rl.py --timesteps 500000 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from uffizi_rl import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from uffizi_rl import config  # noqa: E402
from uffizi_rl.agents.train_ablations import train_ablations  # noqa: E402
from uffizi_rl.agents.train_maskable_ppo import train_maskable_ppo  # noqa: E402
from uffizi_rl.pipeline._paths import ensure_outputs_dir  # noqa: E402


def main() -> None:
    """Train MaskablePPO and run PPO/DQN ablations without masking."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=15000,
                        help="PPO timesteps. Use 500_000+ for converged results.")
    parser.add_argument("--ablation-timesteps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of random seeds for PPO. Use 3 for paper-quality error bars.")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = parser.parse_args()

    output_dir = ensure_outputs_dir()

    print(f"[1/2] Training MaskablePPO ({args.timesteps} steps, {args.seeds} seeds)")
    ppo_returns = []
    for s in range(args.seeds):
        result = train_maskable_ppo(
            timesteps=args.timesteps, seed=args.seed + s * 100, eval_episodes=50,
            save_path=f"{output_dir}/models/ppo_seed{s}.zip",
        )
        ppo_returns.append(result.mean_episode_reward)
        # Per-seed mean is now over 50 common-random-number episodes; the
        # within-seed std is eval noise (similar across seeds), while the
        # spread of the per-seed MEANS is what we want to be small.
        print(f"  seed {s}: return={result.mean_episode_reward:.1f} "
              f"(within-seed std={result.std_episode_reward:.1f} over "
              f"{result.episodes_eval} CRN episodes)")

    ppo_mean = float(np.mean(ppo_returns))
    ppo_std = float(np.std(ppo_returns, ddof=1)) if len(ppo_returns) > 1 else 0.0
    # Robust statistics: with several seeds, the median and IQR describe the
    # typical converged policy without being dragged down by an occasional
    # unstable seed, which the mean is sensitive to.
    ppo_median = float(np.median(ppo_returns))
    ppo_q1 = float(np.percentile(ppo_returns, 25))
    ppo_q3 = float(np.percentile(ppo_returns, 75))
    print(f"  MaskablePPO: mean {ppo_mean:.1f} +/- {ppo_std:.1f} | "
          f"median {ppo_median:.1f} [IQR {ppo_q1:.1f}-{ppo_q3:.1f}] | "
          f"per-seed {[round(x,1) for x in ppo_returns]}")

    print("[2/2] Ablations: PPO and DQN without action masking")
    ablations = train_ablations(timesteps=args.ablation_timesteps, seed=args.seed + 10)
    for r in ablations:
        print(f"  {r.algorithm:30s} return={r.mean_episode_reward:.1f}")

    summary = {
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "maskable_ppo": {
            "mean_episode_reward": ppo_mean,
            "std_episode_reward": ppo_std,
            "median_episode_reward": ppo_median,
            "iqr": [ppo_q1, ppo_q3],
            "per_seed": ppo_returns,
        },
        "ablations": [r.__dict__ for r in ablations],
    }
    out_path = output_dir / "03_deep_rl.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
