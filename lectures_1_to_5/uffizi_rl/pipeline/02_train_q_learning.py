"""SCRIPT 02: Tabular Q-learning on the 12-room toy graph (Phase 2).

[READING ORDER: file 10 of 12 - read after 01_check_environment.py]

Trains a tabular Q-learning agent with epsilon-greedy exploration on a
simplified 12-room version of the Uffizi. Compares against five
handcrafted baselines (default path, random, greedy least crowded,
greedy value ratio, peak avoidance) so the reader can judge whether
RL discovers strategies a human designer would miss.

Run AFTER 01. Writes:
  outputs/02_q_learning.json (agent return + baseline returns + episode rewards)

Usage:
    uv run python uffizi_rl/pipeline/02_train_q_learning.py
    uv run python uffizi_rl/pipeline/02_train_q_learning.py --episodes 25000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from uffizi_rl import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from uffizi_rl import s02_config as config  # noqa: E402
from uffizi_rl.agents.s08_baselines import evaluate_baselines  # noqa: E402
from uffizi_rl.agents.s07_q_learning import (  # noqa: E402
    evaluate_policy,
    greedy_q_policy,
    train_q_learning,
)
from uffizi_rl.pipeline._paths import ensure_outputs_dir  # noqa: E402


def main() -> None:
    """Train the tabular Q-learning agent and evaluate the full baseline suite."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2500,
                        help="Training episodes. Use 25_000+ for converged results.")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = parser.parse_args()

    output_dir = ensure_outputs_dir()

    print(f"[1/2] Training tabular Q-learning ({args.episodes} episodes, seed={args.seed})")
    agent, history, env = train_q_learning(episodes=args.episodes, seed=args.seed)
    q_metrics = evaluate_policy(env, greedy_q_policy(agent), episodes=40, seed=args.seed + 1)
    print(f"  Q-learning mean return: {q_metrics['mean_return']:.2f}")

    print("[2/2] Evaluating five baseline policies on the same toy graph")
    baselines = evaluate_baselines(episodes=40, seed=args.seed + 2)
    for b in baselines:
        print(f"  {b.name:25s} return={b.metrics['mean_return']:.2f}")

    summary = {
        "episodes": args.episodes,
        "seed": args.seed,
        "q_learning_eval": q_metrics,
        "baseline_results": [{"name": b.name, **b.metrics} for b in baselines],
        "episode_rewards": history.episode_rewards,
        "episode_lengths": history.episode_lengths,
        "offpeak_botticelli_visits": history.botticelli_offpeak_visits,
    }
    out_path = output_dir / "02_q_learning.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
