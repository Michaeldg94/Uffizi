"""SCRIPT 02: Tabular Q-learning on the 12-room toy graph (Phase 2).

Trains two variants of tabular Q-learning with epsilon-greedy exploration
on a simplified 12-room version of the Uffizi:

  - Vanilla Q-learning (Watkins, 1989)
  - Double Q-learning (Hasselt, 2010): decouples action selection from
    action evaluation, removing the systematic overestimation bias of
    vanilla Q-learning.

Compares both against five handcrafted baselines (default path, random,
greedy least crowded, greedy value ratio, peak avoidance) so the reader
can judge whether RL discovers strategies a human designer would miss
and whether the Double-Q bias fix helps on this problem.

Run AFTER 01. Writes:
  outputs/02_q_learning.json (returns + per-episode rewards for both variants)

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

from uffizi_rl import config  # noqa: E402
from uffizi_rl.agents.baselines import evaluate_baselines  # noqa: E402
from uffizi_rl.agents.q_learning import (  # noqa: E402
    evaluate_policy,
    greedy_double_q_policy,
    greedy_q_policy,
    train_double_q_learning,
    train_q_learning,
)
from uffizi_rl.pipeline._paths import ensure_outputs_dir  # noqa: E402


def main() -> None:
    """Train both Q-learning variants and evaluate the full baseline suite."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2500,
                        help="Training episodes. Use 25_000+ for converged results.")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = parser.parse_args()

    output_dir = ensure_outputs_dir()

    print(f"[1/3] Training vanilla Q-learning ({args.episodes} episodes, seed={args.seed})")
    q_agent, q_history, env = train_q_learning(episodes=args.episodes, seed=args.seed)
    q_metrics = evaluate_policy(env, greedy_q_policy(q_agent), episodes=40, seed=args.seed + 1)
    print(f"  Q-learning mean return: {q_metrics['mean_return']:.2f}")

    print(f"[2/3] Training Double Q-learning ({args.episodes} episodes, seed={args.seed})")
    dq_agent, dq_history, dq_env = train_double_q_learning(
        episodes=args.episodes, seed=args.seed
    )
    dq_metrics = evaluate_policy(
        dq_env, greedy_double_q_policy(dq_agent), episodes=40, seed=args.seed + 1
    )
    print(f"  Double Q-learning mean return: {dq_metrics['mean_return']:.2f}")

    print("[3/3] Evaluating five baseline policies on the same toy graph")
    baselines = evaluate_baselines(episodes=40, seed=args.seed + 2)
    for b in baselines:
        print(f"  {b.name:25s} return={b.metrics['mean_return']:.2f}")

    summary = {
        "episodes": args.episodes,
        "seed": args.seed,
        "q_learning_eval": q_metrics,
        "double_q_learning_eval": dq_metrics,
        "baseline_results": [{"name": b.name, **b.metrics} for b in baselines],
        "episode_rewards": q_history.episode_rewards,
        "episode_lengths": q_history.episode_lengths,
        "offpeak_botticelli_visits": q_history.botticelli_offpeak_visits,
        "double_q_episode_rewards": dq_history.episode_rewards,
        "double_q_episode_lengths": dq_history.episode_lengths,
        "double_q_offpeak_botticelli_visits": dq_history.botticelli_offpeak_visits,
    }
    out_path = output_dir / "02_q_learning.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
