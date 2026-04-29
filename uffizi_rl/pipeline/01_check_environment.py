"""SCRIPT 01: Environment construction and sanity checks (Phase 0 + Phase 1).

Builds the museum graph, validates its topology, runs a smoke calibration
of the crowd simulator, and exercises the Gymnasium environment with a
random rollout. Writes a JSON summary and the density snapshot used by
script 07 (figures).

Run BEFORE any other numbered script.

Outputs:
  outputs/01_environment.json    (graph validation, calibration, rollout summary)
  outputs/01_density_matrix.npy  (density-by-minute matrix for the heatmap)

Usage:
    uv run python uffizi_rl/pipeline/01_check_environment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from uffizi_rl import ...`
# works whether the script is invoked directly or via run_project.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from uffizi_rl import config  # noqa: E402
from uffizi_rl.environment.crowd_simulator import CrowdSimulator  # noqa: E402
from uffizi_rl.environment.museum_graph import build_uffizi_graph, validate_uffizi_graph  # noqa: E402
from uffizi_rl.environment.uffizi_env import UffiziEnv  # noqa: E402
from uffizi_rl.pipeline._paths import ensure_outputs_dir  # noqa: E402


def main() -> None:
    """Build and validate the environment, then write artifacts."""

    output_dir = ensure_outputs_dir()

    print("[1/4] Capacity sanity check (Phase 0)")
    capacity = config.capacity_math_check()
    print(f"  avg_occupancy={capacity['avg_occupancy']:.0f} / capacity={capacity['capacity']:.0f}")

    print("[2/4] Building and validating the 98-room graph (Phase 1)")
    graph = build_uffizi_graph()
    validation = validate_uffizi_graph(graph).as_dict()
    print(f"  nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")
    print(f"  validation passed: {validation['all_pass']}")

    print("[3/4] Calibrating the crowd simulator (5 days)")
    sim = CrowdSimulator(daily_total=config.DAILY_VISITORS_NORMAL, seed=config.DEFAULT_SEED)
    calibration = sim.run_many_days(n_days=5)
    print(f"  mean_completed_visitors={calibration['completed_visitors_mean']:.1f}")
    print(f"  mean_peak_inside={calibration['peak_inside_mean']:.1f}")

    print("[4/4] Smoke run with the Gymnasium environment + density snapshot")
    sim_day = CrowdSimulator(daily_total=config.DAILY_VISITORS_NORMAL,
                             seed=config.DEFAULT_SEED + 100)
    sim_day.run_day()
    density_matrix = sim_day.export_density_matrix()
    np.save(output_dir / "01_density_matrix.npy", density_matrix)

    env = UffiziEnv(seed=config.DEFAULT_SEED, episode_minutes=180)
    env.reset()
    rng = config.get_rng(config.DEFAULT_SEED)
    total_reward, steps = 0.0, 0
    done = trunc = False
    while not (done or trunc) and steps < 240:
        mask = env.get_action_mask()
        action = int(rng.choice(np.flatnonzero(mask)))
        _, r, done, trunc, _ = env.step(action)
        total_reward += r
        steps += 1
    print(f"  random rollout: {steps} steps, total reward {total_reward:.1f}")

    summary = {
        "phase0_capacity": capacity,
        "graph_validation": validation,
        "calibration_5_days": calibration,
        "env_random_rollout": {
            "steps": steps,
            "total_reward": float(total_reward),
            "terminated": bool(done),
            "truncated": bool(trunc),
        },
    }
    out_path = output_dir / "01_environment.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
