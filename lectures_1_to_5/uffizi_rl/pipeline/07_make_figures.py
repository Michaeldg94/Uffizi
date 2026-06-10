"""SCRIPT 07: Generate all figures from the saved outputs.

[READING ORDER: file 11 of 12 - read after 02_train_q_learning.py]

Reads the JSON and NumPy artifacts written by scripts 01 and 02 and
produces 11 publication-quality figures (PNG + PDF).

Phase 1 figures (environment and crowd dynamics):
  01_graph_density           Museum graph colored by snapshot density
  01_density_heatmap         Time-by-room density heatmap (180 minutes x 98 rooms)
  01_top_congested           Top 15 most-congested rooms by mean density
  01_capacity_distribution   Histogram of room capacities (98 rooms)
  01_arrival_envelope        Daily Gaussian arrival envelope
  01_visitors_over_day       Total visitors inside museum, minute by minute

Phase 2 figures (tabular Q-learning + baselines):
  02_q_learning_curve        Episode-by-episode returns with rolling mean
  02_offpeak_botticelli      Off-peak Botticelli visits (temporal arbitrage)
  02_episode_lengths         Episode length evolution
  02_baseline_comparison     Q-learning vs the 5 handcrafted baselines
  02_epsilon_decay           Epsilon-greedy decay schedule

Run AFTER 01 and 02. Writes only figures (no JSON).

Usage:
    uv run python uffizi_rl/pipeline/07_make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from uffizi_rl import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from uffizi_rl import s02_config as config  # noqa: E402
from uffizi_rl.analysis.visualization import (  # noqa: E402
    plot_arrival_envelope,
    plot_baseline_comparison,
    plot_density_heatmap,
    plot_episode_length_evolution,
    plot_epsilon_decay,
    plot_graph_density,
    plot_learning_curve,
    plot_offpeak_botticelli,
    plot_room_capacity_distribution,
    plot_top_congested_rooms,
    plot_visitors_inside_over_day,
)
from uffizi_rl.environment.s05_crowd_simulator import CrowdSimulator  # noqa: E402
from uffizi_rl.environment.s03_museum_graph import build_uffizi_graph  # noqa: E402
from uffizi_rl.pipeline._paths import ensure_figures_dir, ensure_outputs_dir  # noqa: E402


def _load_json(path: Path):
    """Load a JSON file or return None and print a SKIP message if missing."""

    if not path.exists():
        print(f"  SKIP: {path.name} not found (run earlier scripts first).")
        return None
    with path.open() as f:
        return json.load(f)


def _saved(name: str, ok: bool) -> None:
    """Pretty status line."""

    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {name}")


def main() -> None:
    """Build all figures listed in the module docstring."""

    output_dir = ensure_outputs_dir()
    figs = ensure_figures_dir()
    print(f"Writing figures to {figs}/\n")

    # =====================================================================
    # PHASE 1 figures: environment and crowd dynamics
    # =====================================================================
    print("=== Phase 1: Environment and MDP framing ===")

    sim = CrowdSimulator(daily_total=config.DAILY_VISITORS_NORMAL,
                         seed=config.DEFAULT_SEED + 100)
    sim.run_day()
    g = build_uffizi_graph()
    _saved("01_graph_density",
           plot_graph_density(g, sim.current_density_dict(),
                              str(figs / "01_graph_density.png")))

    matrix_path = output_dir / "01_density_matrix.npy"
    if matrix_path.exists():
        density_matrix = np.load(matrix_path)
        _saved("01_density_heatmap",
               plot_density_heatmap(density_matrix,
                                    str(figs / "01_density_heatmap.png"),
                                    room_labels=config.ROOM_IDS))
        capacities = [config.ROOM_DATA[r]["capacity"] for r in config.ROOM_IDS]
        _saved("01_visitors_over_day",
               plot_visitors_inside_over_day(density_matrix, capacities,
                                              str(figs / "01_visitors_over_day.png")))
        room_means = density_matrix.mean(axis=0)
        pairs = sorted(
            ((config.ROOM_IDS[i], float(room_means[i])) for i in range(len(config.ROOM_IDS))),
            key=lambda kv: kv[1], reverse=True,
        )
        _saved("01_top_congested",
               plot_top_congested_rooms(pairs, str(figs / "01_top_congested.png"),
                                         top_n=15))
    else:
        print("  SKIP: density matrix missing.")

    rooms_data = [config.ROOM_DATA[r] for r in config.ROOM_IDS]
    _saved("01_capacity_distribution",
           plot_room_capacity_distribution(rooms_data,
                                            str(figs / "01_capacity_distribution.png")))
    _saved("01_arrival_envelope",
           plot_arrival_envelope(str(figs / "01_arrival_envelope.png")))

    # =====================================================================
    # PHASE 2 figures: tabular Q-learning + exploration
    # =====================================================================
    print("\n=== Phase 2: Tabular Q-Learning and baselines ===")

    q_data = _load_json(output_dir / "02_q_learning.json")
    if q_data is not None:
        rewards = q_data.get("episode_rewards", [])
        if rewards:
            _saved("02_q_learning_curve",
                   plot_learning_curve(rewards,
                                       str(figs / "02_q_learning_curve.png"),
                                       title="Tabular Q-Learning on Toy Graph"))
        offpeak = q_data.get("offpeak_botticelli_visits", [])
        if offpeak:
            _saved("02_offpeak_botticelli",
                   plot_offpeak_botticelli(offpeak,
                                            str(figs / "02_offpeak_botticelli.png")))
        lengths = q_data.get("episode_lengths", [])
        if lengths:
            _saved("02_episode_lengths",
                   plot_episode_length_evolution(lengths,
                                                  str(figs / "02_episode_lengths.png")))
        baselines = q_data.get("baseline_results", [])
        q_return = q_data.get("q_learning_eval", {}).get("mean_return", 0.0)
        if baselines:
            _saved("02_baseline_comparison",
                   plot_baseline_comparison(baselines, q_return,
                                             str(figs / "02_baseline_comparison.png")))

    _saved("02_epsilon_decay",
           plot_epsilon_decay(str(figs / "02_epsilon_decay.png")))

    print(f"\nAll figures written to {figs}/")


if __name__ == "__main__":
    main()
