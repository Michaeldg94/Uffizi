"""SCRIPT 07: Generate all figures from the saved phase outputs.

Reads the JSON and NumPy artifacts written by scripts 01-06 and produces
publication-quality PNG and PDF figures:
  outputs/figures/01_graph_density.png    (museum graph colored by occupancy)
  outputs/figures/01_density_heatmap.png  (time-by-room density heatmap)
  outputs/figures/02_q_learning_curve.png (Q-learning training curve)
  outputs/figures/05_interventions.png    (intervention welfare-gain bar chart)
  outputs/figures/06_typeb_transition.png (Type B fraction sweep curve)

Run AFTER 01-06. Writes only figures (no JSON).

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

from uffizi_rl import config  # noqa: E402
from uffizi_rl.analysis.visualization import (  # noqa: E402
    plot_density_heatmap,
    plot_graph_density,
    plot_intervention_comparison,
    plot_learning_curve,
    plot_phase_transition,
)
from uffizi_rl.environment.crowd_simulator import CrowdSimulator  # noqa: E402
from uffizi_rl.environment.museum_graph import build_uffizi_graph  # noqa: E402
from uffizi_rl.pipeline._paths import ensure_figures_dir, ensure_outputs_dir  # noqa: E402


def _load_json(path: Path):
    if not path.exists():
        print(f"  SKIP: {path} not found (run earlier scripts first).")
        return None
    with path.open() as f:
        return json.load(f)


def main() -> None:
    """Build all figures listed in the module docstring."""

    output_dir = ensure_outputs_dir()
    figs = ensure_figures_dir()

    print("[1/5] Museum graph density snapshot")
    sim = CrowdSimulator(daily_total=config.DAILY_VISITORS_NORMAL, seed=config.DEFAULT_SEED + 100)
    sim.run_day()
    g = build_uffizi_graph()
    ok = plot_graph_density(g, sim.current_density_dict(), str(figs / "01_graph_density.png"))
    print(f"  saved: {ok}")

    print("[2/5] Density heatmap (time x room)")
    matrix_path = output_dir / "01_density_matrix.npy"
    if matrix_path.exists():
        density_matrix = np.load(matrix_path)
        ok = plot_density_heatmap(density_matrix, str(figs / "01_density_heatmap.png"),
                                  room_labels=config.ROOM_IDS)
        print(f"  saved: {ok}")
    else:
        print(f"  SKIP: {matrix_path} not found.")

    print("[3/5] Q-learning training curve")
    q_data = _load_json(output_dir / "02_q_learning.json")
    if q_data is not None:
        ok = plot_learning_curve(q_data["episode_rewards"],
                                 str(figs / "02_q_learning_curve.png"),
                                 title="Tabular Q-Learning on Toy Graph")
        print(f"  saved: {ok}")

    print("[4/5] Intervention comparison bar chart")
    iv_data = _load_json(output_dir / "05_interventions.json")
    if iv_data is not None:
        ok = plot_intervention_comparison(iv_data["table"],
                                          str(figs / "05_interventions.png"))
        print(f"  saved: {ok}")

    print("[5/5] Type B fraction phase transition")
    sw_data = _load_json(output_dir / "06_sweeps.json")
    if sw_data is not None:
        ok = plot_phase_transition(sw_data["type_b_ratio_sweep"],
                                   x_key="type_b_ratio",
                                   out_path=str(figs / "06_typeb_transition.png"))
        print(f"  saved: {ok}")

    print(f"\nFigures written to {figs}/")


if __name__ == "__main__":
    main()
