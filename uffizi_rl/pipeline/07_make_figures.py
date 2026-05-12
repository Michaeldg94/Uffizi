"""SCRIPT 07: Generate all figures from the saved phase outputs.

Reads the JSON and NumPy artifacts written by scripts 01-06 and produces
~25 publication-quality figures (PNG + PDF), organized by course lecture.

Figure naming convention: NN_short_name.{png,pdf}, where NN matches the
script that produced the data.

The figures are grouped by phase:

  Phase 1 (Lectures 1-3, MDPs and environment)
    01_graph_density           Museum graph colored by snapshot density
    01_density_heatmap         Time-by-room density heatmap (180 minutes)
    01_top_congested           Top 15 most-congested rooms by mean density
    01_capacity_distribution   Histogram of room capacities (98 rooms)
    01_arrival_envelope        Gaussian arrival envelope (one day)
    01_visitors_over_day       Total visitors inside museum, minute by minute

  Phase 2 (Lectures 4-5, 9-10, value-based methods + exploration)
    02_q_learning_curve        Episode-by-episode returns with rolling mean
    02_offpeak_botticelli      Off-peak Botticelli visits (temporal arbitrage)
    02_episode_lengths         Episode length evolution
    02_baseline_comparison     Q-learning vs the 5 handcrafted baselines
    02_epsilon_decay           Epsilon-greedy schedule

  Phase 3 (Lectures 6-8, policy-based methods + deep RL)
    03_deep_rl_comparison      MaskablePPO vs PPO-no-mask vs DQN

  Phase 5 (equilibrium analysis)
    04_equilibrium_convergence Iterated best-response welfare and Botticelli

  Phase 6 (intervention design)
    05_interventions           All curated interventions, welfare deltas
    05_top_interventions       Top 10 interventions detailed
    05_welfare_revenue         Welfare-vs-revenue scatter (Pareto view)
    05_portfolio_breakdown     Portfolio gains over baseline

  Phase 7 (population sweeps)
    06_typeb_transition        Welfare across Type B fraction sweep
    06_typeb_gini              Gini coefficient across Type B fraction
    06_volume_welfare          Welfare across daily visitor volume
    06_volume_botticelli       Botticelli density across volume
    06_heterogeneity           Welfare across preference heterogeneity

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
    plot_arrival_envelope,
    plot_baseline_comparison,
    plot_deep_rl_comparison,
    plot_density_heatmap,
    plot_episode_length_evolution,
    plot_epsilon_decay,
    plot_equilibrium_convergence,
    plot_graph_density,
    plot_intervention_comparison,
    plot_intervention_welfare_revenue,
    plot_learning_curve,
    plot_offpeak_botticelli,
    plot_phase_transition,
    plot_portfolio_breakdown,
    plot_room_capacity_distribution,
    plot_sweep_with_band,
    plot_top_congested_rooms,
    plot_top_interventions,
    plot_visitors_inside_over_day,
)
from uffizi_rl.environment.crowd_simulator import CrowdSimulator  # noqa: E402
from uffizi_rl.environment.museum_graph import build_uffizi_graph  # noqa: E402
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
    print("=== Phase 1: Environment (Lectures 1-3) ===")

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
        # Visitors-inside-over-day: needs capacity per room (same order as ROOM_IDS).
        capacities = [config.ROOM_DATA[r]["capacity"] for r in config.ROOM_IDS]
        _saved("01_visitors_over_day",
               plot_visitors_inside_over_day(density_matrix, capacities,
                                              str(figs / "01_visitors_over_day.png")))
        # Top-congested rooms from the density matrix.
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

    # Room capacity distribution: read directly from config (no simulation needed).
    rooms_data = [config.ROOM_DATA[r] for r in config.ROOM_IDS]
    _saved("01_capacity_distribution",
           plot_room_capacity_distribution(rooms_data,
                                            str(figs / "01_capacity_distribution.png")))
    # Arrival envelope: theoretical curve, no data needed.
    _saved("01_arrival_envelope",
           plot_arrival_envelope(str(figs / "01_arrival_envelope.png")))

    # =====================================================================
    # PHASE 2 figures: tabular Q-learning + exploration
    # =====================================================================
    print("\n=== Phase 2: Tabular Q-Learning (Lectures 4-5, 9-10) ===")

    q_data = _load_json(output_dir / "02_q_learning.json")
    if q_data is not None:
        # The learning curve: per-episode return with rolling mean.
        rewards = q_data.get("episode_rewards", [])
        if rewards:
            _saved("02_q_learning_curve",
                   plot_learning_curve(rewards,
                                       str(figs / "02_q_learning_curve.png"),
                                       title="Tabular Q-Learning on Toy Graph"))

        # Off-peak Botticelli visits: the temporal arbitrage discovery.
        offpeak = q_data.get("offpeak_botticelli_visits", [])
        if offpeak:
            _saved("02_offpeak_botticelli",
                   plot_offpeak_botticelli(offpeak,
                                            str(figs / "02_offpeak_botticelli.png")))

        # Episode length evolution: agent should learn shorter routes over time.
        lengths = q_data.get("episode_lengths", [])
        if lengths:
            _saved("02_episode_lengths",
                   plot_episode_length_evolution(lengths,
                                                  str(figs / "02_episode_lengths.png")))

        # Baseline comparison: 5 baselines plus Q-learning.
        baselines = q_data.get("baseline_results", [])
        q_return = q_data.get("q_learning_eval", {}).get("mean_return", 0.0)
        if baselines:
            _saved("02_baseline_comparison",
                   plot_baseline_comparison(baselines, q_return,
                                             str(figs / "02_baseline_comparison.png")))

    # Epsilon decay: schedule visualization, doesn't need data.
    _saved("02_epsilon_decay",
           plot_epsilon_decay(str(figs / "02_epsilon_decay.png")))

    # =====================================================================
    # PHASE 3 figures: deep RL
    # =====================================================================
    print("\n=== Phase 3: Deep RL (Lectures 6-8) ===")

    dr_data = _load_json(output_dir / "03_deep_rl.json")
    if dr_data is not None:
        ppo = dr_data.get("maskable_ppo", {})
        ablations = dr_data.get("ablations", [])
        _saved("03_deep_rl_comparison",
               plot_deep_rl_comparison(ppo.get("mean_episode_reward", 0.0),
                                        ppo.get("std_episode_reward", 0.0),
                                        ablations,
                                        str(figs / "03_deep_rl_comparison.png")))

    # =====================================================================
    # PHASE 5 figures: equilibrium analysis
    # =====================================================================
    print("\n=== Phase 5: Equilibrium and Price of Anarchy ===")

    eq_data = _load_json(output_dir / "04_equilibrium.json")
    if eq_data is not None and eq_data.get("iterations"):
        _saved("04_equilibrium_convergence",
               plot_equilibrium_convergence(eq_data["iterations"],
                                             str(figs / "04_equilibrium_convergence.png")))

    # =====================================================================
    # PHASE 6 figures: intervention design
    # =====================================================================
    print("\n=== Phase 6: Interventions and Portfolio ===")

    iv_data = _load_json(output_dir / "05_interventions.json")
    if iv_data is not None:
        table = iv_data.get("table", [])
        if table:
            _saved("05_interventions",
                   plot_intervention_comparison(table,
                                                 str(figs / "05_interventions.png")))
            _saved("05_top_interventions",
                   plot_top_interventions(table,
                                           str(figs / "05_top_interventions.png"),
                                           top_n=10))
            _saved("05_welfare_revenue",
                   plot_intervention_welfare_revenue(table,
                                                      str(figs / "05_welfare_revenue.png")))

        # Portfolio breakdown: requires baseline values from the table.
        portfolio = iv_data.get("portfolio")
        baseline_w = float(iv_data.get("baseline_welfare",
                                       table[0]["total_welfare"] if table else 0))
        baseline_r = float(iv_data.get("baseline_revenue",
                                       table[0].get("revenue", 0) if table else 0))
        if portfolio:
            _saved("05_portfolio_breakdown",
                   plot_portfolio_breakdown(portfolio, baseline_w, baseline_r,
                                             str(figs / "05_portfolio_breakdown.png")))

    # =====================================================================
    # PHASE 7 figures: population sweeps
    # =====================================================================
    print("\n=== Phase 7: Population Sweeps ===")

    sw_data = _load_json(output_dir / "06_sweeps.json")
    if sw_data is not None:
        if sw_data.get("type_b_ratio_sweep"):
            _saved("06_typeb_transition",
                   plot_phase_transition(sw_data["type_b_ratio_sweep"],
                                          x_key="type_b_ratio",
                                          out_path=str(figs / "06_typeb_transition.png")))
            _saved("06_typeb_gini",
                   plot_sweep_with_band(sw_data["type_b_ratio_sweep"],
                                         x_key="type_b_ratio",
                                         y_key="occupancy_gini",
                                         y_label="Occupancy Gini Coefficient",
                                         title="Spatial Inequality vs. Type B Fraction",
                                         out_path=str(figs / "06_typeb_gini.png")))

        if sw_data.get("volume_sweep"):
            _saved("06_volume_welfare",
                   plot_sweep_with_band(sw_data["volume_sweep"],
                                         x_key="daily_total",
                                         y_key="total_welfare",
                                         y_label="Total Welfare",
                                         title="Welfare vs. Daily Visitor Volume",
                                         out_path=str(figs / "06_volume_welfare.png")))
            _saved("06_volume_botticelli",
                   plot_sweep_with_band(sw_data["volume_sweep"],
                                         x_key="daily_total",
                                         y_key="peak_botticelli_density",
                                         y_label="Peak Botticelli Density",
                                         title="Botticelli Congestion vs. Daily Volume",
                                         out_path=str(figs / "06_volume_botticelli.png")))

        if sw_data.get("heterogeneity_sweep"):
            _saved("06_heterogeneity",
                   plot_sweep_with_band(sw_data["heterogeneity_sweep"],
                                         x_key="heterogeneity_scale",
                                         y_key="total_welfare",
                                         y_label="Total Welfare",
                                         title="Welfare vs. Preference Heterogeneity",
                                         out_path=str(figs / "06_heterogeneity.png")))

    print(f"\nAll figures written to {figs}/")


if __name__ == "__main__":
    main()
