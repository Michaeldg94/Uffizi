# Uffizi RL: Reinforcement Learning for Museum Visitor Flow Optimization

Course project for Reinforcement Learning, Barcelona School of Economics, Spring 2026.

## Overview

The Uffizi Gallery in Florence receives approximately 5,000 visitors per day across 98 rooms. Demand concentrates on a handful of famous rooms (Botticelli, Leonardo, Raphael/Michelangelo), producing severe congestion while most of the collection goes undervisited. This project treats the museum as a reinforcement learning problem: a visitor navigating the museum faces a sequential decision problem in which the reward depends on both the art quality and the crowd density of each room.

We build an agent-based crowd simulator of the entire museum, train RL agents on it (first on a small toy graph, then on the full museum), evaluate 30+ museum management interventions, and analyze the population dynamics that emerge when thousands of heterogeneous visitors interact.

## How the Course Maps to the Project

| Course Lecture | Topic | Where it appears |
|---|---|---|
| 1 | Introduction to RL | Museum visitor MDP: states, actions, rewards, transitions (Phase 1) |
| 2-3 | Dynamic Programming | Foundation for value functions; shown infeasible on 98-room museum (state space > 10^30) |
| 4-5 | Value-based methods | Tabular Q-learning on 12-room toy graph discovers temporal arbitrage (Phase 2) |
| 6-7 | Policy-based methods | MaskablePPO on full 98-room graph with action masking (Phase 3) |
| 8 | Deep RL | DQN and PPO ablations; curriculum learning; action masking analysis (Phase 3) |
| 9 | Multi-armed bandits | Epsilon-greedy exploration in Q-learning |
| 10 | Exploration and exploitation | Epsilon decay schedule; temporal arbitrage as exploitation of learned timing |

## How to Run the Project

The whole pipeline lives inside `uffizi_rl/pipeline/`, with seven numbered scripts that you run in order. From the project root:

```bash
uv sync                                 # install dependencies
uv run python run_project.py            # smoke test: minutes on a laptop
uv run python run_project.py --medium   # converged results: ~2 hours on M3 Pro
```

Or invoke each numbered script by hand:

```bash
uv run python uffizi_rl/pipeline/01_check_environment.py
uv run python uffizi_rl/pipeline/02_train_q_learning.py --episodes 25000
uv run python uffizi_rl/pipeline/03_train_deep_rl.py --timesteps 500000 --seeds 3
uv run python uffizi_rl/pipeline/04_run_equilibrium.py
uv run python uffizi_rl/pipeline/05_evaluate_interventions.py
uv run python uffizi_rl/pipeline/06_run_sweeps.py --resolution 7 --seeds 3
uv run python uffizi_rl/pipeline/07_make_figures.py
```

All artifacts (JSON summaries, density matrices, PNG/PDF figures) land in `outputs/` at the project root.

## Repository Layout

```
uffizi_submission/
  pyproject.toml              # Python dependencies (numpy, gymnasium, sb3-contrib, etc.)
  uv.lock                     # Locked dependency versions for reproducibility
  README.md                   # This file
  run_project.py              # End-to-end driver: runs all 7 numbered scripts in order
  tests/                      # Pytest test suite (run with: uv run pytest tests/)

  uffizi_rl/                  # Main package (library code, imported by pipeline scripts)
    config.py                 # 98 rooms, 119 edges, capacities from floor plan, all assumptions
    pipeline/                 # 7 numbered run scripts (where the project flow lives)
      01_check_environment.py    Phase 0+1: capacity check, graph validation, calibration
      02_train_q_learning.py     Phase 2: tabular Q-learning on toy graph + 5 baselines
      03_train_deep_rl.py        Phase 3: MaskablePPO + PPO/DQN ablations on full graph
      04_run_equilibrium.py      Phase 5: iterated best-response, Price of Anarchy
      05_evaluate_interventions.py  Phase 6: score 30 interventions, find best portfolio
      06_run_sweeps.py           Phase 7: Type-B / volume / heterogeneity sweeps
      07_make_figures.py         Read 01-06 outputs, write PNG/PDF figures
      _paths.py                  Shared helper: resolves outputs/ relative to project root
      README.md                  Pipeline documentation (inputs/outputs per script)
    environment/              # The world the agents inhabit
      museum_graph.py           NetworkX graph builder; toy graph; topology validation
      crowd_simulator.py        Minute-step simulator with 30+ intervention mechanisms
      visitor_profiles.py       Type A (art lovers) and Type B (guidebook tourists)
      uffizi_env.py             Gymnasium environment wrapper with action masking
    agents/                   # RL algorithms
      q_learning.py             Tabular Q-learning on the 12-room toy graph
      baselines.py              5 handcrafted policies (default, random, greedy variants, peak avoidance)
      train_maskable_ppo.py     MaskablePPO with 3-stage curriculum
      train_ablations.py        Vanilla PPO + DQN without masking (controls)
    interventions/            # Mechanism design building blocks (read by crowd_simulator)
      intervention_config.py    Frozen dataclass with 62 intervention parameters
      timed_entry.py            Slot-based staggered entry helper
      congestion_pricing.py     Reservation book for Botticelli room access
      hidden_gem_trails.py      Preference reshaping via themed trails
      dynamic_info.py           Kiosk visibility model
    analysis/                 # Metrics, sweeps, optimization, plotting
      metrics.py                Gini, Theil, PoA, welfare, experience quality
      phase_transition.py       Population sweeps + iterated best-response equilibrium proxy
      portfolio.py              30 curated interventions + greedy/local-search optimization
      visualization.py          Density heatmaps, learning curves, intervention bar charts
      pareto.py                 Pareto frontier over (welfare, revenue)

  outputs/                    # Created at runtime, written to by every numbered script
    01_environment.json
    01_density_matrix.npy
    02_q_learning.json
    03_deep_rl.json
    04_equilibrium.json
    05_interventions.json
    06_sweeps.json
    figures/
      01_graph_density.png / .pdf
      01_density_heatmap.png / .pdf
      02_q_learning_curve.png / .pdf
      05_interventions.png / .pdf
      06_typeb_transition.png / .pdf
```

## Where Are All 30+ Interventions?

Q: There are only four files in `interventions/`. Where are the other 26?

A: The 62 intervention parameters are defined in `interventions/intervention_config.py` as a single frozen dataclass. The four standalone files in `interventions/` are helpers for the most complex mechanisms (slot assignment, reservation books, trail assignment, kiosk visibility). The actual logic for every intervention lives inside `environment/crowd_simulator.py`: the simulator reads an InterventionConfig in `__init__` and `step()` and modifies visitor behavior accordingly. This keeps the intervention surface small (a single dataclass) while the implementations are co-located with the simulator code that they affect.

## Reproducibility

Every script is deterministic given a seed. The submission ships with `uv.lock` to pin exact dependency versions. To reproduce paper-quality results from scratch:

```bash
uv sync
uv run python run_project.py --medium
```

This trains MaskablePPO for 500,000 timesteps across 3 seeds, runs all sweeps at resolution 7, and produces the full set of figures. Expected wall-clock: ~2 hours on M3 Pro.

For just a smoke test that proves everything imports and runs end-to-end:

```bash
uv sync
uv run python run_project.py            # ~5 minutes
```

## Key Results (from --medium run)

| Metric | Value |
|---|---|
| Q-learning mean return (toy graph, 25k episodes) | 181 |
| MaskablePPO mean return (full museum, 3 seeds) | 368 +/- 5 |
| PPO without action masking | 11 (masking is essential) |
| PPO without curriculum | ~200 (curriculum gives +84%) |
| DQN ablation | 68 |
| Price of Anarchy | ~1.004 (capacity problem, not coordination failure) |
| Best portfolio: welfare gain over baseline | +27% |
| Best portfolio: revenue gain over baseline | +25% |
| Top single interventions | Last-Hour Locals, Timed Entry, Resident Annual Pass, Lunch Free Entry |

The central finding: the optimal strategy is not "restrict access to crowded rooms" (which hurts revenue) but "give visitors reasons to go to empty rooms" (which maintains revenue while reducing congestion).

## Testing

```bash
uv run pytest tests/ -q
```

The test suite validates: graph topology and connectivity, simulator capacity constraints, intervention effects, metric computations.

## Design Decisions

1. **Why a simulator, not real data?** We do not have access to real visitor tracking data. The simulator, grounded in Attanasio et al. (2022) and the official floor plan, replaces the unknown real-world transition function P(s' | s, a). Every behavioral parameter is labeled `[assumption]` in `config.py` and could be replaced by measured values from a tracking system.

2. **Why action masking?** The museum is a graph with variable degree (2 to 15 neighbors per room). Without masking, the agent wastes training on impossible actions (moving to disconnected rooms). With masking, training focuses on choosing among valid options.

3. **Why curriculum learning?** With 5,000 visitors from the start, congestion penalties dominate the reward signal. Starting with 1,800 visitors lets the agent first learn the museum topology, then refine timing as crowds increase.

4. **Why so many interventions?** The 30+ intervention space lets the project explore mechanism design as a continuous design problem rather than a binary on/off question. Greedy forward selection plus local search finds the best portfolio (4-6 interventions deployed together).

5. **Why room capacities from the floor plan?** Initial hand-estimated capacities produced misleading congestion rankings (e.g., the Tribune appeared as the most congested room because its capacity was set to 20 instead of 37). Pixel-area analysis of the official floor plan provides proportionally accurate capacities.

## References

- Attanasio et al. (2022). Visitors flow management at Uffizi Gallery. *Information Technology & Tourism*, 24(3), 409-434.
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
