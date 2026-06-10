# Uffizi RL: Reinforcement Learning for Museum Visitor Flow Optimization

Course project for **Reinforcement Learning**, Barcelona School of Economics, Spring 2026.

## What this project does

The Uffizi Gallery in Florence receives roughly 5,000 visitors per day across 98 rooms. Demand concentrates on a handful of famous rooms (Botticelli, Leonardo, Raphael/Michelangelo) while most of the collection goes undervisited. We treat the museum as a reinforcement learning problem and demonstrate the core RL pipeline on this single, coherent application.

## How to run

```bash
uv sync
uv run python run_project.py                      # smoke run (~minutes)
uv run python run_project.py --episodes 25000     # converged Q-learning
```

Artifacts land in `outputs/` (JSON, NumPy) and `outputs/figures/` (11 figures, PNG + PDF). The committed `outputs/` already contains the results of a converged run, so you can inspect everything without re-running.

## The project, end to end

The project is organized as a 3-step pipeline that you can run from start to finish with one command, or step by step. Each step builds on the previous one.

### Step 1 (Phase 0 + Phase 1): Environment and MDP formulation
**Script**: `uffizi_rl/pipeline/01_check_environment.py`

The museum is formulated as a Markov Decision Process: states (room, time, visited mask, density), actions (stay or move to a neighbor), the reward function `R = importance / (1 + alpha * density)`, and the transition function. The crowd simulator (`uffizi_rl/environment/s05_crowd_simulator.py`) implements the environment with 5,000 stochastic visitors over a 615-minute day.

This step also performs a capacity sanity check showing that the museum operates near its 900-visitor fire-safety limit, and a graph validation step that confirms all 98 rooms are reachable through the 119 corridors and doorways of the floor plan.

The exact MDP is also where the conceptual link to dynamic programming sits: the state space approximately 10^30 elements (98 rooms x 615 time steps x 2^98 visited masks x continuous density vectors), so exact dynamic programming is infeasible. This motivates the move to model-free methods in Step 2.

### Step 2 (Phase 2): Tabular Q-learning + baselines on the toy graph
**Script**: `uffizi_rl/pipeline/02_train_q_learning.py`

Tabular Q-learning with epsilon-greedy exploration on a simplified 12-room toy version of the museum (the state space is small enough for a Q-table to fit in memory; approximately 120,000 entries). Training runs for up to 25,000 episodes with epsilon decaying from 1.0 to 0.01.

The Q-learning agent is compared against 5 handcrafted baseline policies:
1. **default_path**: follow the recommended guidebook route.
2. **random**: choose actions uniformly at random.
3. **greedy_least_crowded**: always move to the least crowded neighbor.
4. **greedy_value_ratio**: move to the neighbor maximizing importance / (1 + alpha * density).
5. **peak_avoidance**: avoid Botticelli rooms during high-density periods.

**Key finding**: the agent discovers **temporal arbitrage**, learning to visit the Botticelli rooms during off-peak windows rather than during the midday crush. This timing strategy is not programmed in; it emerges purely from the Q-value updates and beats every handcrafted baseline.

### Step 3: Figures
**Script**: `uffizi_rl/pipeline/07_make_figures.py`

Reads the outputs of Step 1 and Step 2 and produces 11 publication-quality figures (PNG + PDF), grouped by phase.

## Suggested reading order of the source code

1. `uffizi_rl/s02_config.py` -- the MDP definition: 98 rooms (with names, capacities, importance, magnetism), 119 edges between rooms, all modeling parameters with their sources cited. Sections are clearly labeled top to bottom.
2. `uffizi_rl/environment/s03_museum_graph.py` -- builds the NetworkX graph from `s02_config.py` and validates the topology.
3. `uffizi_rl/environment/s04_visitor_profiles.py` -- the heterogeneous visitor model (Type A art lovers vs Type B guidebook tourists).
4. `uffizi_rl/environment/s05_crowd_simulator.py` -- the minute-step simulator that models 5,000 visitors per day.
5. `uffizi_rl/environment/s06_uffizi_env.py` -- the Gymnasium environment wrapper used in the Step 1 smoke test.
6. `uffizi_rl/agents/s07_q_learning.py` -- the tabular Q-learning agent + the 12-room `ToyTabularEnv`.
7. `uffizi_rl/agents/s08_baselines.py` -- the 5 handcrafted baseline policies.

## Repository layout

```
.
  README.md                              <- you are here
  pyproject.toml                         <- dependencies (numpy, networkx, gymnasium, matplotlib, seaborn)
  uv.lock                                <- locked versions for reproducibility
  run_project.py                         <- one-shot driver: runs scripts 01, 02, 07 in order
  .gitignore

  uffizi_rl/
    s02_config.py                        <- (#2) MDP definition: rooms, edges, capacities, parameters
    environment/
      s03_museum_graph.py                <- (#3) Full graph (98 rooms) and toy graph (12 rooms)
      s04_visitor_profiles.py            <- (#4) Type A (art lovers) + Type B (guidebook tourists)
      s05_crowd_simulator.py             <- (#5) Minute-step simulator
      s06_uffizi_env.py                  <- (#6) Gymnasium env wrapper
    agents/
      s07_q_learning.py                  <- (#7) Tabular Q-learning + the 12-room toy environment
      s08_baselines.py                   <- (#8) 5 handcrafted baseline policies
    pipeline/
      01_check_environment.py            <- (#9)  Phase 0+1: capacity check, environment
      02_train_q_learning.py             <- (#10) Phase 2: tabular Q-learning + 5 baselines
      07_make_figures.py                 <- (#11) Generate the 11 figures
      _paths.py                          <- Shared path helpers
    interventions/
      intervention_config.py             <- Internal: imported by s05_crowd_simulator
      hidden_gem_trails.py               <- Internal: imported by s05_crowd_simulator
    analysis/
      metrics.py                         <- Gini, Theil, welfare, experience quality
      visualization.py                   <- All 11 plot functions

  tests/
    test_config.py                       <- Validates constants and parameters
    test_graph.py                        <- Validates graph topology

  outputs/
    01_environment.json                  <- Phase 0 capacity + Phase 1 calibration
    01_density_matrix.npy                <- One day of density-by-room time series
    02_q_learning.json                   <- Q-learning return + 5 baseline returns + training history
    figures/                             <- 11 figures, each as PNG + PDF
      01_arrival_envelope.png            <- Daily Gaussian arrival rate (theoretical)
      01_capacity_distribution.png       <- Histogram of room capacities
      01_density_heatmap.png             <- Time-by-room density heatmap (180 min x 98 rooms)
      01_graph_density.png               <- Museum graph colored by occupancy
      01_top_congested.png               <- Top 15 most-congested rooms (mean density)
      01_visitors_over_day.png           <- Total visitors inside vs 900 fire-safety cap
      02_baseline_comparison.png         <- Q-learning vs 5 baselines (bar chart)
      02_episode_lengths.png             <- Episode length over training
      02_epsilon_decay.png               <- Epsilon-greedy schedule (theoretical curve)
      02_offpeak_botticelli.png          <- Off-peak Botticelli visits (temporal arbitrage discovery)
      02_q_learning_curve.png            <- Per-episode returns with rolling mean and 95% CI
```

## Key results (committed in `outputs/`)

| Metric | Value |
|---|---|
| Museum size | 98 rooms, 119 corridors/doorways |
| Fire-safety capacity | 900 simultaneous visitors |
| Average daily occupancy (5,000 visitors) | ~745 |
| Toy graph for Q-learning | 12 rooms |
| Q-learning training (25k episodes) | ~3 min on a laptop |
| Q-learning mean return | 181 |
| Best baseline (peak avoidance) | 153 |
| Worst baseline (random) | 25 |
| Q-learning improvement over best baseline | +18% |

The headline finding: tabular Q-learning, given only the reward signal and exploration, **discovers a timing strategy that no human designer programmed**. It learns to visit the Botticelli rooms when the crowd is low rather than when the crowd is high.

## Testing

```bash
uv run pytest tests/ -q
```

## References

- Attanasio et al. (2022). Visitors flow management at Uffizi Gallery. *Information Technology & Tourism*, 24(3), 409-434.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. Chapters 3-6.
