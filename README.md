# Uffizi RL: Reinforcement Learning for Museum Visitor Flow Optimization

Course project for **Reinforcement Learning**, Barcelona School of Economics, Spring 2026.

## What this project does

The Uffizi Gallery in Florence receives ~5,000 visitors per day across 98 rooms. Demand concentrates on a handful of famous rooms (Botticelli, Leonardo, Raphael/Michelangelo); the rest of the collection goes undervisited. This is a sequential decision problem (the visitor) and a population game (the museum management).

We build an agent-based crowd simulator of the entire museum, train RL agents on it, evaluate 30+ intervention mechanisms, and analyze the population dynamics. Every lecture of the course is exercised on the same project.

## How to run the project

```bash
git clone https://github.com/mocl20/Uffizi.git -b mocl
cd Uffizi
uv sync
uv run python run_project.py            # smoke run (5 minutes)
uv run python run_project.py --medium   # converged (~2 hours)
```

All artifacts land in `outputs/` (JSONs, NumPy arrays) and `outputs/figures/` (21 PNG + 21 PDF figures). The committed `outputs/` already contains the results of a `--medium` run, so you can inspect everything without re-running.

## Course-aligned structure: lectures, scripts, outputs

The 7 numbered scripts in `uffizi_rl/pipeline/` follow the course lecture order. Run them sequentially (or use `run_project.py` which calls them in turn).

### Lectures 1-3: Introduction to RL + Dynamic Programming
**Script**: `uffizi_rl/pipeline/01_check_environment.py`
**Phase**: Phase 0 (capacity check) + Phase 1 (environment construction)

**What it demonstrates**:
- MDP formulation: states (room, time, visited, density), actions (stay or move), rewards (importance / (1 + alpha * density)), transitions (graph + crowd dynamics).
- The state space exceeds 10^30 elements, demonstrating that exact dynamic programming is infeasible. This motivates the model-free methods used in later phases.

**Outputs**:
- `outputs/01_environment.json` (graph validation, 5-day simulator calibration, random rollout)
- `outputs/01_density_matrix.npy` (one full day of density-by-room time series)

**Figures** (`outputs/figures/01_*.png`/`.pdf`):
- `01_graph_density` -- Museum graph colored by snapshot occupancy
- `01_density_heatmap` -- Time-by-room density heatmap (180 minutes x 98 rooms)
- `01_top_congested` -- Top 15 most-congested rooms by mean density
- `01_capacity_distribution` -- Histogram of room capacities (right-skewed)
- `01_arrival_envelope` -- Daily Gaussian arrival rate
- `01_visitors_over_day` -- Total visitors inside vs the 900 fire-safety cap

### Lectures 4-5 + 9-10: Value-based methods + Exploration
**Script**: `uffizi_rl/pipeline/02_train_q_learning.py`
**Phase**: Phase 2 (tabular Q-learning on 12-room toy graph)

**What it demonstrates**:
- Tabular Q-learning with epsilon-greedy exploration (Lectures 4-5).
- Epsilon decay schedule (Lectures 9-10).
- Comparison against 5 handcrafted baselines: default path, random, greedy least crowded, greedy value ratio, peak avoidance.
- Key finding: the agent discovers **temporal arbitrage** (visit Botticelli when crowd density is low). No baseline captures this without being told.

**Outputs**:
- `outputs/02_q_learning.json` (Q-learning return, baseline returns, full training history)

**Figures**:
- `02_q_learning_curve` -- Per-episode returns with 25-episode rolling mean and 95% CI band
- `02_offpeak_botticelli` -- Off-peak Botticelli visits over training (temporal arbitrage emerging)
- `02_episode_lengths` -- Episode length over training (agent learns shorter routes)
- `02_baseline_comparison` -- Q-learning vs all 5 baselines (bar chart)
- `02_epsilon_decay` -- The exploration schedule

### Lectures 6-7 + 8: Policy-based methods + Deep RL
**Script**: `uffizi_rl/pipeline/03_train_deep_rl.py`
**Phase**: Phase 3 (MaskablePPO on the full 98-room museum)

**What it demonstrates**:
- Policy-based RL: MaskablePPO with stochastic policy and action masking (Lectures 6-7).
- Deep RL: neural-network function approximation, 3-stage curriculum learning (Lecture 8).
- Ablations: vanilla PPO and DQN without action masking serve as controls. Without masking, PPO collapses to ~11; with masking and curriculum, it reaches 368 +/- 5 across 3 seeds.

**Outputs**:
- `outputs/03_deep_rl.json` (MaskablePPO mean and std across seeds, PPO/DQN ablation returns)

**Figures**:
- `03_deep_rl_comparison` -- MaskablePPO (368) vs PPO-no-mask (11) vs DQN (68), bar chart with error bars

### Population Game and Mechanism Design (course extension)
**Scripts**: `04_run_equilibrium.py`, `05_evaluate_interventions.py`
**Phases**: 5 (equilibrium) and 6 (interventions)

**What it demonstrates** (going beyond the core syllabus):
- Iterated best-response dynamics to approximate the population game equilibrium.
- Price of Anarchy: the welfare ratio between selfish equilibrium and a coordinator-optimized outcome. Result: PoA ~1.004, meaning the Uffizi's problem is **capacity-bound**, not a coordination failure.
- 30 curated interventions (timed entry, Botticelli gating, hidden-gem trails, last-hour locals, etc.) evaluated individually and combined.
- Greedy + local-search portfolio optimization finds the optimal subset.

**Outputs**:
- `outputs/04_equilibrium.json` (iterated best-response, social optimum proxy, PoA)
- `outputs/05_interventions.json` (33-row intervention table, optimal portfolio)

**Figures**:
- `04_equilibrium_convergence` -- Welfare and Botticelli overcrowding across iterations (twin axes)
- `05_interventions` -- Welfare delta for all interventions (bar chart)
- `05_top_interventions` -- Top 10 interventions ranked by welfare gain
- `05_welfare_revenue` -- Scatter of welfare vs revenue change (Pareto view)
- `05_portfolio_breakdown` -- Baseline vs optimal portfolio (welfare and revenue side by side)

### Phase Transitions (course extension)
**Script**: `06_run_sweeps.py`
**Phase**: 7 (population sweeps)

**What it demonstrates**:
- Sweeps over (a) Type B fraction (guidebook tourists vs art lovers), (b) daily visitor volume, (c) preference heterogeneity.
- These are comparative statics: how does welfare change as we vary one parameter, holding the others fixed?

**Outputs**:
- `outputs/06_sweeps.json` (three sweep series with means and 95% CIs)

**Figures**:
- `06_typeb_transition` -- Welfare across Type B fraction (0 -> 1)
- `06_typeb_gini` -- Spatial inequality (Gini) across Type B fraction
- `06_volume_welfare` -- Welfare across daily visitor volume (1,200 -> 12,000)
- `06_volume_botticelli` -- Botticelli congestion across volume
- `06_heterogeneity` -- Welfare across preference heterogeneity

### Figures driver
**Script**: `07_make_figures.py`

Reads the JSON/NumPy outputs from scripts 01-06 and produces all 21 figures. Run last (after the data is in place), or whenever you change a plot style.

## Repository layout

```
uffizi_submission/
  README.md                              <- you are here
  pyproject.toml                         <- 8 pinned dependencies
  uv.lock                                <- locked versions for reproducibility
  run_project.py                         <- one-shot driver: runs all 7 scripts in order
  .gitignore

  uffizi_rl/                             <- the package (imported by pipeline scripts)
    config.py                            <- 98 rooms, 119 edges, capacities from floor plan
    pipeline/                            <- 7 numbered run scripts (in course order)
      01_check_environment.py            <- Lectures 1-3: MDP, environment
      02_train_q_learning.py             <- Lectures 4-5, 9-10: value-based + exploration
      03_train_deep_rl.py                <- Lectures 6-8: policy-based + deep RL
      04_run_equilibrium.py              <- Population game equilibrium
      05_evaluate_interventions.py       <- 30 interventions + portfolio search
      06_run_sweeps.py                   <- Comparative statics sweeps
      07_make_figures.py                 <- Generates all 21 figures
      _paths.py                          <- Shared path helpers
      README.md
    environment/                         <- The simulator
      museum_graph.py                    <- NetworkX graph + validation
      crowd_simulator.py                 <- Minute-step simulator (1,100 lines)
      visitor_profiles.py                <- Type A / Type B visitor model
      uffizi_env.py                      <- Gymnasium env with action masking
    agents/                              <- RL algorithms
      q_learning.py                      <- Tabular Q-learning + ToyTabularEnv
      baselines.py                       <- 5 handcrafted baseline policies
      train_maskable_ppo.py              <- MaskablePPO with curriculum
      train_ablations.py                 <- PPO and DQN without masking (controls)
    interventions/                       <- Mechanism design building blocks
      intervention_config.py             <- 62 intervention parameters (1 dataclass)
      timed_entry.py                     <- Slot assignment helper
      congestion_pricing.py              <- Reservation book helper
      hidden_gem_trails.py               <- Trail preference reshaping helper
      dynamic_info.py                    <- Kiosk visibility helper
    analysis/                            <- Metrics, sweeps, optimization, plotting
      metrics.py                         <- Gini, Theil, Price of Anarchy, welfare
      phase_transition.py                <- Sweep runners + best-response equilibrium
      portfolio.py                       <- Greedy + 1-opt local search
      pareto.py                          <- Pareto frontier algorithm
      visualization.py                   <- 21 plot functions

  tests/                                 <- 112 tests, all passing
    test_config.py
    test_graph.py
    test_simulation.py
    test_analysis.py

  outputs/                               <- All committed results (regenerable)
    01_environment.json
    01_density_matrix.npy
    02_q_learning.json
    03_deep_rl.json
    04_equilibrium.json
    05_interventions.json
    06_sweeps.json
    figures/                             <- 21 figures, each as PNG + PDF
      01_arrival_envelope.png
      01_capacity_distribution.png
      01_density_heatmap.png
      01_graph_density.png
      01_top_congested.png
      01_visitors_over_day.png
      02_baseline_comparison.png
      02_episode_lengths.png
      02_epsilon_decay.png
      02_offpeak_botticelli.png
      02_q_learning_curve.png
      03_deep_rl_comparison.png
      04_equilibrium_convergence.png
      05_interventions.png
      05_portfolio_breakdown.png
      05_top_interventions.png
      05_welfare_revenue.png
      06_heterogeneity.png
      06_typeb_gini.png
      06_typeb_transition.png
      06_volume_botticelli.png
      06_volume_welfare.png
```

## Where are the 30+ interventions?

There are only four standalone files in `interventions/`. The 62 intervention parameters live in `interventions/intervention_config.py` as a single frozen dataclass. The actual implementation of every intervention lives inside `environment/crowd_simulator.py`: the simulator reads an InterventionConfig at construction time and modifies visitor behavior accordingly (room attributes, arrival rates, movement, gating). This keeps the intervention surface small (one dataclass) while the mechanism logic stays co-located with the simulator code that it modifies.

## Key results (from the committed --medium run)

| Metric | Value |
|---|---|
| Q-learning return (toy graph, 25k episodes) | 181 |
| MaskablePPO return (full museum, 3 seeds) | 367.76 +/- 5.1 |
| PPO without action masking | 11 |
| DQN without action masking | 68 |
| Price of Anarchy | ~1.004 |
| Optimal portfolio: welfare gain | +27% |
| Optimal portfolio: revenue gain | +25% |
| Top single interventions | Last-Hour Locals, Timed Entry, Resident Annual Pass, Lunch Free Entry |

Headline finding: the optimal strategy is NOT to restrict access to crowded rooms (which kills revenue) but to **give visitors reasons to visit empty rooms** (last-hour locals, lunch-free entry, hidden-gem trails). Temporal redistribution beats access restriction.

## Testing

```bash
uv run pytest tests/ -q     # 112 tests, all passing
```

## References

- Attanasio et al. (2022). Visitors flow management at Uffizi Gallery. *Information Technology & Tourism*, 24(3), 409-434.
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
