# Uffizi RL: Reinforcement Learning for Museum Visitor-Flow Optimization

Course project for **Reinforcement Learning**, Barcelona School of Economics, Spring 2026.

## The project, in two streams

The Uffizi Gallery receives ~5,000 visitors per day across 98 rooms. Demand concentrates on four masterpieces (Botticelli's *Spring* and *Venus* in A11/A12, Leonardo in A35, Raphael/Michelangelo in A38); the rest of the collection goes undervisited. The visitor population is a mix of **60% Instagram tourists** (famous-room selfie run), **30% normal tourists**, and **10% art lovers**.

We attack the problem in two distinct streams:

1. **Crowd simulator + interventions (museum-wide).** An agent-based simulator runs the whole population for a day. We compare the museum *as is today* (status quo) against the museum *under the 11-lever intervention bundle* (RAMA reservations, secondary-attractor enrichment, extended hours, dynamic pricing, tour-group cap, timed entry, quiet hours, group surcharge, resident annual pass). The gate: a director will only adopt interventions that do not collapse revenue. **Result: welfare rises (+31%) while revenue is preserved; the optimized subset raises both.**

2. **RL on top.** Given that intervened world, what does the *optimal individual visitor* do? We train RL agents (an art lover and a normal tourist) whose central decision is the **RAMA booking**: which masterpiece slots to reserve, and how far ahead. The lesson that emerges from learning: **book early, and earlier when the day is busier.** The intervened agents beat the no-intervention baseline by **+32-41% (art lover)** and **+36-38% (tourist)**, securing all three masterpieces at every crowd level.

The crowd simulator produces the museum-wide numbers; the RL agents are the individual optimum built on top of those crowds. They are separate streams.

## How to run

```bash
uv sync

# Stream 1 foundations + economics (pipeline 01-06)
uv run python run_project.py            # smoke run; --medium for converged

# Stream 1 museum-wide + Stream 2 RL (pipeline 07-12), CPU-capped, resumable
uv run python uffizi_rl/pipeline/12_museum_wide.py        # crowd-sim: base vs intervened
uv run python uffizi_rl/pipeline/07_train_baselines.py    # matched no-intervention baselines
uv run python uffizi_rl/pipeline/08_train_booking.py      # RAMA booking grid
uv run python uffizi_rl/pipeline/09_algorithm_comparison.py
uv run python uffizi_rl/pipeline/10_evaluate_booking.py   # the deliverable eval
uv run python uffizi_rl/pipeline/11_make_figures.py       # all figures, 01..N
```

Stages 07-12 reuse any trained model already in `outputs/models/newenv/` (resumable) and cap themselves to 2 CPU cores. Results land in `outputs/` (`results_rl.json`, `results_museum_wide.json`, `RESULTS.md`) and `outputs/figures/` (numbered PNG + PDF).

## Pipeline (`uffizi_rl/pipeline/`)

| Stage | Stream | What it does |
|---|---|---|
| `01_check_environment` | foundations | MDP / environment construction, density baseline |
| `02_train_q_learning` | foundations | tabular Q-learning + Double-Q vs 5 baselines (toy graph) |
| `03_train_deep_rl` | foundations | MaskablePPO vs unmasked PPO/DQN (the masking ablation) |
| `04_run_equilibrium` | Stream 1 | iterated best-response, Price of Anarchy |
| `05_evaluate_interventions` | Stream 1 | intervention screening + portfolio optimizer |
| `06_run_sweeps` | Stream 1 | population sweeps (Type-B / volume / heterogeneity) |
| `07_train_baselines` | Stream 2 | matched no-intervention walk baselines (art + tourist x 3 crowds) |
| `08_train_booking` | Stream 2 | RAMA booking grid (art + tourist MaskablePPO x 3 crowds) |
| `09_algorithm_comparison` | Stream 2 | PPO / MaskablePPO / DQN x baseline/intervened matrix |
| `10_evaluate_booking` | Stream 2 | deterministic intervened-vs-baseline grid + path sanity |
| `12_museum_wide` | Stream 1 | crowd-sim base vs intervened welfare/revenue, per segment |
| `11_make_figures` | all | the single figure generator -> all numbered figures |

## Repository layout

```
uffizi_submission/
  README.md  ·  pyproject.toml  ·  uv.lock  ·  run_project.py
  uffizi_rl/
    config.py                         98 rooms, capacities, 60/30/10 split, prices
    environment/
      crowd_simulator.py              THE simulator (calibrated; do not modify)
      visitor_profiles.py             art_lover / standard / instagram segments
      museum_graph.py                 graph build + validation
      uffizi_env.py                   base Gymnasium navigation env
      planned_route_env.py            walk baseline (Stream 2)
      rama_art_lover_env.py           RAMA booking, art lover (RL centerpiece)
      rama_tourist_env.py             RAMA booking, normal tourist (subclass)
    agents/
      q_learning.py · baselines.py    tabular foundations
      train_maskable_ppo.py · train_ablations.py   MaskablePPO + masking controls
    interventions/
      intervention_config.py          the intervention flags (1 dataclass)
      timed_entry.py · dynamic_info.py · hidden_gem_trails.py   lever logic used by the sim
    analysis/
      portfolio.py                    11 curated interventions + bundle + optimizer
      phase_transition.py             museum-wide eval (simulate_day_metrics) + sweeps
      metrics.py                      welfare / congestion / inequality / experience
      visualization.py                chart plotters
    figures/
      make_all_figures.py             the one figure generator (01..N, DPI 300)
      floor_plan_coords.py            room -> floor-plan pixel coordinates
    pipeline/                         01-12 numbered stages + _paths.py
  outputs/
    results_rl.json                   RL stream: booking grid + algorithm matrix
    results_museum_wide.json          crowd-sim stream: per-segment welfare/revenue
    RESULTS.md                        human-readable summary of both streams
    12_museum_wide.json · 0X_*.json · *.npy   stage artifacts
    figures/                          numbered figures (PNG + PDF)
    models/newenv/                    trained policies
  tests/
  _legacy_scripts/                    retired generators + archived dead modules
```

## Key results

**Stream 1 — museum-wide (all 11 vs status quo, ~5,000/day, canonical ~94-95k revenue baseline):**

| metric | result |
|---|---|
| Welfare (all 11) | **+31%**, all three segments improve |
| Revenue (all 11) | **preserved** (no collapse) |
| Optimized portfolio | welfare **+~16%**, revenue **+~15%** (both up) |
| Per-segment welfare | Instagram throughput up; normal tourist +49-58%; art lover +9-33% |
| Peak Botticelli density | 1.82 -> 1.00 at max crowd |

**Stream 2 — RL booking agent (intervened vs matched baseline, deterministic):**

| profile | 500 | 2,500 | max |
|---|---|---|---|
| Art lover | +39% (lead 7d) | +32% (35d) | +41% (35d) |
| Normal tourist | +37% (7d) | +36% (7d) | +38% (35d) |

All cells secure 3/3 masterpieces; **book-early emerges** (lead rises 7 -> 35 days as the crowd grows). MaskablePPO is the clear algorithm winner; masking is what lets PPO discover book-early; DQN is crowd-fragile.

## Note on the simulator

`environment/crowd_simulator.py` (and `visitor_profiles.py`, the split constants in `config.py`) is carefully calibrated. Treat it as fixed; everything else (RL, museum-wide eval, figures) is built on top of it.

## Testing

```bash
uv run pytest tests/ -q
```

## References

- Attanasio et al. (2022). Visitors flow management at Uffizi Gallery. *Information Technology & Tourism*, 24(3).
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed.
