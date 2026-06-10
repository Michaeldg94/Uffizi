# Reading Order

Show the files to the professor in this exact order. Every source file is
prefixed with its reading-order number (e.g. `s02_config.py`,
`s03_museum_graph.py`).

| #  | File                                                | What it is                                                  |
|----|-----------------------------------------------------|-------------------------------------------------------------|
| 1  | `README.md`                                         | Project overview, results, run instructions                 |
| 2  | `uffizi_rl/s02_config.py`                           | The MDP: rooms, edges, capacities, all modeling parameters  |
| 3  | `uffizi_rl/environment/s03_museum_graph.py`         | Builds the NetworkX graph from `s02_config.py`              |
| 4  | `uffizi_rl/environment/s04_visitor_profiles.py`     | Type A (art lovers) vs Type B (guidebook tourists)          |
| 5  | `uffizi_rl/environment/s05_crowd_simulator.py`      | The minute-step simulator                                   |
| 6  | `uffizi_rl/environment/s06_uffizi_env.py`           | Gymnasium environment wrapper                               |
| 7  | `uffizi_rl/agents/s07_q_learning.py`                | Tabular Q-learning agent + the 12-room toy environment      |
| 8  | `uffizi_rl/agents/s08_baselines.py`                 | 5 handcrafted baseline policies                             |
| 9  | `uffizi_rl/pipeline/01_check_environment.py`        | Runs Phase 0 (capacity) + Phase 1 (environment)             |
| 10 | `uffizi_rl/pipeline/02_train_q_learning.py`         | Runs Phase 2 (Q-learning + baselines)                       |
| 11 | `uffizi_rl/pipeline/07_make_figures.py`             | Generates the 11 figures                                    |
| 12 | `outputs/figures/`                                  | The 11 PNG/PDF figures (the visual deliverables)            |

## Why this order

Files 1-2 set the stage (what is this project, what is the MDP).
Files 3-6 build the environment (graph, visitors, simulator, RL wrapper).
Files 7-8 implement the agent (Q-learning + baselines to compare against).
Files 9-11 are the runners (scripts you actually execute).
File 12 is the visual output that shows the agent works.

If the professor only has 5 minutes, show them: README, then
`outputs/figures/02_baseline_comparison.png`, then
`uffizi_rl/agents/s07_q_learning.py`.

## Note on the `s` prefix

Python module names cannot start with a digit, so the imported files use an
`s` prefix (for "step") plus the two-digit reading-order number. The
pipeline scripts in `uffizi_rl/pipeline/` are NOT imported (they are
executed as scripts), so they can use the bare `01_`, `02_`, `07_` prefix.

Support modules that are not on the suggested reading path
(`uffizi_rl/analysis/`, `uffizi_rl/interventions/`) keep plain names because
they are internal helpers, not part of the narrative the professor follows.
