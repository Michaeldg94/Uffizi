# Pipeline scripts (A to Z)

Numbered scripts that drive the whole project end-to-end. Run in order. Stages
01-06 are the foundations (environment, tabular RL, equilibrium, interventions,
sweeps); 07-10 are the RAMA booking centerpiece and the algorithm comparison; 11
renders every figure.

| Script | What it does | Outputs |
|---|---|---|
| 01_check_environment.py | Capacity/graph sanity, simulator calibration, random rollout | `outputs/01_environment.json`, `01_density_matrix.npy` |
| 02_train_q_learning.py | Tabular Q-learning (+ Double-Q) on the toy graph vs 5 baselines | `outputs/02_q_learning.json` |
| 03_train_deep_rl.py | MaskablePPO vs unmasked PPO/DQN (the masking ablation) | `outputs/03_deep_rl.json` |
| 04_run_equilibrium.py | Iterated best-response, Price of Anarchy | `outputs/04_equilibrium.json` |
| 05_evaluate_interventions.py | Score curated interventions, find best portfolio | `outputs/05_interventions.json` |
| 06_run_sweeps.py | Type-B fraction / volume / heterogeneity sweeps | `outputs/06_sweeps.json` |
| 07_train_baselines.py | Matched no-intervention baselines (art + tourist × 3 crowds) | `models/newenv/opt_{Art,Tourist}Walk_{c}.zip` |
| 08_train_booking.py | RAMA booking grid (art + tourist MaskablePPO × 3 crowds) | `models/newenv/ramabook_{profile}_{c}.zip` |
| 09_algorithm_comparison.py | PPO / MaskablePPO / DQN × baseline/intervened matrix (18 cells) | `models/newenv/{ppo_book,dqn_book,dqn_base}_*.zip` |
| 10_evaluate_booking.py | Deterministic intervened-vs-baseline grid + path sanity | prints the deliverable grid |
| 11_make_figures.py | ALL figures, numbered 01..N, DPI 300 (PNG+PDF) | `outputs/figures/01..N` |

## Usage

From the project root (`uffizi_submission/`), in order:

```bash
python uffizi_rl/pipeline/01_check_environment.py
python uffizi_rl/pipeline/02_train_q_learning.py --episodes 25000
python uffizi_rl/pipeline/03_train_deep_rl.py --timesteps 500000 --seeds 3
python uffizi_rl/pipeline/04_run_equilibrium.py
python uffizi_rl/pipeline/05_evaluate_interventions.py
python uffizi_rl/pipeline/06_run_sweeps.py --resolution 7 --seeds 3
python uffizi_rl/pipeline/07_train_baselines.py        # matched baselines
python uffizi_rl/pipeline/08_train_booking.py          # RAMA booking grid (RAMA_TS to set budget)
python uffizi_rl/pipeline/09_algorithm_comparison.py   # heavy; resumable, run in sittings
python uffizi_rl/pipeline/10_evaluate_booking.py        # the deliverable numbers
python uffizi_rl/pipeline/11_make_figures.py            # all figures, 01..N
```

## Design notes

- **Resumable.** Stages 07-09 reuse any saved model in `outputs/models/newenv/`; delete a `.zip` to force a retrain. They do not retrain what already exists.
- **CPU-capped.** Stages 07-11 cap BLAS/torch threads to 2 (set before numpy/torch import) and should be run with `nice`; this is a laptop, not a server. Never run several uncapped trainers at once.
- **Deterministic eval.** Booking policies are evaluated with `deterministic=True` on the fixed common-random-number crowds (seeds 900000-5); baselines are trained and evaluated at `random_start=False` (fixed entrance) to match the intervened agent.
- **Figures.** All figure generation is consolidated in `uffizi_rl/figures/make_all_figures.py` (driven by stage 11). It wipes `outputs/figures/` and re-emits the full numbered set; the old per-figure generators are retired.
- `_paths.py` resolves `outputs/` relative to the project root regardless of cwd; the `sys.path` block at the top of each script lets you run them directly.
