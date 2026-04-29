# Pipeline scripts

Numbered scripts that drive the project end-to-end. Run them in order.

| Script | Phase | What it does | Inputs | Outputs |
|---|---|---|---|---|
| 01_check_environment.py | 0 + 1 | Capacity sanity check; graph validation; 5-day simulator calibration; random rollout | none | outputs/01_environment.json, outputs/01_density_matrix.npy |
| 02_train_q_learning.py | 2 | Tabular Q-learning on 12-room toy graph + five baselines | none | outputs/02_q_learning.json |
| 03_train_deep_rl.py | 3 | MaskablePPO on full 98-room graph + DQN/PPO ablations without masking | none | outputs/03_deep_rl.json |
| 04_run_equilibrium.py | 5 | Iterated best-response, Price of Anarchy | none | outputs/04_equilibrium.json |
| 05_evaluate_interventions.py | 6 | Score 30 curated interventions, find best portfolio | none | outputs/05_interventions.json |
| 06_run_sweeps.py | 7 | Type-B fraction, volume, heterogeneity sweeps | none | outputs/06_sweeps.json |
| 07_make_figures.py | viz | Read JSONs from 01-06, write PNG/PDF figures | outputs/01-06 | outputs/figures/*.png, *.pdf |

## Usage

From the project root (`uffizi_submission/`), run scripts in order:

```bash
uv run python uffizi_rl/pipeline/01_check_environment.py
uv run python uffizi_rl/pipeline/02_train_q_learning.py --episodes 25000
uv run python uffizi_rl/pipeline/03_train_deep_rl.py --timesteps 500000 --seeds 3
uv run python uffizi_rl/pipeline/04_run_equilibrium.py
uv run python uffizi_rl/pipeline/05_evaluate_interventions.py
uv run python uffizi_rl/pipeline/06_run_sweeps.py --resolution 7 --seeds 3
uv run python uffizi_rl/pipeline/07_make_figures.py
```

Or run them all at once:

```bash
uv run python run_project.py            # smoke run, minutes
uv run python run_project.py --medium   # converged, ~2 hours
```

## Design notes

- Each script writes to `outputs/<NN>_<name>.json`. Numbering keeps artifacts in execution order.
- Path resolution lives in `_paths.py`: every script writes to `<project_root>/outputs/` regardless of cwd.
- Scripts can be invoked individually (after running 01); only script 07 depends on artifacts from earlier scripts.
- The `sys.path.insert` block at the top of each script lets you run them directly without setting `PYTHONPATH`.
