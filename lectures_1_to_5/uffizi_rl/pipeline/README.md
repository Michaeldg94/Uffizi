# Pipeline scripts

Three scripts run in order to demonstrate the project end-to-end.

| Script | Phase | What it does | Outputs |
|---|---|---|---|
| 01_check_environment.py | 0 + 1 | Capacity sanity check, graph validation, 5-day simulator calibration, random rollout | outputs/01_environment.json, outputs/01_density_matrix.npy |
| 02_train_q_learning.py | 2 | Tabular Q-learning on the 12-room toy graph + 5 handcrafted baselines | outputs/02_q_learning.json |
| 07_make_figures.py | viz | Reads outputs/01-02, writes 11 PNG/PDF figures | outputs/figures/01_*.png + 02_*.png |

## Usage

From the project root, run the scripts in order:

```bash
uv run python uffizi_rl/pipeline/01_check_environment.py
uv run python uffizi_rl/pipeline/02_train_q_learning.py --episodes 25000
uv run python uffizi_rl/pipeline/07_make_figures.py
```

Or run them all at once:

```bash
uv run python run_project.py                   # smoke
uv run python run_project.py --episodes 25000  # converged Q-learning
```
