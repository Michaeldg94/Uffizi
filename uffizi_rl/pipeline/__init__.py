"""Numbered pipeline scripts for the course project.

Run them in order from the project root:

    uv run python uffizi_rl/pipeline/01_check_environment.py
    uv run python uffizi_rl/pipeline/02_train_q_learning.py
    ...
    uv run python uffizi_rl/pipeline/07_make_figures.py

Or run them all at once with the top-level driver:

    uv run python run_project.py            # quick smoke run
    uv run python run_project.py --medium   # converged run

Outputs are written to `<project_root>/outputs/`. The `_paths` module
exposes the path helpers used by every numbered script so they all
write to the same location regardless of the caller's cwd.
"""
