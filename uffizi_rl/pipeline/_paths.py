"""Shared path helpers for the numbered pipeline scripts.

Numbered scripts (01_..., 02_..., etc.) live in `uffizi_rl/pipeline/`
but write artifacts to `<project_root>/outputs/`. This module resolves
the project root from the script's __file__ location, so artifacts
land in the same place regardless of the directory the user invoked
the script from.
"""

from __future__ import annotations

from pathlib import Path


# uffizi_rl/pipeline/_paths.py -> uffizi_rl/pipeline -> uffizi_rl -> project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def ensure_outputs_dir() -> Path:
    """Create the outputs/ directory if it does not exist and return it."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def ensure_figures_dir() -> Path:
    """Create the outputs/figures/ directory if it does not exist and return it."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR
