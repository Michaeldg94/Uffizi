"""PIPELINE 11: generate ALL figures (the single unified generator).

Thin entry point for uffizi_rl/figures/make_all_figures.py, which wipes
outputs/figures/ and regenerates every figure numbered 01..N at DPI 300 (PNG+PDF):
world / crowd / learning / interventions / RAMA booking / algorithm comparison /
floor-plan maps / sweeps. Run LAST, after the data + training stages (01-10).

Usage:
  python uffizi_rl/pipeline/11_make_figures.py            # full clean rebuild (01..N)
  python uffizi_rl/pipeline/11_make_figures.py maps       # only the floor-plan maps
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uffizi_rl.figures import make_all_figures  # noqa: E402  (sets thread caps + cwd on import)

if __name__ == "__main__":
    make_all_figures.main()
