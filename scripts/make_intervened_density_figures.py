"""Density heatmap + peak-congestion figures for the ALL-11 intervened crowd.

Mirrors the museum-wide pipeline exactly: builds CrowdSimulator at the as-is crowd
(daily_total = DAILY_VISITORS_NORMAL = 5000, seed = DEFAULT_SEED + 100), once with no
interventions (status quo) and once with the full curated bundle
(combined_intervention_kwargs()), runs a day each, and exports the (T, N_ROOMS)
occupancy-ratio matrix (the same object behind figures 05 and 06).

Produces:
  outputs/figures/36_density_heatmap_intervened.{png,pdf}   status quo vs all 11, shared scale
  outputs/figures/37_top_congested_intervened.{png,pdf}     PEAK occupancy ratio, status-quo worst 15

Figure 37 uses the PEAK occupancy ratio over the day (max occupancy / capacity), not the
daily mean: a masterpiece is jammed only at midday and near-empty otherwise, so the mean
hides the spike. By peak, only the terrace and the three masterpieces exceed capacity in
the status quo; the interventions pull the masterpieces down to the RAMA cap (1.0), while
the un-gated selfie terrace actually gets busier.

CPU-capped (2 threads). Run:  nice -n 10 uv run python scripts/make_intervened_density_figures.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uffizi_rl import config
from uffizi_rl.environment.crowd_simulator import CrowdSimulator
from uffizi_rl.analysis.portfolio import combined_intervention_kwargs

SEED = config.DEFAULT_SEED + 100          # same crowd seed as pipeline 01 (the baseline matrix)
DT = config.DAILY_VISITORS_NORMAL          # 5000, the as-is crowd
ROOMS = config.ROOM_IDS
FIG = ROOT / "outputs" / "figures"
INK = "#9AA7BD"
ACCENT = "#C0552B"


def run_day_matrix(intervened: bool) -> np.ndarray:
    """One simulated day; returns the (T, N_ROOMS) occupancy-ratio matrix."""
    kwargs = combined_intervention_kwargs() if intervened else {}
    sim = CrowdSimulator(daily_total=DT, seed=SEED, **kwargs)
    sim.run_day()
    return sim.export_density_matrix()


def fig36_heatmap(m_base: np.ndarray, m_int: np.ndarray) -> None:
    """Density heatmap, status quo vs all 11, shared colour scale."""
    vmax = float(max(m_base.max(), m_int.max()))
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    panels = [
        (axes[0], m_base, f"Status quo (no interventions),  day length {m_base.shape[0]} min"),
        (axes[1], m_int, f"All 11 interventions,  day length {m_int.shape[0]} min"),
    ]
    im = None
    for ax, m, title in panels:
        im = ax.imshow(m.T, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax,
                       interpolation="nearest")
        ax.set_ylabel("Room ID")
        ax.set_xlabel("Minute of day (0 = 08:15)")
        ticks = np.linspace(0, len(ROOMS) - 1, 12, dtype=int)
        ax.set_yticks(ticks)
        ax.set_yticklabels([ROOMS[t] for t in ticks])
        ax.set_title(title, fontsize=11)
    fig.suptitle("Crowd density heatmap: status quo vs all 11 interventions "
                 f"(~{DT} visitors). Shared colour scale.", y=0.99, fontsize=13)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="Occupancy ratio (occupancy / capacity)")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"36_density_heatmap_intervened.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig37_peak(m_base: np.ndarray, m_int: np.ndarray) -> None:
    """Top-congested rooms by PEAK occupancy ratio, status quo vs all 11."""
    base_peak = m_base.max(axis=0)
    int_peak = m_int.max(axis=0)
    order = np.argsort(base_peak)[::-1][:15]          # the status-quo worst 15, by peak
    rooms_top = [ROOMS[i] for i in order]
    b = base_peak[order]
    iv = int_peak[order]
    y = np.arange(len(order))
    h = 0.4

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.barh(y - h / 2, b, h, label="status quo", color=INK, edgecolor="white")
    ax.barh(y + h / 2, iv, h, label="all 11 interventions", color=ACCENT, edgecolor="white")
    ax.axvline(1.0, color="gray", ls="--", lw=1.2)
    ax.text(1.02, len(order) - 0.4, "capacity", color="gray", fontsize=9, va="bottom")

    # label the over-capacity rooms (terrace + the three masterpieces) with their values
    for k in range(len(order)):
        if b[k] >= 1.0 or iv[k] >= 1.0:
            ax.text(b[k] + 0.03, y[k] - h / 2, f"{b[k]:.2f}", va="center", fontsize=8, color=INK)
            ax.text(iv[k] + 0.03, y[k] + h / 2, f"{iv[k]:.2f}", va="center", fontsize=8, color=ACCENT)

    ax.set_yticks(y)
    ax.set_yticklabels(rooms_top)
    ax.invert_yaxis()
    ax.set_xlim(0, float(max(b.max(), iv.max())) + 0.45)
    ax.set_xlabel("peak occupancy ratio over the day (peak occupancy / capacity)")
    ax.set_title("Peak occupancy by room: status quo vs all 11 interventions "
                 f"(~{DT} visitors)\n"
                 "The three masterpieces fall to capacity (1.0); only the un-gated selfie "
                 "terrace gets busier.", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"37_top_congested_intervened.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"[1/3] status-quo day (daily_total={DT}, seed={SEED}) ...")
    m_base = run_day_matrix(intervened=False)
    print(f"      matrix {m_base.shape}")

    int_path = ROOT / "outputs" / "01_density_matrix_intervened.npy"
    if int_path.exists():
        m_int = np.load(int_path)
        print(f"[2/3] loaded cached intervened matrix {m_int.shape}")
    else:
        print("[2/3] all-11 intervened day ...")
        m_int = run_day_matrix(intervened=True)
        np.save(int_path, m_int)
    print(f"      bundle keys: {sorted(combined_intervention_kwargs())}")

    print("      peak occupancy ratio (status quo -> intervened):")
    for r in ("PANORAMIC_TERRACE", "A11", "A35", "A38"):
        j = config.ROOM_TO_IDX[r]
        print(f"        {r:>18}: {m_base[:, j].max():.2f} -> {m_int[:, j].max():.2f}")

    # Figure 36 is already approved; regenerate only if it is missing.
    if not (FIG / "36_density_heatmap_intervened.png").exists():
        fig36_heatmap(m_base, m_int)
        print("[3/3] saved 36_density_heatmap_intervened (was missing)")
    else:
        print("[3/3] 36 left as-is (already approved)")

    fig37_peak(m_base, m_int)
    print("      saved 37_top_congested_intervened (PEAK occupancy)")


if __name__ == "__main__":
    main()
