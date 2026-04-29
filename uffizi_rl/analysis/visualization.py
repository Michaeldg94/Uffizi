"""Visualization utilities for Uffizi RL experiments.

Provides publication-ready plot functions for five core visualization types:

1. **Graph density snapshot**: the museum's room-adjacency graph colored
   by current occupancy ratio. Shows spatial congestion patterns at a
   glance (which rooms are red, which are empty).

2. **Learning curve**: RL training progress with a rolling mean and
   approximate 95% confidence band. Shows whether the agent is converging
   and how noisy the training signal is.

3. **Density heatmap**: a time-by-room matrix showing how crowd pressure
   evolves over the museum day. Reveals temporal patterns (the 10:00-12:00
   Botticelli crunch) and spatial patterns (corridor congestion).

4. **Intervention comparison**: bar chart of welfare gains relative to
   the status quo baseline. Immediately shows which interventions help,
   which hurt, and by how much.

5. **Phase transition curve**: welfare as a function of a swept parameter
   (Type B ratio, volume, heterogeneity) with confidence bands. Reveals
   tipping points where the system's behavior changes qualitatively.

All plot functions follow a common pattern:
  - Return False immediately if matplotlib is not installed (graceful
    degradation in environments without display capabilities).
  - Apply a consistent publication theme (font sizes, DPI).
  - Save both PNG (for slides/web) and PDF (for LaTeX inclusion).
  - Return True on success.

Color palette:
  - "#1F4E5F" (dark teal): primary data series, positive gains.
  - "#B14D4D" (muted red): negative gains, warnings.
  - "#8B6F47" (warm brown): raw/noisy data (e.g., episode rewards).
  - "YlOrRd" colormap: occupancy density (yellow=empty, red=crowded).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np

# Plotting libraries are optional; the module degrades gracefully without them.
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import seaborn as sns

    HAS_PLOT = True
except Exception:  # pragma: no cover
    plt = nx = sns = None
    HAS_PLOT = False


# =============================================================================
# Internal helpers
# =============================================================================


def _ensure_dir(path: Path) -> None:
    """Create the parent directory of ``path`` if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _apply_theme() -> None:
    """Set a consistent publication-quality matplotlib theme.

    Font sizes are chosen for readability at the typical figure widths
    used in two-column academic papers (~3.5 inches) and slide decks.
    """
    if not HAS_PLOT:
        return
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def _save_figure(out: Path) -> None:
    """Save the current figure as both PNG and PDF, then close it.

    PNG is saved at 220 DPI for high-resolution screen display.
    PDF is vector-based for lossless LaTeX inclusion.
    ``bbox_inches="tight"`` trims whitespace around the figure.
    """
    _ensure_dir(out)
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


# =============================================================================
# Plot 1: Museum graph density snapshot
# =============================================================================


def plot_graph_density(g, density_by_room: Dict[str, float], out_path: str) -> bool:
    """Plot the museum's room-adjacency graph colored by occupancy density.

    Each node represents a room; edges represent physical connections
    (doors, corridors, staircases). Node color encodes the occupancy
    ratio (room occupancy / room capacity) on a Yellow-Orange-Red scale:
    yellow = empty, orange = moderate, red = overcrowded.

    This visualization shows spatial congestion patterns at a single
    point in time. It answers: "Where are all the visitors right now,
    and which parts of the museum are empty?"

    Parameters
    ----------
    g : networkx.Graph
        The museum adjacency graph (nodes = rooms, edges = connections).
    density_by_room : dict mapping room ID -> occupancy ratio.
        Values in [0, 1+]. Rooms not in the dict default to 0.0.
    out_path : str
        Output file path (without extension; both .png and .pdf are saved).

    Returns
    -------
    bool
        True if the plot was saved; False if matplotlib is unavailable.
    """

    if not HAS_PLOT:
        return False

    out = Path(out_path)
    _apply_theme()

    plt.figure(figsize=(14, 8))
    # spring_layout with fixed seed for reproducible node positions.
    pos = nx.spring_layout(g, seed=1)
    vals = np.array([density_by_room.get(n, 0.0) for n in g.nodes], dtype=float)
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        node_size=240,
        node_color=vals,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=max(1.0, vals.max()),  # scale so that max is at least 1.0
    )
    nx.draw_networkx_edges(g, pos, alpha=0.3)  # faint edges to not overpower nodes
    nx.draw_networkx_labels(g, pos, font_size=7)
    cbar = plt.colorbar(nodes)
    cbar.set_label("Occupancy Ratio (room occupancy / room capacity)")
    plt.title("Uffizi Graph Density Snapshot")
    plt.axis("off")  # no axes for graph plots
    _save_figure(out)
    return True


# =============================================================================
# Plot 2: RL learning curve
# =============================================================================


def plot_learning_curve(rewards: Sequence[float], out_path: str, title: str = "Learning Curve") -> bool:
    """Plot episode rewards with a 25-episode rolling mean and variability band.

    The raw episode rewards are plotted in faint brown to show per-episode
    variance. The rolling mean (dark teal) shows the training trend. The
    shaded band is an approximate 95% confidence interval for the mean,
    computed as 1.96 * rolling_std / sqrt(window_size). This uses a
    normal approximation, which is reasonable for window_size=25.

    This visualization answers: "Is the agent learning? Has it converged?
    How noisy is the training signal?"

    Parameters
    ----------
    rewards : sequence of floats
        Episode returns from RL training, in chronological order.
    out_path : str
        Output file path.
    title : str
        Plot title (default "Learning Curve").

    Returns
    -------
    bool
        True if saved; False if matplotlib unavailable.
    """

    if not HAS_PLOT:
        return False

    out = Path(out_path)
    _apply_theme()

    y = np.asarray(rewards, dtype=float)
    x = np.arange(len(y))

    plt.figure(figsize=(10, 4))
    # Raw episode rewards: faint to avoid visual clutter.
    plt.plot(x, y, alpha=0.25, color="#8B6F47", label="Episode reward")
    if len(y) >= 25:
        w = 25  # rolling window size
        # Convolution-based rolling mean (valid mode = no padding artifacts).
        ma = np.convolve(y, np.ones(w) / w, mode="valid")
        # x-axis aligned to the center of each window.
        centered_x = np.arange(w - 1, len(y))
        # Rolling standard deviation for the confidence band.
        rolling_std = np.array([y[max(0, i - w + 1): i + 1].std(ddof=0) for i in centered_x])
        # Approximate 95% CI for the rolling mean.
        band = 1.96 * rolling_std / np.sqrt(w)
        plt.plot(centered_x, ma, linewidth=2.0, color="#1F4E5F", label="25-episode mean")
        plt.fill_between(centered_x, ma - band, ma + band, color="#1F4E5F", alpha=0.15, label="Approx. 95% CI")
    plt.xlabel("Episode Index")
    plt.ylabel("Episode Return")
    plt.title(title)
    plt.legend(loc="best", frameon=False)
    _save_figure(out)
    return True


# =============================================================================
# Plot 3: Time-by-room density heatmap
# =============================================================================


def plot_density_heatmap(density_matrix: np.ndarray, out_path: str, room_labels: Sequence[str]) -> bool:
    """Plot the time-by-room density heatmap.

    The x-axis is time (minute of museum day, 0 = 08:15). The y-axis
    lists rooms. Color encodes occupancy ratio (Yellow-Orange-Red).

    This visualization reveals both temporal and spatial congestion
    patterns: the horizontal bands show when each room is busy, and
    vertical bands show which time windows are crowded museum-wide.
    The Botticelli rooms (A11, A12) typically show a bright red band
    from ~10:00 to ~14:00.

    Parameters
    ----------
    density_matrix : np.ndarray, shape (T, N_ROOMS)
        Occupancy ratios. Rows = timesteps, columns = rooms.
    out_path : str
        Output file path.
    room_labels : sequence of str
        Room IDs for the y-axis labels, matching column order.

    Returns
    -------
    bool
        True if saved; False if matplotlib unavailable.
    """

    if not HAS_PLOT:
        return False

    out = Path(out_path)
    _apply_theme()

    plt.figure(figsize=(14, 8))
    # Transpose: seaborn expects rows=rooms, columns=time for this layout.
    sns.heatmap(
        density_matrix.T,
        cmap="YlOrRd",
        cbar_kws={"label": "Occupancy Ratio (room occupancy / room capacity)"},
    )
    plt.xlabel("Minute of Day")
    plt.ylabel("Room ID")
    # Show at most 12 y-tick labels to avoid overlapping text.
    ticks = np.linspace(0, len(room_labels) - 1, min(12, len(room_labels)), dtype=int)
    plt.yticks(ticks + 0.5, [room_labels[t] for t in ticks], rotation=0)  # +0.5 centers labels
    plt.title("Crowd Density Heatmap")
    _save_figure(out)
    return True


# =============================================================================
# Plot 4: Intervention comparison bar chart
# =============================================================================


def plot_intervention_comparison(table: Sequence[Dict[str, float]], out_path: str) -> bool:
    """Plot intervention welfare gains relative to the status quo baseline.

    Each bar shows the welfare difference (intervention minus baseline).
    Positive bars (teal) indicate welfare-improving interventions;
    negative bars (red) indicate welfare-harming interventions.
    A horizontal line at y=0 marks the baseline.

    This visualization directly answers: "Which interventions help,
    which hurt, and by how much?"

    Parameters
    ----------
    table : sequence of dicts
        Each dict must contain "total_welfare" and optionally
        "intervention" (label). The first row is treated as the baseline.
    out_path : str
        Output file path.

    Returns
    -------
    bool
        True if saved; False if matplotlib unavailable or table is empty.
    """

    if not HAS_PLOT or not table:
        return False

    out = Path(out_path)
    _apply_theme()

    # First row is the baseline (status quo).
    baseline = float(table[0].get("total_welfare", 0.0))
    # Exclude the "Social Optimum" row (theoretical upper bound, not an intervention).
    filtered = [row for row in table if "Social Optimum" not in str(row.get("intervention", ""))]
    labels = [str(r.get("intervention", f"i{i}")) for i, r in enumerate(filtered)]
    welfare_gain = [float(r.get("total_welfare", 0.0)) - baseline for r in filtered]

    plt.figure(figsize=(12, 5))
    # Color-code: teal for positive, red for negative.
    colors = ["#1F4E5F" if gain >= 0 else "#B14D4D" for gain in welfare_gain]
    plt.bar(labels, welfare_gain, color=colors)
    plt.axhline(0.0, color="#444444", linewidth=0.8)  # baseline reference line
    plt.ylabel("Welfare Gain vs. Baseline")
    plt.title("Intervention Comparison")
    plt.xticks(rotation=35, ha="right")  # rotated labels for readability
    _save_figure(out)
    return True


# =============================================================================
# Plot 5: Phase transition curve
# =============================================================================


def plot_phase_transition(records: Sequence[Dict[str, float]], x_key: str, out_path: str) -> bool:
    """Plot a phase transition curve with confidence bands.

    The x-axis is the swept parameter (Type B ratio, daily total, or
    heterogeneity scale). The y-axis is mean total welfare. The shaded
    band shows the 95% confidence interval across seeds.

    This visualization reveals tipping points: a sharp drop in the
    curve indicates that a small increase in the swept parameter
    causes a disproportionate welfare loss. For Type B ratio sweeps,
    the typical tipping point is around 0.4-0.6, where Botticelli
    congestion transitions from manageable to catastrophic.

    Parameters
    ----------
    records : sequence of dicts
        Sweep output from one of the run_*_sweep functions. Each dict
        must contain ``x_key``, ``total_welfare_mean``, and
        ``total_welfare_ci``.
    x_key : str
        The sweep parameter key (e.g., "type_b_ratio", "daily_total").
    out_path : str
        Output file path.

    Returns
    -------
    bool
        True if saved; False if matplotlib unavailable or records is empty.
    """

    if not HAS_PLOT or not records:
        return False

    out = Path(out_path)
    _apply_theme()

    x = np.array([float(r[x_key]) for r in records], dtype=float)
    y = np.array([float(r.get("total_welfare_mean", 0.0)) for r in records], dtype=float)
    ci = np.array([float(r.get("total_welfare_ci", 0.0)) for r in records], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o", linewidth=2, color="#1F4E5F", label="Mean welfare")
    plt.fill_between(x, y - ci, y + ci, alpha=0.2, color="#1F4E5F", label="95% CI")
    plt.xlabel(x_key.replace("_", " ").title())
    plt.ylabel("Total Welfare")
    plt.title("Phase Transition Sweep")
    plt.legend(loc="best", frameon=False)
    _save_figure(out)
    return True
