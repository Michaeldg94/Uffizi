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

from uffizi_rl import config

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

    PNG is saved at 300 DPI for high-resolution display.
    PDF is vector-based for lossless LaTeX inclusion.
    ``bbox_inches="tight"`` trims whitespace around the figure.
    """
    _ensure_dir(out)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
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


# =============================================================================
# Extended plot library: per-phase visualizations for the course report.
# Added to give the professor many figures per lecture topic. Each function
# is independent and degrades gracefully if matplotlib is unavailable.
# =============================================================================

def plot_room_capacity_distribution(rooms_data: Sequence[Dict], out_path: str) -> bool:
    """Histogram of room capacities across the 98-room museum."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    caps = [r["capacity"] for r in rooms_data]
    plt.figure(figsize=(10, 5))
    plt.hist(caps, bins=20, color="#1F4E5F", edgecolor="white", alpha=0.85)
    plt.axvline(np.mean(caps), color="#B14D4D", linestyle="--",
                label=f"Mean = {np.mean(caps):.1f}")
    plt.axvline(np.median(caps), color="#8B6F47", linestyle=":",
                label=f"Median = {np.median(caps):.0f}")
    plt.xlabel("Room Capacity (visitors)")
    plt.ylabel("Number of Rooms")
    plt.title("Distribution of Room Capacities (98-Room Museum)")
    plt.legend(loc="upper right", frameon=False)
    _save_figure(out)
    return True


def plot_top_congested_rooms(room_density_pairs: Sequence, out_path: str,
                              top_n: int = 15, label_overrides: dict = None) -> bool:
    """Horizontal bar chart of the most congested rooms (peak density).

    The y-axis shows each room's human-readable name (looked up from
    config.ROOM_DATA) rather than the bare room ID, so the reader can
    immediately see which famous galleries are at the top.
    """

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    top = list(room_density_pairs)[:top_n]
    rooms = [p[0] for p in top]
    dens = [p[1] for p in top]
    # Resolve each room ID to its display name. Fall back to the ID if
    # the room is not in ROOM_DATA (defensive).
    label_overrides = label_overrides or {}
    labels = []
    for rid in rooms:
        if rid in label_overrides:
            labels.append(label_overrides[rid]); continue
        name = config.ROOM_DATA.get(rid, {}).get("name")
        labels.append(f"{name} ({rid})" if name else rid)
    colors = ["#B14D4D" if d > 1.0 else "#D49B6C" if d > 0.5 else "#1F4E5F" for d in dens]
    plt.figure(figsize=(11, 6))
    plt.barh(labels[::-1], dens[::-1], color=colors[::-1], edgecolor="white")
    plt.axvline(1.0, color="black", linestyle="--", alpha=0.6,
                label="Capacity (density = 1.0)")
    plt.xlabel("Peak Density (occupancy / capacity)")
    plt.ylabel("Room")
    plt.title(f"Top {top_n} Most Congested Rooms (peak occupancy)")
    plt.legend(loc="lower right", frameon=False)
    _save_figure(out)
    return True


def plot_arrival_envelope(out_path: str, peak_minutes: int = 135,
                           sigma_minutes: int = 90, last_entry: int = 555,
                           total_minutes: int = 615) -> bool:
    """Plot the Gaussian arrival envelope used by the simulator."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    t = np.arange(0, total_minutes)
    envelope = np.exp(-0.5 * ((t - peak_minutes) / sigma_minutes) ** 2)
    envelope[t > last_entry] = 0.0
    plt.figure(figsize=(11, 4.5))
    plt.plot(t / 60.0 + 8.25, envelope, color="#1F4E5F", linewidth=2)
    plt.fill_between(t / 60.0 + 8.25, 0, envelope, alpha=0.2, color="#1F4E5F")
    plt.axvline(8.25 + last_entry / 60, color="#B14D4D", linestyle="--",
                label="Last entry (17:30)")
    plt.axvline(8.25 + peak_minutes / 60, color="#8B6F47", linestyle=":",
                label="Peak (10:30)")
    plt.xlabel("Time of Day (24h)")
    plt.ylabel("Relative Arrival Rate")
    plt.title("Daily Visitor Arrival Envelope (Gaussian)")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(alpha=0.3)
    _save_figure(out)
    return True


def plot_visitors_inside_over_day(density_matrix: np.ndarray,
                                   capacities: Sequence[float],
                                   out_path: str) -> bool:
    """Plot total visitors inside the museum minute by minute over one day."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    caps = np.asarray(capacities, dtype=float)
    visitors = (density_matrix * caps[None, :]).sum(axis=1)
    t = np.arange(len(visitors))
    hours = t / 60.0 + 8.25
    plt.figure(figsize=(11, 4.5))
    plt.plot(hours, visitors, color="#1F4E5F", linewidth=1.8)
    plt.fill_between(hours, 0, visitors, alpha=0.15, color="#1F4E5F")
    plt.axhline(900, color="#B14D4D", linestyle="--",
                label="Fire-safety capacity (900)")
    plt.xlabel("Time of Day (24h)")
    plt.ylabel("Total Visitors Inside Museum")
    plt.title("Museum Occupancy Throughout One Day")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(alpha=0.3)
    _save_figure(out)
    return True


def plot_baseline_comparison(baselines: Sequence[Dict], q_learning_return: float,
                              out_path: str,
                              double_q_return: float | None = None) -> bool:
    """Bar chart comparing the Q-learning family against the 5 baselines.

    Always shows the five baselines and vanilla Q-learning. If
    ``double_q_return`` is provided, a second blue bar is added so the
    reader can see both members of the tabular Q-learning family on the
    same chart.
    """

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()

    names = [b["name"] for b in baselines] + ["q_learning"]
    returns = [b["mean_return"] for b in baselines] + [q_learning_return]
    colors = ["#8B6F47"] * len(baselines) + ["#1F4E5F"]
    if double_q_return is not None:
        names.append("double_q_learning")
        returns.append(double_q_return)
        colors.append("#3F7E8F")  # lighter blue to distinguish from vanilla Q

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, returns, color=colors, edgecolor="white")
    for bar, val in zip(bars, returns):
        plt.text(bar.get_x() + bar.get_width() / 2, val + max(returns) * 0.01,
                 f"{val:.1f}", ha="center", fontsize=9)
    plt.xlabel("Policy")
    plt.ylabel("Mean Episode Return")
    plt.title("Q-Learning Family vs Handcrafted Baselines (12-Room Toy Graph)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(alpha=0.3, axis="y")
    _save_figure(out)
    return True


def plot_offpeak_botticelli(offpeak_visits: Sequence[int], out_path: str) -> bool:
    """Off-peak Botticelli visits per episode (temporal arbitrage discovery)."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    y = np.asarray(offpeak_visits, dtype=float)
    x = np.arange(len(y))
    plt.figure(figsize=(11, 4.5))
    plt.plot(x, y, alpha=0.25, color="#8B6F47", linewidth=0.8, label="Per episode")
    if len(y) >= 100:
        w = 100
        rolling = np.convolve(y, np.ones(w) / w, mode="valid")
        plt.plot(np.arange(w - 1, len(y)), rolling, color="#1F4E5F",
                 linewidth=2.0, label=f"{w}-episode rolling mean")
    plt.xlabel("Episode")
    plt.ylabel("Off-Peak Botticelli Visits")
    plt.title("Temporal Arbitrage: Visiting Botticelli When the Crowd Is Low")
    plt.legend(loc="best", frameon=False)
    plt.grid(alpha=0.3)
    _save_figure(out)
    return True


def plot_episode_length_evolution(episode_lengths: Sequence[int], out_path: str) -> bool:
    """Plot episode length over training."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    y = np.asarray(episode_lengths, dtype=float)
    x = np.arange(len(y))
    plt.figure(figsize=(11, 4.5))
    plt.plot(x, y, alpha=0.3, color="#8B6F47", linewidth=0.6, label="Per episode")
    if len(y) >= 100:
        w = 100
        rolling = np.convolve(y, np.ones(w) / w, mode="valid")
        plt.plot(np.arange(w - 1, len(y)), rolling, color="#1F4E5F",
                 linewidth=2.0, label=f"{w}-episode rolling mean")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (steps)")
    plt.title("Episode Length Evolution During Q-Learning")
    plt.legend(loc="best", frameon=False)
    plt.grid(alpha=0.3)
    _save_figure(out)
    return True


def plot_epsilon_decay(out_path: str, episodes: int = 25000,
                        eps_init: float = 1.0, eps_min: float = 0.01,
                        decay: float = 0.9995) -> bool:
    """Plot the epsilon-greedy exploration schedule over training."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    eps = np.maximum(eps_min, eps_init * decay ** np.arange(episodes))
    plt.figure(figsize=(10, 4.5))
    plt.plot(eps, color="#1F4E5F", linewidth=2)
    plt.fill_between(np.arange(episodes), eps_min, eps, alpha=0.2, color="#1F4E5F")
    plt.axhline(eps_min, color="#B14D4D", linestyle="--", label=f"eps_min = {eps_min}")
    plt.xlabel("Episode")
    plt.ylabel("Exploration Rate (epsilon)")
    plt.title("Epsilon-Greedy Decay Schedule")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(alpha=0.3)
    _save_figure(out)
    return True


def plot_deep_rl_comparison(ppo_mean: float, ppo_std: float,
                              ablations: Sequence[Dict], out_path: str) -> bool:
    """Bar chart: MaskablePPO vs PPO-no-mask vs DQN."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    names = ["MaskablePPO"] + [a["algorithm"] for a in ablations]
    returns = [ppo_mean] + [a["mean_episode_reward"] for a in ablations]
    errors = [ppo_std] + [a.get("std_episode_reward", 0.0) for a in ablations]
    colors = ["#1F4E5F"] + ["#8B6F47"] * len(ablations)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, returns, yerr=errors, capsize=6, color=colors,
                    edgecolor="white", error_kw={"ecolor": "#444444"})
    for bar, val in zip(bars, returns):
        spread = max(returns) - min(returns) if max(returns) > min(returns) else 1.0
        plt.text(bar.get_x() + bar.get_width() / 2, val + spread * 0.02,
                 f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
    plt.ylabel("Mean Episode Return")
    plt.title("Deep RL Comparison on the Full 98-Room Museum")
    plt.grid(alpha=0.3, axis="y")
    _save_figure(out)
    return True


def plot_equilibrium_convergence(iterations: Sequence[Dict], out_path: str) -> bool:
    """Plot welfare over iterated best-response rounds with twin axis for Botticelli."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    welfare = [it["total_welfare"] for it in iterations]
    bott = [it["botticelli_over80_frac"] for it in iterations]
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(welfare, marker="o", color="#1F4E5F", linewidth=2,
             label="Total welfare")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total Welfare", color="#1F4E5F")
    ax1.tick_params(axis="y", labelcolor="#1F4E5F")
    ax2 = ax1.twinx()
    ax2.plot(bott, marker="s", color="#B14D4D", linewidth=2,
             label="Botticelli overcrowded fraction")
    ax2.set_ylabel("Botticelli Overcrowded Fraction", color="#B14D4D")
    ax2.tick_params(axis="y", labelcolor="#B14D4D")
    plt.title("Iterated Best Response Convergence")
    fig.tight_layout()
    _save_figure(out)
    return True


def plot_intervention_welfare_revenue(table: Sequence[Dict], out_path: str) -> bool:
    """Welfare vs revenue scatter for all interventions (Pareto frontier view)."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    baseline_w = float(table[0].get("total_welfare", 0))
    baseline_r = float(table[0].get("revenue", 0))
    rows = [r for r in table[1:] if "Social Optimum" not in str(r.get("intervention", ""))]
    dw = [float(r.get("total_welfare", 0)) - baseline_w for r in rows]
    dr = [float(r.get("revenue", 0)) - baseline_r for r in rows]
    names = [str(r.get("intervention", "")) for r in rows]
    plt.figure(figsize=(10, 6))
    plt.scatter(dw, dr, s=50, c="#1F4E5F", alpha=0.75, edgecolor="white")
    plt.axhline(0, color="black", linestyle=":", alpha=0.5)
    plt.axvline(0, color="black", linestyle=":", alpha=0.5)
    for i, name in enumerate(names):
        if abs(dw[i]) > 300 or abs(dr[i]) > 1500:
            plt.annotate(name[:20], (dw[i], dr[i]), fontsize=7, alpha=0.85,
                         xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Welfare Change vs Baseline")
    plt.ylabel("Revenue Change vs Baseline (EUR)")
    plt.title("Welfare-Revenue Trade-off Across Interventions")
    plt.grid(alpha=0.3)
    _save_figure(out)
    return True


def plot_top_interventions(table: Sequence[Dict], out_path: str, top_n: int = 10) -> bool:
    """Horizontal bar chart: top N interventions by welfare gain."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    baseline_w = float(table[0].get("total_welfare", 0))
    rows = [r for r in table[1:]
            if "Social Optimum" not in str(r.get("intervention", ""))
            and "Combined" not in str(r.get("intervention", ""))]
    rows = sorted(rows, key=lambda r: float(r.get("total_welfare", 0)), reverse=True)[:top_n]
    names = [str(r.get("intervention", "")) for r in rows]
    deltas = [float(r.get("total_welfare", 0)) - baseline_w for r in rows]
    colors = ["#1F4E5F" if d >= 0 else "#B14D4D" for d in deltas]
    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], deltas[::-1], color=colors[::-1], edgecolor="white")
    plt.axvline(0, color="black", linewidth=0.6)
    plt.xlabel("Welfare Gain vs Baseline")
    plt.title(f"Top {top_n} Interventions by Welfare Gain")
    _save_figure(out)
    return True


def plot_sweep_with_band(records: Sequence[Dict], x_key: str, y_key: str,
                          y_label: str, title: str, out_path: str) -> bool:
    """Generic sweep plot with mean line and CI band."""

    if not HAS_PLOT or not records:
        return False
    out = Path(out_path)
    _apply_theme()
    x = np.array([float(r[x_key]) for r in records])
    y = np.array([float(r[f"{y_key}_mean"]) for r in records])
    ci = np.array([float(r.get(f"{y_key}_ci", 0)) for r in records])
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o", linewidth=2, color="#1F4E5F", label="Mean")
    plt.fill_between(x, y - ci, y + ci, alpha=0.2, color="#1F4E5F", label="95% CI")
    plt.xlabel(x_key.replace("_", " ").title())
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="best", frameon=False)
    plt.grid(alpha=0.3)
    _save_figure(out)
    return True


def plot_portfolio_breakdown(portfolio: Dict, baseline_w: float, baseline_r: float,
                               out_path: str) -> bool:
    """Bar chart: baseline vs optimal portfolio (welfare and revenue side by side)."""

    if not HAS_PLOT:
        return False
    out = Path(out_path)
    _apply_theme()
    active = portfolio.get("active") or portfolio.get("active_interventions") or []
    w = float(portfolio.get("welfare", 0))
    r = float(portfolio.get("revenue", 0))
    plt.figure(figsize=(10, 5))
    labels = ["Baseline", "Optimal Portfolio"]
    welfare_vals = [baseline_w, w]
    revenue_vals = [baseline_r, r]
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, welfare_vals, width, color="#1F4E5F",
            label="Welfare", edgecolor="white")
    plt.bar(x + width / 2, np.array(revenue_vals) / 100, width, color="#8B6F47",
            label="Revenue / 100 (EUR)", edgecolor="white")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    welfare_pct = (w / baseline_w - 1) * 100 if baseline_w else 0
    revenue_pct = (r / baseline_r - 1) * 100 if baseline_r else 0
    plt.title(f"Optimal Portfolio ({len(active)} interventions): "
              f"Welfare {welfare_pct:+.1f}%, Revenue {revenue_pct:+.1f}%")
    plt.legend(loc="upper left", frameon=False)
    plt.grid(alpha=0.3, axis="y")
    _save_figure(out)
    return True
