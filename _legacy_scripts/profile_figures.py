"""Per-profile walking figures on the actual Uffizi floor plans.

Replaces the old K-means cluster map. Instead of behavioural clusters, this
shows the three real visitor profiles (Instagram tourist, Standard tourist,
Art lover) side by side, on both floors, two ways:

  profile_walks_map.png     -- 12 sample real walks per profile per floor
  profile_heatmaps_map.png  -- per-room visit frequency per profile per floor

Both are 2 x 3 grids: rows are the 2nd floor (top) and 1st floor (bottom),
columns are the three profiles. Instagram tourists never descend, so their
1st-floor panel is intentionally empty.

Scenario: RAMA + Audio (the full intervention bundle), 5000 visitors.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uffizi_rl.environment.crowd_simulator import CrowdSimulator
from uffizi_rl.interventions.intervention_config import InterventionConfig
import uffizi_rl.config as config
from floor_plan_coords import (
    FLOOR2, FLOOR1, FLOOR2_ROOMS, FLOOR1_ROOMS, CORRIDORS, EDGE_WAYPOINTS,
)

OUT = Path("outputs/figures/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Run the scenario.
# ---------------------------------------------------------------------------
iv = InterventionConfig(
    revenue_model=True, rama=True, extended_hours=True,
    secondary_attractor_enrichment=True, dynamic_pricing=True,
    per_person_group_surcharge=150.0, audio_guide_revenue=True,
)
sim = CrowdSimulator(daily_total=5000, seed=42, interventions=iv)
sim.run_day()

by_seg: dict[str, list] = {"instagram": [], "standard": [], "art_lover": []}
for v in sim.completed_visitors:
    seg = getattr(getattr(v, "profile", None), "segment", None)
    if seg in by_seg:
        by_seg[seg].append(v)

# ---------------------------------------------------------------------------
# Floor-plan backgrounds.
# ---------------------------------------------------------------------------
floor2_img = np.asarray(Image.open(Path("outputs/assets/uffizi_floor2.png").resolve()))
floor1_img = np.asarray(Image.open(Path("outputs/assets/uffizi_floor1.png").resolve()))

# ---------------------------------------------------------------------------
# Walking-path expansion (2nd floor: corridors + teleport jumps).
# ---------------------------------------------------------------------------
gu = sim.g.to_undirected()
_sp_cache: dict[tuple[str, str], list[str]] = {}


def shortest_walk(a: str, b: str) -> list[str]:
    key = (a, b)
    if key in _sp_cache:
        return _sp_cache[key]
    if a == b:
        _sp_cache[key] = [a]
        return [a]
    try:
        path = nx.shortest_path(gu, a, b)
    except nx.NetworkXNoPath:
        path = [a, b]
    _sp_cache[key] = path
    return path


def _manhattan_waypoint(a: str, b: str) -> list[tuple[float, float]]:
    key = (a, b)
    if key in EDGE_WAYPOINTS:
        return list(EDGE_WAYPOINTS[key])
    if a in FLOOR2 and b in FLOOR2:
        ax, ay = FLOOR2[a]
        bx, by = FLOOR2[b]
        if b in CORRIDORS:
            orient, fixed = CORRIDORS[b]
            return [(ax, fixed)] if orient == "h" else [(fixed, ay)]
        if a in CORRIDORS:
            orient, fixed = CORRIDORS[a]
            return [(bx, fixed)] if orient == "h" else [(fixed, by)]
    return []


def expand_floor2(rooms_seq: list[str]) -> list[tuple[float, float]]:
    """Coords tracing the 2nd-floor corridors, expanding teleport jumps."""
    seq = [r for r in rooms_seq if r in FLOOR2_ROOMS]
    if not seq:
        return []
    expanded = [seq[0]]
    for prev, curr in zip(seq[:-1], seq[1:]):
        walk = [r for r in shortest_walk(prev, curr) if r in FLOOR2_ROOMS]
        expanded.extend(walk[1:] if walk[:1] == [prev] else walk)
    coords: list[tuple[float, float]] = [FLOOR2[expanded[0]]]
    for prev, curr in zip(expanded[:-1], expanded[1:]):
        for wp in _manhattan_waypoint(prev, curr):
            coords.append(wp)
        coords.append(FLOOR2[curr])
    return coords


def expand_floor1(rooms_seq: list[str]) -> list[tuple[float, float]]:
    """Coords for the 1st-floor portion: straight lines between the rooms
    the visitor actually walked (no teleports happen down here)."""
    seq = [r for r in rooms_seq if r in FLOOR1_ROOMS and r in FLOOR1]
    return [FLOOR1[r] for r in seq]


FLOORS = [
    ("2nd floor", FLOOR2, FLOOR2_ROOMS, floor2_img, expand_floor2,
     ["A11", "A12", "A35", "A38"], "masterpieces"),
    ("1st floor", FLOOR1, FLOOR1_ROOMS, floor1_img, expand_floor1,
     ["E4", "E5"], "Caravaggio"),
]
PROFILES = [
    ("instagram", "Instagram tourist (60%)"),
    ("standard", "Standard tourist (30%)"),
    ("art_lover", "Art lover (10%)"),
]


def style_axis(ax, img):
    ax.imshow(img, extent=(0, 1, 1, 0), aspect="auto", alpha=0.55)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])


# ===========================================================================
# FIGURE 1: sample walks.
# ===========================================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 15))
for col, (seg, seg_label) in enumerate(PROFILES):
    visitors = by_seg[seg]
    rng = np.random.RandomState(42)
    sample = (rng.choice(len(visitors), size=min(12, len(visitors)), replace=False)
              if visitors else [])
    for row, (fl_name, COORDS, FL_ROOMS, img, expand, stars, _) in enumerate(FLOORS):
        ax = axes[row, col]
        style_axis(ax, img)
        n_on_floor = 0
        for idx in sample:
            rooms = [r for _, r in visitors[idx].path]
            coords = expand(rooms)
            if len(coords) < 2:
                continue
            n_on_floor += 1
            pts = np.array(coords) + rng.normal(scale=0.004, size=(len(coords), 2))
            ax.plot(pts[:, 0], pts[:, 1], "-", color="#2C5282", alpha=0.55, linewidth=1.1)
            ax.plot(pts[0, 0], pts[0, 1], "o", color="#38A169",
                    markersize=6, markeredgecolor="white")
            ax.plot(pts[-1, 0], pts[-1, 1], "s", color="#C53030",
                    markersize=6, markeredgecolor="white")
        for r in stars:
            if r in COORDS:
                ax.plot(COORDS[r][0], COORDS[r][1], "*", color="#D69E2E",
                        markersize=22, markeredgecolor="black", markeredgewidth=1.0)
        if row == 0:
            ax.set_title(f"{seg_label}\n{fl_name}", fontsize=12, pad=6)
        else:
            ax.set_title(fl_name, fontsize=11, pad=4)
        if n_on_floor == 0:
            ax.text(0.5, 0.5, "does not reach this floor",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, style="italic", color="#666666")
fig.suptitle("How the three visitor profiles walk the Uffizi -- 12 sample "
             "paths each (RAMA + Audio). Green = start, red = end; gold stars "
             "= masterpieces / Caravaggio.", fontsize=13, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT / "profile_walks_map.png", dpi=140, bbox_inches="tight")
plt.close()

# ===========================================================================
# FIGURE 2: per-room visit-frequency heatmaps.
# ===========================================================================
cmap = plt.get_cmap("YlOrRd")
norm = Normalize(vmin=0.0, vmax=1.0)
fig, axes = plt.subplots(2, 3, figsize=(22, 15))
for col, (seg, seg_label) in enumerate(PROFILES):
    visitors = by_seg[seg]
    n = max(1, len(visitors))
    # Fraction of this profile's visitors who enter each room.
    freq = Counter()
    for v in visitors:
        for r in {room for _, room in v.path}:
            freq[r] += 1
    for row, (fl_name, COORDS, FL_ROOMS, img, _expand, stars, _) in enumerate(FLOORS):
        ax = axes[row, col]
        style_axis(ax, img)
        xs, ys, cs = [], [], []
        for r, (x, y) in COORDS.items():
            if r in ("ENTRY", "EXIT") or "STAIRCASE" in r:
                continue
            f = freq.get(r, 0) / n
            xs.append(x)
            ys.append(y)
            cs.append(f)
        ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm, s=420, alpha=0.85,
                   edgecolors="black", linewidths=0.5, zorder=3)
        for r in stars:
            if r in COORDS:
                ax.plot(COORDS[r][0], COORDS[r][1], "*", color="#2B6CB0",
                        markersize=16, markeredgecolor="white", markeredgewidth=0.8,
                        zorder=4)
        if row == 0:
            ax.set_title(f"{seg_label}\n{fl_name}", fontsize=12, pad=6)
        else:
            ax.set_title(fl_name, fontsize=11, pad=4)
        if sum(cs) == 0:
            ax.text(0.5, 0.5, "does not reach this floor",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, style="italic", color="#666666")
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01)
cbar.set_label("share of profile's visitors who enter the room", fontsize=11)
fig.suptitle("Where each visitor profile goes -- per-room visit frequency "
             "(RAMA + Audio). Darker = a larger share of that profile enters "
             "the room. Blue stars = masterpieces / Caravaggio.",
             fontsize=13, y=0.995)
plt.savefig(OUT / "profile_heatmaps_map.png", dpi=140, bbox_inches="tight")
plt.close()

print("Wrote:")
print("  ", OUT / "profile_walks_map.png")
print("  ", OUT / "profile_heatmaps_map.png")
for seg, label in PROFILES:
    print(f"  {label}: {len(by_seg[seg])} visitors")
