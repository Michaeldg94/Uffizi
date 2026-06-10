"""Comprehensive diagnostic dashboard.

Outputs:
  - Demographics stacked bar (proper segment-based)
  - All-rooms heatmap per scenario
  - Visitor path clustering with K-Means on room-visit feature vectors
  - Cluster paths plotted on the museum map (networkx graph layout)
  - Equilibrium sanity checks: slot utilization, demographic revenue, etc.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uffizi_rl.environment.crowd_simulator import CrowdSimulator
from uffizi_rl.interventions.intervention_config import InterventionConfig
import uffizi_rl.config as config

OUT = Path("outputs/figures/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "Baseline": dict(revenue_model=True),
    "RAMA only": dict(revenue_model=True, rama=True),
    "RAMA + Audio": dict(
        revenue_model=True, rama=True, extended_hours=True,
        secondary_attractor_enrichment=True, dynamic_pricing=True,
        per_person_group_surcharge=150.0, audio_guide_revenue=True,
    ),
    "Non-RAMA bundle": dict(
        revenue_model=True, secondary_attractor_enrichment=True,
        extended_hours=True, per_person_group_surcharge=150.0,
        audio_guide_revenue=True,
    ),
}

runs = {}
for label, kw in SCENARIOS.items():
    print(f"Running: {label}")
    iv = InterventionConfig(**kw)
    sim = CrowdSimulator(daily_total=5000, seed=42, interventions=iv)
    m = sim.run_day()
    runs[label] = (sim, m)


# ---------------------------------------------------------------------------
# Demographic breakdown (segment-based, not price-based)
# ---------------------------------------------------------------------------
print("\n=== DEMOGRAPHICS (segment-based) ===")
print(f"{'Scenario':<22} {'Total':>7} {'Kid':>6} {'Disab':>6} {'Stud':>6} "
      f"{'Pens':>6} {'Adult':>7} {'Group':>6} {'Rev EUR':>10}")
segment_data = {}
for label, (sim, m) in runs.items():
    vs = sim.completed_visitors
    counts = Counter(v.demographic_segment for v in vs if v.group_id is None)
    group_ct = sum(1 for v in vs if v.group_id is not None)
    rev = sum(v.ticket_price for v in vs)
    print(f"{label:<22} {len(vs):>7d} {counts.get('kid', 0):>6d} "
          f"{counts.get('disabled_or_other_free', 0):>6d} "
          f"{counts.get('student', 0):>6d} "
          f"{counts.get('pensioner', 0):>6d} "
          f"{counts.get('adult', 0):>7d} "
          f"{group_ct:>6d} {rev:>10.0f}")
    segment_data[label] = (counts, group_ct)

# Stacked bar figure
fig, ax = plt.subplots(figsize=(13, 6))
labels = list(runs.keys())
cats = ["kid", "disabled_or_other_free", "student", "pensioner", "adult", "group"]
cat_labels = ["Kids (free)", "Disabled / other free", "Students 18-25 (EUR 2)",
              "Pensioners 65+", "Adults", "Tour groups"]
colors = ["#A0AEC0", "#CBD5E0", "#68D391", "#F6AD55", "#4299E1", "#9F7AEA"]
data = np.zeros((len(cats), len(labels)))
for j, label in enumerate(labels):
    counts, group_ct = segment_data[label]
    for i, cat in enumerate(cats[:-1]):
        data[i, j] = counts.get(cat, 0)
    data[-1, j] = group_ct
bottom = np.zeros(len(labels))
for i, cat in enumerate(cat_labels):
    ax.bar(labels, data[i], bottom=bottom, label=cat, color=colors[i], alpha=0.9)
    bottom += data[i]
ax.set_ylabel("Visitors per day")
ax.set_title("Visitor demographics by scenario (segment-based)")
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(alpha=0.3, axis="y")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUT / "demographics_stacked.png", dpi=140)
plt.close()


# ---------------------------------------------------------------------------
# All-rooms heatmap per scenario
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
focus_rooms = [r for r in config.ROOM_IDS
                if r not in {"ENTRY", "EXIT", "GRANDUCAL_STAIRCASE",
                              "LANZI_STAIRCASE", "BUONTALENTI_STAIRCASE"}]
base_dh = np.array(runs["Baseline"][0].density_history)
focus_rooms.sort(key=lambda r: -base_dh[:, config.ROOM_TO_IDX[r]].max())
focus_rooms = focus_rooms[:40]
for ax, (label, (sim, _)) in zip(axes.flat, runs.items()):
    dh = np.array(sim.density_history)
    sub = np.stack([dh[:, config.ROOM_TO_IDX[r]] for r in focus_rooms])
    im = ax.imshow(sub, aspect="auto", cmap="YlOrRd", vmin=0, vmax=2.0,
                    origin="upper", interpolation="nearest")
    ax.set_title(label, fontsize=12)
    ax.set_yticks(np.arange(len(focus_rooms)))
    ax.set_yticklabels(focus_rooms, fontsize=6)
    ax.set_xlabel("Minute of day")
fig.colorbar(im, ax=axes.ravel().tolist(), label="density (occupancy / cap)", shrink=0.7)
fig.suptitle("All-room density by scenario", fontsize=14, y=0.995)
plt.savefig(OUT / "all_rooms_heatmap.png", dpi=140, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Path clustering: K-means on room-visit feature vectors
# ---------------------------------------------------------------------------
print("\n=== PATH CLUSTERING ===")
ROOM_LIST = [r for r in config.ROOM_IDS if r != "EXIT"]
ROOM_TO_FEAT_IDX = {r: i for i, r in enumerate(ROOM_LIST)}

def path_feature(path_rooms):
    """One-hot vector of rooms visited (binary). 1 if visited, else 0."""
    vec = np.zeros(len(ROOM_LIST))
    for r in path_rooms:
        if r in ROOM_TO_FEAT_IDX:
            vec[ROOM_TO_FEAT_IDX[r]] = 1.0
    return vec

# Use the SCENARIO of interest for clustering (RAMA + Audio).
CLUSTER_SCENARIO = "RAMA + Audio"
sim_c = runs[CLUSTER_SCENARIO][0]
trajectories = [v.path for v in sim_c.completed_visitors if len(v.path) > 3]
X = np.array([path_feature([r for _, r in tr]) for tr in trajectories])
print(f"Clustering {len(X)} trajectories from '{CLUSTER_SCENARIO}'...")

K = 6
km = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = km.fit_predict(X)

# Print cluster summary
print(f"\nCluster summary (K={K}):")
for k in range(K):
    members = np.where(labels == k)[0]
    centroid = km.cluster_centers_[k]
    # Top rooms in centroid (most visited rooms in this cluster)
    top_room_idx = np.argsort(-centroid)[:8]
    top_rooms = [ROOM_LIST[i] for i in top_room_idx if centroid[i] > 0.5]
    mp = [r for r in top_rooms if r in ("A11", "A12", "A35", "A38")]
    print(f"  Cluster {k}: {len(members)} visitors  "
          f"masterpieces=[{','.join(mp) if mp else 'none'}]  "
          f"sample_top_rooms={top_rooms[:5]}")


# ---------------------------------------------------------------------------
# Cluster paths on the ACTUAL Uffizi floor plan
# ---------------------------------------------------------------------------
from floor_plan_coords import (
    FLOOR2, ALL_COORDS, FLOOR2_ROOMS, CORRIDORS, EDGE_WAYPOINTS,
)
from PIL import Image

floor2_img = np.asarray(Image.open(Path("outputs/assets/uffizi_floor2.png").resolve()))
H, W = floor2_img.shape[:2]

masterpiece_set = {"A11", "A12", "A35", "A38"}

# Build undirected version of the museum graph so we can compute walking
# distances even from teleport-only-reachable rooms (the actual visitor walked
# back via the corridor; teleport jumps in the data are an artifact of RAMA
# placing them straight in their slot).
gu = sim_c.g.to_undirected()
sp_cache: dict[tuple[str, str], list[str]] = {}
def shortest_walk(a: str, b: str) -> list[str]:
    """Return the sequence of rooms along the shortest walking path
    from a to b in the museum, including a and b. Caches results."""
    key = (a, b)
    if key in sp_cache:
        return sp_cache[key]
    if a == b:
        sp_cache[key] = [a]
        return [a]
    try:
        path = nx.shortest_path(gu, a, b)
    except nx.NetworkXNoPath:
        path = [a, b]
    sp_cache[key] = path
    return path


def _manhattan_waypoint(a: str, b: str) -> list[tuple[float, float]]:
    """Insert L-shaped corner waypoint(s) between rooms a and b.

    The Uffizi corridors are long straight strips (A2 horizontal at top,
    A23 vertical on right, A24 horizontal at bottom). When a path
    transitions between a corridor and any other room, we route along
    the corridor's axis first, so the drawn line stays inside the
    corridor strip instead of cutting diagonally across the courtyard.
    """
    key = (a, b)
    if key in EDGE_WAYPOINTS:
        return list(EDGE_WAYPOINTS[key])
    if a in FLOOR2 and b in FLOOR2:
        ax, ay = FLOOR2[a]
        bx, by = FLOOR2[b]
        # If b is a corridor, route to the corridor axis at a's other coord.
        if b in CORRIDORS:
            orient, fixed = CORRIDORS[b]
            if orient == "h":
                return [(ax, fixed)]
            return [(fixed, ay)]
        # If a is a corridor, route along the corridor axis to b's coord.
        if a in CORRIDORS:
            orient, fixed = CORRIDORS[a]
            if orient == "h":
                return [(bx, fixed)]
            return [(fixed, by)]
    return []


def expand_walked_coords(rooms_seq: list[str]) -> list[tuple[float, float]]:
    """Convert a sequence of visited rooms into a list of (x, y) coords
    that traces the corridors correctly.

    Steps:
      1. Expand teleport jumps via shortest walking path on the graph.
      2. For each adjacent step in the expanded path, insert L-shaped
         corner waypoints whenever a corridor is involved.
    """
    if not rooms_seq:
        return []
    # First, expand teleport jumps.
    expanded = [rooms_seq[0]]
    for prev, curr in zip(rooms_seq[:-1], rooms_seq[1:]):
        walk = shortest_walk(prev, curr)
        expanded.extend(walk[1:])
    # Then, build coordinate list with Manhattan waypoints.
    coords: list[tuple[float, float]] = []
    if expanded[0] in FLOOR2:
        coords.append(FLOOR2[expanded[0]])
    for prev, curr in zip(expanded[:-1], expanded[1:]):
        if curr not in FLOOR2:
            continue
        for wp in _manhattan_waypoint(prev, curr):
            coords.append(wp)
        coords.append(FLOOR2[curr])
    return coords


fig, axes = plt.subplots(2, 3, figsize=(22, 14))
for k, ax in enumerate(axes.flat):
    if k >= K:
        ax.axis("off")
        continue
    members_idx = np.where(labels == k)[0]
    centroid = km.cluster_centers_[k]
    # Only mark a room as a "cluster masterpiece" if MAJORITY of cluster
    # members actually visited it (centroid >= 0.5). 0.3 was too lax and
    # picked up A12 in every cluster because it's a single-row gate.
    visited_rooms = {ROOM_LIST[i] for i in range(len(centroid)) if centroid[i] >= 0.5}
    mp = [r for r in ("A11", "A12", "A35", "A38") if r in visited_rooms]

    # Floor plan background.
    ax.imshow(floor2_img, extent=(0, 1, 1, 0), aspect="auto", alpha=0.55)

    # Sample trajectories.
    rng = np.random.RandomState(k)
    sample_idx = rng.choice(members_idx, size=min(12, len(members_idx)), replace=False)
    for sample_i, idx in enumerate(sample_idx):
        tr = trajectories[idx]
        rooms_in_order = [r for _, r in tr if r in FLOOR2_ROOMS]
        if len(rooms_in_order) < 2:
            continue
        # Expand teleport jumps into walking paths AND insert corridor
        # corner waypoints so the line follows the museum corridors.
        coord_seq = expand_walked_coords(rooms_in_order)
        if len(coord_seq) < 2:
            continue
        coords = np.array(coord_seq)
        jitter = rng.normal(scale=0.004, size=coords.shape)
        coords = coords + jitter
        ax.plot(coords[:, 0], coords[:, 1], "-",
                color="#2C5282", alpha=0.55, linewidth=1.0)
        ax.plot(coords[0, 0], coords[0, 1], "o",
                color="#38A169", markersize=6, markeredgecolor="white")
        ax.plot(coords[-1, 0], coords[-1, 1], "s",
                color="#C53030", markersize=6, markeredgecolor="white")

    # Star the masterpieces visited by this cluster.
    for r in mp:
        if r in FLOOR2:
            ax.plot(FLOOR2[r][0], FLOOR2[r][1], "*", color="#D69E2E",
                    markersize=24, markeredgecolor="black", markeredgewidth=1.2)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    mp_str = ", ".join(mp) if mp else "no masterpieces"
    ax.set_title(f"Cluster {k}: {len(members_idx)} visitors  [{mp_str}]",
                 fontsize=11, pad=4)

fig.suptitle(f"Visitor path clusters under '{CLUSTER_SCENARIO}' (K={K}) -- "
              "12 sample walking paths per cluster on the Uffizi 2nd-floor plan",
              fontsize=13, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT / "path_clusters_map.png", dpi=140, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Equilibrium sanity checks
# ---------------------------------------------------------------------------
print("\n=== EQUILIBRIUM CHECKS ===")
print(f"{'Scenario':<22} {'Welfare':>8} {'Revenue':>9} {'Bott visits':>12} "
      f"{'Leo':>7} {'Raph':>7} {'Avg path len':>12}")
for label, (sim, m) in runs.items():
    vs = sim.completed_visitors
    bott = sum(1 for v in vs if "A11" in v.rooms_visited)
    leo = sum(1 for v in vs if "A35" in v.rooms_visited)
    raph = sum(1 for v in vs if "A38" in v.rooms_visited)
    avg_len = np.mean([len(v.path) for v in vs if v.path])
    welfare = m.get("mean_welfare_per_attempted", 0)
    rev = m.get("revenue", 0)
    print(f"{label:<22} {welfare:>8.1f} {rev:>9.0f} {bott:>12d} "
          f"{leo:>7d} {raph:>7d} {avg_len:>12.1f}")

# Per-segment welfare
print(f"\n=== PER-SEGMENT WELFARE & REVENUE (RAMA + Audio) ===")
sim_main = runs["RAMA + Audio"][0]
seg_welfare = {}
seg_revenue = {}
seg_count = {}
for v in sim_main.completed_visitors:
    seg = "group" if v.group_id is not None else v.demographic_segment
    seg_welfare.setdefault(seg, []).append(v.experienced_welfare)
    seg_revenue[seg] = seg_revenue.get(seg, 0) + v.ticket_price
    seg_count[seg] = seg_count.get(seg, 0) + 1
print(f"{'Segment':<22} {'Count':>7} {'Avg welfare':>12} {'Total rev EUR':>14}")
for seg, ws in sorted(seg_welfare.items(), key=lambda x: -seg_count[x[0]]):
    print(f"{seg:<22} {seg_count[seg]:>7d} {np.mean(ws):>12.1f} {seg_revenue[seg]:>14.0f}")

print(f"\nFigures saved to {OUT}")
