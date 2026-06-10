"""Generate presentation-ready figures for the RAMA paper.

Produces a curated set of figures designed to tell the visual story:
chaos baseline → RAMA controls → enrichment lifts → extended hours +
dynamic pricing seal the win. Each figure is sized for a 16:9 beamer
slide and uses a clean colour palette.

Run AFTER the main pipeline (scripts 03-05) has populated the JSON
artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from uffizi_rl import config
from uffizi_rl.environment.crowd_simulator import CrowdSimulator
from uffizi_rl.analysis.phase_transition import simulate_day_metrics


# Output directory.
OUT = ROOT / "outputs" / "figures" / "presentation"
OUT.mkdir(parents=True, exist_ok=True)

# Colour palette (colour-blind safe, presentation friendly).
COLOR_BASELINE = "#c0392b"   # red - chaos
COLOR_RAMA = "#2c3e50"        # dark slate - control
COLOR_ENRICH = "#27ae60"      # green - supply
COLOR_FULL = "#8e44ad"        # purple - full bundle
COLOR_NEUTRAL = "#7f8c8d"     # grey

# Common run kwargs.
COMMON = dict(
    daily_total=config.DAILY_VISITORS_NORMAL,
    seed=config.DEFAULT_SEED + 40,
    type_a_fraction=config.TYPE_A_FRACTION_DEFAULT,
    revenue_model=True,
)


def run_sim(label: str, **kwargs):
    """Run one day, return (sim, density_matrix, metrics)."""
    print(f"  running {label} ...")
    sim = CrowdSimulator(**COMMON, **kwargs)
    sim.run_day()
    dh = sim.export_density_matrix()
    m = simulate_day_metrics(
        daily_total=config.DAILY_VISITORS_NORMAL,
        seed=config.DEFAULT_SEED + 40,
        type_b_fraction=config.TYPE_B_FRACTION_DEFAULT,
        revenue_model=True,
        **kwargs,
    )
    return sim, dh, m


# Pre-compute the four scenarios we'll re-use.
print("Simulating scenarios...")
SCENARIOS = {
    "Baseline": run_sim("baseline"),
    "RAMA": run_sim("rama", rama=True),
    "RAMA + Enrichment": run_sim("rama_enr", rama=True, secondary_attractor_enrichment=True),
    "Full bundle": run_sim(
        "full",
        rama=True,
        secondary_attractor_enrichment=True,
        extended_hours=True,
        dynamic_pricing=True,
        per_person_group_surcharge=150.0,
        audio_guide_revenue=True,
    ),
}


# =============================================================================
# Figure 1: Botticelli (A11) density over the day, 4 lines
# =============================================================================
print("[1/8] hero_density_over_day")
fig, ax = plt.subplots(figsize=(10, 5.5))
A11_idx = config.ROOM_TO_IDX["A11"]
A11_cap = config.ROOM_DATA["A11"]["capacity"]
colors = [COLOR_BASELINE, COLOR_RAMA, COLOR_ENRICH, COLOR_FULL]
for (label, (sim, dh, _)), col in zip(SCENARIOS.items(), colors):
    times = np.arange(dh.shape[0])
    counts = dh[:, A11_idx] * A11_cap
    ax.plot(times, counts, label=label, color=col, linewidth=2, alpha=0.85)
ax.axhline(A11_cap, color="black", linestyle="--", linewidth=1.2, label=f"Room capacity ({int(A11_cap)})")
ax.set_xlabel("Minute of museum day (0 = opening)")
ax.set_ylabel("People inside Botticelli (room A11)")
ax.set_title("Botticelli room occupancy through the day", fontsize=14)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "01_botticelli_over_day.png", dpi=150)
plt.savefig(OUT / "01_botticelli_over_day.pdf")
plt.close()


# =============================================================================
# Figure 1b: Four-panel grid of masterpiece occupancy over the day,
# baseline vs full bundle, showing how RAMA + Enrichment caps each
# masterpiece at its capacity.
# =============================================================================
print("[1b/8] masterpieces_baseline_vs_full")
master_rooms = [("A11", "Botticelli (Spring)", 55),
                ("A12", "Botticelli (Venus)", 55),
                ("A35", "Leonardo", 52),
                ("A38", "Raphael & Michelangelo", 53)]
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
panel_scenarios = [("Baseline", COLOR_BASELINE),
                   ("Full bundle", COLOR_FULL)]
for ax, (rid, rname, cap) in zip(axes.flat, master_rooms):
    # Spring and Venus share the Botticelli-zone occupancy series
    # because they are modeled as a single experience zone (visitor
    # dwells 10 min in A11, walks through A12). Show A11 in both
    # panels so the rooms display identically as the user requires.
    plot_rid = "A11" if rid in ("A11", "A12") else rid
    idx = config.ROOM_TO_IDX[plot_rid]
    for sc_label, col in panel_scenarios:
        if sc_label not in SCENARIOS:
            continue
        sim, dh, _ = SCENARIOS[sc_label]
        counts = dh[:, idx] * cap
        # Faint raw line (slot-cycle noise)
        ax.plot(np.arange(dh.shape[0]), counts, color=col,
                linewidth=0.6, alpha=0.30)
        # Bold smoothed line (15-min rolling mean) reveals trend
        w = 15
        if len(counts) > w:
            smoothed = np.convolve(counts, np.ones(w)/w, mode="valid")
            t_smooth = np.arange(len(smoothed)) + w // 2
            ax.plot(t_smooth, smoothed, color=col,
                    linewidth=2.2, alpha=0.95, label=sc_label)
    ax.axhline(cap, color="black", linestyle="--", linewidth=1.0,
               label=f"Capacity {cap}")
    ax.set_title(rname, fontsize=12)
    ax.set_ylabel("People in room")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
axes[1, 0].set_xlabel("Minute of museum day (0 = opening)")
axes[1, 1].set_xlabel("Minute of museum day (0 = opening)")
fig.suptitle("Masterpiece occupancy: Baseline vs Full Bundle", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT / "01b_masterpieces_baseline_vs_full.png", dpi=150)
plt.savefig(OUT / "01b_masterpieces_baseline_vs_full.pdf")
plt.close()


# =============================================================================
# Figure 2: Peak occupancy bar chart, masterpiece rooms
# =============================================================================
print("[2/8] peak_masterpiece_bars")
fig, ax = plt.subplots(figsize=(10, 5.5))
rooms = [("A11", "Botticelli\nSpring", 55),
         ("A12", "Botticelli\nVenus", 55),
         ("A35", "Leonardo", 52),
         ("A38", "Raphael &\nMichelangelo", 53)]
labels = [r[1] for r in rooms]
x = np.arange(len(rooms))
width = 0.2
for i, (label, (sim, dh, _)) in enumerate(SCENARIOS.items()):
    peaks = [float(dh[:, config.ROOM_TO_IDX[r[0]]].max()) * r[2] for r in rooms]
    ax.bar(x + (i - 1.5) * width, peaks, width, label=label, color=colors[i], alpha=0.85)
# Capacity lines (one per room).
for j, (_, _, cap) in enumerate(rooms):
    ax.hlines(cap, j - 2 * width, j + 2 * width, color="black", linestyle="--", linewidth=1.5)
ax.set_ylabel("Peak occupancy (people)")
ax.set_title("Peak masterpiece occupancy under each policy\n(dashed line = room capacity)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT / "02_peak_masterpiece_bars.png", dpi=150)
plt.savefig(OUT / "02_peak_masterpiece_bars.pdf")
plt.close()


# =============================================================================
# Figure 3: Density heatmap 2x2 grid
# =============================================================================
print("[3/8] heatmap_grid")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
items = list(SCENARIOS.items())
# Subset of rooms with meaningful traffic (skip dead ones).
focus_rooms = [r for r in config.ROOM_IDS
               if r not in {"ENTRY", "EXIT", "GRANDUCAL_STAIRCASE",
                            "LANZI_STAIRCASE", "BUONTALENTI_STAIRCASE"}
               and any(
                   item[1][1][:, config.ROOM_TO_IDX[r]].max() > 0.05
                   for item in items
               )]
# Sort focus rooms by their max baseline density (most-traveled first).
base_dh = SCENARIOS["Baseline"][1]
focus_rooms.sort(key=lambda r: -base_dh[:, config.ROOM_TO_IDX[r]].max())
focus_rooms = focus_rooms[:30]  # top 30
vmax = 2.2  # consistent scale across panels
for ax, (label, (sim, dh, _)) in zip(axes.flat, items):
    sub = np.stack([dh[:, config.ROOM_TO_IDX[r]] for r in focus_rooms])
    im = ax.imshow(sub, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
                   origin="upper", interpolation="nearest")
    ax.set_title(label, fontsize=12)
    ax.set_yticks(np.arange(len(focus_rooms)))
    ax.set_yticklabels(focus_rooms, fontsize=7)
    ax.set_xticks(np.linspace(0, sub.shape[1] - 1, 6))
    ax.set_xticklabels([f"{int(t)}" for t in np.linspace(0, sub.shape[1] - 1, 6)], fontsize=8)
    ax.set_xlabel("Minute of day")
fig.colorbar(im, ax=axes.ravel().tolist(), label="density (occupancy / capacity)", shrink=0.7)
fig.suptitle("Per-room density through the day", fontsize=14, y=0.995)
plt.savefig(OUT / "03_heatmap_grid.png", dpi=140, bbox_inches="tight")
plt.savefig(OUT / "03_heatmap_grid.pdf", bbox_inches="tight")
plt.close()


# =============================================================================
# Figure 4: Welfare + Revenue comparison
# =============================================================================
print("[4/8] welfare_revenue_bars")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
labels = list(SCENARIOS.keys())
welfares = [SCENARIOS[k][2]["total_welfare"] for k in labels]
revenues = [SCENARIOS[k][2]["revenue"] for k in labels]
ax1.bar(labels, welfares, color=colors, alpha=0.85)
ax1.set_title("Welfare (per attempted visitor)", fontsize=12)
ax1.set_ylabel("Mean welfare")
ax1.grid(alpha=0.3, axis="y")
for i, v in enumerate(welfares):
    ax1.text(i, v + 5, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
ax2.bar(labels, revenues, color=colors, alpha=0.85)
ax2.set_title("Daily revenue", fontsize=12)
ax2.set_ylabel("Euro")
ax2.grid(alpha=0.3, axis="y")
for i, v in enumerate(revenues):
    ax2.text(i, v + 1000, f"{int(v):,}", ha="center", fontsize=10, fontweight="bold")
plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT / "04_welfare_revenue.png", dpi=150)
plt.savefig(OUT / "04_welfare_revenue.pdf")
plt.close()


# =============================================================================
# Figure 5: Masterpiece throughput (visits per day per masterpiece)
# =============================================================================
print("[5/8] masterpiece_throughput")
fig, ax = plt.subplots(figsize=(10, 5.5))
rooms_t = [("A11", "Botticelli", 55), ("A35", "Leonardo", 52), ("A38", "Raphael", 53)]
x = np.arange(len(rooms_t))
width = 0.2
for i, (label, (sim, dh, _)) in enumerate(SCENARIOS.items()):
    all_v = sim.completed_visitors + sim.active_visitors + list(sim.outside_queue)
    visits = []
    for rid, _, _ in rooms_t:
        v = sum(1 for vv in all_v if rid in vv.rooms_visited)
        visits.append(v)
    ax.bar(x + (i - 1.5) * width, visits, width, label=label, color=colors[i], alpha=0.85)
ax.set_ylabel("Visitors per day")
ax.set_title("Daily masterpiece throughput\n(unique visitors who entered the room)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([r[1] for r in rooms_t])
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT / "05_throughput.png", dpi=150)
plt.savefig(OUT / "05_throughput.pdf")
plt.close()


# =============================================================================
# Figure 6: Intervention welfare ranking (from JSON, horizontal bar)
# =============================================================================
print("[6/8] intervention_ranking")
import json
with open(ROOT / "outputs/05_interventions.json") as f:
    iv = json.load(f)
rows = sorted(iv["ranked_screen"], key=lambda r: r["delta_w"])
rows = [r for r in rows if r["name"] != "baseline"]
fig, ax = plt.subplots(figsize=(11, 6))
labels_r = [r["label"] for r in rows]
deltas = [r["delta_w"] for r in rows]
bar_colors = [COLOR_BASELINE if d < 0 else COLOR_RAMA for d in deltas]
y = np.arange(len(rows))
ax.barh(y, deltas, color=bar_colors, alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(labels_r, fontsize=9)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("ΔWelfare vs baseline")
ax.set_title("Intervention ranking by welfare delta\n(baseline welfare = {:.0f})".format(iv["baseline_welfare"]), fontsize=13)
for i, d in enumerate(deltas):
    ax.text(d + (1 if d > 0 else -1), i, f"{d:+.1f}",
            va="center", ha="left" if d > 0 else "right", fontsize=9)
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(OUT / "06_intervention_ranking.png", dpi=150)
plt.savefig(OUT / "06_intervention_ranking.pdf")
plt.close()


# =============================================================================
# Figure 7: Welfare vs Revenue scatter (Pareto view)
# =============================================================================
print("[7/8] welfare_revenue_scatter")
fig, ax = plt.subplots(figsize=(9, 6))
for r in rows + [{"name": "baseline", "label": "Baseline",
                  "welfare": iv["baseline_welfare"], "revenue": iv["baseline_revenue"]}]:
    rw = r.get("welfare", iv["baseline_welfare"] + r.get("delta_w", 0))
    rr = r.get("revenue", iv["baseline_revenue"] + r.get("delta_r", 0))
    is_baseline = r["name"] == "baseline"
    is_winner = r["name"] == "rama_enrich_extended_priced"
    c = COLOR_BASELINE if is_baseline else (COLOR_FULL if is_winner else COLOR_RAMA)
    size = 200 if is_winner else (180 if is_baseline else 90)
    ax.scatter(rw, rr, s=size, c=c, alpha=0.85, edgecolors="black",
               linewidth=1 if is_winner or is_baseline else 0.3)
    if is_winner or is_baseline or r.get("delta_w", -999) > 20:
        ax.annotate(r["label"], (rw, rr),
                    xytext=(8, 8), textcoords="offset points", fontsize=9)
ax.set_xlabel("Welfare (per attempted visitor)")
ax.set_ylabel("Revenue (EUR)")
ax.set_title("Welfare vs Revenue across interventions\n(Pareto frontier toward top-right)", fontsize=13)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "07_welfare_revenue_scatter.png", dpi=150)
plt.savefig(OUT / "07_welfare_revenue_scatter.pdf")
plt.close()


# =============================================================================
# Figure 8: Booking schedule by segment (RAMA only)
# =============================================================================
print("[8/8] booking_schedule_by_segment")
sim_full = SCENARIOS["Full bundle"][0]
all_v = sim_full.completed_visitors + sim_full.active_visitors + list(sim_full.outside_queue)
# Bin Bot slot start times by segment.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, room, label in zip(axes, ("A11", "A35", "A38"),
                            ("Botticelli", "Leonardo", "Raphael/Mich.")):
    for seg, col in (("art_lover", COLOR_ENRICH),
                     ("standard", COLOR_RAMA),
                     ("instagram", COLOR_BASELINE)):
        slot_starts = []
        for v in all_v:
            if v.profile.segment != seg:
                continue
            if room in ("A11",):
                w = v.botticelli_window
            elif room == "A35":
                w = v.leonardo_window
            else:
                w = v.raphael_window
            if w is not None:
                slot_starts.append(w[0])
        if slot_starts:
            ax.hist(slot_starts, bins=24, color=col, alpha=0.6, label=seg)
    ax.set_xlabel("Booked slot start (min of day)")
    ax.set_ylabel("Number of visitors")
    ax.set_title(f"{label} bookings")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "08_booking_schedule_by_segment.png", dpi=150)
plt.savefig(OUT / "08_booking_schedule_by_segment.pdf")
plt.close()

print(f"\nAll figures saved to {OUT}")
