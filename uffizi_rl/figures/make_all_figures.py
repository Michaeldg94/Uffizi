"""THE figure generator for the Uffizi RL project. One script, one folder.

Generates EVERY figure, numbered sequentially 01..N, into a single clean folder
(outputs/figures/), at a uniform high resolution (DPI 300, PNG + PDF). Replaces
the old scattered set (visualization.py + pipeline/07_make_figures.py +
scripts/{diagnostics,policy_walks_map,profile_figures,make_presentation_figures}.py
that fed three folders at ten different sizes).

Order of figures (the narrative):
  01 graph density            10 baseline comparison      19-26 floor-plan maps
  02 room capacities          11 algorithm comparison      (walks + heatmaps,
  03 arrival envelope         12 equilibrium convergence    art/tourist,
  04 visitors over day        13 interventions              intervened/baseline)
  05 density heatmap          14 top interventions        27 sweep: type-B welfare
  06 top congested            15 welfare vs revenue        28 sweep: type-B gini
  07 q-learning curve         16 portfolio breakdown       29 sweep: volume welfare
  08 episode lengths          17 booking grid              30 sweep: volume botticelli
  09 epsilon decay            18 book-early curve          31 sweep: heterogeneity

Run:  python scripts/make_all_figures.py
CPU-capped to 2 cores (some figures roll out trained policies; never peg the Mac).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]   # repo root (uffizi_rl/figures -> uffizi_rl -> root)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402
torch.set_num_threads(2)

from uffizi_rl import config  # noqa: E402
from uffizi_rl.analysis import visualization as V  # noqa: E402
from uffizi_rl.figures.floor_plan_coords import (  # noqa: E402
    FLOOR2, FLOOR1, FLOOR2_ROOMS, FLOOR1_ROOMS, CORRIDORS, EDGE_WAYPOINTS,
)

# ---------------------------------------------------------------------------
# Uniform style + sequential numbering into one clean folder.
# ---------------------------------------------------------------------------
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)
DPI = 300
OUT = Path("outputs")          # JSON / npy artifacts live here
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 13, "figure.titlesize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK = "#1F3A5F"; ACCENT = "#C0552B"; MUTED = "#6B7280"
PALETTE = ["#1F3A5F", "#C0552B", "#2E8B57", "#8E6FB3", "#D4A017"]

_N = [0]
def num(name: str) -> str:
    """Next sequential numbered stem, e.g. '07_q_learning_curve'."""
    _N[0] += 1
    return f"{_N[0]:02d}_{name}"

def pngpath(name: str) -> str:
    """Numbered .png path for the reusable plotters (they also emit .pdf)."""
    return str(FIG / f"{num(name)}.png")

def save(fig, name: str) -> None:
    """Save one of OUR figures (already numbered) as PNG + PDF at DPI 300."""
    stem = num(name)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  {stem}", flush=True)

def _json(name):
    p = OUT / name
    if not p.exists():
        print(f"  SKIP {name} (missing)"); return None
    return json.loads(p.read_text())


# ===========================================================================
# Embedded RL results: 5 training seeds {1,3,7,42,202606}, deterministic eval on
# the fixed CRN crowds 900000-5. Values are mean (and std) ACROSS the training
# seeds. Source: outputs/results_rl_booking_grandavg.json and
# outputs/results_toy_grandavg.json (scripts/average_seed_runs.py).
# ===========================================================================
CROWDS = (500, 2500, 5000)
CROWD_LABEL = {500: "quiet\n(500)", 2500: "busy\n(2500)", 5000: "packed\n(max)"}
# Booking headline: (PPO baseline, MaskablePPO intervened) points, mean then std.
BOOKING = {
    "art":     {500: (5112, 7113), 2500: (4780, 6167), 5000: (4010, 4800)},
    "tourist": {500: (2694, 3686), 2500: (2708, 3695), 5000: (2680, 3685)},
}
BOOKING_STD = {
    "art":     {500: (0, 0), 2500: (0, 291), 5000: (0, 1891)},
    "tourist": {500: (0, 0), 2500: (0, 0), 5000: (0, 0)},
}
# Dominant MaskablePPO lead-days across the 5 seeds.
LEAD = {"art": {500: 7, 2500: 35, 5000: 35}, "tourist": {500: 7, 2500: 7, 5000: 21}}
ALGO = {
    "art": {
        500:  {"PPO-base": 5112, "DQN-base": 4527, "PPO-int": 7113, "MaskPPO-int": 7113, "DQN-int": 4872},
        2500: {"PPO-base": 4780, "DQN-base": 3986, "PPO-int": 5646, "MaskPPO-int": 6167, "DQN-int": 5190},
        5000: {"PPO-base": 4010, "DQN-base": 2044, "PPO-int": 4832, "MaskPPO-int": 4800, "DQN-int": 2691},
    },
    "tourist": {
        500:  {"PPO-base": 2694, "DQN-base": 2694, "PPO-int": 3686, "MaskPPO-int": 3686, "DQN-int": 3686},
        2500: {"PPO-base": 2708, "DQN-base": 2708, "PPO-int": 3695, "MaskPPO-int": 3695, "DQN-int": 3694},
        5000: {"PPO-base": 2680, "DQN-base": 2680, "PPO-int": 2093, "MaskPPO-int": 3685, "DQN-int": 2900},
    },
}
ALGO_STD = {
    "art": {
        500:  {"PPO-base": 0, "DQN-base": 1596, "PPO-int": 0, "MaskPPO-int": 0, "DQN-int": 2587},
        2500: {"PPO-base": 0, "DQN-base": 225, "PPO-int": 0, "MaskPPO-int": 291, "DQN-int": 1115},
        5000: {"PPO-base": 0, "DQN-base": 1254, "PPO-int": 0, "MaskPPO-int": 1891, "DQN-int": 2154},
    },
    "tourist": {
        500:  {"PPO-base": 0, "DQN-base": 0, "PPO-int": 0, "MaskPPO-int": 0, "DQN-int": 0},
        2500: {"PPO-base": 0, "DQN-base": 0, "PPO-int": 0, "MaskPPO-int": 0, "DQN-int": 0},
        5000: {"PPO-base": 0, "DQN-base": 0, "PPO-int": 0, "MaskPPO-int": 0, "DQN-int": 1107},
    },
}
ALGO_ORDER = ["PPO-base", "DQN-base", "PPO-int", "MaskPPO-int", "DQN-int"]
ALGO_COLOR = {"PPO-base": "#B8C4D9", "DQN-base": "#9AA7BD", "PPO-int": "#C0552B",
              "MaskPPO-int": "#1F3A5F", "DQN-int": "#2E8B57"}
PROFILE_LABEL = {"art": "Art lover", "tourist": "Normal tourist"}
# Toy tabular results: (mean, std) over the 5 seeds. Deterministic baselines have
# std 0; only the stochastic random policy varies.
TOY_Q = (83.3, 3.8)
TOY_DQ = (83.1, 3.7)
TOY_BL = [("default_path", -61.4, 0.0), ("greedy_value_ratio", -72.0, 0.0),
          ("greedy_least_crowded", -43.1, 0.0), ("peak_avoidance", -33.7, 0.0),
          ("random", 35.4, 3.7)]


# ===========================================================================
# GROUP A: environment / crowd  (reuses V plotters + saved artifacts)
# ===========================================================================
def g_world():
    from uffizi_rl.environment.crowd_simulator import CrowdSimulator
    from uffizi_rl.environment.museum_graph import build_uffizi_graph
    sim = CrowdSimulator(daily_total=config.DAILY_VISITORS_NORMAL, seed=config.DEFAULT_SEED + 100)
    sim.run_day()
    V.plot_graph_density(build_uffizi_graph(), sim.current_density_dict(), pngpath("graph_density"))
    V.plot_room_capacity_distribution([config.ROOM_DATA[r] for r in config.ROOM_IDS], pngpath("room_capacities"))
    V.plot_arrival_envelope(pngpath("arrival_envelope"))
    mpath = OUT / "01_density_matrix.npy"
    if mpath.exists():
        m = np.load(mpath)
        caps = [config.ROOM_DATA[r]["capacity"] for r in config.ROOM_IDS]
        V.plot_visitors_inside_over_day(m, caps, pngpath("visitors_over_day"))
        V.plot_density_heatmap(m, pngpath("density_heatmap"), room_labels=config.ROOM_IDS)
        means = m.mean(axis=0)
        pairs = sorted(((config.ROOM_IDS[i], float(means[i])) for i in range(len(config.ROOM_IDS))),
                       key=lambda kv: kv[1], reverse=True)
        V.plot_top_congested_rooms(pairs, pngpath("top_congested"), top_n=15)


# ===========================================================================
# GROUP A2: museum-wide (crowd-simulator stream) -- the director gate
# ===========================================================================
# As-is crowd (5000), clean 5-seed totals: status quo / all-11 / optimized portfolio.
MUSEUM = {
    "status quo":     {"welfare": 310.8, "revenue": 95913},
    "all 11":         {"welfare": 407.6, "revenue": 94809},
    "opt. portfolio": {"welfare": 361.3, "revenue": 108702},
}
MUSEUM_COLOR = {"status quo": "#9AA7BD", "all 11": "#1F3A5F", "opt. portfolio": "#C0552B"}
SEG_LABEL_MW = {"instagram": "Instagram\ntourist", "standard": "Normal\ntourist", "art_lover": "Art\nlover"}


def g_museum():
    # director gate: welfare + revenue, status quo vs all-11 vs optimized portfolio
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), squeeze=False)
    names = list(MUSEUM); x = np.arange(len(names))
    for j, metric in enumerate(("welfare", "revenue")):
        ax = axes[0][j]
        vals = [MUSEUM[n][metric] for n in names]
        base = MUSEUM["status quo"][metric]
        ax.bar(x, vals, color=[MUSEUM_COLOR[n] for n in names], edgecolor="white")
        for i, n in enumerate(names):
            if i > 0:
                ax.text(x[i], vals[i], f"{100*(vals[i]-base)/base:+.0f}%", ha="center", va="bottom",
                        fontsize=10, color=MUSEUM_COLOR[n])
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("welfare / visitor" if metric == "welfare" else "revenue (EUR/day)")
        ax.set_title(metric.capitalize()); ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Museum-wide director gate (as-is crowd): all 11 lift welfare with revenue preserved; "
                 "the optimized subset raises both", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "museum_director_gate")

    # per-segment welfare per visitor + throughput, status quo vs intervened (as-is crowd)
    mw = _json("12_museum_wide.json")
    if mw and "intervened_5000" in mw.get("cells", {}):
        b = mw["cells"]["base_5000"]; iv = mw["cells"]["intervened_5000"]
        segs = ["instagram", "standard", "art_lover"]; xs = np.arange(len(segs)); w = 0.38
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), squeeze=False)
        for j, (key, ylab, title) in enumerate([("welfare_mean", "welfare per visitor", "Experience per visitor"),
                                                 ("completed", "visitors completed", "Throughput")]):
            ax = axes[0][j]
            ax.bar(xs - w/2, [b[s][key] for s in segs], w, label="status quo", color="#9AA7BD", edgecolor="white")
            ax.bar(xs + w/2, [iv[s][key] for s in segs], w, label="intervened", color=ACCENT, edgecolor="white")
            ax.set_xticks(xs); ax.set_xticklabels([SEG_LABEL_MW[s] for s in segs], fontsize=9)
            ax.set_ylabel(ylab); ax.set_title(title); ax.grid(axis="y", alpha=0.25)
            if j == 0:
                ax.legend(fontsize=9)
        fig.suptitle("Museum-wide by segment (as-is crowd): normal tourists gain most per visit; "
                     "more Instagram tourists and art lovers get through", y=1.02, fontsize=12)
        fig.tight_layout()
        save(fig, "museum_per_segment")


# ===========================================================================
# GROUP B: tabular learning  (02_q_learning.json)
# ===========================================================================
def g_learning():
    q = _json("02_q_learning.json")
    if q:
        if q.get("episode_rewards"):
            V.plot_learning_curve(q["episode_rewards"], pngpath("q_learning_curve"),
                                  title="Tabular Q-Learning")
        if q.get("offpeak_botticelli_visits"):
            V.plot_offpeak_botticelli(q["offpeak_botticelli_visits"], pngpath("offpeak_botticelli"))
        if q.get("episode_lengths"):
            V.plot_episode_length_evolution(q["episode_lengths"], pngpath("episode_lengths"))
        V.plot_epsilon_decay(pngpath("epsilon_decay"))
        # baseline comparison: 5-seed mean +/- std (the deterministic baselines have
        # std 0; only learned Q / Double-Q and the random policy carry variance).
        names = ["Q-learning", "Double-Q"] + [b[0] for b in TOY_BL]
        means = [TOY_Q[0], TOY_DQ[0]] + [b[1] for b in TOY_BL]
        errs = [TOY_Q[1], TOY_DQ[1]] + [b[2] for b in TOY_BL]
        cols = [ACCENT, "#1F3A5F"] + ["#9AA7BD"] * len(TOY_BL)
        figb, axb = plt.subplots(figsize=(8.4, 4.4))
        xb = np.arange(len(names))
        axb.bar(xb, means, 0.66, yerr=errs, color=cols, edgecolor="white", capsize=3,
                error_kw={"elinewidth": 0.9, "ecolor": "#444"})
        axb.axhline(0, color="#888", lw=0.8)
        axb.set_xticks(xb); axb.set_xticklabels(names, rotation=28, ha="right", fontsize=11)
        axb.set_ylabel("mean episode return"); axb.grid(axis="y", alpha=0.25)
        axb.set_title("Toy: tabular Q-learning vs handcrafted baselines (5 seeds, mean $\\pm$ std)")
        figb.tight_layout()
        save(figb, "baseline_comparison")


# ===========================================================================
# GROUP C: deep RL  (masking ablation on the itinerary task) + the fresh
# 30-cell RAMA algorithm matrix.
# ===========================================================================
def g_algo():
    dr = _json("03_deep_rl.json")
    if dr:
        ppo = dr.get("maskable_ppo", {})
        V.plot_deep_rl_comparison(ppo.get("mean_episode_reward", 0.0), ppo.get("std_episode_reward", 0.0),
                                  dr.get("ablations", []), pngpath("deep_rl_masking_ablation"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), squeeze=False)
    x = np.arange(len(CROWDS)); w = 0.16
    for j, p in enumerate(("art", "tourist")):
        ax = axes[0][j]
        for k, alg in enumerate(ALGO_ORDER):
            vals = [ALGO[p][c][alg] for c in CROWDS]
            errs = [ALGO_STD[p][c][alg] for c in CROWDS]
            ax.bar(x + (k - 2) * w, vals, w, yerr=errs, label=alg, color=ALGO_COLOR[alg],
                   edgecolor="white", linewidth=0.4, capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": "#444"})
        ax.set_xticks(x); ax.set_xticklabels([CROWD_LABEL[c] for c in CROWDS])
        ax.set_title(PROFILE_LABEL[p]); ax.set_ylabel("points (deterministic eval)")
        ax.grid(axis="y", alpha=0.25)
        if j == 1:
            ax.legend(fontsize=11, ncol=1, loc="upper right", framealpha=0.9)
    fig.suptitle("Algorithm comparison: value-based (DQN) vs policy-gradient (PPO / MaskablePPO), "
                 "baseline vs intervened (5 seeds, mean $\\pm$ std)", y=1.02)
    fig.tight_layout()
    save(fig, "algorithm_comparison")


# ===========================================================================
# GROUP D: equilibrium  (04_equilibrium.json)
# ===========================================================================
def g_equilibrium():
    eq = _json("04_equilibrium.json")
    if eq and eq.get("iterations"):
        V.plot_equilibrium_convergence(eq["iterations"], pngpath("equilibrium_convergence"))


# ===========================================================================
# GROUP E: interventions  (05_interventions.json)
# ===========================================================================
def g_interventions():
    iv = _json("05_interventions.json")
    if not iv:
        return
    table = iv.get("table", [])
    if table:
        V.plot_intervention_comparison(table, pngpath("interventions"))
        V.plot_top_interventions(table, pngpath("top_interventions"), top_n=10)
        V.plot_intervention_welfare_revenue(table, pngpath("welfare_vs_revenue"))
    portfolio = iv.get("portfolio")
    if portfolio:
        bw = float(iv.get("baseline_welfare", table[0]["total_welfare"] if table else 0))
        br = float(iv.get("baseline_revenue", table[0].get("revenue", 0) if table else 0))
        V.plot_portfolio_breakdown(portfolio, bw, br, pngpath("portfolio_breakdown"))


# ===========================================================================
# GROUP F: RAMA booking centerpiece  (NEW)
# ===========================================================================
def g_booking():
    # booking grid: baseline vs intervened, per crowd, per profile
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), squeeze=False)
    x = np.arange(len(CROWDS)); w = 0.36
    for j, p in enumerate(("art", "tourist")):
        ax = axes[0][j]
        base = [BOOKING[p][c][0] for c in CROWDS]
        intv = [BOOKING[p][c][1] for c in CROWDS]
        base_e = [BOOKING_STD[p][c][0] for c in CROWDS]
        intv_e = [BOOKING_STD[p][c][1] for c in CROWDS]
        ax.bar(x - w / 2, base, w, yerr=base_e, label="no interventions", color="#9AA7BD",
               edgecolor="white", capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "#444"})
        ax.bar(x + w / 2, intv, w, yerr=intv_e, label="RAMA + interventions", color=ACCENT,
               edgecolor="white", capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "#444"})
        for i, c in enumerate(CROWDS):
            pct = 100 * (intv[i] - base[i]) / base[i]
            ax.text(x[i] + w / 2, intv[i] + intv_e[i], f"+{pct:.0f}%", ha="center", va="bottom",
                    fontsize=12, color=ACCENT)
        ax.set_xticks(x); ax.set_xticklabels([CROWD_LABEL[c] for c in CROWDS])
        ax.set_title(PROFILE_LABEL[p]); ax.set_ylabel("points"); ax.grid(axis="y", alpha=0.25)
        panel_max = max(max(intv[i] + intv_e[i], base[i] + base_e[i]) for i in range(len(CROWDS)))
        ax.set_ylim(0, panel_max * 1.30)
        if j == 0:
            ax.legend(fontsize=11, loc="upper right")
    fig.suptitle("RAMA booking: intervened vs matched baseline (5 seeds, mean $\\pm$ std; "
                 "3/3 masterpieces secured)", y=1.02)
    fig.tight_layout()
    save(fig, "booking_grid")

    # book-early curve: lead days chosen vs crowd
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for p, col in (("art", INK), ("tourist", ACCENT)):
        ax.plot(range(len(CROWDS)), [LEAD[p][c] for c in CROWDS], "-o", color=col,
                linewidth=2.2, markersize=8, label=PROFILE_LABEL[p])
    ax.set_xticks(range(len(CROWDS))); ax.set_xticklabels([CROWD_LABEL[c] for c in CROWDS])
    ax.set_yticks([1, 7, 21, 35]); ax.set_ylabel("lead time chosen (days ahead)")
    ax.set_title("Book-early-when-busier emerges from learning")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    save(fig, "book_early_curve")


# ===========================================================================
# GROUP G: floor-plan maps  (NEW; agent walks + visit-time heatmaps)
# ===========================================================================
PROFILES = ("art", "tourist")
CONDITIONS = ("intervened", "baseline")
COND_LABEL = {"intervened": "RAMA + interventions", "baseline": "no interventions"}
MAP_CROWD_LABEL = {500: "quiet (500)", 2500: "busy (2500)", 5000: "packed (max)"}
N_PATHS = 8
PATH_SEEDS = [900000 + i for i in range(N_PATHS)]
STARS = {"floor2": ["A11", "A12", "A35", "A38"], "floor1": ["E4", "E5"]}


def _graph_sp():
    from uffizi_rl.environment.crowd_simulator import CrowdSimulator
    import networkx as nx
    g = CrowdSimulator(daily_total=10, seed=0).g.to_undirected()
    cache = {}
    def sp(a, b):
        if (a, b) in cache:
            return cache[(a, b)]
        if a == b:
            cache[(a, b)] = [a]; return [a]
        try:
            pth = nx.shortest_path(g, a, b)
        except Exception:
            pth = [a, b]
        cache[(a, b)] = pth; return pth
    return sp


def _wp(a, b):
    if (a, b) in EDGE_WAYPOINTS:
        return list(EDGE_WAYPOINTS[(a, b)])
    if a in FLOOR2 and b in FLOOR2:
        ax, ay = FLOOR2[a]; bx, by = FLOOR2[b]
        if b in CORRIDORS:
            o, f = CORRIDORS[b]; return [(ax, f)] if o == "h" else [(f, ay)]
        if a in CORRIDORS:
            o, f = CORRIDORS[a]; return [(bx, f)] if o == "h" else [(f, by)]
    return []


def _exp2(seq, sp):
    seq = [r for r in seq if r in FLOOR2_ROOMS and r not in {"ENTRY", "EXIT"}]
    if not seq:
        return []
    exp = [seq[0]]
    for p, c in zip(seq[:-1], seq[1:]):
        walk = [r for r in sp(p, c) if r in FLOOR2_ROOMS]
        exp.extend(walk[1:] if walk[:1] == [p] else walk)
    coords = [FLOOR2[exp[0]]]
    for p, c in zip(exp[:-1], exp[1:]):
        coords.extend(_wp(p, c)); coords.append(FLOOR2[c])
    return coords


def _exp1(seq):
    return [FLOOR1[r] for r in seq if r in FLOOR1_ROOMS and r in FLOOR1 and r not in {"ENTRY", "EXIT"}]


def _floor_of(r):
    """2 = second floor only, 1 = first floor only, 0 = staircase (both)."""
    f2, f1 = r in FLOOR2_ROOMS, r in FLOOR1_ROOMS
    if f2 and not f1:
        return 2
    if f1 and not f2:
        return 1
    return 0


def _endpoints(seq):
    """A visit is ONE continuous walk. Return (global_start, global_end,
    last_floor2_room, first_floor1_room): the true start, the true end, and the
    floor-change point, so each is marked exactly once."""
    g = [r for r in seq if r not in {"ENTRY", "EXIT"} and (r in FLOOR2_ROOMS or r in FLOOR1_ROOMS)]
    if not g:
        return None
    pure2 = [r for r in g if _floor_of(r) == 2]
    pure1 = [r for r in g if _floor_of(r) == 1]
    return g[0], g[-1], (pure2[-1] if pure2 else None), (pure1[0] if pure1 else None)


def _ax_img(ax, img):
    ax.imshow(img, extent=(0, 1, 1, 0), aspect="auto", alpha=0.55)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_xticks([]); ax.set_yticks([])


def _rollout(profile, crowd, condition, seed, models, IV, priors):
    if condition == "intervened":
        from uffizi_rl.environment.rama_art_lover_env import RamaArtLoverEnv
        from uffizi_rl.environment.rama_tourist_env import RamaTouristEnv
        EnvCls = RamaArtLoverEnv if profile == "art" else RamaTouristEnv
        env = EnvCls(seed=seed, daily_total=crowd, interventions=IV, random_start=False, compute_plan=False)
        env._learned_arrival = list(priors[(profile, crowd)]); env._arr_alpha = 0.0
        env.seed_value = seed; env._episode_counter = 0
        model = models[("intervened", profile, crowd)]; masked = True
    else:
        from uffizi_rl.environment.planned_route_env import PlannedRouteEnv
        from uffizi_rl.environment.visitor_profiles import sample_type_b_profile
        env = PlannedRouteEnv(seed=seed, episode_minutes=config.MUSEUM_OPEN_MINUTES,
                              daily_total=crowd, random_start=False)
        if profile == "tourist":
            base = np.array([config.ROOM_DATA[r]["importance"] for r in config.ROOM_IDS], dtype=float)
            recog = base.copy()
            for i, r in enumerate(config.ROOM_IDS):
                if r.startswith("A") or r in {"E4", "E5", "E7"}:
                    continue
                recog[i] = 7.0 if r in {"C6", "C7", "C8", "C9", "C10", "C11"} else 2.0
            def _s(rng):
                pr = sample_type_b_profile(rng); pr.importance_vector = recog.copy(); return pr
            env.inner.art_crowd_alpha = 1.0
            env.inner._profile_sampler = _s
            env.inner.agent_profile = _s(env.inner.rng)
            env.inner.dwell_per_importance = 1.4; env.inner.dwell_importance_floor = 5.0
        env.seed_value = seed; env._episode_counter = 0
        model = models[("baseline", profile, crowd)]; masked = False
    obs, _ = env.reset()
    inner = env.inner
    seq = []; dwell = Counter(); in_walk = (condition == "baseline")
    last_t = inner.time_elapsed; last_room = inner.current_room
    if in_walk:
        seq.append(inner.current_room)
    d = t = False
    while not (d or t):
        if masked:
            a, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, r, d, t, _ = env.step(int(a))
        if condition == "intervened" and env._phase != "walk":
            continue
        if not in_walk:
            in_walk = True; last_t = inner.time_elapsed; last_room = inner.current_room
            seq.append(inner.current_room); continue
        cur = inner.current_room; now = inner.time_elapsed
        dwell[last_room] += max(0, now - last_t); last_t = now; last_room = cur
        if cur != seq[-1]:
            seq.append(cur)
    return seq, dwell


def g_maps():
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    from uffizi_rl.analysis.portfolio import combined_intervention_kwargs
    from uffizi_rl.interventions.intervention_config import InterventionConfig
    from uffizi_rl.environment.rama_art_lover_env import RamaArtLoverEnv
    from uffizi_rl.environment.rama_tourist_env import RamaTouristEnv
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    IV = InterventionConfig.from_kwargs(**combined_intervention_kwargs())
    SEED = config.DEFAULT_SEED
    models = {}; priors = {}
    for p in PROFILES:
        EnvCls = RamaArtLoverEnv if p == "art" else RamaTouristEnv
        bstem = "opt_ArtWalk" if p == "art" else "opt_TouristWalk"
        for c in CROWDS:
            models[("intervened", p, c)] = MaskablePPO.load(f"outputs/models/newenv/ramabook_{p}_{c}.zip")
            models[("baseline", p, c)] = PPO.load(f"outputs/models/newenv/{bstem}_{c}.zip")
            priors[(p, c)] = list(EnvCls(seed=SEED, daily_total=c, interventions=IV, random_start=False)._learned_arrival)

    sp = _graph_sp()
    f2 = np.asarray(Image.open("outputs/assets/uffizi_floor2.png"))
    f1 = np.asarray(Image.open("outputs/assets/uffizi_floor1.png"))
    FLOORS = [("2nd floor", f2, "floor2"), ("1st floor", f1, "floor1")]
    PANEL = (6.8, 4.3)

    runs = {}
    for cond in CONDITIONS:
        for p in PROFILES:
            for c in CROWDS:
                runs[(p, c, cond)] = [_rollout(p, c, cond, s, models, IV, priors) for s in PATH_SEEDS]
            print(f"    rolled out {p} {cond}", flush=True)

    # walks: one figure per (condition, profile) => 4 figures, 6 panels each.
    # A visit is ONE continuous walk: ONE green start + ONE red end, plus a grey
    # 'stairs' diamond where it descends from floor 2 to floor 1 (no phantom
    # per-floor endpoints).
    from matplotlib.lines import Line2D
    rng = np.random.RandomState(42)
    for cond in CONDITIONS:
        for p in PROFILES:
            fig, axes = plt.subplots(2, 3, figsize=(PANEL[0] * 3, PANEL[1] * 2), squeeze=False)
            for row, (fl, img, fk) in enumerate(FLOORS):
                COORDS = FLOOR2 if fk == "floor2" else FLOOR1
                on2 = (fk == "floor2")
                for col, c in enumerate(CROWDS):
                    ax = axes[row][col]; _ax_img(ax, img); non = 0
                    for seq, _dw in runs[(p, c, cond)]:
                        coords = _exp2(seq, sp) if on2 else _exp1(seq)
                        if len(coords) < 2:
                            continue
                        non += 1
                        pts = np.array(coords) + rng.normal(scale=0.004, size=(len(coords), 2))
                        ax.plot(pts[:, 0], pts[:, 1], "-", color=INK, alpha=0.45, linewidth=1.7)
                    ep = _endpoints(runs[(p, c, cond)][0][0])
                    if ep:
                        start, end, last2, first1 = ep
                        g_args = dict(marker="o", color="#2E8B57", markersize=14, markeredgecolor="white", markeredgewidth=1.3, zorder=6, linestyle="None")
                        r_args = dict(marker="s", color="#B22222", markersize=13, markeredgecolor="white", markeredgewidth=1.3, zorder=6, linestyle="None")
                        d_args = dict(marker="D", color="#555555", markersize=11, markeredgecolor="white", markeredgewidth=1.1, zorder=6, linestyle="None")
                        if on2:
                            if _floor_of(start) == 2 and start in COORDS:
                                ax.plot(*COORDS[start], **g_args)
                            if _floor_of(end) == 2 and end in COORDS:
                                ax.plot(*COORDS[end], **r_args)
                            elif last2 and first1 and last2 in COORDS:   # descends here
                                ax.plot(*COORDS[last2], **d_args)
                        else:
                            if first1 and _floor_of(start) != 1 and first1 in COORDS:  # arrives here
                                ax.plot(*COORDS[first1], **d_args)
                            if _floor_of(start) == 1 and start in COORDS:
                                ax.plot(*COORDS[start], **g_args)
                            if _floor_of(end) == 1 and end in COORDS:
                                ax.plot(*COORDS[end], **r_args)
                    for s_ in STARS[fk]:
                        if s_ in COORDS:
                            ax.plot(*COORDS[s_], marker="*", color="#D4A017", markersize=23,
                                    markeredgecolor="black", markeredgewidth=0.9, zorder=5, linestyle="None")
                    if row == 0:
                        ax.set_title(MAP_CROWD_LABEL[c], fontsize=14)
                    if col == 0:
                        ax.set_ylabel(fl, fontsize=14, fontweight="bold")
                    if non == 0:
                        ax.text(0.5, 0.5, "does not reach this floor", transform=ax.transAxes,
                                ha="center", va="center", fontsize=13, style="italic", color=MUTED)
            handles = [
                Line2D([], [], linestyle="None", marker="o", markerfacecolor="#2E8B57", markeredgecolor="white", markersize=12, label="visit start"),
                Line2D([], [], linestyle="None", marker="s", markerfacecolor="#B22222", markeredgecolor="white", markersize=11, label="visit end"),
                Line2D([], [], linestyle="None", marker="D", markerfacecolor="#555555", markeredgecolor="white", markersize=10, label="stairs (floor change)"),
                Line2D([], [], linestyle="None", marker="*", markerfacecolor="#D4A017", markeredgecolor="black", markersize=16, label="masterpieces / Caravaggio"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.01))
            fig.suptitle(f"{PROFILE_LABEL[p]} walking routes ({COND_LABEL[cond]}) -- one continuous visit, "
                         f"{N_PATHS} rollouts/panel; it descends floor 2 -> floor 1.", y=1.0, fontsize=15)
            fig.tight_layout(rect=[0.02, 0.04, 1, 0.95])
            save(fig, f"walks_{p}_{cond}")

    # heatmaps: one figure per (condition, profile), shared colour scale per figure
    for cond in CONDITIONS:
        for p in PROFILES:
            fig, axes = plt.subplots(2, 3, figsize=(PANEL[0] * 3, PANEL[1] * 2), squeeze=False)
            pv = {}; gmax = 0.0
            for row, (fl, img, fk) in enumerate(FLOORS):
                COORDS = FLOOR2 if fk == "floor2" else FLOOR1
                for col, c in enumerate(CROWDS):
                    agg = Counter()
                    for _seq, dw in runs[(p, c, cond)]:
                        for room, mnt in dw.items():
                            agg[room] += mnt
                    vals = {r: agg.get(r, 0) / N_PATHS for r in COORDS if r not in {"ENTRY", "EXIT"}}
                    pv[(row, col)] = vals
                    if vals:
                        gmax = max(gmax, max(vals.values()))
            norm = Normalize(0, gmax or 1.0)
            for row, (fl, img, fk) in enumerate(FLOORS):
                COORDS = FLOOR2 if fk == "floor2" else FLOOR1
                for col, c in enumerate(CROWDS):
                    ax = axes[row][col]; _ax_img(ax, img); drew = False
                    for r, v in pv[(row, col)].items():
                        if v <= 0:
                            continue
                        drew = True
                        ax.scatter(COORDS[r][0], COORDS[r][1], s=180, c=[plt.cm.YlOrRd(norm(v))],
                                   edgecolor="black", linewidth=0.5, zorder=3)
                    for s_ in STARS[fk]:
                        if s_ in COORDS:
                            ax.plot(COORDS[s_][0], COORDS[s_][1], "*", color="#2C5282",
                                    markersize=18, markeredgecolor="white", markeredgewidth=0.8, zorder=4)
                    if row == 0:
                        ax.set_title(MAP_CROWD_LABEL[c], fontsize=14)
                    if col == 0:
                        ax.set_ylabel(fl, fontsize=14, fontweight="bold")
                    if not drew:
                        ax.text(0.5, 0.5, "does not reach this floor", transform=ax.transAxes,
                                ha="center", va="center", fontsize=13, style="italic", color=MUTED)
            sm = ScalarMappable(cmap="YlOrRd", norm=norm); sm.set_array([])
            cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
            cb.set_label("mean minutes in room (per visit)", fontsize=9)
            fig.suptitle(f"{PROFILE_LABEL[p]} -- where time is spent ({COND_LABEL[cond]}). "
                         f"Darker = more minutes. Blue stars = masterpieces/Caravaggio.", y=1.0)
            save(fig, f"heatmap_{p}_{cond}")


# ===========================================================================
# GROUP H: population sweeps  (06_sweeps.json)
# ===========================================================================
def g_sweeps():
    sw = _json("06_sweeps.json")
    if not sw:
        return
    if sw.get("type_b_ratio_sweep"):
        V.plot_phase_transition(sw["type_b_ratio_sweep"], x_key="type_b_ratio",
                                out_path=pngpath("sweep_typeb_welfare"))
        V.plot_sweep_with_band(sw["type_b_ratio_sweep"], x_key="type_b_ratio", y_key="occupancy_gini",
                               y_label="Occupancy Gini", title="Spatial Inequality vs Type-B Fraction",
                               out_path=pngpath("sweep_typeb_gini"))
    if sw.get("volume_sweep"):
        V.plot_sweep_with_band(sw["volume_sweep"], x_key="daily_total", y_key="total_welfare",
                               y_label="Total Welfare", title="Welfare vs Daily Volume",
                               out_path=pngpath("sweep_volume_welfare"))
        V.plot_sweep_with_band(sw["volume_sweep"], x_key="daily_total", y_key="peak_botticelli_density",
                               y_label="Peak Botticelli Density", title="Botticelli Congestion vs Volume",
                               out_path=pngpath("sweep_volume_botticelli"))
    if sw.get("heterogeneity_sweep"):
        V.plot_sweep_with_band(sw["heterogeneity_sweep"], x_key="heterogeneity_scale", y_key="total_welfare",
                               y_label="Total Welfare", title="Welfare vs Preference Heterogeneity",
                               out_path=pngpath("sweep_heterogeneity"))


# (name, fn, figure_count) in display order -> drives sequential numbering.
GROUP_SEQ = [("world", g_world, 6), ("museum", g_museum, 2), ("learning", g_learning, 5), ("algorithm", g_algo, 2),
             ("equilibrium", g_equilibrium, 1), ("interventions", g_interventions, 4),
             ("booking", g_booking, 2), ("maps", g_maps, 8), ("sweeps", g_sweeps, 5)]


def main():
    starts = {}; acc = 0
    for name, _fn, cnt in GROUP_SEQ:
        starts[name] = acc; acc += cnt
    sel = sys.argv[1:]
    if not sel:
        for f in list(FIG.glob("*.png")) + list(FIG.glob("*.pdf")):
            f.unlink()
        print("Generating all figures -> outputs/figures/ (1..N, DPI 300)\n", flush=True)
        for name, fn, _cnt in GROUP_SEQ:
            print(f"[{name}]", flush=True); fn()
        print(f"\nDONE: {_N[0]} figures in outputs/figures/", flush=True)
        return
    # selective: regenerate only chosen groups, keeping their real numbers
    for name, fn, cnt in GROUP_SEQ:
        if name not in sel:
            continue
        lo, hi = starts[name] + 1, starts[name] + cnt
        for f in list(FIG.glob("*.png")) + list(FIG.glob("*.pdf")):
            head = f.name[:2]
            if head.isdigit() and lo <= int(head) <= hi:
                f.unlink()
        _N[0] = starts[name]
        print(f"[{name}] (figures {lo:02d}-{hi:02d})", flush=True); fn()
    print("\nDONE (selective)", flush=True)


if __name__ == "__main__":
    main()
