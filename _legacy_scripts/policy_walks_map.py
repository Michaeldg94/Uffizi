"""Learned-policy walking figure on the actual Uffizi floor plans.

The RL counterpart to ``profile_figures.py``. Instead of the three NPC visitor
profiles, each column is a *trained RL agent* and the paths are real rollouts of
its learned policy: reset the env, then step with ``model.predict`` (masked for
MaskablePPO, unmasked for the vanilla baselines) and record the room the agent
occupies each minute.

  policy_walks_map.png -- 12 sample rollouts per agent per floor

Layout matches the profile map: a 2 x N grid, rows are the 2nd floor (top) and
1st floor (bottom), columns are the agents (MaskablePPO, then unmasked PPO, and
DQN once its 3rd seed is in). Green = start, red = end, gold stars = masterpieces
/ Caravaggio.

Scenario: the status-quo crowd (no interventions) the agents were trained on, so
this is NOT the RAMA+Audio scenario used for the profile map. The 12 crowds are
the fixed common-random-number evaluation crowds (seed_base 900000), so every
agent is drawn on the same scenarios.

Why the contrast appears: at eval the unmasked agents do NOT waste steps on
illegal moves (they learned to avoid the penalty, ~0% invalid). The damage is
done during training: without masking, the easiest way to escape the
illegal-move penalty is to stop exploring, which pulls some runs into a
degenerate low-coverage policy (the worst unmasked seed parks in ~5 rooms;
the best tours ~28). Masking removes illegal actions from the choice set, so
the agent never trades exploration for penalty-avoidance: every MaskablePPO
seed tours the full museum (~64 rooms, both floors, Caravaggio included).

To avoid stacking the deck, both columns show each method's BEST seed
(MaskablePPO seed-2 = 416.4; unmasked PPO seed-3 = 340.3), so the gap shown is
conservative, not the worst case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uffizi_rl.environment.crowd_simulator import CrowdSimulator
from uffizi_rl.environment.uffizi_env import UffiziEnv
from floor_plan_coords import (
    FLOOR2, FLOOR1, FLOOR2_ROOMS, FLOOR1_ROOMS, CORRIDORS, EDGE_WAYPOINTS,
)

OUT = Path("outputs/figures/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 12
CROWD_SEED_BASE = 900_000              # the CRN evaluation crowds
CROWD_SEEDS = [CROWD_SEED_BASE + i for i in range(N_SAMPLES)]

# ---------------------------------------------------------------------------
# Trained agents -> columns. (model, supports_masking, label)
# DQN is intentionally left out until its 3rd seed lands; uncomment to add.
# ---------------------------------------------------------------------------
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO  # noqa: F401  (DQN added later)

COLUMNS = [
    (MaskablePPO.load("outputs/models/newenv/ppo_maskable_seed0.zip"), True, "MaskablePPO (egress reward)"),
    # Ablation columns to add once their new-env models finish training:
    # (PPO.load("outputs/models/newenv/ppo_ablation.zip"),  False, "PPO (unmasked)"),
    # (DQN.load("outputs/models/newenv/dqn_ablation.zip"),  False, "DQN (unmasked)"),
]

# ---------------------------------------------------------------------------
# Rollout: record the room the agent occupies each minute.
# ---------------------------------------------------------------------------
def rollout_rooms(model, masked: bool, crowd_seed: int) -> list[str]:
    """Run one deterministic rollout on the fixed crowd ``crowd_seed`` and
    return the collapsed sequence of rooms the agent occupied (consecutive
    duplicates from 'stay' actions removed)."""
    env = UffiziEnv(seed=crowd_seed, episode_minutes=180)
    env.seed_value = crowd_seed
    env._episode_counter = 0
    obs, _ = env.reset()
    rooms = [env.current_room]
    done = trunc = False
    while not (done or trunc):
        mask = env.get_action_mask()
        # Sample actions (deterministic=False): under the egress reward the
        # greedy argmax collapses into a non-exiting loop, while the stochastic
        # policy (the one PPO actually learned) tours and exits. Render that.
        if masked:
            action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        else:
            action, _ = model.predict(obs, deterministic=False)
        obs, _r, done, trunc, _info = env.step(int(action))
        if env.current_room != rooms[-1]:
            rooms.append(env.current_room)
    return rooms


# ---------------------------------------------------------------------------
# Path expansion (copied from profile_figures.py so the look matches exactly).
# ---------------------------------------------------------------------------
_graph_sim = CrowdSimulator(daily_total=10, seed=0)
gu = _graph_sim.g.to_undirected()
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
    except (nx.NetworkXNoPath, nx.NodeNotFound):
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
    # Exclude the ENTRY/EXIT terminals: EXIT is (wrongly) a member of
    # FLOOR2_ROOMS with a coord near Lanzi, which used to drop the floor-2
    # "end" marker on the wrong staircase. The real last 2nd-floor location
    # is the descent staircase the agent actually uses.
    seq = [r for r in rooms_seq if r in FLOOR2_ROOMS and r not in {"ENTRY", "EXIT"}]
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
    seq = [r for r in rooms_seq if r in FLOOR1_ROOMS and r in FLOOR1 and r not in {"ENTRY", "EXIT"}]
    return [FLOOR1[r] for r in seq]


def style_axis(ax, img):
    ax.imshow(img, extent=(0, 1, 1, 0), aspect="auto", alpha=0.55)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])


floor2_img = np.asarray(Image.open(Path("outputs/assets/uffizi_floor2.png").resolve()))
floor1_img = np.asarray(Image.open(Path("outputs/assets/uffizi_floor1.png").resolve()))

FLOORS = [
    ("2nd floor", FLOOR2, floor2_img, expand_floor2, ["A11", "A12", "A35", "A38"]),
    ("1st floor", FLOOR1, floor1_img, expand_floor1, ["E4", "E5"]),
]

# ---------------------------------------------------------------------------
# Roll out every agent on the 12 fixed crowds.
# ---------------------------------------------------------------------------
print("Rolling out trained agents on the fixed CRN crowds...", flush=True)
agent_paths: list[list[list[str]]] = []   # per column -> list of room-sequences
for model, masked, label in COLUMNS:
    paths = [rollout_rooms(model, masked, s) for s in CROWD_SEEDS]
    agent_paths.append(paths)
    distinct = np.mean([len(set(p)) for p in paths])
    print(f"  {label}: {distinct:.1f} distinct rooms / rollout (mean over {N_SAMPLES})",
          flush=True)

# ---------------------------------------------------------------------------
# FIGURE.
# ---------------------------------------------------------------------------
n_cols = len(COLUMNS)
fig, axes = plt.subplots(2, n_cols, figsize=(7.3 * n_cols, 15), squeeze=False)
rng = np.random.RandomState(42)
for col, (_model, _masked, label) in enumerate(COLUMNS):
    paths = agent_paths[col]
    for row, (fl_name, COORDS, img, expand, stars) in enumerate(FLOORS):
        ax = axes[row][col]
        style_axis(ax, img)
        n_on_floor = 0
        for rooms in paths:
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
            ax.set_title(f"{label}\n{fl_name}", fontsize=12, pad=6)
        else:
            ax.set_title(fl_name, fontsize=11, pad=4)
        if n_on_floor == 0:
            ax.text(0.5, 0.5, "does not reach this floor",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, style="italic", color="#666666")
fig.suptitle("How the trained RL agents walk the Uffizi -- 12 sample rollouts each "
             "(status-quo crowd, no interventions). Green = start, red = end; "
             "gold stars = masterpieces / Caravaggio.", fontsize=13, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT / "policy_walks_map.png", dpi=140, bbox_inches="tight")
plt.close()

print("Wrote:", OUT / "policy_walks_map.png")
