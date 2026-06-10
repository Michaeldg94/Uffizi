"""The normal-tourist RL problem: selective routing under crowd uncertainty.

Unlike the art lover (sees everything, so the itinerary is forced), the normal
tourist has limited time (a 2-3 hour visit) and selective taste (recognition,
driven by importance, not the four Instagram magnets). As it moves through the
museum it must decide, room by room and under crowds it only discovers on
arrival, where to stop and look, where to walk straight through, and what to
skip outright, to get the most from a tight budget, then leave.

This subclass keeps the base env's navigation action (move to a neighbor or
stay) so the agent genuinely chooses its path (corridor vs rooms, skip a loop,
backtrack for the side rooms). What changes:

- Profile: Type B (recognition taste), via the base-env profile hook.
- Dwell: importance-driven and shorter than the art lover, with a skip
  threshold (below ``dwell_importance_floor`` a room earns no dwell and is
  walked straight through).
- Observation: the visitor has a MAP. It sees relative room sizes, the value
  map (which rooms are worth its time), and a per-move routing signal toward
  the nearest unseen valued room (so navigation is informed, not memorized
  blind, which is what made the art-lover navigation collapse). It does NOT see
  the global crowd; only the crowd in the current room and the rooms it can see
  into (its neighbours). Magnetism enters only through that crowd.
- Reward shaping: the tour pull is OFF, so the routing is genuinely learned
  from the map, not handed to the agent by a potential.
"""
from __future__ import annotations

import numpy as np

from uffizi_rl import config
from uffizi_rl.environment.uffizi_env import UffiziEnv
from uffizi_rl.environment.visitor_profiles import sample_type_b_profile

try:
    from gymnasium import spaces
except Exception:  # pragma: no cover
    spaces = None


class NormalTouristEnv(UffiziEnv):
    """Selective normal tourist with a map (see module docstring)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # --- Type B visitor, but with BROAD recognition taste ---
        # Type B's own importance vector is magnet-only (5 rooms: the Instagram
        # blockbusters), which gives a 50-minute checkbox visit. The normal
        # tourist recognizes the famous names broadly (Caravaggio, Giotto, ...),
        # which is the base room importance. So we keep Type B's crowd-tolerant
        # behavior but swap in base importance as what it values.
        self._recognition_vec = np.array(
            [config.ROOM_DATA[r]["importance"] for r in config.ROOM_IDS], dtype=float)

        def _normal_tourist_sampler(rng):
            p = sample_type_b_profile(rng)
            p.importance_vector = self._recognition_vec.copy()
            return p

        self._profile_sampler = _normal_tourist_sampler
        self.agent_profile = self._profile_sampler(self.rng)

        # --- normal-tourist dwell: shorter than the art lover, skip low rooms ---
        self.dwell_per_importance = 1.4     # imp 10 -> ~7 min, well under the art lover's ~30
        self.dwell_importance_floor = 5.0   # below importance 5: no dwell, walk straight through
        self.recog_threshold = 5.0          # rooms worth routing to (the recognition set)
        self.tour_weight = 0.0              # routing is LEARNED from the map, not shaped
        self.completion_k = 0.0

        # --- static map the visitor carries: relative room size, value map ---
        cap = np.array([config.ROOM_DATA[r]["capacity"] for r in config.ROOM_IDS], dtype=float)
        self._size_vec = (cap / cap.max()).astype(np.float32)   # relative size [0,1]

        # --- observation: 6*N + max_degree + 4 ---
        n = config.N_ROOMS
        self._obs_dim = 6 * n + self.max_degree + 4
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(self._obs_dim,), dtype=np.float32)

    # -- routing targets: recognized rooms not yet fully appreciated -----------
    def _valued_remaining(self) -> list[str]:
        iv = self.agent_profile.importance_vector
        out = []
        for r in config.ROOM_IDS:
            idx = config.ROOM_TO_IDX[r]
            imp = float(iv[idx])
            if imp < self.recog_threshold:
                continue
            ideal = max(1.0, self.dwell_per_importance * max(0.0, imp - self.dwell_importance_floor))
            if self._extracted.get(r, 0.0) < ideal:
                out.append(r)
        return out

    def _build_observation(self) -> np.ndarray:
        n = config.N_ROOMS
        cur = self.current_room
        cidx = config.ROOM_TO_IDX[cur]
        iv = self.agent_profile.importance_vector

        one_hot = np.zeros(n, dtype=np.float32)
        one_hot[cidx] = 1.0

        visited_bin = np.zeros(n, dtype=np.float32)
        for r in self.visited:
            visited_bin[config.ROOM_TO_IDX[r]] = 1.0

        progress = self._appreciation_progress().astype(np.float32)   # per-room drain
        value_map = np.clip(np.asarray(iv, dtype=np.float32) / 10.0, 0.0, 1.0)  # what I value
        size_map = self._size_vec                                     # relative sizes

        # Local crowd only: current room + the rooms I can see into (neighbours).
        dens_all = self._density_all_rooms()
        local_crowd = np.full(n, -1.0, dtype=np.float32)              # -1 = can't see it
        for r in [cur] + list(self._sorted_neighbors(cur)):
            j = config.ROOM_TO_IDX[r]
            local_crowd[j] = float(np.clip(dens_all[j], 0.0, 1.0))

        # Per-move routing signal: hop distance from each neighbour to the nearest
        # unseen valued room (this is the map telling me which way to walk). 1.0
        # (far) for invalid moves.
        remaining = self._valued_remaining()
        per_action = np.ones(self.max_degree, dtype=np.float32)
        neighbors = self._sorted_neighbors(cur)
        for i, nb in enumerate(neighbors):
            if remaining:
                d = min(self._distances.get(nb, {}).get(t, 99) for t in remaining)
                per_action[i] = min(1.0, d / 20.0)

        dist_next = 0.0
        if remaining:
            dist_next = min(1.0, min(self._distances.get(cur, {}).get(t, 99) for t in remaining) / 20.0)
        time_norm = self.time_elapsed / max(1, self.episode_minutes)
        t_remaining = self.episode_minutes - self.time_elapsed
        egress = float(np.clip((t_remaining - self._distance_to_exit(cur)) / max(1, self.episode_minutes), -1.0, 1.0))
        fatigue = self.fatigue

        return np.concatenate([
            one_hot, size_map, value_map, visited_bin, progress, local_crowd,
            per_action,
            np.array([dist_next, time_norm, egress, fatigue], dtype=np.float32),
        ]).astype(np.float32)
