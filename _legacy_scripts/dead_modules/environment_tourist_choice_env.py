"""The normal tourist as a value-allocation problem (navigation is free).

There is no navigation problem: the visitor has a map and knows how to reach any
room. The real decision, every minute, is one of:

    0            ENGAGE: spend another minute on the room I'm in (go deeper:
                 linger, read the labels, run the audio guide)
    1 .. K       HEAD FOR valued room k (the staff/map walk me there; one step
                 of the shortest path is taken automatically)
    K+1          LEAVE: head for the exit

So the agent never learns how to walk; it learns WHAT to see, in WHAT ORDER, and
how DEEP to go, to extract the most value from a limited number of hours, while
adapting to crowds it can only see once it arrives (it observes the crowd in its
current room and the rooms it can see into, not across the museum).

This is time-budgeted value collection on a known graph (the orienteering
problem, NP-hard even with no crowds) plus adaptation to a partially observed,
stochastic crowd. That is a legitimate RL problem; the routing never was.

Wraps an inner ``UffiziEnv`` (graph, crowd simulator, reward) configured as the
normal tourist: Type B behaviour but BROAD recognition taste (base room
importance, so Caravaggio and Giotto count, not just the Instagram magnets),
short dwells, and a skip threshold.
"""
from __future__ import annotations

import numpy as np

from uffizi_rl import config
from uffizi_rl.environment.uffizi_env import UffiziEnv
from uffizi_rl.environment.visitor_profiles import sample_type_b_profile

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except Exception:  # pragma: no cover
    gym = None
    spaces = None
    _HAVE_GYM = False


class TouristChoiceEnv(gym.Env if _HAVE_GYM else object):
    """Value-allocation formulation of the normal tourist (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(self, recog_threshold: float = 5.0, **inner_kwargs):
        super().__init__()
        self.inner = UffiziEnv(**inner_kwargs)

        # Configure the inner env as the normal tourist.
        base_imp = np.array(
            [config.ROOM_DATA[r]["importance"] for r in config.ROOM_IDS], dtype=float)
        self._recognition_vec = base_imp

        def _normal_tourist_sampler(rng):
            p = sample_type_b_profile(rng)
            p.importance_vector = base_imp.copy()
            return p

        self.inner._profile_sampler = _normal_tourist_sampler
        self.inner.agent_profile = _normal_tourist_sampler(self.inner.rng)
        self.inner.dwell_per_importance = 1.4    # short dwells (deep look ~7 min, not 30)
        self.inner.dwell_importance_floor = 5.0  # below importance 5: not worth engaging
        self.inner.tour_weight = 0.0             # no routing shaping; selection is learned
        self.inner.completion_k = 0.0

        # Recognized rooms = the candidate destinations the tourist might choose.
        self.recog_threshold = recog_threshold
        self.targets = [r for r in config.ROOM_IDS
                        if base_imp[config.ROOM_TO_IDX[r]] >= recog_threshold]
        self._K = len(self.targets)

        self.action_space = spaces.Discrete(self._K + 2)  # engage, K destinations, leave
        obs_dim = 3 * self._K + 6
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32)

    # -- CRN seeding passthrough ----------------------------------------------
    @property
    def seed_value(self):
        return self.inner.seed_value

    @seed_value.setter
    def seed_value(self, v):
        self.inner.seed_value = v

    @property
    def _episode_counter(self):
        return self.inner._episode_counter

    @_episode_counter.setter
    def _episode_counter(self, v):
        self.inner._episode_counter = v

    @property
    def current_room(self):
        return self.inner.current_room

    # -- helpers ---------------------------------------------------------------
    def _ideal(self, room: str) -> float:
        idx = config.ROOM_TO_IDX.get(room)
        if idx is None:
            return 1.0
        imp = float(self.inner.agent_profile.importance_vector[idx])
        return max(1.0, self.inner.dwell_per_importance * max(0.0, imp - self.inner.dwell_importance_floor))

    def _toward(self, target: str) -> int:
        inner = self.inner
        if inner.current_room == target:
            return 0
        mask = inner.get_action_mask()
        best_a, best_d = 0, 1e9
        for a in range(int(inner.action_space.n)):
            if not mask[a]:
                continue
            room = inner.action_to_room(a)
            if room == inner.current_room:
                continue
            d = inner._distances.get(room, {}).get(target, 1e9)
            if d < best_d:
                best_d, best_a = d, a
        return best_a

    def get_action_mask(self) -> np.ndarray:
        """All choices always legal (engage / any destination / leave)."""
        return np.ones(self._K + 2, dtype=np.int8)

    # -- observation -----------------------------------------------------------
    def _obs(self) -> np.ndarray:
        inner = self.inner
        cur = inner.current_room
        iv = inner.agent_profile.importance_vector
        dens = inner._density_all_rooms()
        visible = set([cur] + list(inner._sorted_neighbors(cur)))

        feats = []
        n_done = 0
        for tk in self.targets:
            idx = config.ROOM_TO_IDX[tk]
            imp = float(iv[idx])
            ideal = self._ideal(tk)
            drain = min(1.0, inner._extracted.get(tk, 0.0) / ideal)
            if inner._extracted.get(tk, 0.0) >= ideal:
                n_done += 1
            remaining_value = (imp / 10.0) * (1.0 - drain)        # value still on offer
            dist = min(1.0, inner._distances.get(cur, {}).get(tk, 99) / 20.0)
            crowd = float(np.clip(dens[idx], 0.0, 1.0)) if tk in visible else -1.0  # only if I can see it
            feats += [remaining_value, dist, crowd]

        cidx = config.ROOM_TO_IDX[cur]
        cdrain = min(1.0, inner._extracted.get(cur, 0.0) / self._ideal(cur))
        ccrowd = float(np.clip(dens[cidx], 0.0, 1.0))
        cimp = float(iv[cidx]) / 10.0
        frac_done = n_done / max(1, self._K)
        time_norm = inner.time_elapsed / max(1, inner.episode_minutes)
        t_remaining = inner.episode_minutes - inner.time_elapsed
        egress = float(np.clip((t_remaining - inner._distance_to_exit(cur)) / max(1, inner.episode_minutes), -1.0, 1.0))

        return np.array(feats + [cdrain, ccrowd, cimp, frac_done, time_norm, egress], dtype=np.float32)

    # -- gym API ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self.inner.reset(seed=seed, options=options)
        return self._obs(), {}

    def step(self, action: int):
        a = int(action)
        if a == 0:                       # ENGAGE current room
            inner_action = 0
        elif a == self._K + 1:           # LEAVE
            inner_action = self._toward("EXIT")
        else:                            # HEAD FOR valued room a
            inner_action = self._toward(self.targets[a - 1])
        _obs, reward, terminated, truncated, info = self.inner.step(inner_action)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
