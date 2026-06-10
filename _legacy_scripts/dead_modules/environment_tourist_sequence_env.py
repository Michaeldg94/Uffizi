"""Normal tourist as a minimal-action forward traversal (the design that worked).

The art lover learned perfectly with a tiny action space (stay / move-on,
navigation fixed). Every formulation that handed PPO a big menu (neighbour-
stepping, 41-way destination choice) collapsed into the same local optima:
greedy oscillates a cluster, stochastic gets lazy. So this gives the normal
tourist the SAME minimal action space, plus the one thing it needs that the art
lover didn't, the ability to skip and to leave early:

    0  ENGAGE   spend another minute on the room I'm in (go deeper)
    1  ADVANCE  walk to the next room along the museum's forward sequence
    2  LEAVE    head for the exit now

Strictly forward: there's no cluster to oscillate in and no 41-way menu to get
lost in. Skipping a room is just ADVANCE without ENGAGE; ending the visit early
is LEAVE. The selective, crowd-aware decisions (engage vs skip, how deep, when
to leave) are exactly what's learned. Navigation is automatic.

Wraps an inner ``UffiziEnv`` configured as the normal tourist: Type B behaviour
with broad recognition taste (base room importance), short dwells, skip
threshold.
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

_NON_GALLERY = {"ENTRY", "EXIT", "GRANDUCAL_STAIRCASE", "PANORAMIC_TERRACE",
                "LANZI_STAIRCASE", "BUONTALENTI_STAIRCASE"}


class TouristSequenceEnv(gym.Env if _HAVE_GYM else object):
    """Minimal-action forward-traversal normal tourist (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(self, recog_threshold: float = 5.0, **inner_kwargs):
        super().__init__()
        self.inner = UffiziEnv(**inner_kwargs)
        # Casual tourist is less crowd-averse than the art lover: a lower uniform
        # per-room crowd penalty, so congestion still costs points but bites less
        # than it does the connoisseur.
        self.inner.art_crowd_alpha = 1.0

        # Recognition (name-recognition, NOT raw art-historical importance):
        # high on the famous second floor and on Caravaggio/Rembrandt; a moderate
        # linger in the early-Renaissance C6-C11; rush-through (low) on the rest
        # of the floor-1 corridor (the mannerists nobody recognizes).
        base_imp = np.array(
            [config.ROOM_DATA[r]["importance"] for r in config.ROOM_IDS], dtype=float)
        recog = base_imp.copy()
        for i, r in enumerate(config.ROOM_IDS):
            if r.startswith("A") or r in {"E4", "E5", "E7"}:
                continue                       # recognized: keep importance
            if r in {"C6", "C7", "C8", "C9", "C10", "C11"}:
                recog[i] = 7.0                 # early Renaissance: moderate linger
            else:
                recog[i] = 2.0                 # unrecognized floor-1: rush through
        self._recognition_vec = recog

        def _normal_tourist_sampler(rng):
            p = sample_type_b_profile(rng)
            p.importance_vector = recog.copy()
            return p

        self.inner._profile_sampler = _normal_tourist_sampler
        self.inner.agent_profile = _normal_tourist_sampler(self.inner.rng)
        self.inner.dwell_per_importance = 1.4
        self.inner.dwell_importance_floor = 5.0
        self.inner.tour_weight = 0.0
        self.inner.completion_k = 0.0
        self.recog_threshold = recog_threshold

        # The agent walks the museum's actual first-floor circuit (the published
        # route), NOT every room: 2nd floor in full, down Lanzi, skip B, through
        # C, the D corridor (D1 D6 D13 D14 D15 D26 D27, side-rooms bypassed), then
        # the E block to the Magliabechi exit. Skipping is structural, built into
        # the circuit, exactly as the building routes a visitor.
        A_rooms = [r for r in config.ROOM_IDS if r.startswith("A")]
        # The circuit is the ordered list of STOPS on the museum's published
        # route: the whole 2nd floor, then C, then the D corridor, then the E
        # block. B and the D side-rooms are simply not on it. Navigation between
        # consecutive stops is automatic (shortest path through the hubs), so the
        # list stays clean (no repeats) and progress can't get confused.
        self.sequence = A_rooms + ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12",
                                   "D1", "D6", "D13", "D14", "D15", "D26", "D27", "E4", "E5", "E6", "E7"]
        self._next_wp = 1  # index of the next stop on the circuit (we start at sequence[0] = A1)

        # 2 actions only: engage / advance. No explicit early-LEAVE, because the
        # LEAVE option let PPO settle into a lazy partial-visit local optimum
        # (short visit + exit bonus). Forcing a forward traversal that exits only
        # at the natural end mirrors the art lover that learned cleanly; the
        # agent is still selective via engage-vs-advance (skip) per room.
        self.action_space = spaces.Discrete(2)  # 0 = engage, 1 = advance (exits at sequence end)
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(9,), dtype=np.float32)

    # -- CRN passthrough -------------------------------------------------------
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
    def _ideal(self, room):
        idx = config.ROOM_TO_IDX.get(room)
        if idx is None:
            return 1.0
        imp = float(self.inner.agent_profile.importance_vector[idx])
        return max(1.0, self.inner.dwell_per_importance * max(0.0, imp - self.inner.dwell_importance_floor))

    def _toward(self, target):
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

    def _advance_target(self):
        if self._next_wp < len(self.sequence):
            return self.sequence[self._next_wp]
        return "EXIT"

    def _update_progress(self):
        # Advance the stop counter only when the agent actually reaches the next
        # stop (no jumps, robust to transiting back through corridor hubs).
        if self._next_wp < len(self.sequence) and self.inner.current_room == self.sequence[self._next_wp]:
            self._next_wp += 1

    def get_action_mask(self):
        return np.ones(2, dtype=np.int8)

    # -- observation -----------------------------------------------------------
    def _obs(self):
        inner = self.inner
        cur = inner.current_room
        iv = inner.agent_profile.importance_vector
        cidx = config.ROOM_TO_IDX.get(cur)
        imp = float(iv[cidx]) if cidx is not None else 0.0
        drain = min(1.0, inner._extracted.get(cur, 0.0) / self._ideal(cur))
        dens = float(np.clip(inner._density_all_rooms()[cidx], 0.0, 1.0)) if cidx is not None else 0.0
        is_valued = 1.0 if imp >= self.recog_threshold else 0.0

        # valued rooms still ahead in the sequence (not yet engaged)
        ahead = 0
        for r in self.sequence[self._next_wp:]:
            ri = config.ROOM_TO_IDX[r]
            if float(iv[ri]) >= self.recog_threshold and inner._extracted.get(r, 0.0) < self._ideal(r):
                ahead += 1
        ahead_frac = min(1.0, ahead / 20.0)

        seq_frac = self._next_wp / max(1, len(self.sequence))
        time_norm = inner.time_elapsed / max(1, inner.episode_minutes)
        t_remaining = inner.episode_minutes - inner.time_elapsed
        egress = float(np.clip((t_remaining - inner._distance_to_exit(cur)) / max(1, inner.episode_minutes), -1.0, 1.0))
        fatigue = inner.fatigue

        return np.array([
            imp / 10.0,    # value of the room I'm in
            drain,         # how much of it I've already taken
            dens,          # crowd right here
            is_valued,     # is this a room worth my time
            ahead_frac,    # how much I still want to see lies ahead
            seq_frac,      # how far through the museum I am
            time_norm,     # time of day
            egress,        # slack to the exit
            fatigue,
        ], dtype=np.float32)

    # -- gym API ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self.inner.reset(seed=seed, options=options)
        self._next_wp = 1
        return self._obs(), {}

    def step(self, action):
        a = int(action)
        if a == 0:                       # ENGAGE
            inner_action = 0
        else:                            # ADVANCE (exits at the end of the sequence)
            inner_action = self._toward(self._advance_target())
        _obs, reward, terminated, truncated, info = self.inner.step(inner_action)
        self._update_progress()
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
