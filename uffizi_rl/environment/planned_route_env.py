"""Plan-following formulation of the art-lover's visit.

An art lover does not decide *which* rooms to see while walking the gallery;
that is fixed before arrival (the guidebook itinerary). Navigation is therefore
not a learned decision. The only live, minute-by-minute decision is **how long
to stay in the room I am in before moving on**, and that interacts with the one
thing the plan cannot foresee: the crowd in the room I am actually standing in.

This wrapper exposes exactly that decision. It holds an inner ``UffiziEnv`` for
the museum graph, crowd simulator, profile and reward, and reduces the action
space to two choices:

    0 = STAY      remain in the current room and keep absorbing it
    1 = MOVE ON   take one step along the shortest path toward the next room on
                  the plan (the nearest must-see not yet fully appreciated), or
                  toward the exit once the whole plan is done

So the agent walks its predetermined route automatically and only learns the
dwell timing. The observation is the small local state a real visitor has: how
drained the current room is, how crowded it is right here, whether I have
arrived at a planned room, how much of the plan is left, the time of day, and my
slack to the exit. It does NOT include the crowd in rooms the visitor cannot
see.

Why this is the right model: forcing PPO to learn navigation from a one-hot room
index with no map is what made every earlier run collapse into camps and loops.
Navigation was never the visitor's decision. Removing it turns an unlearnable
long-horizon control problem into the trivial dwell-timing decision it actually
is.
"""
from __future__ import annotations

import numpy as np

from uffizi_rl import config
from uffizi_rl.environment.uffizi_env import UffiziEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except Exception:  # pragma: no cover - exercised only without gymnasium
    gym = None
    spaces = None
    _HAVE_GYM = False

_OBS_DIM = 8


class PlannedRouteEnv(gym.Env if _HAVE_GYM else object):
    """Stay-or-move-on wrapper around :class:`UffiziEnv` (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(self, **inner_kwargs):
        super().__init__()
        self.inner = UffiziEnv(**inner_kwargs)
        # Art lover is the more crowd-averse visitor: a higher uniform per-room
        # crowd penalty (value = importance / (1 + alpha*density^2)) than the
        # casual tourist, so congestion bites the connoisseur harder.
        self.inner.art_crowd_alpha = 2.0
        self.action_space = spaces.Discrete(2)  # 0 = stay, 1 = move on
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(_OBS_DIM,), dtype=np.float32)

    # -- expose CRN seeding controls so the eval harness can pin scenarios -----
    @property
    def seed_value(self) -> int:
        return self.inner.seed_value

    @seed_value.setter
    def seed_value(self, v: int) -> None:
        self.inner.seed_value = v

    @property
    def _episode_counter(self) -> int:
        return self.inner._episode_counter

    @_episode_counter.setter
    def _episode_counter(self, v: int) -> None:
        self.inner._episode_counter = v

    @property
    def current_room(self) -> str:
        return self.inner.current_room

    # -- plan helpers ----------------------------------------------------------
    def _ideal(self, room: str) -> float:
        idx = config.ROOM_TO_IDX.get(room)
        if idx is None:
            return 1.0
        imp = float(self.inner.agent_profile.importance_vector[idx])
        return max(1.0, self.inner.dwell_per_importance * max(0.0, imp - self.inner.dwell_importance_floor))

    def _remaining_targets(self) -> list[str]:
        return [t for t in self.inner._tour_targets
                if self.inner._extracted.get(t, 0.0) < self._ideal(t)]

    def _next_target(self) -> str:
        """Next room on the plan: the nearest must-see not yet fully appreciated,
        or EXIT once the whole plan is done."""
        remaining = self._remaining_targets()
        if not remaining:
            return "EXIT"
        cur = self.inner.current_room
        return min(remaining, key=lambda t: self.inner._distances.get(cur, {}).get(t, 1e9))

    def _inner_action_toward(self, target: str) -> int:
        """Inner-env action index for the valid neighbor that most reduces hop
        distance to ``target`` (the next step of the planned walk)."""
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

    # -- observation -----------------------------------------------------------
    def _obs(self) -> np.ndarray:
        inner = self.inner
        cur = inner.current_room
        idx = config.ROOM_TO_IDX.get(cur)
        imp = float(inner.agent_profile.importance_vector[idx]) if idx is not None else 0.0
        ideal_cur = self._ideal(cur)
        drain = min(1.0, inner._extracted.get(cur, 0.0) / ideal_cur)
        dens = float(inner._density_all_rooms()[idx]) if idx is not None else 0.0
        target = self._next_target()
        at_target = 1.0 if (target != "EXIT" and cur == target) else 0.0
        dist_next = 0.0 if target == "EXIT" else float(inner._distances.get(cur, {}).get(target, 0))
        n_targets = max(1, len(inner._tour_targets))
        n_done = n_targets - len(self._remaining_targets())
        frac_done = n_done / n_targets
        time_norm = inner.time_elapsed / max(1, inner.episode_minutes)
        t_remaining = inner.episode_minutes - inner.time_elapsed
        egress = float(np.clip((t_remaining - inner._distance_to_exit(cur)) / max(1, inner.episode_minutes), -1.0, 1.0))
        return np.array([
            drain,                       # how drained the current room is [0,1]
            imp / 10.0,                  # how valuable the current room is [0,~1]
            min(1.0, dens),              # crowd right here [0,1]
            at_target,                   # 1 if standing in the next planned room
            frac_done,                   # fraction of the plan completed [0,1]
            time_norm,                   # time of day [0,1]
            egress,                      # slack to the exit [-1,1]
            min(1.0, dist_next / 20.0),  # how far the next planned room is
        ], dtype=np.float32)

    # -- gym API ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self.inner.reset(seed=seed, options=options)
        return self._obs(), {}

    def step(self, action: int):
        if int(action) == 1:  # MOVE ON: one step along the planned walk
            inner_action = self._inner_action_toward(self._next_target())
        else:                 # STAY
            inner_action = 0
        _obs, reward, terminated, truncated, info = self.inner.step(inner_action)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def get_action_mask(self) -> np.ndarray:
        """Both actions are always legal (kept for MaskablePPO compatibility)."""
        return np.ones(2, dtype=np.int8)
