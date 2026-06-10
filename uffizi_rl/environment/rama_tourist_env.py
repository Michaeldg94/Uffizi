"""Normal tourist under RAMA, rebuilt on the art lover's proven structure.

The earlier engage/advance + early-exit circuit gave PPO too many local traps
(rush past everything and leave; or camp in a masterpiece). The art lover's shape
converges cleanly: move toward the next valued/booked stop, dwell it to its ideal,
move on, and you can't leave until your valued stops are done. So the RAMA tourist
is now exactly that env (`RamaArtLoverEnv`) with the tourist's character swapped in:
recognition-based taste (famous rooms only, the rest rushed), short dwells, less
crowd-aversion, and the freedom to decline a masterpiece. The booking decision,
the frozen one-shot plan, the early-entry grace, and the lead-time market are all
inherited unchanged.
"""
from __future__ import annotations

import numpy as np

from uffizi_rl import config
from uffizi_rl.environment.rama_art_lover_env import RamaArtLoverEnv
from uffizi_rl.environment.visitor_profiles import sample_type_b_profile


class RamaTouristEnv(RamaArtLoverEnv):
    """Normal tourist on the art lover's env, with recognition taste + decline."""

    def __init__(self, **inner_kwargs):
        compute_plan = inner_kwargs.pop("compute_plan", True)
        super().__init__(compute_plan=False, **inner_kwargs)  # skip art-lover plan; recompute w/ recognition taste

        # --- swap the connoisseur for the recognition tourist ---
        # Casual tourist is less crowd-averse than the art lover.
        self.inner.art_crowd_alpha = 1.0
        # Recognition (name-recognition, NOT art-historical importance): keep the
        # famous second floor + Caravaggio/Rembrandt at full value, a moderate
        # linger in the early-Renaissance C6-C11, rush-through (low) everywhere
        # else on floor 1.
        base_imp = np.array(
            [config.ROOM_DATA[r]["importance"] for r in config.ROOM_IDS], dtype=float)
        recog = base_imp.copy()
        for i, r in enumerate(config.ROOM_IDS):
            if r.startswith("A") or r in {"E4", "E5", "E7"}:
                continue
            if r in {"C6", "C7", "C8", "C9", "C10", "C11"}:
                recog[i] = 7.0
            else:
                recog[i] = 2.0
        self._recognition_vec = recog

        def _tourist_sampler(rng):
            p = sample_type_b_profile(rng)
            p.importance_vector = recog.copy()
            return p

        self.inner._profile_sampler = _tourist_sampler
        self.inner.agent_profile = _tourist_sampler(self.inner.rng)
        self.inner.dwell_per_importance = 1.4     # short dwells
        self.inner.dwell_importance_floor = 5.0

        # Both visitor types come to the Uffizi FOR the famous three (the connoisseur
        # to study them, the Instagram tourist to photograph them); nobody travels to
        # Florence and skips all of Botticelli, Leonardo and Raphael. So the tourist
        # also books all three. A free decline button is an RL do-nothing trap (PPO
        # collapses to declining everything, 0/3), not a realistic choice. The tourist's
        # character lives in the SHORT dwells and recognition taste set above, plus the
        # lighter no-show penalty: it is readier to lose a slot it can't reach in time.
        self.allow_decline = False
        self.decline_penalty = 50.0
        self.noshow_penalty = 80.0

        # Compute the brief-memory plan with the tourist's (faster, recognition-
        # based) pace. Eval envs skip it (compute_plan=False) and inject the
        # trained plan instead.
        if compute_plan:
            self._compute_brief_memory()
