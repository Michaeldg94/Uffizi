"""Art lover under RAMA: booking is the decision.

The three floor-2 masterpieces (Botticelli A11/A12, Leonardo A35, Raphael A38)
are reservation-only timed-entry rooms. Before the visit the agent makes two
booking choices, then walks the museum trying to keep the three appointments it
made.

Decision 1, LEAD TIME: how far ahead it reserves (month / weeks / few-days /
day-before). Windows for a given day fill over the booking horizon, popular
midday windows selling out first, so a long lead time leaves every window open
while a last-minute booking leaves only fringe early/late windows. Booking early
is free and dominant: the agent must *learn* that booking late forfeits rooms.

Decision 2, WHICH WINDOWS: from whatever is still available at the chosen lead
time, a 10-min entry window for each masterpiece (or decline one). This sets the
spacing.

Then the visit executes (stay / move-on, navigation automatic, as in
``PlannedRouteEnv``), but each masterpiece now pays only if the agent *arrives
within* its booked window. Arrive early and it waits (burned minutes, no value);
arrive after the window closes and the room is forfeit. Once checked in it
appreciates normally, the full ~30-min calm visit. So the windows must be spaced
far enough apart that the agent can finish one room and reach the next before
its window opens, while threading the enriched secondary rooms and ungated
Caravaggio into the gaps under crowd. That scheduling-under-congestion is the
non-trivial decision the RAMA intervention creates.

The booking phase is four pre-visit decisions (no museum time passes): lead time,
then a window for Botticelli, Leonardo, Raphael. Action semantics are
phase-dependent and masked (MaskablePPO): lead levels during the lead step,
available windows + decline during a window step, stay/move-on during the walk.
"""
from __future__ import annotations

import numpy as np

from uffizi_rl import config
from uffizi_rl.environment.planned_route_env import PlannedRouteEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except Exception:  # pragma: no cover
    gym = None
    spaces = None
    _HAVE_GYM = False

# Booking-decision constants.
LEAD_DAYS = [1, 7, 21, 35]          # day-before / few-days / weeks / month
LEAD_MAX = 35.0                     # longest lead time (full availability)
N_WIN = 8                           # candidate entry windows across the day
DECLINE = N_WIN                     # action index meaning "do not book this one"
GROUPS = ["bott", "leo", "raph"]    # Botticelli / Leonardo / Raphael
_LEAD_N = len(LEAD_DAYS)
SLOT_DUR = config.RAMA_SLOT_DURATION_MIN
# Minimum gap between consecutive booked windows. The masterpieces are visited
# in fixed spatial order (A11 -> A35 -> A38), so each window must leave enough
# time to finish one room (~30-min dwell) and walk to the next; booking them
# tighter is physically impossible to keep. Masking enforces this feasibility,
# leaving how-wide-to-space and the lead time as the agent's real choices.
MIN_GAP = 60
CHECKIN_BONUS = 50.0                 # sub-goal reward for keeping an appointment
# Action space (reused per phase, masked): lead step uses the 4 lead options;
# each book step uses 0=book-at-my-estimate / 1=decline; the walk uses 0/1.
_ACTION_N = max(len(LEAD_DAYS), 2)   # 4
_OBS_DIM = 25


class RamaArtLoverEnv(PlannedRouteEnv):
    """Art lover whose decision includes booking the masterpiece windows."""

    def __init__(self, **inner_kwargs):
        compute_plan = inner_kwargs.pop("compute_plan", True)  # eval envs inject the trained plan instead
        super().__init__(**inner_kwargs)
        if not self.inner._rama_active:
            raise ValueError("RamaArtLoverEnv requires interventions with rama=True")
        self.action_space = spaces.Discrete(_ACTION_N)
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(_OBS_DIM,), dtype=np.float32)
        self.inner.rama_checkin_bonus = CHECKIN_BONUS
        # Minimum spacing between booked windows (finish a ~30-min room + walk to
        # the next) and the entry grace band (a real reservation lets you arrive
        # within a window, not on the exact minute). Generous tolerance keeps the
        # execution achievable so the agent engages with the strategic decision,
        # lead-time and which windows, rather than failing on minute-precision.
        # Windows are placed at the agent's true estimated arrival times; the only
        # spacing requirement is that they be distinct slots (>= one slot apart),
        # NOT an artificial 60-min spread (which used to shove adjacent rooms'
        # windows past the agent's pace and manufacture waiting). The natural
        # arrival order already spaces them feasibly.
        self.min_gap = SLOT_DUR
        self.entry_tolerance = 150   # LATE grace: still make the window if a crowd slows you
        self.early_grace = 150       # EARLY grace: arriving ahead lets you in immediately (no idling)
        # Waiting penalty: idling for a window costs time. SAFE now because the
        # plan is frozen (_arr_alpha=0), so it can't feed the old runaway. Its
        # job here is to make LATE booking bite: book too late and your slots get
        # shoved off your pace, so you wait (or miss), which is what makes the
        # agent learn to book further ahead as the crowd grows.
        self.inner.rama_wait_penalty = 2.0
        # Over-dwell penalty: staying in a room past satiation earns nothing and
        # now costs, so the agent maxes each room and moves on (no lingering).
        self.inner.boredom_k = 4.0
        self.decline_penalty = 200.0         # regret for skipping a masterpiece (used only if a subtype allows decline)
        # The art lover IS the connoisseur: a visitor who travelled to Florence for
        # Botticelli, Leonardo and Raphael books all three. "Always books" is the
        # DEFINING TRAIT of the type, like the tourist's recognition taste, not the
        # museum forcing a ticket. A freely-available decline button does not model a
        # connoisseur choosing to skip; empirically PPO collapses to declining all
        # three (a do-nothing trap) even against a -1200 regret and after a curriculum
        # that first taught it booking pays ~+2000. The genuine decline CHOICE belongs
        # to the tourist subtype (allow_decline=True there), which does pick and skip.
        self.allow_decline = False
        self.noshow_penalty = 100.0          # booking then skipping a masterpiece is penalized
        # Off-pace booking penalty, applied IMMEDIATELY at booking time: how far
        # each secured slot lands from when the agent will actually arrive (its
        # estimate). Booking early -> slots on your pace -> ~0; booking late ->
        # the slots you wanted are sold out and you get scraps far from your pace
        # -> large penalty. Bigger the busier the day, so the agent learns to book
        # earlier when it's busier. This bridges the once-per-visit, delayed lead
        # decision with a same-step signal so PPO can actually learn it.
        self.mismatch_k = 2.0
        # The agent's arrival prior = its 'brief memory' before ever visiting: a
        # mental walkthrough of the route (given the rooms and the crowd) that
        # estimates when it would reach Botticelli / Leonardo / Raphael at an
        # efficient max-out pace. Computed once and FIXED, so the booked windows
        # are a stable target; PPO then refines only the pacing to hit them. (An
        # earlier version updated this from the agent's own arrivals each episode,
        # which let the target drift away from the pacing chasing it -> runaway.)
        self._learned_arrival = [0.25, 0.5, 0.75]
        # One-shot plan: the slot is the tourist's pre-visit estimate (computed
        # once in _compute_brief_memory from the route), FROZEN. A real visitor
        # books once and does not re-book across a thousand visits, so the slot is
        # not updated online (that online update has an alarm-clock feedback that
        # drifts even at constant crowd). The genuine learning is the lead time
        # and the in-visit pacing that PPO learns to keep these appointments.
        self._arr_alpha = 0.0
        self._ep_first_arrival: dict[str, int] = {}

        # Candidate window start minutes, spread across the museum day, each
        # snapped to a 10-min slot boundary.
        em = self.inner.episode_minutes
        fracs = np.linspace(0.08, 0.92, N_WIN)
        self._win_starts = [int(round(f * em / SLOT_DUR)) * SLOT_DUR for f in fracs]
        # Window popularity: a midday bump. The most popular windows sell out
        # earliest (largest required lead time).
        self._peak = 0.45 * em
        self._width = 0.22 * em

        self._phase = "booking"
        self._booking_step = 0
        self._lead_days = LEAD_MAX
        self._chosen: dict[str, int | None] = {}

        if compute_plan:
            self._compute_brief_memory()     # the visitor's pre-visit timing plan

    # -- brief-memory timing plan ----------------------------------------------
    def _ref_next_target(self) -> str:
        """Nearest unfinished must-see for the unconstrained reference walk."""
        cur = self.inner.current_room
        rem = [t for t in self.inner._tour_targets
               if self.inner._extracted.get(t, 0.0) < self._ideal(t)]
        if not rem:
            return "EXIT"
        return min(rem, key=lambda t: self.inner._distances.get(cur, {}).get(t, 1e9))

    def _compute_brief_memory(self, n_runs: int = 3) -> None:
        """Estimate when the agent reaches each masterpiece at an efficient
        max-out pace, by mentally walking the route a few times with the windows
        wide open (so the booking gate never distorts the pace). The average
        becomes the fixed booking plan. This is the visitor's map-based estimate,
        measured on clean (unconstrained) walks rather than on booking-contaminated
        ones, which is what keeps it stable."""
        em = self.inner.episode_minutes
        got: dict[str, list[int]] = {g: [] for g in GROUPS}
        for _ in range(n_runs):
            self.inner.reset()
            self.inner.set_secured_windows({g: (0, em) for g in GROUPS})  # never gated
            first: dict[str, int] = {}
            steps = 0
            done = trunc = False
            while not (done or trunc) and steps < em + 5:
                cur = self.inner.current_room
                tgt = self._ref_next_target()
                if tgt != "EXIT" and cur == tgt and self.inner._extracted.get(cur, 0.0) < self._ideal(cur):
                    act = 0                                   # max out this room
                else:
                    act = self._inner_action_toward(tgt if tgt != "EXIT" else "EXIT")
                self.inner.step(act); steps += 1
                cur = self.inner.current_room
                if cur in config.MASTERPIECE_ROOMS:
                    g = self.inner._rama_group(cur)
                    if g not in first:
                        first[g] = self.inner.time_elapsed
            for g in GROUPS:
                if g in first:
                    got[g].append(first[g])
        for i, g in enumerate(GROUPS):
            if got[g]:
                self._learned_arrival[i] = (sum(got[g]) / len(got[g])) / max(1, em)

    # -- planned targets -------------------------------------------------------
    def _remaining_targets(self) -> list[str]:
        """Must-sees still worth navigating to. A RAMA masterpiece is dropped if
        it was declined (never bookable) or if its window has closed without a
        check-in (forfeit), so the agent never strands itself on a room it can no
        longer appreciate. A booked-but-not-yet-open masterpiece stays a target
        (the agent walks there and waits for the window)."""
        out = []
        for t in self.inner._tour_targets:
            if self.inner._extracted.get(t, 0.0) >= self._ideal(t):
                continue  # already fully appreciated
            g = self.inner._rama_group(t) if t in config.MASTERPIECE_ROOMS else None
            if g is not None:
                start = self._chosen.get(g)
                if start is None:
                    continue  # declined / not booked
                if (self.inner.time_elapsed > start + self.entry_tolerance
                        and not self.inner._rama_checkin.get(g, False)):
                    continue  # window closed, forfeit
            out.append(t)
        return out

    # -- fill curve ------------------------------------------------------------
    def _popularity(self, t: float) -> float:
        return float(np.exp(-(((t - self._peak) / self._width) ** 2)))

    def _available(self, slot_start: int, lead_days: float) -> bool:
        """A window is still open at this lead time iff the lead time meets the
        window's sell-out horizon. Popular (midday) windows need a longer lead,
        and a BUSIER DAY fills every window sooner, so the required lead scales
        with the crowd: at a quiet 500-visitor day even prime slots are bookable
        late, at a packed day they sell out a month ahead. This is what makes
        'book early' bite as the crowd grows."""
        crowd_factor = self.inner.sim.daily_total / 2500.0
        return lead_days >= LEAD_MAX * self._popularity(slot_start) * crowd_factor

    def _resolve_window_from(self, estimate: float, floor: int) -> int:
        """The slot the agent actually gets for a masterpiece: the still-open
        10-min slot nearest to its rough estimate of when it will arrive, subject
        to coming after the previous booking (floor, keeps the three ordered and
        spaced). Book early and the estimate slot is open (booked right on the
        mark); book late and the prime midday slots are sold out, so the nearest
        open slot is far off, which the agent must absorb during the visit
        (waiting, or a near-miss)."""
        em = self.inner.episode_minutes
        cands = [c for c in range(0, em - SLOT_DUR + 1, SLOT_DUR)
                 if c >= floor and self._available(c, self._lead_days)]
        if not cands:                       # nothing open past the floor: ignore availability
            cands = [c for c in range(0, em - SLOT_DUR + 1, SLOT_DUR) if c >= floor]
        if not cands:
            return int(round(estimate / SLOT_DUR) * SLOT_DUR)
        return min(cands, key=lambda c: abs(c - estimate))

    # -- masking ---------------------------------------------------------------
    def get_action_mask(self) -> np.ndarray:
        m = np.zeros(_ACTION_N, dtype=np.int8)
        if self._phase == "booking":
            if self._booking_step == 0:                  # choose lead time
                m[: len(LEAD_DAYS)] = 1
            else:                                        # book this masterpiece at my estimate, or decline
                m[0] = 1                                  # book
                if self.allow_decline:
                    m[1] = 1                              # decline
        else:                                            # walk: stay / move-on
            m[0] = 1
            m[1] = 1
        return m

    # -- observation -----------------------------------------------------------
    def _avail_by_lead(self) -> list[float]:
        """The booking screen: fraction of windows still open at each lead time
        (longer lead -> more open). Lets the policy read 'book further ahead =
        more availability' directly, not only through the action mask."""
        return [sum(1 for w in self._win_starts if self._available(w, d)) / N_WIN
                for d in LEAD_DAYS]

    def _obs(self) -> np.ndarray:
        inner = self.inner
        em = max(1, inner.episode_minutes)
        cur = inner.current_room
        idx = config.ROOM_TO_IDX.get(cur)
        imp = float(inner.agent_profile.importance_vector[idx]) if idx is not None else 0.0
        drain = min(1.0, inner._extracted.get(cur, 0.0) / self._ideal(cur))
        dens = float(inner._density_all_rooms()[idx]) if idx is not None else 0.0
        t_remaining = inner.episode_minutes - inner.time_elapsed
        egress = float(np.clip((t_remaining - inner._distance_to_exit(cur)) / em, -1.0, 1.0))

        avail = self._avail_by_lead()
        win_norm = [(self._chosen.get(g) / em if self._chosen.get(g) is not None else 0.0) for g in GROUPS]
        checkin = [1.0 if inner._rama_checkin.get(g, False) else 0.0 for g in GROUPS]
        is_mast, unbooked, access_now, ttl = inner.rama_access_features()

        return np.array([
            1.0 if self._phase == "booking" else 0.0,
            self._booking_step / 4.0,
            self._lead_days / LEAD_MAX,
            avail[0], avail[1], avail[2], avail[3],   # booking screen: slots open by lead time
            inner.time_elapsed / em,
            win_norm[0], win_norm[1], win_norm[2],    # the windows I hold
            checkin[0], checkin[1], checkin[2],        # which I've used
            is_mast, unbooked, access_now, ttl,        # access read of the room I'm standing in
            self._learned_arrival[0], self._learned_arrival[1], self._learned_arrival[2],  # learned prior: when I'll reach each
            drain,
            imp / 10.0,
            min(1.0, dens),
            egress,
        ], dtype=np.float32)

    # -- gym API ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self.inner.reset(seed=seed, options=options)
        self._phase = "booking"
        self._booking_step = 0
        self._lead_days = LEAD_MAX
        self._book_decision: dict[str, bool] = {}   # per masterpiece: book vs decline
        self._chosen = {}                            # resolved window starts, filled at commit
        self._ep_first_arrival = {}                  # when (this episode) it first reached each masterpiece
        return self._obs(), {}

    def _update_learned_arrival(self) -> None:
        """Learn the agent's pace as a running average of when it actually reaches
        each masterpiece. With the waiting penalty removed the agent no longer
        dawdles, so these arrivals are the clean max-out-each-room pace, and a
        plain average is both correct and robust to crowd noise (no runaway,
        because nothing pulls arrivals later). Crowd-specific (one crowd per env)."""
        em = max(1, self.inner.episode_minutes)
        for i, g in enumerate(GROUPS):
            t = self._ep_first_arrival.get(g)
            if t is not None:
                self._learned_arrival[i] = ((1 - self._arr_alpha) * self._learned_arrival[i]
                                            + self._arr_alpha * (t / em))

    def _commit(self) -> None:
        """Resolve the booked masterpieces into actual entry windows, in order, so
        they stay sequenced and spaced: each is placed at the open slot nearest to
        the agent's arrival estimate but after the previous one. Then install them
        on the inner env. Declined masterpieces are absent (inaccessible)."""
        self._chosen = {}
        prev = 0
        for i, g in enumerate(GROUPS):
            if not self._book_decision.get(g, False):
                continue
            est = self._learned_arrival[i] * self.inner.episode_minutes
            floor = prev + self.min_gap if prev else 0
            start = self._resolve_window_from(est, floor)
            self._chosen[g] = start
            prev = start
        # Window the agent may enter: [start - early_grace, start + entry_tolerance].
        # Early grace means it enters on arrival instead of idling before its slot.
        windows = {g: (s - self.early_grace, s + self.entry_tolerance) for g, s in self._chosen.items()}
        self.inner.set_secured_windows(windows)
        self._phase = "walk"

    def step(self, action: int):
        a = int(action)
        if self._phase == "booking":
            if self._booking_step == 0:
                self._lead_days = LEAD_DAYS[min(a, len(LEAD_DAYS) - 1)]
                self._booking_step = 1
            else:
                g = GROUPS[self._booking_step - 1]
                declined = (a == 1 and self.allow_decline)
                self._book_decision[g] = not declined  # book unless declined
                self._booking_step += 1
                # Regret is felt AT the moment of skipping, not lumped at the end:
                # this gives the book-vs-decline choice a sharp same-step gradient,
                # so the agent learns to choose booking instead of lazily declining.
                step_r = -self.decline_penalty if declined else 0.0
                if self._booking_step > len(GROUPS):
                    self._commit()
                    em = self.inner.episode_minutes
                    # How far each secured slot is from the agent's estimated arrival
                    # (in minutes): ~0 if booked early on-pace, large if booked late
                    # and shoved off-pace. Penalized immediately so the lead choice
                    # gets a same-step gradient.
                    mismatch = sum(abs(self._chosen[g] - self._learned_arrival[i] * em)
                                   for i, g in enumerate(GROUPS) if self._chosen.get(g) is not None)
                    step_r += -self.mismatch_k * mismatch
                return self._obs(), step_r, False, False, {}
            # No museum time passes during booking.
            return self._obs(), 0.0, False, False, {}

        # Walk phase: stay (0) or move-on (1) toward the next planned room.
        if a == 1:
            inner_action = self._inner_action_toward(self._next_target())
        else:
            inner_action = 0
        _obs, reward, terminated, truncated, info = self.inner.step(inner_action)
        # Note when it first reaches each masterpiece (its natural pace), and fold
        # those into the learned arrival prior at episode end.
        cur = self.inner.current_room
        if cur in config.MASTERPIECE_ROOMS:
            g = self.inner._rama_group(cur)
            if g not in self._ep_first_arrival:
                self._ep_first_arrival[g] = self.inner.time_elapsed
        reward = float(reward)
        if terminated or truncated:
            self._update_learned_arrival()
            noshow = sum(1 for g in GROUPS if self._chosen.get(g) is not None
                         and not self.inner._rama_checkin.get(g, False))
            reward += -self.noshow_penalty * noshow   # booked-but-skipped is penalized
        return self._obs(), reward, bool(terminated), bool(truncated), info
