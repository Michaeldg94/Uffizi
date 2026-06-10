"""Crowd simulation with capacity-gated arrivals and heterogeneous visitors.

[READING ORDER: file 5 of 12 - read after s04_visitor_profiles.py]

This is the CORE simulation engine for the Uffizi RL project. It models one
full museum day as a sequence of minute-by-minute timesteps. Each minute:
  1. New visitors arrive according to a Gaussian intensity envelope [A22]
  2. Arrivals queue outside if the museum is at capacity (900-person legal cap)
  3. Each active visitor decides: stay in current room or move to a neighbor
  4. Movement uses weighted stochastic neighbor selection combining:
     - Route bias (tendency to follow the recommended itinerary)
     - Importance pull (attraction toward culturally significant rooms)
     - Anti-crowd avoidance (Type A visitors steer away from density)
  5. Visitors who exhaust their time budget or reach EXIT are removed

The simulator serves two purposes:
  - As a standalone day-level model (run_day) for policy evaluation
  - As the transition function inside the Gymnasium RL environment

Visitor heterogeneity follows a two-type model inspired by Veron &
Levasseur (1983) [VL83]: Type A ("art lovers" / Butterfly visitors) are
crowd-sensitive and explore freely; Type B ("checkbox tourists" / Ant
visitors) follow the guidebook route to canonical masterpieces.

Interventions (timed entry, photography bans, dynamic pricing, etc.) modify
graph attributes (importance, magnetism), arrival curves, or movement rules
at init time or per-step, without changing the core simulation loop.

References
----------
[A22]  Attanasio et al. (2022). "Visitors flow management at Uffizi Gallery
       in Florence, Italy." Information Technology & Tourism, 24(3), 409-434.
[VL83] Veron & Levasseur (1983). "Ethnographie de l'exposition." BPI,
       Centre Georges Pompidou.
[UFF]  Official Uffizi website and visitor FAQ (uffizi.it).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np

from uffizi_rl import s02_config as config
from uffizi_rl.analysis.metrics import experience_quality_components
from uffizi_rl.interventions.intervention_config import InterventionConfig
from uffizi_rl.environment.s03_museum_graph import (
    all_pairs_shortest_paths,      # Floyd-Warshall precomputed O(1) lookups
    build_uffizi_graph,            # constructs the NetworkX museum graph
    recommended_next_map,          # room -> next-room along the standard route
)
from uffizi_rl.environment.s04_visitor_profiles import (
    VisitorProfile,                # frozen dataclass of per-visitor behavioral params
    sample_profile,                # stochastic visitor profile generator
    sample_type_b_profile,         # forced Type B profile for tour group members
    type_a_effective_density,      # asymmetric density: Type A perceives Type B as noisier
)
from uffizi_rl.interventions.hidden_gem_trails import apply_trail_to_profile

# =============================================================================
# Arrival envelope parameters
# =============================================================================
# The Gaussian arrival curve is centered at ~10:30 AM (135 minutes after the
# 08:15 opening) with sigma=90 minutes, producing a broad morning-peaked
# bell shape consistent with [A22] Figure 3. The Gaussian is evaluated at
# each minute t in [0, last_entry_minutes]; its integral over that range
# is normalized so total arrivals sum to daily_total.
_ARRIVAL_PEAK_MINUTES = 135   # minutes after opening [A22]
_ARRIVAL_SIGMA_MINUTES = 90   # Gaussian width [assumption]


# =============================================================================
# Visitor state container
# =============================================================================

@dataclass
class NPCVisitor:
    """Mutable simulator state for one non-controlled museum visitor.

    Each visitor carries both immutable behavioral parameters (in ``profile``)
    and mutable simulation state (current room, time in room, budget).
    The separation keeps behavioral sampling in visitor_profiles.py and
    state evolution in this file.

    Parameters
    ----------
    visitor_id : int
        Unique monotonic identifier assigned at creation time.
    visitor_type : str
        "A" (art lover) or "B" (checkbox tourist). Determines crowd
        sensitivity, route-following tendency, and dwell behavior.
    profile : VisitorProfile
        Frozen behavioral parameters: importance vector, crowd alpha,
        route bias, backtrack probability, dwell multiplier, etc.
    current_room : str or None
        Room ID where the visitor currently stands. None only before
        they enter the museum (while in the outside queue).
    remaining_budget : int
        Minutes of visit time left. Decremented by 1 each step.
        When it reaches 0, the visitor exits regardless of location.
    entry_slot : int
        Index of the 15-minute entry slot (0-36) when the visitor arrived.
        Used for time-dependent budget sampling and pricing.
    time_in_room : int
        Minutes spent in the current room so far. Resets to 0 on each
        room transition. Feeds the exponential stay-probability decay.
    rooms_visited : set of str
        Accumulator of all room IDs visited during this day. Used for
        experience quality (surprise, narrative coherence) computation.
    checklist : dict of str to float
        The visitor's top-8 "must-see" rooms with their personal importance
        scores. Rooms on the checklist get a route bonus in _pick_next_room.
    group_id : int or None
        Tour group identifier. None for individual visitors. Group members
        share a profile and checklist and are created together.
    botticelli_window : tuple of (int, int) or None
        (start_minute, end_minute) window during which this visitor is
        allowed to enter Botticelli rooms (A11/A12). Used by the
        decoupled gating and smart defaults interventions.
    ticket_price : float
        Ticket price in EUR paid by this visitor. Populated only when
        the revenue model intervention is active. [assumption]
    is_walk_in : bool
        Whether this visitor arrived without a reservation (walk-in).
        Walk-ins pay a premium price under the walk-in intervention.
    """

    visitor_id: int
    visitor_type: str
    profile: VisitorProfile
    current_room: str | None
    remaining_budget: int
    entry_slot: int
    time_in_room: int = 0
    rooms_visited: set[str] = field(default_factory=set)
    checklist: Dict[str, float] = field(default_factory=dict)
    group_id: int | None = None
    botticelli_window: tuple[int, int] | None = None
    ticket_price: float = 0.0
    is_walk_in: bool = False


# =============================================================================
# Core simulator
# =============================================================================

class CrowdSimulator:
    """Minute-step crowd simulator for Uffizi population dynamics.

    This class is the single source of truth for how a museum day unfolds.
    It owns the museum graph, the visitor population, the occupancy counters,
    and the intervention logic. The RL environment (UffiziEnv) wraps this
    class and exposes step/reset through the Gymnasium interface.

    The simulation proceeds in discrete one-minute timesteps. At each step:
      1. step_arrivals() draws Poisson arrivals from the Gaussian envelope,
         creates NPCVisitor objects, and admits them up to the 900-person cap.
      2. _step_visitor() is called for every active visitor: each decides
         whether to stay (exponential decay) or move (weighted neighbor
         selection). Budget is decremented; exhausted visitors exit.
      3. Density snapshots are recorded for metrics and RL observations.

    All interventions are applied either at init time (modifying graph
    attributes like importance/magnetism) or at step time (modifying arrival
    rates or movement weights). The core loop itself is intervention-agnostic.
    """

    def __init__(
        self,
        daily_total: int = config.DAILY_VISITORS_NORMAL,
        seed: int = config.DEFAULT_SEED,
        type_a_fraction: float = config.TYPE_A_FRACTION_DEFAULT,
        heterogeneity_scale: float = 1.0,
        trail_acceptance_prob: float = 0.30,
        interventions: InterventionConfig | None = None,
        **kwargs,
    ) -> None:
        """Initialize the simulator with museum graph, arrival model, and interventions.

        Parameters
        ----------
        daily_total : int
            Total visitors expected over the full day. The Gaussian arrival
            envelope is normalized so its integral equals this number.
            Default 5000 represents a typical non-peak weekday [assumption].
        seed : int
            Random seed for reproducibility. All stochastic operations
            (arrivals, movement, dwell) use a single Generator seeded here.
        type_a_fraction : float
            Fraction of visitors who are Type A ("art lovers"). The remainder
            are Type B ("checkbox tourists"). Default 0.30 [assumption].
        heterogeneity_scale : float
            Scales the variance of visitor-specific importance vectors.
            At 1.0, visitors have the default spread of preferences.
            Higher values increase behavioral diversity. [assumption]
        trail_acceptance_prob : float
            Base probability that a Type A visitor accepts a hidden-gem
            trail (alternative itinerary through underused rooms). The
            adaptive_trail intervention overrides this dynamically.
        interventions : InterventionConfig or None
            Structured intervention configuration. If None and kwargs
            are provided, backward-compatible construction from kwargs.
            If both are None, no interventions are active (baseline).
        **kwargs
            Legacy keyword arguments forwarded to InterventionConfig.
        """
        # ----- Museum graph -----
        # Build the full Uffizi graph: 100+ rooms as nodes with attributes
        # (importance, magnetism, capacity), edges from the floor plan [MAP].
        self.g = build_uffizi_graph()

        # ----- Core parameters -----
        self.daily_total = int(daily_total)
        self.seed = int(seed)
        self.rng = config.get_rng(seed)          # single Generator for all stochastic ops
        self.day_minutes = config.MUSEUM_OPEN_MINUTES  # 615 min = 08:15-18:30 [UFF]
        self.type_a_fraction = float(type_a_fraction)
        self.heterogeneity_scale = float(heterogeneity_scale)
        self.trail_acceptance_prob = float(trail_acceptance_prob)

        # ----- Intervention config -----
        # Accept either the structured InterventionConfig or old-style keyword
        # arguments for backward compatibility with early experiment scripts.
        if interventions is not None:
            self.iv = interventions
        elif kwargs:
            self.iv = InterventionConfig.from_kwargs(**kwargs)
        else:
            self.iv = InterventionConfig()  # baseline: no interventions active

        # =================================================================
        # Init-time intervention modifications (Wave 1)
        # =================================================================
        # Interventions modify the simulation in two ways:
        #   (a) Init-time: permanently alter graph attributes (importance,
        #       magnetism) or simulation parameters (daily_total, day_minutes).
        #       These changes persist for the entire day.
        #   (b) Step-time: conditionally alter arrival rates or movement
        #       weights each minute (handled in step_arrivals, _pick_next_room,
        #       _step_visitor).
        # All init-time modifications are applied below.

        # Last entry cutoff: either overridden by the intervention or the
        # standard 17:30 (555 min after 08:15 opening) [UFF].
        self.last_entry_minutes = (
            int(self.iv.last_entry_override) if self.iv.last_entry_override is not None
            else config.LAST_ENTRY_MINUTES
        )

        # Photography ban: banning photography in bottleneck rooms (A11, A12)
        # reduces dwell time by ~40%. Visitors look, absorb, and move on
        # instead of posing, waiting for clear shots, and taking multiple
        # photos. Implemented as a magnetism reduction because magnetism
        # directly scales expected dwell time. [assumption]
        if self.iv.photography_ban:
            for rid in config.PHOTO_BAN_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["magnetism"] *= config.PHOTO_BAN_MAGNETISM_FACTOR

        # Temporary exhibit: place a headline work from storage in an
        # underused room, creating a new demand attractor. The importance
        # boost makes the room compete with magnet rooms for visitor
        # attention, redistributing flow. Capped at 10.0 (max scale).
        if self.iv.temporary_exhibit_room and self.iv.temporary_exhibit_room in self.g.nodes:
            current_imp = self.g.nodes[self.iv.temporary_exhibit_room]["importance"]
            self.g.nodes[self.iv.temporary_exhibit_room]["importance"] = min(
                10.0, current_imp + config.TEMPORARY_EXHIBIT_IMPORTANCE_BOOST
            )

        # Room nobody knows: daily featured underused room announced on
        # social media. Creates FOMO and redirects attention to overlooked
        # galleries. Different from temporary exhibit because no physical
        # change is needed; the intervention is purely informational.
        if self.iv.room_nobody_knows and self.iv.room_nobody_knows in self.g.nodes:
            self.g.nodes[self.iv.room_nobody_knows]["importance"] = min(
                10.0, self.g.nodes[self.iv.room_nobody_knows]["importance"]
                + config.ROOM_NOBODY_KNOWS_IMPORTANCE_BOOST
            )

        # Conservation theater: a restorer works visibly in an underused room.
        # Fascination with the craft creates dwell time exactly where you want
        # it. Modeled as a magnetism boost (not importance) because the
        # intervention increases how long people stay, not whether they come.
        if self.iv.conservation_theater and self.iv.conservation_theater in self.g.nodes:
            self.g.nodes[self.iv.conservation_theater]["magnetism"] += (
                config.CONSERVATION_THEATER_MAGNETISM_BOOST
            )

        # Seating strategy: benches in underused rooms increase dwell time
        # (magnetism up), while removing seating from bottleneck rooms
        # (A11, A12) decreases dwell time (magnetism down). Sculpts flow
        # with furniture at near-zero cost. [assumption]
        if self.iv.seating_strategy:
            for rid in config.SEATING_BOOST_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["magnetism"] *= config.SEATING_BOOST_FACTOR
            for rid in config.SEATING_REMOVE_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["magnetism"] *= config.SEATING_REMOVE_FACTOR

        # Sound design: period-appropriate music in underused rooms makes them
        # feel like destinations rather than pass-throughs. Modeled as an
        # importance boost because the intervention changes whether visitors
        # choose to enter the room, not how long they stay once inside.
        if self.iv.sound_design:
            for rid in config.SOUND_DESIGN_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["importance"] = min(
                        10.0, self.g.nodes[rid]["importance"]
                        + config.SOUND_DESIGN_IMPORTANCE_BOOST
                    )

        # Social media spot: weekly featured "Instagram room" gets an
        # importance boost. The social-media crowd follows the designated
        # spot, steering the herd with a hashtag.
        if self.iv.social_media_spot and self.iv.social_media_spot in self.g.nodes:
            self.g.nodes[self.iv.social_media_spot]["importance"] = min(
                10.0, self.g.nodes[self.iv.social_media_spot]["importance"]
                + config.SOCIAL_MEDIA_IMPORTANCE_BOOST
            )

        # Courtyard programming: temporary installations or performances
        # in the Uffizi courtyard absorb a fraction of visitors outdoors,
        # effectively reducing the indoor daily_total. Every visitor who
        # lingers outside for 20 minutes is 20 minutes of reduced indoor
        # pressure. Modeled as a simple daily_total reduction. [assumption]
        if self.iv.courtyard_programming:
            self.daily_total = int(self.daily_total * (1.0 - config.COURTYARD_ABSORPTION_FRACTION))

        # Weather routing: outdoor/view rooms are weather-dependent.
        # Rain penalizes outdoor rooms (terrace, maps room); sun boosts them.
        # This is a coarse environmental modifier. [assumption]
        if self.iv.weather_routing == "rain":
            for rid in config.OUTDOOR_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["importance"] = max(
                        1.0, self.g.nodes[rid]["importance"] - config.WEATHER_RAIN_IMPORTANCE_PENALTY
                    )
        elif self.iv.weather_routing == "sun":
            for rid in config.OUTDOOR_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["importance"] = min(
                        10.0, self.g.nodes[rid]["importance"] + config.WEATHER_SUN_IMPORTANCE_BONUS
                    )

        # =================================================================
        # Init-time intervention modifications (Wave 2)
        # =================================================================
        # Wave 2 interventions focus on content enrichment, themed
        # programming, and temporal demand management.

        # Designated photo spots: curated rooms with professional lighting
        # where photography is actively encouraged. Draws photo-seeking
        # visitors away from bottleneck rooms (where photos are banned).
        if self.iv.designated_photo_spots:
            for rid in config.PHOTO_SPOT_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["importance"] = min(
                        10.0, self.g.nodes[rid]["importance"] + config.PHOTO_SPOT_IMPORTANCE_BOOST)

        # "The Painting Talks": first-person audio narration where the
        # painting "speaks" to the visitor. Boosts importance because it
        # makes rooms feel like must-visit experiences. [assumption]
        if self.iv.painting_talks:
            for rid in config.PAINTING_TALKS_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["importance"] = min(
                        10.0, self.g.nodes[rid]["importance"] + config.PAINTING_TALKS_IMPORTANCE_BOOST)

        # Comparative displays: physical panels linking underused rooms to
        # famous works (e.g., "see how Lorenzetti influenced Botticelli").
        # Makes minor galleries feel connected to the canonical narrative.
        if self.iv.comparative_displays:
            for rid in config.COMPARATIVE_DISPLAY_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["importance"] = min(
                        10.0, self.g.nodes[rid]["importance"] + config.COMPARATIVE_DISPLAY_IMPORTANCE_BOOST)

        # Tactile art station: room with touchable replicas of famous works.
        # Magnetism boost (not importance) because the intervention increases
        # dwell time through physical interaction, not arrival probability.
        if self.iv.tactile_art_station and self.iv.tactile_art_station in self.g.nodes:
            self.g.nodes[self.iv.tactile_art_station]["magnetism"] += config.TACTILE_STATION_MAGNETISM_BOOST

        # Themed weeks: rotating weekly themes (e.g., "Caravaggio week",
        # "Women of the Uffizi") boost both importance and magnetism for
        # theme-relevant rooms. The dual boost reflects both increased
        # visitor interest (importance) and enhanced programming that
        # encourages lingering (magnetism). [assumption]
        if self.iv.themed_weeks and self.iv.themed_weeks in config.THEMED_WEEKS:
            for rid in config.THEMED_WEEKS[self.iv.themed_weeks]:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["importance"] = min(
                        10.0, self.g.nodes[rid]["importance"] + config.THEMED_WEEK_IMPORTANCE_BOOST)
                    # Magnetism boost is 25% of the importance boost [assumption]
                    self.g.nodes[rid]["magnetism"] += 0.25 * config.THEMED_WEEK_IMPORTANCE_BOOST

        # Artist-in-residence: a living artist works in a gallery room.
        # Similar to conservation theater but with a contemporary art angle.
        # Magnetism boost because visitors stop to watch the creative process.
        if self.iv.artist_in_residence and self.iv.artist_in_residence in self.g.nodes:
            self.g.nodes[self.iv.artist_in_residence]["magnetism"] += config.ARTIST_RESIDENCE_MAGNETISM_BOOST

        # Achievement system: gamification via a digital "passport" that
        # rewards visiting underused rooms. Importance boost makes these
        # rooms more attractive in the movement model. [assumption]
        if self.iv.achievement_system:
            for rid in config.ACHIEVEMENT_UNDERUSED_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["importance"] = min(
                        10.0, self.g.nodes[rid]["importance"] + config.ACHIEVEMENT_IMPORTANCE_BOOST)

        # Cross-venue day pass: a combined ticket with Palazzo Pitti and
        # Boboli Gardens redirects some afternoon visitors to other venues.
        # The 0.5 multiplier on the reduction rate reflects that only half
        # of the shifted visitors would have been at the Uffizi otherwise.
        if self.iv.cross_venue_day_pass:
            self.daily_total = int(self.daily_total * (1.0 - config.CROSS_VENUE_AFTERNOON_REDUCTION * 0.5))

        # Seasonal hours: summer extends to 14 hours (7:00-21:00),
        # winter contracts to 8 hours (9:00-17:00). Longer hours spread
        # the same daily_total over more minutes, reducing peak density.
        # Last entry is always 60 minutes before closing.
        if self.iv.seasonal_hours == "summer":
            self.day_minutes = config.SUMMER_OPEN_MINUTES
            self.last_entry_minutes = config.SUMMER_OPEN_MINUTES - 60
        elif self.iv.seasonal_hours == "winter":
            self.day_minutes = config.WINTER_OPEN_MINUTES
            self.last_entry_minutes = config.WINTER_OPEN_MINUTES - 60

        # Last-hour locals: extend the entry window to cover the locals-only
        # discount window, so late-arriving locals can still enter.
        if self.iv.last_hour_locals:
            self.last_entry_minutes = max(self.last_entry_minutes, config.LAST_HOUR_LOCALS_WINDOW[1])

        # Resident annual pass: Florentine residents get a cheap annual pass
        # for short off-peak visits. Adds extra daily visitors with short
        # budgets (45 min). These visitors add to daily_total here; their
        # short budgets are handled in create_npc_visitor via profile override.
        if self.iv.resident_annual_pass:
            self.daily_total += config.RESIDENT_PASS_DAILY_VISITORS

        # =================================================================
        # Routing and distance precomputation
        # =================================================================
        # next_room maps each room to its successor on the recommended route.
        # Used to compute route_bonus in _pick_next_room: visitors following
        # the default itinerary get a weight boost toward the "next" room.
        self.next_room = recommended_next_map()

        # Vasari narration: an alternative chronological route designed by
        # Vasari (the architect) that follows art history rather than the
        # topological layout. Visitors on this trail use _vasari_next
        # instead of next_room for their route bias.
        if self.iv.vasari_narration:
            self._vasari_next = recommended_next_map(route=config.VASARI_ROUTE)
        else:
            self._vasari_next = None

        # Inverse map: given a room, find which room precedes it on the route.
        # Used for backtracking behavior (visitors occasionally reverse).
        self.prev_room = {b: a for a, b in self.next_room.items()}

        # All-pairs shortest paths: precomputed at init so that distance
        # lookups in _pick_next_room and low-budget exit-seeking are O(1).
        self._distances = all_pairs_shortest_paths(self.g)

        # =================================================================
        # Arrival normalization constants
        # =================================================================
        # Each arrival model (Gaussian, timed-entry, forecast) needs a
        # normalization constant so that the sum of per-minute rates over
        # [0, last_entry_minutes] equals daily_total. These are computed
        # once at init time; the per-minute rate is then envelope(t) * norm.
        self._arrival_norm = self._compute_arrival_normalization()
        if self.iv.timed_entry:
            self._timed_entry_norm = self._compute_timed_entry_normalization()
        if self.iv.dynamic_pricing:
            self._price_schedule = self._build_price_schedule()
        if self.iv.crowd_forecast:
            self._forecast_norm = self._compute_forecast_normalization()

        # =================================================================
        # Per-step tracking state
        # =================================================================
        # Botticelli entry counter: reset each step. Used by the
        # botticelli_slot_cap intervention to enforce a per-minute
        # admission limit into rooms A11/A12.
        self._botticelli_entries_this_step = 0

        # Tour group tracking: monotonic group ID counter and a map
        # from group_id to member visitor_ids. Groups share a profile
        # and checklist and are created together in step_arrivals.
        self._tour_group_counter = 0
        self._active_tour_groups: Dict[int, List[int]] = {}

        # Monotonic visitor ID counter, incremented in create_npc_visitor.
        self._visitor_counter = 0

        # Initialize the within-day state (occupancy, visitor lists, etc.).
        self.reset_day()

    # =================================================================
    # Arrival model: normalization helpers
    # =================================================================

    def _compute_arrival_normalization(self) -> float:
        """Compute the scalar that normalizes the Gaussian arrival envelope.

        The Gaussian envelope g(t) = exp(-0.5 * ((t - peak) / sigma)^2)
        is evaluated at each minute t in [0, last_entry_minutes]. The
        normalization constant is daily_total / sum(g(t)), so that
        sum(g(t) * norm) = daily_total, i.e., the expected total arrivals
        over the full entry window equal daily_total.

        Returns
        -------
        float
            Normalization constant. The per-minute arrival rate at time t
            is envelope(t) * norm, which is then passed to Poisson sampling.
        """
        times = np.arange(self.last_entry_minutes + 1)
        envelope = np.exp(-0.5 * ((times - _ARRIVAL_PEAK_MINUTES) / _ARRIVAL_SIGMA_MINUTES) ** 2)
        denom = float(envelope.sum())
        if denom <= 0:
            return 0.0
        return self.daily_total / denom

    def _compute_timed_entry_normalization(self) -> float:
        """Flat arrival distribution for timed-entry intervention.

        Instead of a Gaussian peak, visitors are spread uniformly across
        the entry window with a slight morning bias (slot weights follow
        a broad, flat-topped distribution). This reduces peak-hour
        concentration, which is the mechanism by which timed entry works.
        """

        times = np.arange(self.last_entry_minutes + 1)
        # Broad plateau with gentle taper at edges: much flatter than Gaussian.
        peak = self.last_entry_minutes / 2.0
        sigma = self.last_entry_minutes / 1.5
        envelope = np.exp(-0.5 * ((times - peak) / sigma) ** 2)
        denom = float(envelope.sum())
        if denom <= 0:
            return 0.0
        return self.daily_total / denom

    def _build_price_schedule(self) -> np.ndarray:
        """Build per-minute price multiplier array for dynamic pricing.

        Each minute gets a price multiplier from PRICE_WINDOWS. Visitors
        with willingness-to-pay below the price are deterred (reduce
        arrival rate for that minute). The effect redistributes demand
        from expensive (peak) slots to cheap (off-peak) slots.
        """

        schedule = np.ones(self.last_entry_minutes + 1, dtype=float)
        for start, end, mult in config.PRICE_WINDOWS:
            lo = min(start, self.last_entry_minutes)
            hi = min(end, self.last_entry_minutes + 1)
            schedule[lo:hi] = mult
        return schedule

    def _compute_forecast_normalization(self) -> float:
        """Crowd forecast at booking: visitors see predicted density per slot
        and a fraction shift away from peak hours. The predicted density is
        approximated by the standard Gaussian arrival envelope. Visitors who
        respond to the forecast avoid high-density slots, flattening demand.
        """

        times = np.arange(self.last_entry_minutes + 1)
        base_envelope = np.exp(-0.5 * ((times - _ARRIVAL_PEAK_MINUTES) / _ARRIVAL_SIGMA_MINUTES) ** 2)
        # Forecast response: visitors who see high predicted density shift
        # to lower-density times. Effect: invert and mix with original.
        response = config.CROWD_FORECAST_RESPONSE_RATE
        adjusted = base_envelope * (1.0 - response) + np.mean(base_envelope) * response
        denom = float(adjusted.sum())
        if denom <= 0:
            return 0.0
        return self.daily_total / denom

    # =================================================================
    # Day lifecycle
    # =================================================================

    def reset_day(self) -> None:
        """Reset all within-day simulator state while preserving configuration.

        Called at the start of each day (both in run_day and when the RL
        environment calls reset). Graph attributes, intervention config,
        and precomputed constants are preserved; only mutable per-day
        state is cleared.
        """

        self.current_time = 0  # minutes since 08:15 opening

        # Visitor population pools:
        #   outside_queue: visitors waiting to enter (capacity-gated FIFO)
        #   active_visitors: visitors currently inside the museum
        #   completed_visitors: visitors who have exited (for end-of-day metrics)
        self.outside_queue: Deque[NPCVisitor] = deque()
        self.active_visitors: List[NPCVisitor] = []
        self.completed_visitors: List[NPCVisitor] = []

        # Per-room occupancy counters. Three separate dicts to enable
        # type-specific density lookups (Type A perceives Type B as noisier
        # via the asymmetric externality model in type_a_effective_density).
        self.occ = {room: 0 for room in self.g.nodes}    # total occupancy
        self.occ_a = {room: 0 for room in self.g.nodes}  # Type A only
        self.occ_b = {room: 0 for room in self.g.nodes}  # Type B only
        self.total_inside = 0  # sum of all occ values; cached for O(1) capacity check

        # Day-level summary statistics, computed incrementally.
        self.max_total_inside = 0    # peak simultaneous occupancy
        self.capacity_violations = 0  # minutes where total_inside > 900

        # Time-series logs for metrics and RL observation construction.
        self.density_history: List[np.ndarray] = []      # N_ROOMS-dim vector per step
        self.total_inside_history: List[int] = []         # scalar per step
        self.queue_history: List[int] = []                # outside queue length per step

    # =================================================================
    # Arrival model: per-minute rate computation
    # =================================================================

    def desired_arrival_rate(self, t: int) -> float:
        """Return the expected Poisson arrival intensity for minute ``t``.

        This is the core arrival model. It computes the expected number of
        new visitors per minute by:
          1. Evaluating the base envelope (Gaussian, timed-entry, or forecast)
          2. Applying multiplicative intervention modifiers (dynamic pricing,
             lunch free entry, last-hour locals, etc.)
          3. Applying hard intervention overrides (breathing pause, occupancy cap)

        The returned rate is passed to rng.poisson() in step_arrivals to
        sample the actual number of arrivals (integer, stochastic).

        Parameters
        ----------
        t : int
            Current minute since museum opening (0 = 08:15).

        Returns
        -------
        float
            Expected arrivals per minute (Poisson lambda).
        """

        # No arrivals after the last entry cutoff (default: minute 555 = 17:30).
        if t > self.last_entry_minutes:
            return 0.0

        # ----- Base arrival envelope -----
        # Three mutually exclusive modes:
        if self.iv.timed_entry:
            # Timed entry: a broad, flat-topped distribution that spreads
            # visitors uniformly across the entry window. The peak is at the
            # midpoint and sigma is very wide (last_entry/1.5), so the
            # curve is nearly flat. This is the mechanism by which timed
            # entry reduces peak-hour concentration.
            peak = self.last_entry_minutes / 2.0
            sigma = self.last_entry_minutes / 1.5
            envelope = np.exp(-0.5 * ((t - peak) / sigma) ** 2)
            rate = float(envelope * self._timed_entry_norm)
        elif self.iv.crowd_forecast:
            # Crowd forecast: visitors see predicted density per slot at
            # booking time and a fraction (CROWD_FORECAST_RESPONSE_RATE)
            # shift away from peak hours. The adjusted envelope is a mix
            # of the original Gaussian and a flat mean, partially flattening
            # the peak while preserving total volume.
            envelope = np.exp(-0.5 * ((t - _ARRIVAL_PEAK_MINUTES) / _ARRIVAL_SIGMA_MINUTES) ** 2)
            response = config.CROWD_FORECAST_RESPONSE_RATE
            times = np.arange(self.last_entry_minutes + 1)
            base_mean = float(
                np.exp(-0.5 * ((times - _ARRIVAL_PEAK_MINUTES) / _ARRIVAL_SIGMA_MINUTES) ** 2).mean()
            )
            # Mix: (1-response)*original + response*flat_mean
            adjusted = envelope * (1.0 - response) + base_mean * response
            rate = float(adjusted * self._forecast_norm)
        else:
            # Default: standard Gaussian envelope centered at 10:30 AM [A22].
            envelope = np.exp(-0.5 * ((t - _ARRIVAL_PEAK_MINUTES) / _ARRIVAL_SIGMA_MINUTES) ** 2)
            rate = float(envelope * self._arrival_norm)

        # ----- Multiplicative intervention modifiers -----

        # Dynamic pricing: scale arrival rate by inverse price sensitivity.
        # High price -> fewer arrivals at this time, redistributed to
        # cheaper windows via the overall normalization.
        if self.iv.dynamic_pricing and t < len(self._price_schedule):
            price_mult = self._price_schedule[t]
            # Demand elasticity: rate scales as 1/price^elasticity.
            # Elasticity of 0.5 means a 2x price reduces demand by ~30%.
            # This is a standard constant-elasticity demand model. [assumption]
            elasticity = 0.5
            rate *= float(price_mult ** (-elasticity))

        # Lunch free entry: free admission during 12:30-14:00 boosts
        # arrivals, especially for local office workers. [assumption]
        if self.iv.lunch_free_entry:
            lo, hi = config.LUNCH_FREE_WINDOW
            if lo <= t <= hi:
                rate *= config.LUNCH_FREE_ARRIVAL_BOOST

        # Last-hour locals: reduced-price entry in the final 90 minutes
        # attracts Florentine residents for short visits. Both a
        # multiplicative boost and an additive constant (extra visitors
        # spread uniformly across the window). [assumption]
        if self.iv.last_hour_locals:
            lo, hi = config.LAST_HOUR_LOCALS_WINDOW
            if lo <= t <= hi:
                rate *= config.LAST_HOUR_LOCALS_BOOST
                rate += config.LAST_HOUR_LOCALS_EXTRA_VISITORS / max(1, hi - lo + 1)

        # ----- Hard overrides -----

        # Breathing pause: 30-minute closure at 13:00. Zero arrivals.
        if self.iv.breathing_pause:
            if config.BREATHING_PAUSE_START <= t < config.BREATHING_PAUSE_START + config.BREATHING_PAUSE_DURATION:
                rate = 0.0

        # Quiet hours: tour group suppression during the quiet window.
        # (Tour group fraction is zeroed in step_arrivals, not here;
        # individual arrival rate is unaffected.)

        # Real-time occupancy pricing: when the museum exceeds 800 people
        # (89% of the 900 cap), the ticket office slows admission by 30%.
        # This creates a soft pre-cap throttle. [assumption]
        if self.iv.realtime_occupancy_pricing and self.total_inside > 800:
            rate *= 0.7

        return rate

    # =================================================================
    # Visitor creation
    # =================================================================

    def create_npc_visitor(self, entry_slot: int) -> NPCVisitor:
        """Sample a visitor profile, time budget, and ticketing attributes.

        This is the visitor factory. Each call produces one NPCVisitor with:
          - A behavioral profile (Type A or B, with importance vector,
            crowd sensitivity, route bias, dwell multiplier, etc.)
          - A time budget (minutes until the visitor must leave)
          - A checklist of top-8 must-see rooms (drives route bonus)
          - Optional Botticelli window, trail assignment, ticket price

        The profile is modified by active interventions (adaptive trail,
        skip-the-famous, themed weeks, Vasari narration) before the
        visitor is created. These modifications change the visitor's
        importance vector or route bias, altering their movement behavior.

        Parameters
        ----------
        entry_slot : int
            Index of the 15-minute entry slot (0-36). Determines time
            budget distribution and ticket pricing.

        Returns
        -------
        NPCVisitor
            Fully initialized visitor ready to enter the outside queue.
        """

        # Trail acceptance probability: under adaptive_trail, the base
        # probability is zero; acceptance is computed dynamically below
        # based on current Botticelli density.
        base_trail_prob = self.trail_acceptance_prob
        if self.iv.adaptive_trail:
            base_trail_prob = 0.0

        # ----- Profile sampling -----
        # Two models: the six-segment model (richer heterogeneity with
        # named segments like "family", "student", "connoisseur") or the
        # two-type model (Type A/B). Both produce a VisitorProfile.
        if self.iv.six_segment_model:
            from uffizi_rl.environment.visitor_segments import sample_segmented_profile

            profile = sample_segmented_profile(
                self.rng,
                heterogeneity_scale=self.heterogeneity_scale,
                trail_acceptance_prob=base_trail_prob,
                treasure_hunt=self.iv.children_treasure_hunt,
            )
        else:
            profile = sample_profile(
                self.rng,
                type_a_fraction=self.type_a_fraction,
                heterogeneity_scale=self.heterogeneity_scale,
                trail_acceptance_prob=base_trail_prob,
            )

        # ----- Adaptive trail assignment -----
        # When Botticelli is congested, Type A visitors are more likely to
        # accept a hidden-gem trail. The acceptance probability rises with
        # current Botticelli density (congestion makes alternatives more
        # appealing). The trail chosen is whichever has the lowest current
        # average density across its rooms, steering visitors toward the
        # least-crowded alternative itinerary.
        if self.iv.adaptive_trail and profile.name == "A":
            adaptive_acceptance = min(
                0.85,  # cap at 85% to preserve some organic behavior [assumption]
                self.trail_acceptance_prob + 0.25 + 0.20 * min(1.0, self.botticelli_density()),
            )
            if self.rng.random() < adaptive_acceptance:
                # Pick the trail with the lowest average current density.
                trail_name = min(
                    config.HIDDEN_GEM_TRAILS,
                    key=lambda name: np.mean(
                        [self.density(room) for room in config.HIDDEN_GEM_TRAILS[name]]
                    ),
                )
                profile = apply_trail_to_profile(profile, trail_name=trail_name)

        # ----- Skip-the-famous intervention -----
        # A fraction of visitors have magnet rooms (Botticelli, Leonardo,
        # Caravaggio, etc.) downgraded to background importance in their
        # personal importance vector. Simulates discounted tickets that
        # exclude access to top rooms, or visitors who have already seen
        # the famous works and want to explore other galleries.
        if self.iv.skip_the_famous > 0 and self.rng.random() < self.iv.skip_the_famous:
            imp = profile.importance_vector.copy()
            for room in config.TYPE_B_MAGNET_ROOMS:
                imp[config.ROOM_TO_IDX[room]] = config.TYPE_B_BACKGROUND_IMPORTANCE
            profile = VisitorProfile(
                name=profile.name,
                crowd_alpha=profile.crowd_alpha,
                route_bias=profile.route_bias,
                backtrack_prob=profile.backtrack_prob,
                dwell_multiplier=profile.dwell_multiplier,
                anti_crowd_bonus=profile.anti_crowd_bonus,
                importance_vector=imp,
                trail_name=profile.trail_name,
                segment=profile.segment,
                time_budget_override=profile.time_budget_override,
            )

        # ----- Themed weeks: per-visitor importance boost -----
        # 45% of visitors respond to the themed week by boosting their
        # personal importance for theme-relevant rooms. Their route bias
        # is slightly reduced (-0.03) to encourage exploration toward
        # themed rooms even if they are not on the standard route. [assumption]
        if self.iv.themed_weeks and self.iv.themed_weeks in config.THEMED_WEEKS:
            if self.rng.random() < 0.45:
                imp = profile.importance_vector.copy()
                for room in config.THEMED_WEEKS[self.iv.themed_weeks]:
                    imp[config.ROOM_TO_IDX[room]] = np.clip(
                        imp[config.ROOM_TO_IDX[room]] + config.THEMED_WEEK_IMPORTANCE_BOOST,
                        0.5,
                        10.0,
                    )
                profile = VisitorProfile(
                    name=profile.name,
                    crowd_alpha=profile.crowd_alpha,
                    route_bias=max(0.2, profile.route_bias - 0.03),
                    backtrack_prob=profile.backtrack_prob,
                    dwell_multiplier=profile.dwell_multiplier,
                    anti_crowd_bonus=profile.anti_crowd_bonus,
                    importance_vector=imp,
                    trail_name=profile.trail_name or f"theme:{self.iv.themed_weeks}",
                    segment=profile.segment,
                    time_budget_override=profile.time_budget_override,
                )

        # ----- Botticelli time window assignment -----
        # Smart defaults / decoupled gating: each visitor is assigned a
        # 30-minute window during which they may enter Botticelli rooms
        # (A11/A12). The window starts at a random offset (30-120 min)
        # after their entry time, ensuring they explore other rooms first.
        # This spreads Botticelli demand across the day.
        bott_window = None
        if self.iv.decoupled_botticelli_gating or self.iv.smart_defaults:
            window_dur = config.SMART_DEFAULT_BOTTICELLI_WINDOW  # 30 min [assumption]
            offset = int(self.rng.integers(30, 120))  # random delay before Botticelli
            window_start = entry_slot * config.ENTRY_SLOT_MINUTES + offset
            window_start = min(window_start, self.day_minutes - window_dur)  # clamp to closing
            bott_window = (window_start, window_start + window_dur)

        # ----- Vasari narration assignment -----
        # 25% of visitors without an existing trail adopt the Vasari
        # chronological route. Their route bias is reduced (-0.1, floor 0.3)
        # because the Vasari route crosses the standard route multiple times,
        # requiring more willingness to deviate from the default path.
        if self.iv.vasari_narration and profile.trail_name is None:
            if self.rng.random() < config.VASARI_ADOPTION:
                profile = VisitorProfile(
                    name=profile.name,
                    crowd_alpha=profile.crowd_alpha,
                    route_bias=max(0.3, profile.route_bias - 0.1),
                    backtrack_prob=profile.backtrack_prob,
                    dwell_multiplier=profile.dwell_multiplier,
                    anti_crowd_bonus=profile.anti_crowd_bonus,
                    importance_vector=profile.importance_vector,
                    trail_name="vasari",
                    segment=profile.segment,
                    time_budget_override=profile.time_budget_override,
                )

        # ----- Checklist construction -----
        # The visitor's top-8 rooms by personal importance (excluding
        # non-gallery nodes like ENTRY, EXIT, staircases, terrace).
        # Rooms on the checklist get a +1.5 route bonus in _pick_next_room,
        # guiding the visitor toward personally important destinations.
        room_scores = {
            rid: float(profile.importance_vector[config.ROOM_TO_IDX[rid]])
            for rid in config.ROOM_IDS
            if rid not in {"ENTRY", "EXIT", "LANZI_STAIRCASE", "BUONTALENTI_STAIRCASE", "PANORAMIC_TERRACE"}
        }
        top_rooms = dict(sorted(room_scores.items(), key=lambda kv: kv[1], reverse=True)[:8])

        # ----- Time budget -----
        # If the profile has a segment-specific override (e.g., resident
        # annual pass holders get 45-minute budgets), use that. Otherwise,
        # sample from the slot-dependent log-normal distribution [A22].
        if profile.time_budget_override is not None:
            budget = profile.time_budget_override
        else:
            budget = config.sample_visit_duration(entry_slot, self.rng)

        # Themed week visitors stay 10% longer (more to see). [assumption]
        if self.iv.themed_weeks and profile.trail_name == f"theme:{self.iv.themed_weeks}":
            budget = int(round(1.10 * budget))

        # ----- Revenue model -----
        # Compute ticket price from entry time and visitor type.
        # Only active when revenue_model intervention is enabled.
        price = 0.0
        is_walk_in = False
        if self.iv.revenue_model:
            price, is_walk_in = self.sample_ticket_price(entry_slot)

        # ----- Assemble and return the visitor -----
        self._visitor_counter += 1
        return NPCVisitor(
            visitor_id=self._visitor_counter,
            visitor_type=profile.name,
            profile=profile,
            current_room=None,        # None until admitted past the capacity gate
            remaining_budget=budget,
            entry_slot=entry_slot,
            checklist=top_rooms,
            botticelli_window=bott_window,
            ticket_price=price,
            is_walk_in=is_walk_in,
        )

    # =================================================================
    # Revenue model: ticket pricing
    # =================================================================

    @staticmethod
    def scheduled_ticket_price(entry_minute: int) -> float:
        """Return the scheduled full-price ticket for a given entry minute.

        Looks up the time-of-day price from TICKET_PRICE_SCHEDULE.
        Falls back to the last window's price if the minute is beyond
        all defined windows.

        Parameters
        ----------
        entry_minute : int
            Minute since opening (0 = 08:15). Converted from entry_slot
            by the caller.

        Returns
        -------
        float
            Ticket price in EUR.
        """

        for start, end, price in config.TICKET_PRICE_SCHEDULE:
            if start <= entry_minute < end:
                return float(price)
        return float(config.TICKET_PRICE_SCHEDULE[-1][2])

    def sample_ticket_price(self, entry_slot: int) -> tuple[float, bool]:
        """Sample one visitor's ticket price under the revenue model.

        The visitor population is stratified into:
          - Free visitors (30%): under-18, students, disabled [assumption]
          - Reduced visitors (10%): EU 18-25 at EUR 2 [UFF]
          - Walk-in visitors (15% of the remainder): pay EUR 35 premium
          - Last-hour locals: pay EUR 10 reduced rate
          - Everyone else: pay the scheduled time-of-day price

        Parameters
        ----------
        entry_slot : int
            Entry slot index (0-36).

        Returns
        -------
        tuple of (float, bool)
            (price_eur, is_walk_in). is_walk_in is True only for
            unplanned visitors paying the walk-in premium.
        """

        roll = self.rng.random()
        # Free entry: ~30% of visitors (under-18, students, disabled, etc.)
        if roll < config.FREE_VISITOR_FRACTION:
            return 0.0, False
        # Reduced entry: ~10% of visitors (EU 18-25 at EUR 2) [UFF]
        if roll < config.FREE_VISITOR_FRACTION + config.REDUCED_VISITOR_FRACTION:
            return float(config.REDUCED_TICKET_PRICE), False

        # Walk-in premium: unplanned visitors without a reservation [assumption]
        if self.iv.walk_in_premium and self.rng.random() < config.WALK_IN_FRACTION:
            return float(config.WALK_IN_TICKET_PRICE), True

        # Last-hour locals: reduced evening price for Florentine residents
        entry_minute = entry_slot * config.ENTRY_SLOT_MINUTES
        if self.iv.last_hour_locals and entry_minute >= config.LAST_HOUR_LOCALS_WINDOW[0]:
            return float(config.LAST_HOUR_LOCALS_PRICE), False

        # Standard time-of-day pricing
        return self.scheduled_ticket_price(entry_minute), False

    # =================================================================
    # Arrival processing
    # =================================================================

    def step_arrivals(self) -> None:
        """Generate arrivals for the current minute and admit up to capacity.

        This method implements the full arrival pipeline:
          1. Compute the Poisson rate from the arrival envelope + modifiers
          2. Apply walk-in deterrence and quiet-hours rate reduction
          3. Compute the effective tour group fraction (surcharge, quiet hours)
          4. Sample the actual arrival count from Poisson(rate)
          5. Create visitors (individuals or tour groups) and enqueue them
          6. Admit queued visitors FIFO up to MAX_MUSEUM_CAPACITY (900)

        The capacity gate (step 6) is the mechanism that creates the outside
        queue. When the museum is full, new arrivals wait in the queue until
        space opens up (visitors exit). This is the hard constraint that
        prevents the 900-person legal limit from being violated.
        """

        rate = self.desired_arrival_rate(self.current_time)

        # Walk-in premium: the existence of a premium price deters a fraction
        # of unplanned walk-ins, reducing the overall arrival rate.
        # WALK_IN_FRACTION * WALK_IN_DETERRENCE = 0.15 * 0.40 = 6% reduction.
        if self.iv.walk_in_premium:
            rate *= (1.0 - config.WALK_IN_FRACTION * config.WALK_IN_DETERRENCE)

        # Quiet hours: during the quiet window (08:15-11:00), overall
        # arrival rate is reduced by 15% because the quiet atmosphere
        # is marketed to a more selective audience. [assumption]
        if self.iv.quiet_hours:
            lo, hi = config.QUIET_HOURS_WINDOW
            if lo <= self.current_time <= hi:
                rate *= 0.85

        # ----- Tour group fraction -----
        # Tour groups create corridor "shock waves" by moving large
        # coordinated blocks of Type B visitors together. The fraction
        # of arrivals that form groups is modulated by interventions.
        base_group_frac = config.TOUR_GROUP_FRACTION  # 15% baseline [assumption]
        if self.iv.quiet_hours:
            lo, hi = config.QUIET_HOURS_WINDOW
            if lo <= self.current_time <= hi:
                base_group_frac = 0.0  # no tour groups during quiet hours
        # Per-person surcharge: higher surcharge reduces group demand.
        # At surcharge = reference_price (EUR 25), group fraction drops to 0.
        # The 20% overall rate reduction reflects that groups generate
        # administrative overhead that limits total throughput. [assumption]
        if self.iv.per_person_group_surcharge > 0:
            reduction = self.iv.per_person_group_surcharge / config.GROUP_SURCHARGE_REFERENCE_PRICE
            base_group_frac = max(0.0, base_group_frac * (1.0 - reduction))
            rate *= max(0.0, 1.0 - 0.20 * min(1.0, reduction))

        # ----- Sample arrival count -----
        # Poisson draw: stochastic integer count of new arrivals this minute.
        new_arrivals = int(self.rng.poisson(max(0.0, rate)))

        # Map current_time to the 15-minute entry slot index (0-36).
        entry_slot = min(config.N_ENTRY_SLOTS - 1, self.current_time // config.ENTRY_SLOT_MINUTES)

        # Tour group size: capped by intervention or default (30). [assumption]
        group_size = (
            self.iv.tour_group_cap if self.iv.tour_group_cap is not None
            else config.TOUR_GROUP_SIZE_DEFAULT
        )

        # ----- Create visitors: individuals and tour groups -----
        i = 0
        while i < new_arrivals:
            if self.rng.random() < base_group_frac and (new_arrivals - i) >= group_size:
                # Tour group: all members share the leader's profile and
                # checklist. They move together (same Type B behavior with
                # high route bias). This models the correlated movement
                # pattern of guided tours.
                self._tour_group_counter += 1
                gid = self._tour_group_counter
                leader = self.create_npc_visitor(entry_slot=entry_slot)
                leader.group_id = gid
                # Force Type B behavior: tour groups follow the guide,
                # not personal preferences.
                leader.profile = sample_type_b_profile(self.rng)
                leader.visitor_type = "B"
                if self.iv.revenue_model and self.iv.per_person_group_surcharge > 0:
                    leader.ticket_price += self.iv.per_person_group_surcharge
                self.outside_queue.append(leader)
                for _ in range(group_size - 1):
                    follower = self.create_npc_visitor(entry_slot=entry_slot)
                    follower.group_id = gid
                    # Clone leader's profile and checklist to ensure
                    # synchronized movement within the group.
                    follower.profile = leader.profile
                    follower.visitor_type = "B"
                    follower.checklist = dict(leader.checklist)
                    if self.iv.revenue_model and self.iv.per_person_group_surcharge > 0:
                        follower.ticket_price += self.iv.per_person_group_surcharge
                    self.outside_queue.append(follower)
                i += group_size
            else:
                # Individual visitor: independent profile and checklist.
                self.outside_queue.append(self.create_npc_visitor(entry_slot=entry_slot))
                i += 1

        # ----- Capacity-gated admission -----
        # Admit visitors from the FIFO queue as long as the museum is
        # below the 900-person legal capacity limit [UFF]. All visitors
        # start in room A1 (Lorenese Vestibule, the museum entrance).
        while self.outside_queue and self.total_inside < config.MAX_MUSEUM_CAPACITY:
            v = self.outside_queue.popleft()
            v.current_room = "A1"       # all visitors enter through the vestibule
            v.time_in_room = 0
            v.rooms_visited.add("A1")
            self.active_visitors.append(v)
            self._increment_occ("A1", v.visitor_type)

    # =================================================================
    # Occupancy bookkeeping
    # =================================================================
    # Three parallel counters (occ, occ_a, occ_b) are maintained because
    # the asymmetric externality model requires knowing the type-specific
    # occupancy: Type A visitors perceive Type B visitors as 1.5x noisier
    # than fellow Type A visitors (TYPE_A_CROSS_TYPE_EXTERNALITY) [assumption].

    def _increment_occ(self, room: str, visitor_type: str) -> None:
        """Record a visitor entering a room. Updates all three counters."""
        self.occ[room] += 1
        if visitor_type == "A":
            self.occ_a[room] += 1
        else:
            self.occ_b[room] += 1
        self.total_inside += 1

    def _decrement_occ(self, room: str, visitor_type: str) -> None:
        """Record a visitor leaving a room. Clamped at 0 for safety."""
        self.occ[room] = max(0, self.occ[room] - 1)
        if visitor_type == "A":
            self.occ_a[room] = max(0, self.occ_a[room] - 1)
        else:
            self.occ_b[room] = max(0, self.occ_b[room] - 1)
        self.total_inside = max(0, self.total_inside - 1)

    # =================================================================
    # Stay/leave decision model
    # =================================================================

    @staticmethod
    def npc_stay_probability(
        room_magnetism: float,
        time_in_room: int,
        dwell_multiplier: float,
        base_dwell: float = config.BASE_DWELL_MINUTES,
    ) -> float:
        """Probability of staying one more minute in the current room.

        The stay probability follows an exponential decay model:
            p(stay) = exp(-time_in_room / expected_dwell)

        where expected_dwell = base_dwell * room_magnetism * dwell_multiplier.

        Intuition: the longer a visitor has been in a room, the less likely
        they are to stay another minute. High-magnetism rooms (e.g., Botticelli
        with magnetism=5.0) have longer expected dwell times, so visitors
        linger naturally. The dwell_multiplier encodes type-specific behavior:
        Type A (1.0) lingers when uncrowded; Type B (0.3) takes a quick photo
        and moves on. [assumption]

        The probability is clamped to [0.05, 0.95]:
          - Floor of 0.05 ensures visitors eventually leave every room
          - Ceiling of 0.95 prevents visitors from being "stuck" forever
            in the first minute of high-magnetism rooms

        Parameters
        ----------
        room_magnetism : float
            Room's magnetism attribute from the graph (1.0 = neutral).
        time_in_room : int
            Minutes already spent in this room.
        dwell_multiplier : float
            Visitor-type dwell scaling (1.0 for A, 0.3 for B) [assumption].
        base_dwell : float
            Base expected dwell in a magnetism-1.0 room (default 3 min).

        Returns
        -------
        float
            Probability in [0.05, 0.95] of staying one more minute.
        """

        # Floor of 0.5 minutes prevents division-by-near-zero when
        # magnetism and/or dwell_multiplier are very small.
        expected_dwell = max(0.5, base_dwell * room_magnetism * dwell_multiplier)
        p = float(np.exp(-time_in_room / expected_dwell))
        return float(np.clip(p, 0.05, 0.95))

    # =================================================================
    # Density and distance helpers
    # =================================================================

    def density(self, room: str) -> float:
        """Room density: occupancy / capacity. Values > 1.0 = overcrowded."""
        cap = self.g.nodes[room]["capacity"]
        return self.occ[room] / max(1.0, cap)

    def _effective_density_for(self, room: str, visitor_type: str) -> float:
        """Type-specific perceived density.

        Type A visitors perceive Type B visitors as 1.5x noisier (the
        asymmetric cross-type externality from config). This makes Type A
        visitors more strongly repelled by rooms full of tour groups.
        Type B visitors perceive raw density (crowd-insensitive). [assumption]
        """
        cap = self.g.nodes[room]["capacity"]
        if visitor_type == "A":
            return type_a_effective_density(self.occ_a[room], self.occ_b[room], cap)
        return self.occ[room] / max(1.0, cap)

    def _distance(self, src: str, dst: str) -> int:
        """O(1) shortest-path lookup from precomputed all-pairs distances.

        Returns 999 (effectively infinity) for unreachable pairs, which
        ensures the exit-seeking heuristic never divides by zero.
        """

        src_dists = self._distances.get(src)
        if src_dists is None:
            return 999
        return src_dists.get(dst, 999)

    # =================================================================
    # Movement model: next-room selection
    # =================================================================

    def _pick_next_room(self, visitor: NPCVisitor) -> str:
        """Choose which adjacent room the visitor moves to next.

        This is the core movement model. It computes a weight for each
        neighboring room and samples one proportionally. The weight for
        each neighbor n combines three additive components:

          weight(n) = route_bonus(n) + importance_pull(n) + anti_crowd(n)

        1. Route bonus: baseline 1.0 for all neighbors, +3.0*route_bias if
           n is the "next" room on the recommended itinerary, +1.5 if n is
           on the visitor's personal checklist. This is the main force
           keeping visitors on the standard path.

        2. Importance pull: 0.05 + 0.08*room_importance + 0.10*personal_importance.
           Blends the room's objective cultural significance (graph attribute)
           with the visitor's personal taste (profile importance vector).
           This ensures that content interventions (which modify graph-level
           importance) change movement incentives while still respecting
           heterogeneous preferences.

        3. Anti-crowd avoidance: Type A visitors have a positive anti_crowd
           bonus that rewards moving toward less-dense rooms. Type B visitors
           are crowd-indifferent (anti_crowd_bonus = 0) unless they are at
           a kiosk room with dynamic info displays or carry an adaptive
           audio guide.

        Special cases:
          - Low budget (<15 min): visitor ignores normal weights and instead
            biases toward rooms closest to the EXIT (exit-seeking heuristic).
          - Backtracking: with probability backtrack_prob, the visitor
            reverses direction along the recommended route (returns to
            the previous room).

        Parameters
        ----------
        visitor : NPCVisitor
            The visitor choosing their next room.

        Returns
        -------
        str
            Room ID of the chosen next room.
        """
        if visitor.current_room is None:
            return "A1"  # should not happen, but safe fallback
        room = visitor.current_room

        neighbors = sorted(self.g.neighbors(room))
        if not neighbors:
            return room  # dead-end with no neighbors (should not occur)

        # ----- Exit-seeking heuristic for low-budget visitors -----
        # When a visitor has fewer than 15 minutes of budget left, they
        # prioritize reaching the EXIT. The weight for each neighbor is
        # inversely proportional to its shortest-path distance to EXIT.
        # This prevents visitors from wandering into dead ends when they
        # should be heading out.
        if visitor.remaining_budget < 15:
            exit_scores = []
            for n in neighbors:
                d = self._distance(n, "EXIT")
                exit_scores.append(1.0 / (1.0 + d))  # higher score = closer to exit
            probs = config.normalize_weights(exit_scores)
            return neighbors[int(self.rng.choice(np.arange(len(neighbors)), p=probs))]

        # ----- Occasional backtracking -----
        # With small probability (5% for Type A, 2% for Type B [assumption]),
        # the visitor reverses direction. This prevents perfectly deterministic
        # flow patterns and adds realistic randomness.
        if self.rng.random() < visitor.profile.backtrack_prob:
            prev = self.prev_room.get(room)
            if prev in neighbors:
                return prev

        # ----- Compute per-neighbor weights -----
        weights = []

        # Determine which route map to use for the route bonus.
        # Vasari narration visitors follow the chronological route.
        if self._vasari_next and visitor.profile.trail_name == "vasari":
            next_pref = self._vasari_next.get(room)
        else:
            next_pref = self.next_room.get(room)

        # Pre-check step-time intervention flags (computed once per call,
        # not once per neighbor, for efficiency).

        # Dynamic info: NPCs at kiosk rooms gain crowd-avoidance behavior
        # regardless of type, because they can see real-time density displays.
        info_anti_crowd = (
            self.iv.dynamic_info and room in config.KIOSK_ROOMS
        )
        # Adaptive audio guide: a fraction of visitors (60% [assumption])
        # carry an audio guide that provides crowd-adjusted movement
        # recommendations. Sampled per movement decision.
        has_audio_guide = (
            self.iv.adaptive_audio_guide
            and self.rng.random() < config.AUDIO_GUIDE_ADOPTION
        )
        # Micro-events: during peak hours (10:00-13:00), scheduled docent
        # talks in specific underused rooms create temporary attractors.
        micro_event_active = (
            self.iv.micro_events
            and config.MICRO_EVENT_PEAK_WINDOW[0] <= self.current_time <= config.MICRO_EVENT_PEAK_WINDOW[1]
        )

        for n in neighbors:
            idx = config.ROOM_TO_IDX[n]

            # --- Component 1: Importance pull ---
            # Blends graph-level room importance (modified by interventions)
            # with the visitor's personal importance vector. The coefficients
            # (0.05 base, 0.08 room, 0.10 personal) were tuned so that a
            # room with importance=10 and personal=10 contributes ~1.85 to
            # the weight, comparable to the route bonus. [assumption]
            room_importance = float(self.g.nodes[n]["importance"])
            personal_importance = float(visitor.profile.importance_vector[idx])
            importance_pull = 0.05 + 0.08 * room_importance + 0.10 * personal_importance

            # Micro-events: rooms hosting a docent talk get a bonus.
            if micro_event_active and n in config.MICRO_EVENT_ROOMS:
                importance_pull += 0.15 * config.MICRO_EVENT_IMPORTANCE_BOOST

            # --- Component 2: Anti-crowd avoidance ---
            # Positive when the room is uncrowded (density < 1.0),
            # negative when overcrowded (density > 1.0). Capped at
            # density=1.5 to prevent extreme negative weights.
            anti_crowd = 0.0
            if visitor.visitor_type == "A":
                # Type A: strong crowd avoidance using asymmetric density
                # (perceives Type B as 1.5x noisier).
                anti_crowd = visitor.profile.anti_crowd_bonus * (1.0 - min(1.5, self._effective_density_for(n, "A")))
            elif info_anti_crowd:
                # Type B at a kiosk: gains moderate crowd avoidance from
                # seeing the real-time density display.
                dens_n = self.density(n)
                anti_crowd = 0.5 * (1.0 - min(1.5, dens_n))

            # Adaptive audio guide: additional crowd avoidance for any
            # visitor carrying the guide. Stacks with type-specific avoidance.
            if has_audio_guide:
                dens_n = self.density(n)
                anti_crowd += config.AUDIO_GUIDE_CROWD_SENSITIVITY * (1.0 - min(1.5, dens_n))

            # Real-time room boost: nearly-empty rooms (density < 0.2) get
            # an importance bonus on digital displays, making them more
            # visible to visitors passing through corridors.
            if self.iv.realtime_room_boost:
                dens_n = self.density(n)
                if dens_n < config.REALTIME_BOOST_DENSITY_THRESHOLD:
                    importance_pull += 0.15 * config.REALTIME_BOOST_FACTOR

            # Crowd-responsive lighting: physically dim overcrowded rooms
            # (discouraging entry) and brighten empty ones (inviting entry).
            # Modeled as an importance_pull multiplier.
            if self.iv.crowd_responsive_lighting:
                dens_n = self.density(n)
                if dens_n > config.LIGHTING_DENSITY_THRESHOLD:
                    importance_pull *= config.LIGHTING_OVERCROWDED_FACTOR  # 0.8x
                elif dens_n < 0.3:
                    importance_pull *= config.LIGHTING_EMPTY_FACTOR        # 1.3x

            # Social proof nudge: "highly rated" badge on rooms with low
            # density and high importance. Only effective when the room
            # is not crowded, creating a self-correcting feedback loop.
            if self.iv.social_proof_nudge:
                dens_n = self.density(n)
                imp_n = self.g.nodes[n]["importance"]
                if dens_n < 0.5 and imp_n >= 5:
                    importance_pull += 0.3

            # Progress bar: gamification that rewards visiting new rooms.
            # Unvisited rooms get a 1.5x importance boost.
            if self.iv.progress_bar and n not in visitor.rooms_visited:
                importance_pull *= config.PROGRESS_BAR_UNVISITED_BOOST

            # Predictive routing: detect rooms where density is trending
            # upward (congestion building) and add a crowd penalty.
            # Uses the last 3 density snapshots to estimate the trend.
            if self.iv.predictive_routing and len(self.density_history) >= 3:
                recent = [h[config.ROOM_TO_IDX[n]] for h in self.density_history[-3:]]
                trend = recent[-1] - recent[0]  # positive = density rising
                if trend > 0.1:
                    anti_crowd += 0.3 * trend  # penalty proportional to trend

            # --- Component 3: Route bonus ---
            # Baseline of 1.0 ensures every neighbor has nonzero weight.
            # The recommended-route neighbor gets +3.0*route_bias (e.g.,
            # +2.64 for Type B with route_bias=0.88). Checklist rooms
            # get an additional +1.5 flat bonus.
            route_bonus = 1.0
            if next_pref == n:
                route_bonus += 3.0 * visitor.profile.route_bias
            if n in visitor.checklist:
                route_bonus += 1.5

            # --- Final weight ---
            # Floor of 0.01 ensures no neighbor has exactly zero probability.
            weight = max(0.01, route_bonus + importance_pull + anti_crowd)
            weights.append(weight)

        # Normalize weights to a probability distribution and sample.
        probs = config.normalize_weights(weights)
        choice_idx = int(self.rng.choice(np.arange(len(neighbors)), p=probs))
        return neighbors[choice_idx]

    # =================================================================
    # Per-visitor step logic
    # =================================================================

    def _step_visitor(self, visitor: NPCVisitor) -> bool:
        """Advance one visitor by one minute.

        This is the per-visitor inner loop. Each call:
          1. Decrements budget (every minute costs 1 unit)
          2. Checks for exit conditions (budget exhausted, at EXIT)
          3. Computes stay probability (exponential decay with magnetism)
          4. If staying: increment time_in_room, return True
          5. If moving: call _pick_next_room, apply intervention gates
             (Botticelli slot cap, decoupled gating, reciprocal access),
             update occupancy counters, return True
          6. If the visitor reaches EXIT: complete them, return False

        Parameters
        ----------
        visitor : NPCVisitor
            The visitor to advance.

        Returns
        -------
        bool
            True if the visitor remains active (still inside the museum),
            False if they have exited (moved to completed_visitors).
        """

        if visitor.current_room is None:
            return False  # safety: visitor not yet admitted

        # ----- Budget tick -----
        # Every visitor loses one minute of budget per step, regardless
        # of whether they stay or move. This is a hard time constraint.
        visitor.remaining_budget -= 1

        # Budget exhausted: the visitor leaves immediately from wherever
        # they are. This is a simplification; in reality visitors would
        # walk to the exit, but tracking exit-walk time adds complexity
        # without changing aggregate dynamics significantly.
        if visitor.remaining_budget <= 0:
            self._decrement_occ(visitor.current_room, visitor.visitor_type)
            self.completed_visitors.append(visitor)
            return False

        # Already at EXIT: leave immediately (reached via _pick_next_room
        # in a previous step).
        if visitor.current_room in {"EXIT"}:
            self._decrement_occ(visitor.current_room, visitor.visitor_type)
            self.completed_visitors.append(visitor)
            return False

        # ----- Stay/leave decision -----
        # Compute effective magnetism for the current room. Interventions
        # can temporarily boost magnetism to create absorption buffers.
        room_magnetism = self.g.nodes[visitor.current_room]["magnetism"]

        # Queue-to-content: when Botticelli rooms are congested (density >
        # 0.7), the buffer room A10 (Pollaiolo) activates contextual
        # content about Botticelli, increasing its magnetism. Visitors
        # in A10 perceive enrichment (pre-Botticelli context), not waiting.
        # This absorbs the queue into a meaningful experience.
        if (
            self.iv.queue_to_content
            and visitor.current_room == config.QUEUE_TO_CONTENT_BUFFER_ROOM
            and self.botticelli_density() > config.QUEUE_TO_CONTENT_DENSITY_THRESHOLD
        ):
            room_magnetism += config.QUEUE_TO_CONTENT_MAGNETISM_BOOST

        # Multi-room buffer cascade: extends the queue-to-content logic to
        # multiple rooms (A9, A10, A13). When Botticelli is congested,
        # all buffer rooms activate content, creating a distributed
        # absorption zone rather than a single bottleneck.
        if (
            self.iv.multi_room_buffer
            and visitor.current_room in config.BUFFER_CASCADE_ROOMS
            and self.botticelli_density() > config.QUEUE_TO_CONTENT_DENSITY_THRESHOLD
        ):
            room_magnetism += config.BUFFER_CASCADE_MAGNETISM_BOOST

        # Compute stay probability from effective magnetism and time in room.
        p_stay = self.npc_stay_probability(
            room_magnetism=room_magnetism,
            time_in_room=visitor.time_in_room,
            dwell_multiplier=visitor.profile.dwell_multiplier,
        )

        # ----- Stay: visitor remains in current room -----
        if self.rng.random() < p_stay:
            visitor.time_in_room += 1
            visitor.rooms_visited.add(visitor.current_room)
            return True

        # ----- Move: visitor decides to leave, pick next room -----
        next_room = self._pick_next_room(visitor)
        old_room = visitor.current_room

        # ----- Intervention access gates -----
        # These interventions can BLOCK a move into Botticelli or magnet
        # rooms, forcing the visitor to stay in their current room instead.
        # The visitor's movement model still chose the blocked room, but
        # the gate overrides it. This creates natural queueing behavior.

        # Botticelli slot cap: hard per-step admission limit. Once N
        # visitors have entered A11/A12 this minute, further entries are
        # blocked. This prevents "rush" dynamics where 50 visitors pile
        # into Botticelli simultaneously.
        if (
            self.iv.botticelli_slot_cap is not None
            and next_room in {"A11", "A12"}
            and old_room not in {"A11", "A12"}
            and self._botticelli_entries_this_step >= self.iv.botticelli_slot_cap
        ):
            next_room = old_room  # denied entry, stay put

        # Decoupled Botticelli gating: each visitor has a personal time
        # window (assigned at creation) during which they may enter A11/A12.
        # Outside their window, entry is blocked. This distributes
        # Botticelli demand across the day.
        if (
            self.iv.decoupled_botticelli_gating
            and next_room in {"A11", "A12"}
            and old_room not in {"A11", "A12"}
            and visitor.botticelli_window is not None
        ):
            w_start, w_end = visitor.botticelli_window
            if not (w_start <= self.current_time <= w_end):
                next_room = old_room  # not your window yet

        # Reciprocal access: visitor must have visited at least N hidden-gem
        # trail rooms before being allowed into Botticelli. Rewards
        # exploration of underused galleries with access to the star rooms.
        if (
            self.iv.reciprocal_access > 0
            and next_room in {"A11", "A12"}
            and old_room not in {"A11", "A12"}
        ):
            all_trail_rooms = set()
            for rooms in config.HIDDEN_GEM_TRAILS.values():
                all_trail_rooms.update(rooms)
            visited_trail = len(visitor.rooms_visited & all_trail_rooms)
            if visited_trail < self.iv.reciprocal_access:
                next_room = old_room  # haven't earned Botticelli access yet

        # Magnet room windows: extends decoupled gating to ALL magnet rooms
        # (A11, A12, A35, A38, E4), not just Botticelli. Reuses the same
        # botticelli_window field for simplicity. [assumption]
        if (
            self.iv.magnet_room_windows
            and next_room in config.ALL_MAGNET_WINDOW_ROOMS
            and old_room not in config.ALL_MAGNET_WINDOW_ROOMS
            and visitor.botticelli_window is not None
        ):
            w_start, w_end = visitor.botticelli_window
            if not (w_start <= self.current_time <= w_end):
                next_room = old_room

        # ----- Execute the move (if not blocked) -----
        if next_room != old_room:
            # Track Botticelli entries for the slot cap intervention.
            if next_room in {"A11", "A12"} and old_room not in {"A11", "A12"}:
                self._botticelli_entries_this_step += 1
            # Update occupancy: leave old room, enter new room.
            self._decrement_occ(old_room, visitor.visitor_type)
            visitor.current_room = next_room
            visitor.time_in_room = 0  # reset dwell clock in new room
            self._increment_occ(next_room, visitor.visitor_type)

        visitor.rooms_visited.add(visitor.current_room)

        # Immediate exit: if the visitor moved to EXIT, they leave this
        # step (no dwell time at the exit).
        if visitor.current_room in {"EXIT"}:
            self._decrement_occ(visitor.current_room, visitor.visitor_type)
            self.completed_visitors.append(visitor)
            return False

        return True

    # =================================================================
    # Density snapshot and main simulation step
    # =================================================================

    def _snapshot_densities(self) -> np.ndarray:
        """Capture the current room-density vector (N_ROOMS floats).

        Each element is occupancy/capacity for the corresponding room.
        This vector is appended to density_history every step and is
        used for RL observations, metrics, and trend detection.
        """

        dens = np.zeros(config.N_ROOMS, dtype=float)
        for room, idx in config.ROOM_TO_IDX.items():
            cap = self.g.nodes[room]["capacity"]
            dens[idx] = self.occ[room] / max(1.0, cap)
        return dens

    def step(self) -> Dict[str, float]:
        """Advance the simulator by one minute and return summary stats.

        This is the top-level simulation step called by the RL environment
        (UffiziEnv.step) or by run_day. The sequence each minute is:

          1. Generate arrivals and admit up to capacity (if before last entry)
          2. Reset the per-step Botticelli entry counter
          3. Advance every active visitor by one minute (_step_visitor)
          4. Remove visitors who exited (budget=0 or reached EXIT)
          5. Record capacity violations, peak occupancy, and density snapshot
          6. Increment the clock by 1 minute

        Returns
        -------
        dict of str to float
            Summary stats for this timestep: current time, total inside,
            queue length, and Botticelli density. Used by the RL environment
            for observation construction and reward computation.
        """

        # Step 1: arrivals (only before last entry cutoff).
        if self.current_time <= self.last_entry_minutes:
            self.step_arrivals()

        # Step 2: reset per-step Botticelli entry counter so the slot cap
        # applies fresh each minute.
        self._botticelli_entries_this_step = 0

        # Steps 3-4: advance all active visitors; collect those still inside.
        still_active = []
        for v in self.active_visitors:
            if self._step_visitor(v):
                still_active.append(v)
        self.active_visitors = still_active

        # Step 5: bookkeeping.
        if self.total_inside > config.MAX_MUSEUM_CAPACITY:
            self.capacity_violations += 1  # should not happen under proper gating

        self.max_total_inside = max(self.max_total_inside, self.total_inside)
        self.total_inside_history.append(self.total_inside)
        self.queue_history.append(len(self.outside_queue))
        self.density_history.append(self._snapshot_densities())

        # Step 6: advance the clock.
        self.current_time += config.TIME_STEP_MINUTES

        bott = self.botticelli_density()
        return {
            "time": float(self.current_time),
            "inside": float(self.total_inside),
            "queue": float(len(self.outside_queue)),
            "botticelli_density": float(bott),
        }

    # =================================================================
    # Full-day simulation
    # =================================================================

    def run_day(self) -> Dict[str, float]:
        """Simulate one full museum day and return aggregate outcomes.

        Runs the step() loop from minute 0 to day_minutes (615 = 18:30),
        then continues stepping (without new arrivals) for up to 120
        extra minutes to let remaining visitors exit. This "drain" phase
        ensures all visitors are accounted for in the completed list.

        Returns
        -------
        dict of str to float
            Aggregate metrics for the day:
            - mean_inside: time-averaged occupancy
            - peak_inside: max simultaneous occupancy
            - queue_peak: max outside queue length
            - peak_botticelli_density: worst-case Botticelli congestion
            - capacity_violations: minutes where occupancy exceeded 900
            - completed_visitors: total visitors who exited
            - type_a_completed / type_b_completed: by type
            - revenue: total ticket revenue (EUR)
            - experience_quality: mean composite experience score
            - experience_intimacy/surprise/narrative_coherence/engagement_depth
        """

        self.reset_day()

        # Main simulation loop: step minute by minute through the day.
        while self.current_time < self.day_minutes:
            self.step()

        # Drain phase: continue stepping without new arrivals to let
        # remaining visitors exit. The 120-minute cap prevents infinite
        # loops if visitors are stuck (should not happen with proper graph
        # connectivity, but is a safety measure).
        for _ in range(120):
            if not self.active_visitors:
                break
            still_active = []
            for v in self.active_visitors:
                if self._step_visitor(v):
                    still_active.append(v)
            self.active_visitors = still_active

        # ----- Aggregate metrics -----
        mean_inside = float(np.mean(self.total_inside_history)) if self.total_inside_history else 0.0

        # Peak Botticelli density: the higher of the two Botticelli rooms'
        # peak densities across the entire day. This is the primary
        # congestion metric for the RL reward function.
        peak_bott = float(np.max([d[config.ROOM_TO_IDX["A11"]] for d in self.density_history] + [0.0]))
        peak_bott = max(
            peak_bott,
            float(np.max([d[config.ROOM_TO_IDX["A12"]] for d in self.density_history] + [0.0])),
        )

        # Experience quality: composite metric capturing intimacy, surprise,
        # narrative coherence, and engagement depth.
        experience = self._compute_experience_quality()

        return {
            "mean_inside": mean_inside,
            "peak_inside": float(self.max_total_inside),
            "queue_peak": float(max(self.queue_history) if self.queue_history else 0.0),
            "peak_botticelli_density": peak_bott,
            "capacity_violations": float(self.capacity_violations),
            "completed_visitors": float(len(self.completed_visitors)),
            "type_a_completed": float(sum(v.visitor_type == "A" for v in self.completed_visitors)),
            "type_b_completed": float(sum(v.visitor_type == "B" for v in self.completed_visitors)),
            "revenue": float(sum(v.ticket_price for v in self.completed_visitors)),
            "experience_quality": experience["total"],
            "experience_intimacy": experience["intimacy"],
            "experience_surprise": experience["surprise"],
            "experience_narrative_coherence": experience["narrative_coherence"],
            "experience_engagement_depth": experience["engagement_depth"],
        }

    # =================================================================
    # Experience quality computation
    # =================================================================

    def _compute_experience_quality(self) -> Dict[str, float]:
        """Compute mean experience quality and its component breakdown.

        Experience quality captures what makes a museum visit memorable
        beyond the absence of congestion. It has four components:
          - Intimacy: bonus for spending time in rooms with < 15 people
          - Surprise: bonus for discovering rooms not on the checklist
          - Narrative coherence: bonus for following a thematic trail
          - Engagement depth: bonus for long, uncrowded dwell times

        Each component is computed per-visitor in experience_quality_components
        (from uffizi_rl.analysis.metrics), then averaged across all completed
        visitors.

        Returns
        -------
        dict of str to float
            Keys: "total", "intimacy", "surprise", "narrative_coherence",
            "engagement_depth". Values are population-mean scores.
        """

        if not self.completed_visitors:
            return {
                "total": 0.0,
                "intimacy": 0.0,
                "surprise": 0.0,
                "narrative_coherence": 0.0,
                "engagement_depth": 0.0,
            }

        # Graph-level room importance (potentially modified by interventions).
        room_importance = {
            room: float(self.g.nodes[room]["importance"])
            for room in self.g.nodes
        }

        # Time-averaged room occupancy: convert density history (density =
        # occ/cap) back to absolute occupancy by multiplying by capacity.
        # Used for intimacy scoring (intimate = < 15 people on average).
        room_mean_occupancy = {}
        if self.density_history:
            density_matrix = np.vstack(self.density_history)  # (T, N_ROOMS)
            for room, idx in config.ROOM_TO_IDX.items():
                room_mean_occupancy[room] = float(
                    density_matrix[:, idx].mean() * self.g.nodes[room]["capacity"]
                )

        # Compute per-visitor experience quality components.
        component_records = []
        for v in self.completed_visitors:
            component_records.append(
                experience_quality_components(
                    visited_rooms=v.rooms_visited,
                    checklist_rooms=v.checklist.keys(),
                    trail_name=v.profile.trail_name,
                    room_importance=room_importance,
                    room_mean_occupancy=room_mean_occupancy,
                )
            )

        # Average each component across all visitors.
        keys = component_records[0].keys()
        return {
            key: float(np.mean([record[key] for record in component_records]))
            for key in keys
        }

    # =================================================================
    # Multi-day simulation
    # =================================================================

    def run_many_days(self, n_days: int = 100, seed_offset: int = 0) -> Dict[str, float]:
        """Repeat the day simulation over sequential seeds and aggregate moments.

        Each day uses a different seed (base_seed + offset + day_index),
        producing independent stochastic realizations. Results are
        aggregated as mean and standard deviation across days.

        This is the primary method for policy evaluation: compare
        interventions by their mean and variance over many days.

        Parameters
        ----------
        n_days : int
            Number of independent day simulations to run.
        seed_offset : int
            Added to the base seed for each day. Allows running different
            batches of days without seed collision.

        Returns
        -------
        dict of str to float
            For each metric key K from run_day: K_mean, K_std, plus n_days.
        """

        records = []
        for d in range(n_days):
            # Each day gets a unique seed for independent stochastic draws.
            self.rng = config.get_rng(self.seed + seed_offset + d)
            rec = self.run_day()
            records.append(rec)

        # Aggregate: compute mean and std for each metric.
        keys = sorted(records[0].keys()) if records else []
        out = {}
        for k in keys:
            vals = np.array([r[k] for r in records], dtype=float)
            out[f"{k}_mean"] = float(vals.mean())
            out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        out["n_days"] = float(n_days)
        return out

    # =================================================================
    # Convenience accessors
    # =================================================================

    def botticelli_density(self) -> float:
        """Return the higher of the two Botticelli-room densities.

        The Uffizi has two Botticelli rooms (A11: The Spring, A12: Venus).
        For congestion metrics and intervention triggers, the relevant
        density is the worse of the two, as that determines the visitor
        experience at the bottleneck.
        """

        dens11 = self.density("A11")
        dens12 = self.density("A12")
        return float(max(dens11, dens12))

    def export_density_matrix(self) -> np.ndarray:
        """Export the full time-by-room density matrix for the current day.

        Returns
        -------
        np.ndarray
            Shape (T, N_ROOMS) where T is the number of steps completed.
            Each row is a density snapshot from _snapshot_densities.
        """

        if not self.density_history:
            return np.zeros((0, config.N_ROOMS), dtype=float)
        return np.vstack(self.density_history)

    def current_density_dict(self) -> Dict[str, float]:
        """Return the current room densities keyed by room ID.

        Used by the RL environment for observation construction and
        by adaptive interventions (e.g., adaptive trail selection)
        that need real-time density information.
        """

        return {room: self.density(room) for room in self.g.nodes}
