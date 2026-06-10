"""Crowd simulation with capacity-gated arrivals and heterogeneous visitors.

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
import math
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, List

import numpy as np

from uffizi_rl import config
from uffizi_rl.analysis.metrics import experience_quality_components
from uffizi_rl.interventions.intervention_config import InterventionConfig
from uffizi_rl.environment.museum_graph import (
    all_pairs_shortest_paths,      # Floyd-Warshall precomputed O(1) lookups
    build_uffizi_graph,            # constructs the NetworkX museum graph
    recommended_next_map,          # room -> next-room along the standard route
)
from uffizi_rl.environment.visitor_profiles import (
    VisitorProfile,                # frozen dataclass of per-visitor behavioral params
    sample_profile,                # stochastic visitor profile generator
    sample_type_a_profile,         # art-lover sampler (RAMA early-slot bias)
    sample_type_b_profile,         # forced Type B profile for tour group members
    sample_instagram_profile,      # IG sampler (RAMA late-slot bias)
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
_ARRIVAL_PEAK_MINUTES = 285   # minutes after opening: 13:00
                              # Calibrated against the actual Uffizi
                              # booking website: ticket-slot remaining-
                              # capacity numbers are nearly uniform
                              # across the operating day on peak
                              # summer days. The Gaussian shape is
                              # essentially flat for our purposes;
                              # peak time is mid-day but the curve is
                              # very wide.
_ARRIVAL_SIGMA_MINUTES = 240  # Gaussian width: 4 hours either side.
                              # With sigma=240 and a 555-minute entry
                              # window, the arrival profile is
                              # near-uniform across the whole day,
                              # matching the Uffizi's actual slot-by-
                              # slot booking distribution which shows
                              # demand spread evenly from open to
                              # close (peak summer: every slot 8:15-
                              # 16:45 has 86-124 of ~190 seats left).


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
    # RAMA per-room windows. Under the RAMA intervention each ticket
    # reserves an explicit 20-minute slot for each of Leonardo and
    # Raphael in addition to the Botticelli window. Windows are
    # spread deterministically across the day. Used by the gating
    # logic and by the behavioral-linger code which slows visitors
    # down in non-masterpiece rooms when their next slot is far away.
    leonardo_window: tuple[int, int] | None = None
    raphael_window: tuple[int, int] | None = None
    ticket_price: float = 0.0
    is_walk_in: bool = False
    # Demographic category for analytics. Values: "kid", "student",
    # "pensioner", "adult", "disabled_or_other_free". Set in
    # sample_ticket_price() so diagnostic scripts don't have to infer
    # the segment from the ticket price (which is ambiguous under
    # modular RAMA pricing).
    demographic_segment: str = "adult"
    # Time-stamped path of rooms entered (in order). Used by path
    # clustering. Each entry is a (minute, room) tuple appended when
    # the visitor first enters a room.
    path: list = field(default_factory=list)
    # Visitor-integrated experienced welfare. Accumulated each minute as
    # personal_importance[current_room] / (1 + crowd_alpha * density**2),
    # so a visitor who reaches their preferred rooms, spends time there,
    # and finds them uncrowded scores high. Aggregated across all
    # attempted visitors (completed + active + still-queued) and divided
    # by attempt count at end of day to form per-attempted-visitor mean
    # welfare, which is the social-welfare object the museum operator
    # cares about: how much experience per unit of demand.
    experienced_welfare: float = 0.0


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
        self._daily_cap = int(daily_total)   # hard cap: never admit more than the requested level
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

        # Extended hours intervention: 07:00-19:00 = 12 hour day.
        # 720 min of operation (vs baseline 615), last entry at 18:00
        # (660 min from open). The shift adds 105 extra minutes,
        # spread roughly evenly at the start (45 min earlier opening)
        # and end (60 min later closing).
        if self.iv.extended_hours:
            self.day_minutes = 720
            self.last_entry_minutes = 660

        # Secondary-attractor enrichment: apply the curatorial table to
        # the 14 pre-masterpiece rooms. Each room gets a unique
        # combination of sensory channels (visual, audio, tactile,
        # olfactory, live), and the importance/magnetism deltas sum
        # across active channels via CHANNEL_EFFECTS. Importance is
        # capped at 10.0 to stay below masterpiece level. The full
        # curatorial reasoning per room lives in
        # docs/curatorial_table.md.
        if self.iv.secondary_attractor_enrichment:
            for rid, channels in config.ROOM_CURATION.items():
                if rid not in self.g.nodes:
                    continue
                imp_delta = 0.0
                mag_delta = 0.0
                for channel in channels:
                    eff = config.CHANNEL_EFFECTS.get(channel, {})
                    imp_delta += eff.get("importance", 0.0)
                    mag_delta += eff.get("magnetism", 0.0)
                self.g.nodes[rid]["importance"] = min(
                    10.0, self.g.nodes[rid]["importance"] + imp_delta
                )
                self.g.nodes[rid]["magnetism"] += mag_delta

        # Photography ban: banning photography in bottleneck rooms (A11, A12)
        # reduces dwell time by ~40%. Visitors look, absorb, and move on
        # instead of posing, waiting for clear shots, and taking multiple
        # photos. Implemented as a magnetism reduction because magnetism
        # directly scales expected dwell time. [assumption]
        if self.iv.photography_ban:
            for rid in config.PHOTO_BAN_ROOMS:
                if rid in self.g.nodes:
                    self.g.nodes[rid]["magnetism"] *= config.PHOTO_BAN_MAGNETISM_FACTOR

        # Temporary exhibit: place headline works from storage in
        # underused rooms, creating new demand attractors. Two modes:
        #   - String room ID: boost only that one room (legacy form).
        #   - True (bool sentinel): rotate across all candidate rooms
        #     in config.TEMPORARY_EXHIBIT_ROOMS with a per-room boost
        #     scaled down by the number of candidates, so the total
        #     redirection is comparable to a single-room exhibit.
        # The boost is capped at 10.0 (max importance scale).
        if self.iv.temporary_exhibit_room is True:
            candidates = [r for r in config.TEMPORARY_EXHIBIT_ROOMS if r in self.g.nodes]
            if candidates:
                per_room_boost = config.TEMPORARY_EXHIBIT_IMPORTANCE_BOOST / len(candidates)
                for rid in candidates:
                    self.g.nodes[rid]["importance"] = min(
                        10.0, self.g.nodes[rid]["importance"] + per_room_boost
                    )
        elif isinstance(self.iv.temporary_exhibit_room, str) and self.iv.temporary_exhibit_room in self.g.nodes:
            rid = self.iv.temporary_exhibit_room
            self.g.nodes[rid]["importance"] = min(
                10.0, self.g.nodes[rid]["importance"] + config.TEMPORARY_EXHIBIT_IMPORTANCE_BOOST
            )

        # Room nobody knows: a social-media campaign that features an
        # underused gallery each day, creating FOMO and redirecting
        # attention to overlooked rooms. Different from the temporary
        # exhibit because no physical change is needed; the intervention
        # is purely informational. Two modes:
        #   - String room ID: feature only that one room.
        #   - True (bool sentinel): rotate across all candidate rooms in
        #     config.ROOM_NOBODY_KNOWS_CANDIDATES with a per-room boost
        #     scaled by the candidate count.
        if self.iv.room_nobody_knows is True:
            candidates = [r for r in config.ROOM_NOBODY_KNOWS_CANDIDATES if r in self.g.nodes]
            if candidates:
                per_room_boost = config.ROOM_NOBODY_KNOWS_IMPORTANCE_BOOST / len(candidates)
                for rid in candidates:
                    self.g.nodes[rid]["importance"] = min(
                        10.0, self.g.nodes[rid]["importance"] + per_room_boost
                    )
        elif isinstance(self.iv.room_nobody_knows, str) and self.iv.room_nobody_knows in self.g.nodes:
            rid = self.iv.room_nobody_knows
            self.g.nodes[rid]["importance"] = min(
                10.0, self.g.nodes[rid]["importance"] + config.ROOM_NOBODY_KNOWS_IMPORTANCE_BOOST
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

        # Annual pass holders: the real Uffizi sells a EUR 80/year unlimited
        # pass [UFF]. We model ~200 pass holders visiting per day as baseline
        # reality (they are already part of the visitor mix). The
        # `resident_annual_pass` intervention represents EXPANDING the
        # program (additional take-up), modelled as +200 more daily.
        self.daily_total += config.ANNUAL_PASS_DAILY_VISITORS
        if self.iv.resident_annual_pass:
            self.daily_total += config.RESIDENT_PASS_DAILY_VISITORS
        # Hard cap: the museum never accepts more than the requested daily level.
        # Pass holders count WITHIN the cap (they displace walk-ups), they do not
        # push the total above it. Applies to baseline and intervened identically.
        self.daily_total = min(self.daily_total, self._daily_cap)

        # =================================================================
        # Routing and distance precomputation
        # =================================================================
        # next_room maps each room to the ordered LIST of successor rooms
        # across every occurrence in the recommended route. The simulator's
        # _route_next_pref helper picks the first entry the visitor has not
        # already visited as the route-preferred neighbor, then the
        # _pick_next_room loop adds a route_bonus to that neighbor. A list
        # is used instead of a single value because corridor hubs (A24,
        # A36) appear multiple times in any sensible itinerary; a flat
        # dict with last-write-wins produced incorrect biases.
        self.next_room = recommended_next_map(route=config.STANDARD_ROUTE)

        # Per-profile route maps. The route bonus in _pick_next_room steers
        # each visitor along the route matching their profile.segment:
        # Instagram tourists take the shortest masterpiece-and-terrace path
        # and exit at Lanzi; art lovers walk the whole museum including the
        # first floor; the standard route above is used for everyone else.
        self._ig_next = recommended_next_map(route=config.INSTAGRAM_ROUTE)
        self._artlover_next = recommended_next_map(route=config.ART_LOVER_ROUTE)

        # Vasari narration: an alternative chronological route designed by
        # Vasari (the architect) that follows art history rather than the
        # topological layout. Visitors on this trail use _vasari_next
        # instead of next_room for their route bias.
        if self.iv.vasari_narration:
            self._vasari_next = recommended_next_map(route=config.VASARI_ROUTE)
        else:
            self._vasari_next = None

        # Inverse map: given a room, find one predecessor on the route.
        # Used only for occasional backtracking (Type A 5%, Type B 2%), so
        # taking the first encountered predecessor is sufficient.
        self.prev_room: Dict[str, str] = {}
        for src, dests in self.next_room.items():
            for dest in dests:
                if dest not in self.prev_room:
                    self.prev_room[dest] = src

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
        """Compute the scalar that normalizes the baseline arrival envelope.

        Real Uffizi booking data [UFF] shows near-uniform demand across
        the day, with the final two 15-min slots (last 30 min before
        last-entry cutoff) typically undersold. We model the envelope as:
          - 1.0 for t in [0, last_entry - 30)
          - 0.5 for t in [last_entry - 30, last_entry] (last 2 slots)

        Returns
        -------
        float
            Normalization constant. The per-minute arrival rate at time t
            is envelope(t) * norm, which is then passed to Poisson sampling.
        """
        times = np.arange(self.last_entry_minutes + 1)
        envelope = np.where(times < self.last_entry_minutes - 30, 1.0, 0.5)
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

        # Per-minute arrivals counter. Used to subtract "transit" visitors
        # (those who just walked into the room this minute and may walk
        # straight through) from the density-snapshot metric. Reset to
        # zero at the start of every step.
        self.arrivals_this_min: Dict[str, int] = {room: 0 for room in self.g.nodes}

        # Day-level summary statistics, computed incrementally.
        self.max_total_inside = 0    # peak simultaneous occupancy
        self.capacity_violations = 0  # minutes where total_inside > 900

        # RAMA slot ledger. Tracks how many visitors have booked each
        # 15-minute slot for each masterpiece room. Keys are
        # (room_id, slot_start_minute) -> current booking count. When
        # a visitor tries to book a slot, the booking is accepted if
        # the count is below RAMA_SLOT_CAPACITY. If full, the visitor
        # tries the next adjacent slot. Visitors who exhaust the
        # search window get no slot for that room and won't visit it.
        self.rama_slot_bookings: Dict[tuple[str, int], int] = {}

        # Time-series logs for metrics and RL observation construction.
        self.density_history: List[np.ndarray] = []      # N_ROOMS-dim vector per step
        self.total_inside_history: List[int] = []         # scalar per step
        self.queue_history: List[int] = []                # outside queue length per step

        # RAMA pre-day booking schedule. Maps entry_minute -> list of
        # booking dicts. Each dict carries the visitor's profile spec,
        # their booked masterpiece windows, and whether they are a
        # walk-in. Computed once per day in _precompute_rama_bookings()
        # under RAMA; empty otherwise.
        self._rama_schedule: Dict[int, List[Dict]] = {}
        if self.iv.rama:
            self._precompute_rama_bookings()

    # =================================================================
    # RAMA pre-day booking
    # =================================================================

    def _precompute_rama_bookings(self) -> None:
        """Generate the pre-day RAMA booking schedule.

        Models the real-world booking process: weeks before visit day,
        each potential visitor goes to the Uffizi website. They:
          1. Decide which masterpiece rooms they want to see (each
             with high probability per RAMA_P_WANT_MASTERPIECE).
          2. Pick their preferred slot spacings (type-specific
             distributions: art lovers space slots hours apart;
             IG tourists go back-to-back).
          3. Try to book each wanted slot. If their preferred minute is
             sold out, the website offers nearby slots within
             RAMA_BOOKING_TOLERANCE_MIN. If nothing fits, that
             masterpiece is dropped from their itinerary.
          4. If the visitor can book ALL the masterpieces they wanted,
             they keep the reservation and will visit. If even one
             wanted masterpiece is sold out within tolerance, they
             abandon the booking and visit another day.

        A small WALK_IN_FRACTION represents tourists who didn't pre-book
        any masterpiece. They show up at the museum with a general
        ticket and walk the non-masterpiece rooms; they know at purchase
        time that the masterpieces are sold out, so no exit-rejection
        welfare penalty applies (their importance for masterpieces is
        already zeroed).

        Populates ``self._rama_schedule[entry_minute] = list of booking
        dicts`` for the day's actual entries. Visitors who failed to
        book do NOT appear in the schedule; they have rescheduled to
        another day.
        """

        slot_dur = config.RAMA_SLOT_DURATION_MIN
        caps = {
            "A11": config.RAMA_SLOT_CAPACITY_BY_ROOM.get("A11", 55),
            "A35": config.RAMA_SLOT_CAPACITY_BY_ROOM.get("A35", 52),
            "A38": config.RAMA_SLOT_CAPACITY_BY_ROOM.get("A38", 53),
        }
        # Tracks current booking count per (room, slot_start) cell.
        ledger: Dict[tuple, int] = {}

        def round_to_slot(x: int) -> int:
            return ((max(0, int(x)) + slot_dur - 1) // slot_dur) * slot_dur

        def find_slot(room: str, preferred: int, min_start: int,
                       party_size: int = 1) -> int | None:
            """Find the nearest available slot to `preferred` with at
            least `party_size` capacity remaining. Used by both
            individual bookings (party_size=1) and tour-group bookings
            (party_size=group_size, e.g. 15)."""
            cap = caps[room]
            preferred = round_to_slot(preferred)
            upper = self.day_minutes - 30 - slot_dur
            max_offset = max(preferred, upper - preferred) + slot_dur
            for offset in range(0, max_offset + slot_dur, slot_dur):
                for sign in ([0] if offset == 0 else [1, -1]):
                    cand = preferred + sign * offset
                    if cand < min_start or cand < 0 or cand > upper:
                        continue
                    if ledger.get((room, cand), 0) + party_size <= cap:
                        return cand
            return None

        spacing = config.RAMA_SPACING_DISTRIBUTIONS
        p_want = config.RAMA_P_WANT_MASTERPIECE
        walk_in_count = int(round(self.daily_total * config.RAMA_WALK_IN_FRACTION))
        booker_count = self.daily_total - walk_in_count

        # Build the visitor intent list. Order matters for booking
        # (first-come first-served on the website), so shuffle.
        intents = []
        for vid in range(booker_count):
            # Sample type with the canonical mix.
            r = self.rng.random()
            if r < config.INSTAGRAM_FRACTION:
                vtype = "instagram"
            elif r < config.INSTAGRAM_FRACTION + config.STANDARD_FRACTION:
                vtype = "standard"
            else:
                vtype = "art_lover"
            wants = {
                room: bool(self.rng.random() < p)
                for room, p in p_want.items()
            }
            # If the visitor doesn't want any of the three, treat as walk-in.
            if not any(wants.values()):
                intents.append({"id": vid, "type": vtype, "wants": wants,
                                 "is_walk_in": True})
                continue
            sp = spacing[vtype]
            spc_entry = int(self.rng.integers(sp["entry_to_first"][0],
                                              sp["entry_to_first"][1] + 1))
            spc_b_to_l = int(self.rng.integers(sp["bott_to_leo"][0],
                                                sp["bott_to_leo"][1] + 1))
            spc_l_to_r = int(self.rng.integers(sp["leo_to_raph"][0],
                                                sp["leo_to_raph"][1] + 1))
            arrive_early = int(self.rng.integers(sp["arrive_early"][0],
                                                  sp["arrive_early"][1] + 1))
            intents.append({
                "id": vid, "type": vtype, "wants": wants,
                "is_walk_in": False,
                "spc_entry": spc_entry,
                "spc_b_to_l": spc_b_to_l,
                "spc_l_to_r": spc_l_to_r,
                "arrive_early": arrive_early,
            })

        # Append walk-ins (no booking attempt).
        for wid in range(booker_count, booker_count + walk_in_count):
            r = self.rng.random()
            if r < config.INSTAGRAM_FRACTION:
                vtype = "instagram"
            elif r < config.INSTAGRAM_FRACTION + config.STANDARD_FRACTION:
                vtype = "standard"
            else:
                vtype = "art_lover"
            intents.append({"id": wid, "type": vtype,
                             "wants": {"A11": False, "A35": False, "A38": False},
                             "is_walk_in": True})

        # ----- TOUR GROUP INTENTS (booking competition with individuals) -----
        # Real tour operators book ahead just like individuals. Each
        # group books one slot per masterpiece, consuming group_size
        # capacity (e.g. 15 of the 55-cap Bott slot). Most groups want
        # all three masterpieces - that's the selling point of the tour.
        group_size = (
            self.iv.tour_group_cap if self.iv.tour_group_cap is not None
            else config.TOUR_GROUP_SIZE_DEFAULT
        )
        n_visitors_in_groups = int(round(self.daily_total * config.TOUR_GROUP_FRACTION))
        n_groups = n_visitors_in_groups // max(1, group_size)
        # Group surcharge elasticity: same exponential as individual
        # arrivals so the effect is consistent across both streams.
        if self.iv.per_person_group_surcharge > 0:
            extra_per_head = self.iv.per_person_group_surcharge / max(1, group_size)
            n_groups = int(round(n_groups * math.exp(-extra_per_head / 10.0)))
        if self.iv.quiet_hours or self.iv.quiet_hours_tour_ban:
            n_groups = 0  # no group bookings if quiet-hours intervention bans groups
        # Treat each group as a single "intent" that books with party_size.
        sp_std = spacing["standard"]
        next_id = len(intents)
        for g_idx in range(n_groups):
            spc_entry = int(self.rng.integers(sp_std["entry_to_first"][0],
                                              sp_std["entry_to_first"][1] + 1))
            spc_b_to_l = int(self.rng.integers(sp_std["bott_to_leo"][0],
                                                sp_std["bott_to_leo"][1] + 1))
            arrive_early = int(self.rng.integers(sp_std["arrive_early"][0],
                                                  sp_std["arrive_early"][1] + 1))
            intents.append({
                "id": next_id + g_idx,
                "type": "standard",
                "is_group": True,
                "group_size": group_size,
                "wants": {"A11": True, "A35": True, "A38": True},  # groups always want all 3
                "is_walk_in": False,
                "spc_entry": spc_entry,
                "spc_b_to_l": spc_b_to_l,
                "spc_l_to_r": 0,   # unused (forced)
                "arrive_early": arrive_early,
            })

        # Instagram tourists book first. They come to the Uffizi for one
        # reason - the masterpiece selfies - so they secure those slots
        # ahead of the more flexible standard and art-lover visitors (and
        # groups). Instagram demand (~60% of the day) sits just under the
        # masterpiece slot capacity, so giving them priority means they all
        # get in; everyone else takes the remaining slots and otherwise
        # visits on a general-entry ticket. Within each priority tier the
        # order stays random (stable sort over a random permutation).
        # Shuffle booking order so no type systematically wins early slots.
        # (Masterpiece-slot priority by visitor type is a policy decision that
        # interacts with visit length and completion; left fair for now.)
        order = self.rng.permutation(len(intents))

        schedule: Dict[int, List[Dict]] = {}

        for idx in order:
            intent = intents[idx]
            if intent["is_walk_in"]:
                # Walk-in: random entry across the entry window. They
                # have no masterpiece bookings; their importance for
                # A11/A12/A35/A38 will be zeroed at visitor creation.
                entry_time = int(self.rng.integers(0, self.last_entry_minutes + 1))
                schedule.setdefault(entry_time, []).append({
                    "type": intent["type"],
                    "is_walk_in": True,
                    "bott_window": None, "leo_window": None, "raph_window": None,
                    "wants": intent["wants"],
                })
                continue

            # Pre-booker: pick a preferred entry time uniformly across
            # the entry window (leaving headroom for their full circuit
            # AND a 30-min "no one wants to rush" margin from closure).
            # Real visitors don't pre-book entries that would put them
            # in the museum during the last 30 min of operating hours -
            # they want enough time to actually enjoy the rooms, not
            # be ushered out.
            sp = spacing[intent["type"]]
            # Thorough profiles continue onto the first floor after Raphael,
            # so the booking leaves room for that leg (plus an exit buffer)
            # when it picks an entry time, otherwise the visitor would be
            # scheduled to enter too late to finish the first floor before
            # closing. Instagram tourists exit at the terrace, so just a
            # walk-out buffer.
            floor1_alloc = {"art_lover": 150, "standard": 75}.get(intent["type"], 20)
            rough_visit_len = (intent["spc_entry"] + slot_dur
                               + intent["spc_b_to_l"] + slot_dur
                               + slot_dur     # Raph dwell (forced adjacent)
                               + floor1_alloc)
            # 30-min closure margin: nobody books with intent to be in
            # the museum within 30 min of closing time.
            closure_margin = 30
            max_entry = max(1, self.last_entry_minutes - rough_visit_len - closure_margin)
            preferred_entry = int(self.rng.integers(0, max_entry + 1))

            # Compute preferred slot times in visit order.
            wants = intent["wants"]
            booked = {"A11": None, "A35": None, "A38": None}
            party = intent.get("group_size", 1) if intent.get("is_group") else 1

            # Botticelli first (if wanted). Allowed to fail; sets None
            # rather than aborting the visitor.
            if wants["A11"]:
                pref = preferred_entry + intent["spc_entry"]
                booked["A11"] = find_slot("A11", pref, min_start=preferred_entry,
                                           party_size=party)

            # Leonardo second. Anchored on Bott end if Bott was booked,
            # else on entry time. Type-specific spacing (art lover can
            # spend hours exploring between Bott and Leo).
            if wants["A35"]:
                if booked["A11"] is not None:
                    min_start = booked["A11"] + slot_dur
                    pref = min_start + intent["spc_b_to_l"]
                else:
                    min_start = preferred_entry
                    pref = preferred_entry + intent["spc_entry"]
                booked["A35"] = find_slot("A35", pref, min_start, party_size=party)

            # Raphael third. Forced to be Leo+slot_dur (adjacent rooms).
            if wants["A38"]:
                if booked["A35"] is not None:
                    forced = booked["A35"] + slot_dur
                    if (forced <= self.day_minutes - slot_dur
                            and ledger.get(("A38", forced), 0) + party <= caps["A38"]):
                        booked["A38"] = forced
                elif booked["A11"] is not None:
                    min_start = booked["A11"] + slot_dur
                    pref = min_start + intent["spc_b_to_l"]
                    booked["A38"] = find_slot("A38", pref, min_start, party_size=party)
                else:
                    pref = preferred_entry + intent["spc_entry"]
                    booked["A38"] = find_slot("A38", pref, min_start=preferred_entry,
                                               party_size=party)

            # All visitors visit today. Even if they couldn't book any
            # masterpiece (all three sold out), they still buy a
            # general-entry online ticket and visit the rest of the
            # museum. They pay only the EUR 15 base price (no
            # masterpiece add-ons). This matches the user model: NO
            # walk-ins under RAMA, just modular online booking where
            # the price scales with the number of masterpiece slots
            # secured (0 -> EUR 15, 1 -> EUR 20, 2 -> EUR 25, 3 -> EUR 30).

            for room, s in booked.items():
                if s is not None:
                    ledger[(room, s)] = ledger.get((room, s), 0) + party

            # Personal teleport offset within the slot. With slot cap
            # > room cap (Botticelli zone = 110, room cap = 55),
            # visitors are staggered across the 10-min slot so the
            # room maintains its cap continuously rather than oscillating
            # 0 -> 55 -> 0. Same mechanism for Leo/Raph keeps them
            # smoothly at cap throughout the slot.
            offset = int(self.rng.integers(0, slot_dur))

            # Adjust windows by personal offset. The visitor's effective
            # teleport time is slot_start + offset, dwell ends at
            # slot_start + offset + dwell_target. Different visitors in
            # the same slot have different offsets so the room fills
            # smoothly across the 10-min slot duration.
            bott_window = ((booked["A11"] + offset, booked["A11"] + offset + slot_dur)
                            if booked["A11"] is not None else None)
            leo_window = ((booked["A35"] + offset, booked["A35"] + offset + slot_dur)
                           if booked["A35"] is not None else None)
            raph_window = ((booked["A38"] + offset, booked["A38"] + offset + slot_dur)
                            if booked["A38"] is not None else None)

            # Compute entry time = (earliest booked slot personal teleport)
            # - (arrive_early buffer). Visitors who couldn't book any
            # masterpiece (general-entry-only ticket) pick a uniform
            # entry time within the entry window.
            booked_windows = [w for w in (bott_window, leo_window, raph_window) if w is not None]
            if booked_windows:
                first_personal = min(w[0] for w in booked_windows)
                entry_time = max(0, first_personal - intent["arrive_early"])
            else:
                entry_time = int(self.rng.integers(0, max(1, self.last_entry_minutes + 1)))

            entry_dict = {
                "type": intent["type"],
                "is_walk_in": False,
                "bott_window": bott_window,
                "leo_window": leo_window,
                "raph_window": raph_window,
                "wants": wants,
            }
            if intent.get("is_group"):
                # Schedule the leader + (group_size - 1) followers, all
                # sharing the same booking and entry_time. Followers
                # share the leader's profile so their dwell behavior is
                # synchronized (they move together as a guided tour).
                entry_dict["is_group_leader"] = True
                entry_dict["group_size"] = party
                self._tour_group_counter += 1
                entry_dict["group_id"] = self._tour_group_counter
                schedule.setdefault(entry_time, []).append(entry_dict)
                for _ in range(party - 1):
                    follower = dict(entry_dict)
                    follower["is_group_leader"] = False
                    schedule.setdefault(entry_time, []).append(follower)
            else:
                schedule.setdefault(entry_time, []).append(entry_dict)

        self._rama_schedule = schedule
        self.rama_slot_bookings = ledger  # Expose for downstream introspection.

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

        # RAMA: arrivals are uniformly distributed across the entry
        # window because every visitor books a specific 15-min entrance
        # slot and the slot capacity is set so total bookings equal
        # daily_total. The Gaussian peak does not apply - visitors
        # arrive when their reservation says, and the reservations are
        # spread evenly. Note: we don't early-return here so dynamic
        # pricing can still reshape the uniform base rate below.
        if self.iv.rama:
            rate = float(self.daily_total) / max(1.0, float(self.last_entry_minutes))
            # Skip the Gaussian envelope branches below; jump straight
            # to the multiplicative modifiers.
            if self.iv.dynamic_pricing and t < len(self._price_schedule):
                price_mult = self._price_schedule[t]
                elasticity = 0.5
                rate *= float(price_mult ** (-elasticity))
            return rate

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
            # Default: near-uniform envelope matching real Uffizi booking
            # demand [UFF]. Flat across the day; last 2 slots (30 min
            # before last-entry) are undersold at 50%.
            envelope = 1.0 if t < self.last_entry_minutes - 30 else 0.5
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
    # RAMA: slot booking
    # =================================================================

    def _book_rama_slot(
        self,
        room: str,
        preferred_start: int,
        preferred_end: int,
        max_slot_start: int,
    ) -> tuple[int, int] | None:
        """Find the earliest slot in [preferred_start, max_slot_start] with capacity.

        Slots are quantized to RAMA_SLOT_DURATION_MIN-minute boundaries
        and per-room capped by RAMA_SLOT_CAPACITY_BY_ROOM. The search
        scans forward from preferred_start. If the entire range is sold
        out by max_slot_start, returns None - the visitor will not be
        able to book this masterpiece. This is critical: in the real
        world, when slots are sold out, the visitor accepts not seeing
        the room rather than being assigned a slot beyond their visit
        window. Without this cap, late visitors were getting slots
        hours after their budget ran out and never showed up.

        Parameters
        ----------
        room : str
            Masterpiece room id (A11, A12, A35, A38). A11 and A12
            share the Botticelli slot ledger (single chain entry).
        preferred_start : int
            Earliest acceptable slot start (minutes from museum open).
        preferred_end : int
            Latest preferred slot start. Beyond this we keep searching
            but it's no longer "preferred." (Currently unused; kept
            for future preference-weighted booking logic.)
        max_slot_start : int
            Hard upper bound on the slot start. Bookings beyond this
            are refused. Should be visitor's planned exit time minus a
            safe walking buffer.

        Returns
        -------
        (slot_start, slot_end) or None if no reachable slot has capacity.
        """
        slot_dur = config.RAMA_SLOT_DURATION_MIN
        # Round UP to the next slot boundary so the booked slot is
        # never EARLIER than the visitor's preferred earliest time.
        # Rounding down was putting bookings in the past for visitors
        # created mid-slot, causing them to miss the slot they booked.
        ps = max(0, preferred_start)
        start = ((ps + slot_dur - 1) // slot_dur) * slot_dur
        # Botticelli chain: A11 and A12 share the same ledger key
        # because they're a single forced traversal.
        ledger_room = "A11" if room in ("A11", "A12") else room
        per_room_cap = config.RAMA_SLOT_CAPACITY_BY_ROOM.get(ledger_room, 50)
        # Search forward within the reachable range only.
        upper = min(max_slot_start, self.day_minutes - slot_dur)
        candidate = start
        while candidate <= upper:
            key = (ledger_room, candidate)
            booked = self.rama_slot_bookings.get(key, 0)
            if booked < per_room_cap:
                self.rama_slot_bookings[key] = booked + 1
                return (candidate, candidate + slot_dur)
            candidate += slot_dur
        return None  # no reachable slot available

    # =================================================================
    # Visitor creation
    # =================================================================

    def create_npc_visitor_from_booking(self, booking: Dict) -> NPCVisitor:
        """Wrapper that creates a visitor from a pre-computed RAMA booking.

        Used by step_arrivals under RAMA. The booking dict carries the
        visitor type, masterpiece windows, group flag, and walk-in flag
        computed by ``_precompute_rama_bookings()`` at day start.
        """
        entry_slot = min(config.N_ENTRY_SLOTS - 1,
                          self.current_time // config.ENTRY_SLOT_MINUTES)
        v = self.create_npc_visitor(entry_slot=entry_slot, pre_booking=booking)
        # Apply group affiliation and force Type B behaviour for groups
        # (followers cluster around the guide rather than wandering
        # independently). Group surcharge already factored into the
        # ticket via the modular RAMA price below; here we layer on the
        # baseline EUR 70 group fee + any per-head Pigovian surcharge.
        if booking.get("is_group_leader") is not None:  # i.e. is a group visitor
            v.group_id = booking.get("group_id")
            v.profile = sample_type_b_profile(self.rng)
            v.visitor_type = "B"
            if self.iv.revenue_model:
                gs = max(1, booking.get("group_size", 1))
                v.ticket_price += config.GROUP_SURCHARGE_FLAT / gs
                if self.iv.per_person_group_surcharge > 0:
                    v.ticket_price += self.iv.per_person_group_surcharge / gs
        return v

    def create_npc_visitor(self, entry_slot: int,
                            pre_booking: Dict | None = None) -> NPCVisitor:
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
        elif self.iv.rama:
            # Under RAMA, the visitor's type is set at booking time
            # (pre-day) and carried via pre_booking["type"]. The day-
            # progress segment bias used here previously is obsolete:
            # the entry minute is dictated by the booking, so the type
            # determines the entry pattern, not the other way around.
            vtype = (pre_booking or {}).get("type", "instagram")
            if vtype == "art_lover":
                profile = sample_type_a_profile(
                    self.rng,
                    heterogeneity_scale=self.heterogeneity_scale,
                    trail_acceptance_prob=base_trail_prob,
                )
            elif vtype == "standard":
                profile = sample_type_b_profile(self.rng)
            else:
                profile = sample_instagram_profile(self.rng)
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
            profile = replace(profile, importance_vector=imp)

        # ----- Secondary-attractor enrichment: per-visitor preference shift -----
        # Type B visitors (IG / Standard, 90% of population) respond
        # only to their PERSONAL importance vector when moving, not to
        # the room's graph-level cultural importance. So a graph-level
        # boost from enrichment is invisible to them unless we also
        # shift their personal preference for the curated rooms. The
        # rationale is realistic: the screens / music / scent / live
        # talks in a curated room genuinely make the visitor care more
        # about being there. Boost equals the sum of the room's active
        # channel importance effects, capped so the personal preference
        # stays below masterpiece-level. Applied to every visitor when
        # the intervention is on.
        if self.iv.secondary_attractor_enrichment:
            imp = profile.importance_vector.copy()
            for rid, channels in config.ROOM_CURATION.items():
                if rid not in config.ROOM_TO_IDX:
                    continue
                boost = sum(
                    config.CHANNEL_EFFECTS.get(c, {}).get("importance", 0.0)
                    for c in channels
                )
                idx_r = config.ROOM_TO_IDX[rid]
                imp[idx_r] = min(8.0, imp[idx_r] + boost)
            profile = replace(profile, importance_vector=imp)

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
                profile = replace(
                    profile,
                    route_bias=max(0.2, profile.route_bias - 0.03),
                    importance_vector=imp,
                    trail_name=profile.trail_name or f"theme:{self.iv.themed_weeks}",
                )

        # ----- Botticelli time window assignment -----
        # Smart defaults / decoupled gating: each visitor is assigned a
        # 30-minute window during which they may enter Botticelli rooms
        # (A11/A12). The window starts at a random offset (30-120 min)
        # after their entry time, ensuring they explore other rooms first.
        # This spreads Botticelli demand across the day.
        bott_window = None
        leo_window = None
        raph_window = None
        if self.iv.rama:
            # Bookings come from the pre-day schedule. The visitor's
            # masterpiece windows are already allocated; we just pull
            # them off the pre_booking dict. Walk-ins (no booking) and
            # visitors who skipped individual masterpieces have those
            # windows as None.
            pb = pre_booking or {}
            bott_window = pb.get("bott_window")
            leo_window = pb.get("leo_window")
            raph_window = pb.get("raph_window")
            # Time budget: must be long enough to honour all booked
            # slots plus a walk-out buffer. Use the latest booked slot
            # end + 20 min as the exit time; sample dwell-time will be
            # set to (exit - entry) by the time-budget block below via
            # the override mechanism.
            slots_present = [w[1] for w in (bott_window, leo_window, raph_window) if w is not None]
            if slots_present:
                latest_end = max(slots_present)
                # Minimum stay: long enough to honour the last booked slot
                # plus a walk-out buffer. The thorough profiles continue
                # onto the first floor after their last masterpiece, so they
                # need a real floor-1 allowance, not just a walk to the door.
                if profile.segment == "standard":
                    post_slot_buffer = 100
                elif profile.segment == "art_lover":
                    post_slot_buffer = 150
                else:
                    post_slot_buffer = 20
                rama_min_budget = max(30, latest_end - self.current_time + post_slot_buffer)
                # But respect the profile's own visit length when it is
                # longer: an art lover spends hours and continues to the
                # first floor after their last masterpiece slot, rather than
                # leaving 20 minutes later. Without this, the booking budget
                # truncates the long profiles and they never reach floor 1.
                base_budget = (
                    profile.time_budget_override
                    if profile.time_budget_override is not None
                    else config.sample_visit_duration(entry_slot, self.rng)
                )
                rama_budget = max(rama_min_budget, base_budget)
                profile = replace(profile, time_budget_override=rama_budget)
            # Walk-in / partial-booker: zero their personal importance
            # for un-booked masterpieces so they don't wander toward
            # rooms they can't enter. The "wantedness mass" of those
            # masterpieces is REDISTRIBUTED across their remaining
            # rooms (Tribuna, Caravaggio, terrace, etc.) because the
            # visitor calibrated their expectations at booking time:
            # they bought a Florence trip + Uffizi ticket KNOWING the
            # masterpieces are sold out, so their welfare reference
            # point is the rest of the museum, not the masterpieces
            # they never expected to see. Without this redistribution,
            # the per-attempted-visitor welfare metric would double-
            # penalize RAMA (the metric punishes visitors for missing
            # rooms they had no expectation of seeing).
            unbooked_rooms = []
            if bott_window is None:
                unbooked_rooms.extend(["A11", "A12"])
            if leo_window is None:
                unbooked_rooms.append("A35")
            if raph_window is None:
                unbooked_rooms.append("A38")
            if unbooked_rooms:
                imp_vec = profile.importance_vector.copy()
                # Sum the importance mass being zeroed.
                lost_mass = 0.0
                for rid in unbooked_rooms:
                    if rid in config.ROOM_TO_IDX:
                        idx = config.ROOM_TO_IDX[rid]
                        lost_mass += float(imp_vec[idx])
                        imp_vec[idx] = 0.0
                # Redistribute proportionally across the remaining
                # non-zero importance. Visitors who lose all 3
                # masterpiece slots get a substantial boost on
                # secondary rooms because that's where they will
                # spend their visit.
                remaining_total = float(imp_vec.sum())
                if remaining_total > 0 and lost_mass > 0:
                    scale = (remaining_total + lost_mass) / remaining_total
                    imp_vec = imp_vec * scale
                profile = replace(profile, importance_vector=imp_vec)
        elif self.iv.decoupled_botticelli_gating or self.iv.smart_defaults:
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
                profile = replace(
                    profile,
                    route_bias=max(0.3, profile.route_bias - 0.1),
                    trail_name="vasari",
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

        # Cap the visit at the time remaining until closing (plus a short
        # grace to finish the current room and walk out). Nobody can stay
        # past the museum's closing time: a visitor who arrives in the
        # afternoon simply does a shorter visit instead of an 8-hour one.
        # This keeps completion near 100% in every scenario (no visitors
        # stranded inside at close), which makes the baseline-vs-RAMA
        # comparison fair - otherwise the long-budget art lovers who arrive
        # late never exit and are dropped from the baseline metrics.
        max_remaining = self.day_minutes - self.current_time + config.CLOSING_GRACE_MINUTES
        budget = min(budget, max(1, max_remaining))

        # ----- Revenue model -----
        # Compute ticket price from entry time and visitor type.
        # Only active when revenue_model intervention is enabled.
        # Under RAMA, the modular ticket adds per-masterpiece booking fees
        # on top of the general entry price, so the price depends on how
        # many masterpiece slots the visitor actually has.
        price = 0.0
        is_walk_in = False
        demographic_segment = "adult"
        if self.iv.revenue_model:
            num_masterpiece_bookings = (
                (1 if bott_window is not None else 0)
                + (1 if leo_window is not None else 0)
                + (1 if raph_window is not None else 0)
            )
            price, is_walk_in, demographic_segment = self.sample_ticket_price(
                entry_slot, num_masterpiece_bookings=num_masterpiece_bookings,
            )

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
            leonardo_window=leo_window,
            raphael_window=raph_window,
            ticket_price=price,
            is_walk_in=is_walk_in,
            demographic_segment=demographic_segment,
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

    def sample_ticket_price(self, entry_slot: int,
                             num_masterpiece_bookings: int = 3) -> tuple[float, bool, str]:
        """Sample one visitor's ticket price under the revenue model.

        Under RAMA, the price is MODULAR: a base general-entry ticket
        plus a per-masterpiece booking fee. Visitors who skip individual
        masterpieces pay less. This keeps culture accessible (families
        with children or budget tourists can pay just the base €15 for
        a general walk) while recapturing revenue from full-RAMA
        visitors who pay €30 for guaranteed access to all three
        masterpieces. Avoids the political problem of charging €40+ at
        the gate.

        The visitor population is stratified into:
          - Free visitors (30%): under-18, students, disabled [assumption]
          - Reduced visitors (10%): EU 18-25 at EUR 2 [UFF]
          - Walk-in visitors (15%, baseline only): pay EUR 25 day-of rate
          - Last-hour locals: pay EUR 10 reduced rate
          - Everyone else under baseline: scheduled time-of-day price
          - Everyone else under RAMA: modular base + masterpiece add-ons

        Parameters
        ----------
        entry_slot : int
            Entry slot index (0-36).
        num_masterpiece_bookings : int
            How many of {Bott, Leo, Raph} this visitor booked. Only
            used under RAMA modular pricing.

        Returns
        -------
        tuple of (float, bool)
            (price_eur, is_walk_in). is_walk_in is True only for
            unplanned visitors paying the walk-in premium.
        """

        roll = self.rng.random()
        # Free entry: under-18 (~10%) + disabled / school groups (~5%)
        if roll < config.FREE_VISITOR_FRACTION:
            # Split: 10% kids, 5% disabled-or-other-free
            if roll < (10.0 / 15.0) * config.FREE_VISITOR_FRACTION:
                return 0.0, False, "kid"
            return 0.0, False, "disabled_or_other_free"
        # Reduced entry: ~10% of visitors (EU 18-25 at EUR 2) [UFF]
        if roll < config.FREE_VISITOR_FRACTION + config.REDUCED_VISITOR_FRACTION:
            return float(config.REDUCED_TICKET_PRICE), False, "student"
        # Pensioners (65+): pay reduced rate ONLY if pensioner_pricing
        # intervention active. Otherwise they pay full adult price like
        # everyone else (real Uffizi has no senior discount).
        is_pensioner = roll < (config.FREE_VISITOR_FRACTION
                                + config.REDUCED_VISITOR_FRACTION
                                + config.PENSIONER_FRACTION)
        if is_pensioner and self.iv.pensioner_pricing:
            return float(config.PENSIONER_REDUCED_PRICE), False, "pensioner"
        segment = "pensioner" if is_pensioner else "adult"

        # Baseline: the Uffizi has both walk-in (day-of) and pre-booked
        # online tickets [UFF]. ~15% of visitors walk in [assumption: we
        # don't have official data on the split]. They pay EUR 25
        # versus EUR 29 pre-booked - the EUR 4 difference is the
        # online platform's booking fee. RAMA removes walk-ins entirely
        # (all visitors pre-book).
        if not self.iv.rama and self.rng.random() < config.WALK_IN_FRACTION:
            return float(config.WALK_IN_TICKET_PRICE), True, segment

        # Last-hour locals: reduced evening price for Florentine residents
        entry_minute = entry_slot * config.ENTRY_SLOT_MINUTES
        if self.iv.last_hour_locals and entry_minute >= config.LAST_HOUR_LOCALS_WINDOW[0]:
            return float(config.LAST_HOUR_LOCALS_PRICE), False, segment

        # Modular RAMA pricing: base general-entry price + per-masterpiece
        # add-on. Visitors pay only for what they actually consume. Family
        # discount and audio guide stack on top of the modular ticket.
        if self.iv.rama:
            price = config.RAMA_BASE_GENERAL_PRICE + (
                num_masterpiece_bookings * config.RAMA_MASTERPIECE_ADDON
            )
            if (self.iv.family_discount
                    and self.rng.random() < config.PARENT_WITH_KIDS_FRACTION):
                price = max(0.0, price - config.PARENT_DISCOUNT_EUR)
            if (self.iv.audio_guide_revenue
                    and self.rng.random() < config.AUDIO_GUIDE_TAKE_UP):
                price += config.AUDIO_GUIDE_PRICE
            return float(price), False, segment

        # Standard pre-booked price. Under dynamic_pricing intervention,
        # use the time-of-day schedule. Otherwise the flat EUR 29
        # baseline.
        if self.iv.dynamic_pricing:
            price = self.scheduled_ticket_price(entry_minute)
        else:
            price = float(config.BASELINE_TICKET_PRICE)
        # Family discount: ~30% of paying adults travel with kids and
        # get EUR 5 off their ticket when the intervention is on.
        if (self.iv.family_discount
                and self.rng.random() < config.PARENT_WITH_KIDS_FRACTION):
            price = max(0.0, price - config.PARENT_DISCOUNT_EUR)
        # Audio guide ancillary revenue: ~25% of paying adults add
        # an EUR 6 audio guide to their ticket. This counts in the
        # ticket_price field for simplicity (the simulator sums all
        # ticket_price for revenue).
        if (self.iv.audio_guide_revenue
                and self.rng.random() < config.AUDIO_GUIDE_TAKE_UP):
            price += config.AUDIO_GUIDE_PRICE
        return price, False, segment

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

        # Under RAMA, both individual and group arrivals are
        # pre-scheduled by the pre-day booking phase. Groups book
        # masterpiece slots ahead too, competing with individuals for
        # capacity. Spawn whatever's scheduled for this minute.
        if self.iv.rama:
            scheduled = self._rama_schedule.get(self.current_time, [])
            for booking in scheduled:
                v = self.create_npc_visitor_from_booking(booking)
                if v is not None:
                    self.outside_queue.append(v)
            self._admit_from_queue()
            return

        rate = self.desired_arrival_rate(self.current_time)

        # Walk-in premium: the existence of a premium price deters a fraction
        # of unplanned walk-ins, reducing the overall arrival rate.
        # WALK_IN_FRACTION * WALK_IN_DETERRENCE = 0.15 * 0.40 = 6% reduction.
        if self.iv.walk_in_premium:
            rate *= (1.0 - config.WALK_IN_FRACTION * config.WALK_IN_DETERRENCE)

        # Quiet hours, arrival-reduction component. Active when either
        # the bundled quiet_hours flag is set, or the standalone
        # quiet_hours_arrival_reduction sub-flag is set. During the
        # quiet window (08:15-11:00), overall arrival rate is reduced
        # by 15% because the quiet atmosphere is marketed to a more
        # selective audience. [assumption]
        if self.iv.quiet_hours or self.iv.quiet_hours_arrival_reduction:
            lo, hi = config.QUIET_HOURS_WINDOW
            if lo <= self.current_time <= hi:
                rate *= 0.85

        # ----- Tour group fraction -----
        # Tour groups create corridor "shock waves" by moving large
        # coordinated blocks of Type B visitors together. The fraction
        # of arrivals that form groups is modulated by interventions.
        base_group_frac = config.TOUR_GROUP_FRACTION  # 15% baseline [assumption]
        # Quiet hours, tour-group-ban component. Active when either the
        # bundled flag is set, or the standalone tour-ban sub-flag.
        if self.iv.quiet_hours or self.iv.quiet_hours_tour_ban:
            lo, hi = config.QUIET_HOURS_WINDOW
            if lo <= self.current_time <= hi:
                base_group_frac = 0.0  # no tour groups during quiet hours
        # Group surcharge intervention. The Uffizi charges a flat
        # EUR 70 fee per tour group of 11+ (baseline). The intervention
        # value is the ADDITIONAL Pigovian surcharge on top of this
        # baseline. At intervention = EUR 70 (doubling to EUR 140 total),
        # group demand is significantly suppressed via price elasticity.
        if self.iv.per_person_group_surcharge > 0:
            # Per-head extra cost = total surcharge / group size.
            # Exponential decay: mult = exp(-extra_per_head / REF_10) so
            # demand never fully zeros (some tour operators absorb the
            # higher cost or pass it through to wealthier travellers).
            # At EUR 150 surcharge / 15-head group = EUR 10/head,
            # mult = exp(-1) = 0.37 -> 63% reduction, leaves ~20 groups/day.
            # At EUR 70 / 15 = 4.7/head, mult = 0.63 -> 37% reduction.
            extra_per_head = self.iv.per_person_group_surcharge / max(1, config.TOUR_GROUP_SIZE_DEFAULT)
            mult = math.exp(-extra_per_head / 10.0)
            base_group_frac = max(0.0, base_group_frac * mult)

        # ----- Sample arrival count -----
        # Individual arrival rate = total rate * (1 - group_frac). The
        # remaining (group_frac) of the visitor mass arrives via the
        # separate group-burst stream below, so the two streams together
        # match the configured total demand without double-counting.
        individual_rate = max(0.0, rate * (1.0 - base_group_frac))
        new_arrivals = int(self.rng.poisson(individual_rate))

        # Map current_time to the 15-minute entry slot index (0-36).
        entry_slot = min(config.N_ENTRY_SLOTS - 1, self.current_time // config.ENTRY_SLOT_MINUTES)

        # Tour group size: capped by intervention or default (30). [assumption]
        group_size = (
            self.iv.tour_group_cap if self.iv.tour_group_cap is not None
            else config.TOUR_GROUP_SIZE_DEFAULT
        )

        # ----- Create INDIVIDUAL visitors -----
        # The new_arrivals draw was scaled by (1 - base_group_frac) above
        # (conceptually); individuals enter as Poisson stream. Group
        # arrivals are a SEPARATE Poisson stream so that a tour bus can
        # arrive in a single minute even when per-minute individual
        # arrivals are below group_size.
        for _ in range(new_arrivals):
            self.outside_queue.append(self.create_npc_visitor(entry_slot=entry_slot))

        # ----- Tour group burst stream -----
        self._spawn_group_arrivals(rate=rate, base_group_frac=base_group_frac)

        # ----- Capacity-gated admission -----
        self._admit_from_queue()

    def _spawn_group_arrivals(self, rate: float | None,
                                base_group_frac: float | None) -> None:
        """Spawn tour-group bursts for the current minute.

        Real tour groups arrive as buses dropping off ~15 people at once.
        Modelling them as a separate per-minute Poisson stream (where each
        event spawns a full group_size cohort) lets baseline groups
        actually form (with the original per-arrival approach they almost
        never did because 8 arrivals/minute < 15 group size).

        Under RAMA the rate and base_group_frac are recomputed locally
        because step_arrivals' RAMA branch returns early without
        computing them.
        """
        entry_slot = min(config.N_ENTRY_SLOTS - 1, self.current_time // config.ENTRY_SLOT_MINUTES)
        group_size = (
            self.iv.tour_group_cap if self.iv.tour_group_cap is not None
            else config.TOUR_GROUP_SIZE_DEFAULT
        )

        # Under RAMA, recompute group fraction from scratch (the
        # baseline rate-modification code is bypassed in step_arrivals).
        if self.iv.rama:
            base_group_frac = config.TOUR_GROUP_FRACTION
            rate = (self.daily_total / max(1.0, float(self.last_entry_minutes))
                    if self.current_time <= self.last_entry_minutes else 0.0)
            # Apply group surcharge elasticity here too (independent
            # of individual arrival rate modification).
            if self.iv.per_person_group_surcharge > 0:
                extra_per_head = self.iv.per_person_group_surcharge / max(1, group_size)
                mult = math.exp(-extra_per_head / 10.0)
                base_group_frac = max(0.0, base_group_frac * mult)
            # Quiet hours tour-ban also applies under RAMA.
            if self.iv.quiet_hours or self.iv.quiet_hours_tour_ban:
                lo, hi = config.QUIET_HOURS_WINDOW
                if lo <= self.current_time <= hi:
                    base_group_frac = 0.0

        if rate is None or base_group_frac is None or base_group_frac <= 0:
            return

        group_rate_per_min = rate * base_group_frac / max(1, group_size)
        num_groups = int(self.rng.poisson(max(0.0, group_rate_per_min)))
        for _ in range(num_groups):
            self._tour_group_counter += 1
            gid = self._tour_group_counter
            leader = self.create_npc_visitor(entry_slot=entry_slot)
            leader.group_id = gid
            leader.profile = sample_type_b_profile(self.rng)
            leader.visitor_type = "B"
            if self.iv.revenue_model:
                baseline_group_fee = config.GROUP_SURCHARGE_FLAT / group_size
                extra = (self.iv.per_person_group_surcharge / group_size
                         if self.iv.per_person_group_surcharge > 0 else 0.0)
                leader.ticket_price += baseline_group_fee + extra
            self.outside_queue.append(leader)
            for _ in range(group_size - 1):
                follower = self.create_npc_visitor(entry_slot=entry_slot)
                follower.group_id = gid
                follower.profile = leader.profile
                follower.visitor_type = "B"
                follower.checklist = dict(leader.checklist)
                if self.iv.revenue_model:
                    follower.ticket_price += baseline_group_fee + extra
                self.outside_queue.append(follower)

    def _admit_from_queue(self) -> None:
        """Admit visitors from outside_queue into the museum.

        Under baseline: gated by MAX_MUSEUM_CAPACITY (900-person legal
        limit). Under RAMA: bookings ARE the capacity-control mechanism,
        so visitors enter without queue delay (the booking quota already
        enforces total daily demand). All visitors start at A1.
        """
        cap_check = (not self.iv.rama)
        while self.outside_queue and (
            not cap_check or self.total_inside < config.MAX_MUSEUM_CAPACITY
        ):
            v = self.outside_queue.popleft()
            v.current_room = "A1"       # all visitors enter through the vestibule
            v.time_in_room = 0
            v.rooms_visited.add("A1")
            v.path.append((self.current_time, "A1"))
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
        """Record a visitor entering a room. Updates all three counters.

        Also bumps ``arrivals_this_min[room]`` so the density-snapshot logic
        can distinguish "perpetual mass" (visitors who have stopped) from
        "transit" (visitors who arrived this minute and may walk straight
        through). The arrivals counter is reset to zero at the start of
        each step in ``step()``.
        """
        self.occ[room] += 1
        if visitor_type == "A":
            self.occ_a[room] += 1
        else:
            self.occ_b[room] += 1
        self.total_inside += 1
        self.arrivals_this_min[room] = self.arrivals_this_min.get(room, 0) + 1

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
        dwell_floor: float = 0.0,
    ) -> float:
        """Probability of staying one more minute in the current room.

        The stay probability follows an exponential decay model:
            p(stay) = exp(-time_in_room / expected_dwell)

        where expected_dwell = base_dwell * (room_magnetism * dwell_multiplier
        + dwell_floor). The dwell_floor is an additive term (zero for rushed
        tourists, positive for art lovers) that keeps the art lover lingering
        even in low-magnetism minor rooms, where the multiplicative term
        alone would collapse to seconds.

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
        dwell_floor : float
            Additive minutes-of-magnetism floor (default 0.0). Art lovers
            carry a positive floor so they linger in minor rooms.

        Returns
        -------
        float
            Probability in [0.05, 0.95] of staying one more minute.
        """

        # Floor of 0.1 minutes prevents division-by-near-zero when both
        # magnetism and dwell_multiplier are very small. Lowered from
        # 0.5 so genuine transit rooms (vestibule, corridors) can have
        # truly short expected dwells.
        expected_dwell = max(0.1, base_dwell * (room_magnetism * dwell_multiplier + dwell_floor))
        p = float(np.exp(-time_in_room / expected_dwell))
        # Upper clamp 0.95 prevents visitors from being stuck on t=0
        # (where exp(0)=1.0 would mean p_stay=1 forever). The lower
        # bound is now 0.0 so visitors can actually leave fast rooms;
        # the 0.05 floor of the previous version artificially extended
        # transit-room dwell and inflated their density.
        return float(np.clip(p, 0.0, 0.95))

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

        # Lanzi staircase fork. At the foot of the Lanzi staircase the
        # visitor can step out to the street (EXIT) or continue onto the
        # first floor (B1 / C1). Instagram tourists are done after the
        # terrace and take the direct exit; the thorough profiles (Standard
        # and Art Lover) continue down to the first floor and leave later
        # via the Magliabechi staircase. Forcing the choice here makes the
        # fork robust regardless of the route bonus or the RAMA exit filter.
        if room == "LANZI_STAIRCASE":
            if visitor.profile.segment == "instagram":
                forced = [n for n in neighbors if n == "EXIT"]
            else:
                forced = [n for n in neighbors if n != "EXIT"]
            if forced:
                neighbors = forced

        # Instagram tourists never leave their minimal route. For them the
        # U-loop, the Tribune chain and the dead-end side rooms do not
        # exist: they came only for the masterpiece selfies and the terrace.
        # Restrict their movement to the route + masterpieces so they cannot
        # divert into the rest of the museum even while waiting for a slot.
        if visitor.profile.segment == "instagram":
            on_route = [n for n in neighbors if n in config.INSTAGRAM_ROUTE_ROOMS]
            if on_route:
                neighbors = on_route

        # Buontalenti staircase suppression: A36 is the antechamber to
        # Raphael & Michelangelo (A38). The staircase from A36 descends
        # to the 1st floor with no return. Realistically, no visitor at
        # A36 who has NOT YET seen Raphael & Michelangelo would take
        # the staircase down - they would walk the 2 rooms to A38 to
        # see them. The staircase is for visitors LEAVING after seeing
        # Raphael who descend via A36 -> A38 -> A24 -> ... Or for art
        # lovers who have already done A38 and want the 1st floor.
        # Suppress this exit when A38 unvisited so flow naturally
        # mirrors Leonardo in baseline.
        if (
            room == "A36"
            and "A38" not in visitor.rooms_visited
            and "BUONTALENTI_STAIRCASE" in neighbors
        ):
            neighbors = [n for n in neighbors if n != "BUONTALENTI_STAIRCASE"]

        # Masterpiece re-entry suppression: once a visitor has been in a
        # masterpiece room, they don't re-enter it. Without this, the
        # importance pull (10) on a visited masterpiece can drag them
        # back through the corridor, inflating that room's occupancy
        # relative to rooms whose topology makes re-entry impossible.
        # Specifically: visitors leaving A38 -> A24 would 40% return
        # to A35 (Leo) since A24 -> A35 is a legal edge and importance
        # outweighs route bonus, while no such return exists for A38.
        # Suppressing re-entry makes Leo and Raph throughput symmetric.
        masterpiece_blockers = [
            n for n in neighbors
            if n in config.MASTERPIECE_ROOMS and n in visitor.rooms_visited
        ]
        if masterpiece_blockers and len(neighbors) > len(masterpiece_blockers):
            neighbors = [n for n in neighbors if n not in masterpiece_blockers]

        # Suppress EXIT pull for RAMA visitors with substantial budget.
        # Without this, RAMA visitors who finish the masterpiece chain
        # beeline straight to EXIT and end their visit with most of
        # their time-budget unused.
        if self.iv.rama and "EXIT" in neighbors and visitor.remaining_budget > 30 and len(neighbors) > 1:
            neighbors = [n for n in neighbors if n != "EXIT"]

        # Block exit-direction movement for RAMA visitors with un-honored
        # bookings. The 2nd-floor terrace and Lanzi staircase descend to
        # the 1st floor (B/C wings), from which there is no path back to
        # the 2nd-floor masterpieces. A visitor with a future Botticelli/
        # Leonardo/Raphael slot who wanders down would lose the slot
        # entirely (no walking path back). This filter keeps them on the
        # 2nd floor until they have honored all bookings.
        if self.iv.rama:
            has_unhonored = (
                (visitor.botticelli_window is not None and "A11" not in visitor.rooms_visited)
                or (visitor.leonardo_window is not None and "A35" not in visitor.rooms_visited)
                or (visitor.raphael_window is not None and "A38" not in visitor.rooms_visited)
            )
            if has_unhonored:
                blocked_exits = {"PANORAMIC_TERRACE", "LANZI_STAIRCASE", "BUONTALENTI_STAIRCASE"}
                remaining = [n for n in neighbors if n not in blocked_exits]
                if remaining:
                    neighbors = remaining
                # Find the next un-honored target and filter to only
                # neighbors from which it remains reachable. This
                # prevents wandering into dead-end rooms that have no
                # path back to the booked masterpiece.
                next_t = None
                for room_id, w in (
                    ("A11", visitor.botticelli_window),
                    ("A35", visitor.leonardo_window),
                    ("A38", visitor.raphael_window),
                ):
                    if w is not None and room_id not in visitor.rooms_visited:
                        next_t = room_id
                        break
                if next_t is not None:
                    reachable = [
                        n for n in neighbors
                        if self._distances.get(n, {}).get(next_t, 99) < 99
                    ]
                    if reachable:
                        neighbors = reachable

        # ----- First-floor priority: Caravaggio -----
        # Caravaggio (E4/E5) is the star of the first floor. A visitor who
        # has descended and not yet seen it will give up any other room to
        # get there: once their remaining budget falls within a margin of
        # the walking distance to Caravaggio, they beeline straight to it,
        # ignoring the normal route and importance pulls. This fires before
        # the generic exit-seeking so that, on the first floor, Caravaggio
        # takes priority over the door.
        if room[0] in "BCDE" and "E4" not in visitor.rooms_visited:
            d_car = self._distance(room, "E4")
            if 0 < d_car < 99 and visitor.remaining_budget < d_car + config.CARAVAGGIO_PRIORITY_MARGIN:
                # Deterministic beeline: always step to the neighbour that
                # is strictly closer to Caravaggio (ties broken by node id),
                # so the visitor reaches it in exactly d_car steps instead
                # of drifting. Trading away every other first-floor room.
                best_n, best_d = None, 10**9
                for n in neighbors:
                    dn = self._distance(n, "E4")
                    if dn < best_d or (dn == best_d and (best_n is None or n < best_n)):
                        best_d, best_n = dn, n
                if best_n is not None:
                    return best_n

        # ----- Exit-seeking heuristic for low-budget visitors -----
        # When a visitor has fewer than 15 minutes of budget left, they
        # prioritize reaching the EXIT. The weight for each neighbor is
        # inversely proportional to its shortest-path distance to EXIT.
        # This prevents visitors from wandering into dead ends when they
        # should be heading out.
        #
        # EXCEPTION: if a neighbor is an unvisited Type B magnet room
        # (Botticelli, Tribune, Leonardo, Raphael, Caravaggio) AND it is
        # also on the visitor's personal checklist, they will pop in for
        # a quick look even when running short on time. Lazy tourists
        # who have not seen the masterpiece yet will not leave without
        # at least a glance. This is the "I came all this way, I'm not
        # leaving without seeing Leonardo" effect.
        if visitor.remaining_budget < 15:
            magnet_unseen = [
                n for n in neighbors
                if n in visitor.profile.magnet_rooms
                and n not in visitor.rooms_visited
                and n in visitor.checklist
            ]
            if magnet_unseen:
                return magnet_unseen[int(self.rng.integers(0, len(magnet_unseen)))]
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
            route_choices = self._vasari_next.get(room, [])
        elif visitor.profile.segment == "instagram":
            route_choices = self._ig_next.get(room, [])
        elif visitor.profile.segment == "art_lover":
            route_choices = self._artlover_next.get(room, [])
        else:
            route_choices = self.next_room.get(room, [])

        # Pick the first un-visited room on the route as the route bias
        # target. This is the key fix for the "last-write-wins" bug: at
        # corridor hubs like A24 that appear multiple times in the route,
        # we route the visitor toward whichever branch they have not yet
        # completed (e.g., A25 -> ... -> A35 -> ... -> exit). When every
        # listed successor has been visited, fall back to the last entry
        # (typically the exit direction).
        next_pref = None
        for cand in route_choices:
            if cand in visitor.rooms_visited:
                continue
            # Under RAMA, skip a masterpiece the visitor has no booking for:
            # they cannot enter it, so routing them toward it just makes them
            # bounce off the gate and wander the side rooms forever. Advance
            # the route to the next reachable target instead.
            if self.iv.rama and cand in config.MASTERPIECE_ROOMS:
                if cand in ("A11", "A12"):
                    booked = visitor.botticelli_window is not None
                elif cand == "A35":
                    booked = visitor.leonardo_window is not None
                elif cand == "A38":
                    booked = visitor.raphael_window is not None
                else:
                    booked = True
                if not booked:
                    continue
            next_pref = cand
            break
        if next_pref is None and route_choices:
            next_pref = route_choices[-1]

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
            #
            # Non-art-lover visitors (Instagram and Standard tiers) do NOT
            # care about the graph-level cultural importance of a room.
            # They care only about their personal selfie/checkmark list.
            # For non-art-lover visitors at non-magnet rooms we strip out
            # the room_importance term and use only their personal vector
            # (near-zero except at the checklist magnets). Only art lovers
            # get the full room-importance pull, because they care about
            # every room's cultural value.
            #
            # NOVELTY: once a visitor has visited a room, the pull back to
            # it is drastically reduced (5% of original), for ALL profiles.
            # This stops ping-ponging back into rooms already seen. Art
            # lovers express their depth by lingering longer IN a room (the
            # dwell floor), not by walking back into it; without this they
            # oscillate along corridors and never traverse the museum.
            room_importance = float(self.g.nodes[n]["importance"])
            personal_importance = float(visitor.profile.importance_vector[idx])
            is_art_lover = visitor.profile.segment == "art_lover"
            is_magnet = n in visitor.profile.magnet_rooms
            already_seen = n in visitor.rooms_visited

            # Under RAMA: only the visitor's PREFERENCE-CLEANED state
            # (importance vector already zeroed for unbooked rooms at
            # visitor creation) governs whether they're pulled toward
            # a masterpiece. Booked visitors keep their high importance
            # and naturally path toward the masterpiece via route bonus
            # and importance pull, just like baseline. The slot gate
            # then enforces "you may only enter during your window."
            # If they arrive at the pre-room early, the
            # _pre_masterpiece_holding_boost in the stay-probability
            # code makes them linger there until the slot opens.
            # No artificial pull suppression here.

            if not is_art_lover and not is_magnet:
                importance_pull = 0.05 + 0.10 * personal_importance
            else:
                importance_pull = 0.05 + 0.08 * room_importance + 0.10 * personal_importance
                if already_seen:
                    importance_pull = 0.05 + 0.05 * importance_pull

            # RAMA: don't path to the pre-room of a future masterpiece
            # slot too early. Real visitors look ahead - if their slot
            # is 30 min away, they don't go stand at the door now;
            # they explore other rooms first. Suppress the pull toward
            # the pre-room until the slot is within 10 min.
            # Also: respect crowding at the pre-room - if it's full,
            # they back off (even Type B notices a packed pre-room).
            if self.iv.rama:
                pre_room_check = None
                if n == "A10" and visitor.botticelli_window is not None and "A11" not in visitor.rooms_visited:
                    pre_room_check = visitor.botticelli_window
                elif n == "A24" and visitor.leonardo_window is not None and "A35" not in visitor.rooms_visited:
                    pre_room_check = visitor.leonardo_window
                elif n == "A36" and visitor.raphael_window is not None and "A38" not in visitor.rooms_visited:
                    pre_room_check = visitor.raphael_window
                if pre_room_check is not None:
                    time_to_slot = pre_room_check[0] - self.current_time
                    if time_to_slot > 10:
                        # Too early to head to the pre-room.
                        importance_pull *= 0.1
                    # Also: if pre-room is already crowded, back off.
                    pre_dens = self.density(n)
                    if pre_dens > 0.7:
                        importance_pull *= max(0.1, 1.0 - pre_dens)

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

            # Under RAMA, every visitor avoids crowded rooms regardless
            # of segment. The reservation system encourages forward-
            # looking behaviour: you can see Pollaiolo is packed and
            # know your slot is in 20 min, so you stay back at Piero
            # (or Masaccio) until the pre-room clears. This bonus is
            # applied to ALL visitors and is strong enough to dominate
            # the importance pull when a room is approaching capacity.
            # Transit nodes must not be repelled by crowding. This covers
            # (a) the staircases / terrace / exit (the only ways down and
            # out) and (b) the U-shaped corridors A2/A23/A24, which carry
            # every visitor and are therefore always busy. Without the
            # corridor exemption the anti-crowd penalty cancels the route
            # bonus on a packed corridor, and visitors flee into empty side
            # rooms (the Tribune chain, the U-loop) instead of following the
            # corridor: Instagram tourists end up wandering the whole floor
            # rather than taking the shortest path. You cannot avoid a
            # corridor because it is busy; it is the only way through.
            if self.iv.rama and visitor.profile.segment != "instagram" and n not in {
                "PANORAMIC_TERRACE", "LANZI_STAIRCASE",
                "BUONTALENTI_STAIRCASE", "GRANDUCAL_STAIRCASE",
                "ENTRY", "EXIT", "A2", "A23", "A24",
            }:
                dens_n = self.density(n)
                # Emptiness bonus only for rooms not yet visited: an empty
                # room you have already seen should not pull you back into
                # it. Without this guard, every visited room behind a
                # visitor stays attractive (it is empty) and the visitor
                # ping-pongs along the corridor instead of progressing.
                # The crowding penalty still applies regardless.
                if dens_n < 0.5 and n not in visitor.rooms_visited:
                    anti_crowd += 3.0 * (0.5 - dens_n)
                elif dens_n >= 0.8:
                    anti_crowd -= 8.0 * (dens_n - 0.8)

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

            # Predictive routing: project density forward and penalize
            # rooms that will be crowded by the time the visitor gets
            # there. Uses a 5-minute trailing window to estimate the
            # density derivative, then extrapolates 5 minutes ahead.
            # The projected density (current plus 5x trend, clamped to
            # [0, 1.5]) replaces the present density in the anti_crowd
            # computation, with the same magnitude as the regular crowd
            # avoidance term so the signal is comparable in strength.
            # Differs from adaptive_trail (which redirects at entry by
            # trail assignment) by acting at every per-room movement
            # decision, on every visitor type.
            if self.iv.predictive_routing and len(self.density_history) >= 5:
                idx_n = config.ROOM_TO_IDX[n]
                recent = [h[idx_n] for h in self.density_history[-5:]]
                trend_per_min = (recent[-1] - recent[0]) / 4.0
                projected = recent[-1] + 5.0 * trend_per_min  # density in 5 min
                projected = max(0.0, min(1.5, projected))
                pred_alpha = config.PREDICTIVE_ROUTING_ALPHA
                if visitor.visitor_type == "A":
                    pred_alpha *= visitor.profile.anti_crowd_bonus
                else:
                    pred_alpha *= 0.5
                anti_crowd += pred_alpha * (1.0 - projected)

            # --- Component 3: Route bonus ---
            # Baseline of 1.0 ensures every neighbor has nonzero weight.
            # For art lovers the route bonus is moderate (+3 * route_bias)
            # so they can deviate. For non-art-lovers (Instagram and
            # Standard tiers) the route bonus is much higher (+10 *
            # route_bias) because they do not in practice wander; they
            # stick to the masterpiece path. Without this bump, a
            # non-trivial fraction would divert at corridor hubs (A2 to
            # Giotto, etc.) and skip Botticelli, which the user said
            # never happens in reality. Checklist rooms also keep their
            # +1.5 bonus.
            route_bonus = 1.0
            if next_pref == n:
                # Strong route adherence for all visitors. Even art lovers
                # in practice cluster around the famous rooms; the older
                # 3x multiplier let them wander off and skip Botticelli,
                # which the user said never happens.
                # Strong route adherence for all profiles. The art lover's
                # breadth comes from their long route (which sweeps the
                # whole museum), not from random deviation: without firm
                # adherence they drift backward into empty floor-2 rooms
                # after their masterpiece slots and never descend to the
                # first floor. Their importance pull still breaks ties
                # toward culturally important neighbours.
                bias_mult = 10.0
                route_bonus += bias_mult * visitor.profile.route_bias
            if n in visitor.checklist:
                route_bonus += 1.5

            # --- Final weight ---
            # Floor of 0.01 ensures no neighbor has exactly zero probability.
            weight = max(0.01, route_bonus + importance_pull + anti_crowd)
            # Discourage stepping back into a room already seen when the
            # route still points forward to an unvisited room. Without this,
            # the residual ~9% backward probability accumulates and a
            # visitor who reaches the first floor late never traverses it
            # before their budget runs out. Forced route revisits (where the
            # visited room IS the next route step, e.g. the A24 corridor hub)
            # are exempt, as are dead ends where every neighbour is visited.
            if already_seen and n != next_pref:
                weight *= 0.25
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

        # ----- RAMA booking enforcement -----
        # A booking is a guarantee, by definition. The visitor planned
        # their visit around the slot and shows up on time. If their
        # current position happens to be somewhere other than the
        # masterpiece room at slot start (because walking dynamics had
        # them lingering elsewhere), the booking system places them in
        # the room. This is what the appointment-pacing tries to do
        # via natural walking; the placement here makes it exact, the
        # way a real booking works. They still walked through pre-rooms
        # in the minutes leading up to the slot; the placement is just
        # "you entered the room at your reserved time."
        if self.iv.rama:
            # Forfeit any slot whose window has fully passed without entry
            # (the room stayed at capacity through the slot). Otherwise the
            # visitor is treated as having an un-honored booking forever and
            # is kept inside indefinitely, trying to reach a room it can no
            # longer enter. Clearing the window lets it proceed to the exit.
            if (visitor.botticelli_window is not None
                    and self.current_time > visitor.botticelli_window[1]
                    and "A11" not in visitor.rooms_visited):
                visitor.botticelli_window = None
            if (visitor.leonardo_window is not None
                    and self.current_time > visitor.leonardo_window[1]
                    and "A35" not in visitor.rooms_visited):
                visitor.leonardo_window = None
            if (visitor.raphael_window is not None
                    and self.current_time > visitor.raphael_window[1]
                    and "A38" not in visitor.rooms_visited):
                visitor.raphael_window = None
            target_room = None
            for room, w in (
                ("A11", visitor.botticelli_window),
                ("A35", visitor.leonardo_window),
                ("A38", visitor.raphael_window),
            ):
                if w is None or room in visitor.rooms_visited:
                    continue
                if w[0] <= self.current_time < w[1]:
                    target_room = room
                    break
            if target_room is not None and visitor.current_room != target_room:
                # Respect room capacity. If the room is full (typically
                # because last slot's visitors haven't all left yet),
                # don't push them in - the visitor stays put and will
                # be placed in the next minute when capacity opens up.
                target_cap = self.g.nodes[target_room]["capacity"]
                if self.occ[target_room] < target_cap:
                    self._decrement_occ(visitor.current_room, visitor.visitor_type)
                    visitor.current_room = target_room
                    visitor.time_in_room = 0
                    self._increment_occ(target_room, visitor.visitor_type)
                    visitor.rooms_visited.add(target_room)
                    visitor.path.append((self.current_time, target_room))

        # ----- Experienced welfare accrual -----
        # Each minute the visitor spends in a gallery room contributes:
        #   personal_importance[room] / (1 + crowd_alpha * density[room]**2)
        # The crowd term is QUADRATIC in density. With the linear form,
        # an Instagram tourist (alpha=0.5) in a packed Botticelli at
        # density=2.0 retained a 0.5 discount factor (50% of the full
        # experience). That is unrealistic: at peak crowd the painting
        # is essentially invisible. The quadratic form gives 1/(1+0.5*4)
        # = 0.33 at density=2.0 for the IG tier, and 1/(1+6*4) = 0.04
        # for the art lover, both closer to real perception. Personal
        # importance reflects what THIS visitor wants to see (Botticelli
        # for art lovers, the terrace for IG tourists). Transit nodes
        # (ENTRY/EXIT, staircases) have importance approximately zero in
        # the personal vector, so they contribute negligibly without
        # special-casing.
        room = visitor.current_room
        if room in config.ROOM_TO_IDX:
            idx = config.ROOM_TO_IDX[room]
            personal_importance = float(visitor.profile.importance_vector[idx])
            room_density = self.density(room)
            crowd_alpha = float(visitor.profile.crowd_alpha)
            visitor.experienced_welfare += (
                personal_importance / (1.0 + crowd_alpha * room_density * room_density)
            )

        # ----- Budget tick -----
        # Every visitor loses one minute of budget per step, regardless
        # of whether they stay or move. This is a hard time constraint.
        visitor.remaining_budget -= 1

        # Budget exhausted: the visitor leaves immediately from wherever
        # they are. The previous "stubbornness" extension that granted
        # +30 minutes when magnets were unvisited has been removed, since
        # the now-forced topology (no skip path past Botticelli) combined
        # with the 200-260 min Instagram budget gives visitors enough
        # time to complete the masterpiece tour naturally. The extension
        # was inflating Leonardo's dwell beyond Botticelli's because
        # visitors arriving at Leonardo with low budget kept being
        # extended.
        if visitor.remaining_budget <= 0:
            # Past closing: the museum is shutting; no more budget
            # extensions fire (no booking hold, no mid-chain or Caravaggio
            # grant). The visitor is ushered out wherever they are, so
            # nobody is stranded inside overnight.
            if self.current_time >= self.day_minutes:
                self._decrement_occ(visitor.current_room, visitor.visitor_type)
                self.completed_visitors.append(visitor)
                return False
            # Under RAMA: if the visitor has un-honored bookings, they
            # do NOT exit. A ticket is a guarantee; they stay until
            # they have seen the rooms they paid for. Extend by enough
            # time to honor remaining slots plus a walk out.
            if self.iv.rama and (
                (visitor.botticelli_window is not None and "A11" not in visitor.rooms_visited)
                or (visitor.leonardo_window is not None and "A35" not in visitor.rooms_visited)
                or (visitor.raphael_window is not None and "A38" not in visitor.rooms_visited)
            ):
                visitor.remaining_budget = 1  # keep them inside
            elif (
                visitor.current_room in {"A35", "A36", "A37"}
                and "A38" not in visitor.rooms_visited
                and visitor.profile.importance_vector[config.ROOM_TO_IDX.get("A38", 0)] > 0.05
            ):
                # Mid-chain protection: visitor is inside the
                # Leonardo->Raphael/Michelangelo corridor with A38 still
                # ahead in their checklist. Don't strand them; grant
                # enough time to walk to A38 AND complete the full
                # IG dwell target there (10 min).
                visitor.remaining_budget = 15
            elif (
                visitor.current_room[0] in "BCDE"
                and "E4" not in visitor.rooms_visited
                and self._distances.get(visitor.current_room, {}).get("E4", 99) < 99
            ):
                # Caravaggio guarantee: a visitor on the first floor does not
                # leave without seeing Caravaggio. Grant only a small top-up
                # so the deterministic beeline can finish the last few steps;
                # the beeline (see _pick_next_room) does the real work, so
                # this should rarely fire and the visit does not balloon.
                visitor.remaining_budget = 8
            else:
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

        # RAMA appointment-pacing. Booked visitors plan their pre-room
        # dwell to arrive at the masterpiece exactly at slot start.
        # Algorithm:
        #   1. Find the next un-honored slot and its target room.
        #   2. Compute walking distance current->target.
        #   3. spare_time = (slot_start - current_time) - distance
        #   4. per_room_dwell_budget = spare_time / distance
        #      (each remaining pre-room can be lingered this much)
        #   5. If time_in_room >= per_room_dwell_budget OR spare_time<=0:
        #      MUST move toward target NOW.
        #   6. If at target and slot not yet open: WAIT.
        # When MUST move, the move decision overrides _pick_next_room to
        # pick the neighbor that minimises distance to target. This is
        # the "I have an appointment, I'm getting there on time" rule.
        _rama_wait_for_slot = False
        _rama_must_move = False
        _rama_target = None
        if self.iv.rama:
            for room_id, w in (
                ("A11", visitor.botticelli_window),
                ("A35", visitor.leonardo_window),
                ("A38", visitor.raphael_window),
            ):
                if w is None or room_id in visitor.rooms_visited:
                    continue
                if self.current_time >= w[1]:
                    continue  # slot ended already
                _rama_target = room_id
                next_slot_start = w[0]
                break
            if _rama_target is not None:
                distance = self._distances.get(visitor.current_room, {}).get(_rama_target, 99)
                if distance == 0:
                    # Already in target. Wait if slot not yet open.
                    if self.current_time < next_slot_start:
                        _rama_wait_for_slot = True
                elif distance == 1:
                    # In the pre-room. Real visitors arrive ~3 min before
                    # their slot, not 30 minutes early. If the slot is
                    # still > 3 min away AND the pre-room is uncrowded,
                    # they leave (move back upstream). If close to slot
                    # OR pre-room is already crowded enough, they wait.
                    if self.current_time < next_slot_start:
                        if next_slot_start - self.current_time <= 3:
                            _rama_wait_for_slot = True
                    else:
                        _rama_must_move = True
                else:
                    # Distance >= 2. Backward induction with a
                    # per-visitor random spread so 55 bookers do not
                    # all walk into the pre-room at the same minute.
                    # Each visitor's walking_start is the slot time
                    # minus walking distance minus a stagger of 0-5 min
                    # derived from visitor id (deterministic, evenly
                    # spread across the cohort). Before walking_start,
                    # the visitor is free to wander upstream rooms,
                    # paced naturally by their personal importance
                    # vector and the upstream pre-room suppression
                    # implemented below in _pick_next_room.
                    stagger = (visitor.visitor_id * 31) % 6  # 0..5
                    walking_start = next_slot_start - (distance - 1) - stagger
                    if self.current_time >= walking_start:
                        _rama_must_move = True

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

        # Type B visitors (Instagram and Standard tiers) at their personal
        # magnet rooms lift the rushed dwell_multiplier (0.2 / 0.3) up to
        # a fixed "selfie floor" of 0.5. This is the time needed to take
        # a photo and look at the painting; it does NOT scale with density,
        # because the masterpiece rooms have roughly equal magnetism and
        # visitors spend roughly equal time in each (Botticelli ~15 min,
        # Leonardo ~14 min, Raphael ~14 min, per the user's empirical
        # description). Botticelli ending up more crowded than Leonardo
        # comes from the visit-rate gap (more visitors reach it first),
        # not from any per-visitor dwell difference.
        effective_dwell_mult = visitor.profile.dwell_multiplier
        if (
            visitor.visitor_type == "B"
            and visitor.current_room in visitor.profile.magnet_rooms
        ):
            effective_dwell_mult = max(effective_dwell_mult, 1.0)

        # Per-segment hard dwell targets (e.g. IG tourist: 5 min A11, 5
        # min A12, 10 min Leo, 10 min Raph, 30 min terrace). Overrides
        # the magnetism-based decay so the visitor stays exactly the
        # empirically-observed time, then moves on. Applies in both
        # baseline and RAMA.
        _dwell_target_dict = visitor.profile.magnet_dwell_target
        _dwell_target = (
            _dwell_target_dict.get(visitor.current_room)
            if _dwell_target_dict else None
        )

        # RAMA hard cap at masterpieces. Last-Supper-Milan style: once a
        # visitor has been in a masterpiece room for the cap (10 min),
        # they are forced to move on.
        if (
            self.iv.rama
            and visitor.current_room in config.MASTERPIECE_ROOMS
            and visitor.time_in_room >= config.RAMA_DWELL_CAP_MIN
        ):
            p_stay = 0.0  # forced to leave this step
        elif _dwell_target is not None:
            # Per-segment target: stay until time_in_room hits target.
            p_stay = 1.0 if visitor.time_in_room < _dwell_target else 0.0
        elif _rama_wait_for_slot:
            # Waiting in the pre-room for the slot to open.
            p_stay = 1.0
        elif _rama_must_move:
            # Appointment is imminent; no slack left to linger here.
            p_stay = 0.0
        else:
            # Compute stay probability from effective magnetism and time in room.
            p_stay = self.npc_stay_probability(
                room_magnetism=room_magnetism,
                time_in_room=visitor.time_in_room,
                dwell_multiplier=effective_dwell_mult,
                dwell_floor=visitor.profile.dwell_floor,
            )

        # ----- Stay: visitor remains in current room -----
        if self.rng.random() < p_stay:
            visitor.time_in_room += 1
            visitor.rooms_visited.add(visitor.current_room)
            return True

        # ----- Move: visitor decides to leave, pick next room -----
        if _rama_must_move and _rama_target is not None:
            # Appointment pacing: pick the neighbor that minimises
            # remaining distance to the booked masterpiece. This is the
            # "I have to be at my slot on time" rule. Ties broken by
            # smaller node id for determinism.
            best_n, best_d = None, 10**9
            for n in self.g.successors(visitor.current_room):
                d = self._distances.get(n, {}).get(_rama_target, 99)
                if d < best_d or (d == best_d and (best_n is None or n < best_n)):
                    best_d, best_n = d, n
            next_room = best_n if best_n is not None else self._pick_next_room(visitor)
        else:
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
        # into Botticelli simultaneously. Denied visitors bypass the
        # gated room rather than stand in place (the one-way Botticelli
        # chain has no back-edge, so staying put would trap them).
        if (
            self.iv.botticelli_slot_cap is not None
            and next_room in {"A11", "A12"}
            and old_room not in {"A11", "A12"}
            and self._botticelli_entries_this_step >= self.iv.botticelli_slot_cap
        ):
            bypass = config.MASTERPIECE_BYPASS.get(next_room, old_room)
            next_room = bypass if bypass in self.g.nodes else old_room

        # Masterpiece gating with per-room windows.
        #
        # Under RAMA: each visitor has separate windows for Botticelli
        # (A11/A12), Leonardo (A35) and Raphael (A38). Entry is
        # permitted ONLY when current_time falls within the visitor's
        # booked window for that room. Visitors with NO booking are
        # also blocked: no slot, no entry. Both cases route to bypass.
        # This matches the real-world rule: you cannot see the painting
        # without a ticket for it, full stop. Walk-through visitors
        # don't get phantom credit for glancing at it on their way past.
        # Under decoupled_botticelli_gating (single shared window):
        # same logic but only if visitor has a window.
        if next_room in config.MASTERPIECE_ROOMS and old_room not in config.MASTERPIECE_ROOMS:
            if self.iv.rama:
                relevant_window = None
                if next_room in {"A11", "A12"}:
                    relevant_window = visitor.botticelli_window
                elif next_room == "A35":
                    relevant_window = visitor.leonardo_window
                elif next_room == "A38":
                    relevant_window = visitor.raphael_window
                # Block if there is no booking at all OR if we're outside
                # the booked window.
                blocked = relevant_window is None
                if not blocked:
                    w_start, w_end = relevant_window
                    blocked = not (w_start <= self.current_time <= w_end)
                # Hard room-capacity check: the slot cap equals the
                # room cap, so by construction no more than that many
                # people should ever be inside. If the room is already
                # at cap (last slot's visitors haven't finished
                # leaving), the new visitor waits one minute in the
                # pre-room instead of pushing in.
                if not blocked:
                    room_cap = self.g.nodes[next_room]["capacity"]
                    if self.occ[next_room] >= room_cap:
                        next_room = old_room  # wait one step, try again next minute
                if blocked:
                    bypass = config.MASTERPIECE_BYPASS.get(next_room, old_room)
                    next_room = bypass if bypass in self.g.nodes else old_room
            elif self.iv.decoupled_botticelli_gating and visitor.botticelli_window is not None:
                w_start, w_end = visitor.botticelli_window
                if not (w_start <= self.current_time <= w_end):
                    bypass = config.MASTERPIECE_BYPASS.get(next_room, old_room)
                    next_room = bypass if bypass in self.g.nodes else old_room

        # Reciprocal access: visitor must have visited at least N hidden-gem
        # trail rooms before being allowed into Botticelli. Rewards
        # exploration of underused galleries with access to the star rooms.
        # Denied visitors bypass past Botticelli (one-way chain trap).
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
                bypass = config.MASTERPIECE_BYPASS.get(next_room, old_room)
                next_room = bypass if bypass in self.g.nodes else old_room

        # Magnet room windows: extends decoupled gating to ALL magnet rooms
        # (A11, A12, A35, A38, E4), not just Botticelli. Reuses the same
        # botticelli_window field for simplicity. [assumption] Denied
        # visitors bypass past the gated room.
        if (
            self.iv.magnet_room_windows
            and next_room in config.ALL_MAGNET_WINDOW_ROOMS
            and old_room not in config.ALL_MAGNET_WINDOW_ROOMS
            and visitor.botticelli_window is not None
        ):
            w_start, w_end = visitor.botticelli_window
            if not (w_start <= self.current_time <= w_end):
                # E4 (Caravaggio) is bidirectional so old_room is fine
                # as a fallback; the masterpiece rooms have explicit
                # bypass targets to avoid the one-way trap.
                bypass = config.MASTERPIECE_BYPASS.get(next_room, old_room)
                next_room = bypass if bypass in self.g.nodes else old_room

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
            visitor.path.append((self.current_time, next_room))

        visitor.rooms_visited.add(visitor.current_room)

        # Immediate exit: if the visitor moved to EXIT, they leave this
        # step (no dwell time at the exit).
        if visitor.current_room in {"EXIT"}:
            self._decrement_occ(visitor.current_room, visitor.visitor_type)
            self.completed_visitors.append(visitor)
            return False

        # ----- Fast-transit chain -----
        # Visitors flagged fast_transit (Instagram tier) blitz through
        # non-magnet rooms in the same simulated minute, up to 5 chain
        # moves. The chain stops at any magnet room (so RAMA / capacity
        # gates fire on the next minute's normal step), at EXIT, or
        # when no forward move is possible. This compresses the long
        # transit chain (Botticelli to Leonardo through 10+ rooms) into
        # the real-world ~1 walking minute the small Uffizi actually
        # takes.
        if (
            getattr(visitor.profile, "fast_transit", False)
            and visitor.current_room not in visitor.profile.magnet_rooms
        ):
            STAIRCASE_NODES = {"GRANDUCAL_STAIRCASE", "LANZI_STAIRCASE",
                               "BUONTALENTI_STAIRCASE"}
            for _ in range(5):
                if visitor.current_room in visitor.profile.magnet_rooms:
                    break
                if visitor.current_room in STAIRCASE_NODES:
                    break
                chain_next = self._pick_next_room(visitor)
                if chain_next == visitor.current_room:
                    break
                if chain_next in visitor.profile.magnet_rooms:
                    break  # hand off to next-minute gating
                if chain_next == "EXIT":
                    break
                self._decrement_occ(visitor.current_room, visitor.visitor_type)
                visitor.current_room = chain_next
                visitor.time_in_room = 0
                self._increment_occ(visitor.current_room, visitor.visitor_type)
                visitor.rooms_visited.add(visitor.current_room)
                visitor.path.append((self.current_time, visitor.current_room))

        return True

    # =================================================================
    # Density snapshot and main simulation step
    # =================================================================

    def _snapshot_densities(self) -> np.ndarray:
        """Capture the current room-density vector (N_ROOMS floats).

        The metric is INTENT-WEIGHTED occupancy divided by capacity.
        A visitor contributes their full weight (1.0) to a room's density
        only if (a) they have spent at least one full minute there
        (excludes transit) AND (b) they actually intended to visit it
        (the room is on their personal magnet set, or they are an art
        lover who values every room). Visitors who are just walking
        through a non-magnet room contribute only 0.1. This makes
        density mean "perpetual mass of people stopped here", which is
        what the user actually wants to measure. The raw ``self.occ``
        counter is unchanged and still drives movement, capacity caps,
        and the crowd-feedback dwell formula.
        """

        weighted = np.zeros(config.N_ROOMS, dtype=float)
        for v in self.active_visitors:
            room = v.current_room
            if room is None or v.time_in_room < 1:
                continue  # not yet admitted, or still in their transit minute
            is_art_lover = v.profile.segment == "art_lover"
            is_magnet = room in v.profile.magnet_rooms
            weight = 1.0 if (is_art_lover or is_magnet) else 0.1
            weighted[config.ROOM_TO_IDX[room]] += weight

        dens = np.zeros(config.N_ROOMS, dtype=float)
        for room, idx in config.ROOM_TO_IDX.items():
            cap = self.g.nodes[room]["capacity"]
            dens[idx] = weighted[idx] / max(1.0, cap)
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

        # Reset the per-minute arrivals tracker. ``_increment_occ`` bumps
        # this counter every time a visitor enters a room (either via
        # door admission or by moving through a doorway). The snapshot at
        # the end of the step subtracts it from ``occ`` to compute the
        # "dwelling" density, so transit visitors do not inflate the
        # crowdedness metric for small high-throughput rooms.
        for room in self.arrivals_this_min:
            self.arrivals_this_min[room] = 0

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

        # Visitor-integrated welfare aggregates. Includes every visitor
        # who ATTEMPTED to visit, not just those who completed:
        #   - completed_visitors: finished a full visit, full welfare
        #   - active_visitors: still inside when day ended, partial welfare
        #   - outside_queue: queued but never admitted, welfare = 0
        # The denominator is _visitor_counter, the count of every
        # visitor created across the day. A visitor who was admitted but
        # had a chaotic time scores low; a visitor who was never
        # admitted scores exactly zero. Per-attempted-visitor mean
        # welfare is then the social-welfare object: how much
        # experience the museum produced per unit of demand. With this
        # denominator, gating interventions are scored honestly: the
        # denied visitors weigh into the metric at zero, but the
        # remaining visitors' improved experience is also captured.
        all_attempted = (
            list(self.completed_visitors)
            + list(self.active_visitors)
            + list(self.outside_queue)
        )
        type_a_welfare = float(sum(v.experienced_welfare for v in all_attempted if v.visitor_type == "A"))
        type_b_welfare = float(sum(v.experienced_welfare for v in all_attempted if v.visitor_type == "B"))
        experienced_welfare_total = type_a_welfare + type_b_welfare
        n_attempted = max(1, int(self._visitor_counter))
        mean_welfare_per_attempted = experienced_welfare_total / n_attempted

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
            "experienced_welfare_total": experienced_welfare_total,
            "experienced_welfare_type_a": type_a_welfare,
            "experienced_welfare_type_b": type_b_welfare,
            "mean_welfare_per_attempted": float(mean_welfare_per_attempted),
            "n_attempted": float(n_attempted),
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
