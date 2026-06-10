"""Central intervention configuration dataclass.

All intervention flags and parameters are bundled into a single frozen
dataclass. ``InterventionConfig()`` with no arguments produces the
status quo (all interventions disabled). This replaces the previous
pattern of passing 12+ keyword arguments through CrowdSimulator,
``_simulate_population_day``, and ``simulate_day_metrics``.

Design pattern: frozen dataclass
---------------------------------
The ``@dataclass(frozen=True)`` decorator makes instances immutable
after construction. This provides three guarantees:

  1. **Thread safety**: multiple simulator threads can share a config
     without locks, because no thread can mutate it.
  2. **Hashability**: frozen dataclasses are hashable, enabling use as
     dictionary keys or set members (useful for caching results keyed
     by intervention configuration).
  3. **Accidental mutation protection**: calling ``config.timed_entry = True``
     raises ``FrozenInstanceError``, preventing bugs where a shared
     config is silently modified mid-simulation.

To create a modified copy, use ``dataclasses.replace(config, field=value)``.

Backward compatibility
----------------------
``InterventionConfig.from_kwargs(**kwargs)`` accepts the old-style
keyword arguments used by CrowdSimulator and portfolio.py. It silently
drops any kwargs that do not correspond to a field on the dataclass,
so callers can safely pass a superset of keys (e.g., kwargs that also
include simulator parameters like ``daily_total`` or ``seed``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class InterventionConfig:
    """Frozen container for all supported intervention switches and parameters.

    Each field represents one intervention. Boolean fields are on/off
    switches; ``str | None`` fields specify a target room or theme;
    ``int | None`` and ``float`` fields specify numeric parameters
    (e.g., group cap size, surcharge amount).

    Fields are organized into groups by mechanism category.
    Default values disable every intervention (status quo).
    """

    # =========================================================================
    # Group: Original interventions (Wave 1)
    # =========================================================================
    # These 12 interventions were part of the initial codebase and cover
    # the most common museum management levers: entry scheduling, capacity
    # caps, pricing, and information provision.

    # Assigns visitors to 15-minute entry time slots, smoothing the arrival
    # curve to prevent the morning crush (10:00-12:00).
    timed_entry: bool = False

    # Activates real-time crowd information kiosks at corridor junctions.
    # Visitors at kiosk rooms see full museum density; others see only
    # their room and neighbors. See dynamic_info.py for details.
    dynamic_info: bool = False

    # Hard cap on simultaneous visitors in the Botticelli rooms (A11, A12).
    # None = no cap (visitors flow freely). Typical value: 55 (room capacity).
    botticelli_slot_cap: int | None = None

    # Time-of-day ticket pricing. Shifts demand from peak to off-peak
    # windows via price elasticity. See config.PRICE_WINDOWS.
    dynamic_pricing: bool = False

    # Overrides the last-entry time (minutes after museum open).
    # None = use default (555 minutes = 17:30).
    last_entry_override: int | None = None

    # Caps tour group size at the specified number of people.
    # None = no cap (default group size is 30).
    tour_group_cap: int | None = None

    # Places a temporary high-importance exhibit in the specified room
    # (e.g., "A20"). Creates a new attractor to redistribute demand.
    temporary_exhibit_room: str | None = None

    # Bans photography in bottleneck rooms (A11, A12), reducing dwell
    # time by ~40% and increasing throughput.
    photography_ban: bool = False

    # Audio guide that steers visitors away from congested rooms in
    # real time. Adoption rate is AUDIO_GUIDE_ADOPTION (60%).
    adaptive_audio_guide: bool = False

    # When Botticelli rooms exceed a density threshold, activates
    # contextual content in the preceding room (A10) as a buffer.
    queue_to_content: bool = False

    # Shows predicted crowd levels at booking time, shifting demand
    # to less congested entry slots.
    crowd_forecast: bool = False

    # Scheduled docent talks in underused rooms during peak hours,
    # creating temporary demand attractors.
    micro_events: bool = False

    # =========================================================================
    # Group A: Room attribute modifications
    # =========================================================================
    # These interventions change room-level properties (importance,
    # magnetism) to reshape how visitors perceive and use specific rooms.

    # Daily featured underused room announced on social media.
    # Creates FOMO and redirects attention. Value is the room ID (e.g., "A22").
    room_nobody_knows: str | None = None

    # A restorer works visibly in the specified room, creating dwell
    # time where it is wanted. Value is the room ID.
    conservation_theater: str | None = None

    # Adds benches in underused rooms (increases dwell) and removes them
    # from bottleneck rooms (decreases dwell). See config.SEATING_BOOST_ROOMS.
    seating_strategy: bool = False

    # Period-appropriate music in underused rooms makes them destinations;
    # silence in bottleneck rooms keeps visitors reverent and moving.
    sound_design: bool = False

    # Designates a room as the weekly "Instagram spot," leveraging
    # social media to redirect the selfie crowd. Value is the room ID.
    social_media_spot: str | None = None

    # Temporary installations in the Uffizi courtyard, absorbing a
    # fraction of visitors outdoors to reduce indoor pressure.
    courtyard_programming: bool = False

    # =========================================================================
    # Group B: Visitor segments
    # =========================================================================
    # These interventions target specific visitor sub-populations or
    # alter the composition of the visitor mix.

    # Replaces the two-type model with a six-segment visitor typology.
    # [not used in the curated 30; reserved for future extension]
    six_segment_model: bool = False

    # Fraction of visitors persuaded to skip Botticelli/Leonardo.
    # 0.0 = nobody skips; 0.20 = 20% skip the famous rooms.
    skip_the_famous: float = 0.0

    # Scavenger-hunt for families in underused rooms.
    # Redirects family visitors away from congested galleries.
    children_treasure_hunt: bool = False

    # =========================================================================
    # Group C: Arrival/session modifications
    # =========================================================================
    # These interventions alter when and how visitors enter the museum.

    # Special low-capacity session in an empty museum (e.g., dawn session).
    empty_museum_session: bool = False

    # Premium evening session after normal closing hours.
    night_museum_session: bool = False

    # Walk-in visitors (no reservation) pay more, deterring a fraction.
    walk_in_premium: bool = False

    # Per-person surcharge for tour group members (EUR). Internalizes
    # the congestion externality of groups. 0.0 = no surcharge.
    per_person_group_surcharge: float = 0.0

    # =========================================================================
    # Group D: Movement model
    # =========================================================================
    # These interventions change how visitors move through the museum.

    # Alternative chronological route designed by Giorgio Vasari's
    # architectural logic. See config.VASARI_ROUTE.
    vasari_narration: bool = False

    # Adjusts outdoor room attractiveness based on weather.
    # None = normal; "rain" = penalize outdoor rooms; "sun" = boost them.
    weather_routing: str | None = None

    # A fraction of coupled visitors split into two independent agents
    # with different starting routes, spreading demand.
    group_splitting: bool = False

    # Assigns a Botticelli time window at booking, spreading visits
    # over the day instead of everyone arriving at once.
    smart_defaults: bool = False

    # =========================================================================
    # Group E: Conditional access
    # =========================================================================
    # These interventions control access to specific high-demand rooms.

    # Decoupled time-window access to Botticelli rooms. Visitors are
    # assigned a specific window, preventing the all-at-once rush.
    decoupled_botticelli_gating: bool = False

    # Reciprocal access: visitors who also visit a partner venue
    # (e.g., Palazzo Pitti) get priority Botticelli access. Value is
    # the number of reciprocal slots (0 = disabled).
    reciprocal_access: int = 0

    # =========================================================================
    # Group F: Revenue model
    # =========================================================================

    # When True, the simulator tracks ticket revenue using the price
    # schedule in config.TICKET_PRICE_SCHEDULE. Required for portfolio
    # optimization with a revenue objective.
    revenue_model: bool = False

    # =========================================================================
    # Wave 2: Physical flow design
    # =========================================================================
    # Structural interventions that change the physical movement
    # options within the museum.

    # One-way microloops in bottleneck corridors to prevent bidirectional
    # congestion.
    oneway_microloops: bool = False

    # Express corridor bypass that skips magnet rooms for visitors who
    # have already seen them.
    corridor_express: bool = False

    # Activates content in multiple rooms (A9, A10, A13) when Botticelli
    # is congested, creating an absorption cascade rather than a single
    # buffer point.
    multi_room_buffer: bool = False

    # Decoupled time-window gating for ALL magnet rooms, not just
    # Botticelli. See config.ALL_MAGNET_WINDOW_ROOMS.
    magnet_room_windows: bool = False

    # West wing opens 75 minutes after east wing, spreading the
    # arrival wave across the museum over time.
    staggered_wing_opening: bool = False

    # =========================================================================
    # Wave 2: Time architecture
    # =========================================================================
    # Interventions that reshape the temporal distribution of visits.

    # Free entry during the lunch hour (12:30-14:00) for locals,
    # filling the midday dip with short visits.
    lunch_free_entry: bool = False

    # Reduced-price entry in the final 90 minutes for local residents,
    # filling otherwise-empty evening capacity.
    last_hour_locals: bool = False

    # Adjusts museum hours by season. None = default (08:15-18:30).
    # "summer" = extended hours; "winter" = shorter hours.
    seasonal_hours: str | None = None

    # Two-visit ticket: visitors can split their visit across morning
    # and afternoon, reducing peak occupancy.
    two_visit_ticket: bool = False

    # 30-minute closure at 13:00 ("breathing pause"), allowing rooms
    # to empty before the afternoon wave.
    breathing_pause: bool = False

    # =========================================================================
    # Wave 2: Pricing innovation
    # =========================================================================
    # Pricing mechanisms beyond simple time-of-day differentiation.

    # Annual pass for local residents (EUR equivalent to ~2 visits).
    # Adds 200 off-peak short-visit locals per day.
    resident_annual_pass: bool = False

    # Tour group booking slots sold via auction (highest bidder gets
    # preferred time slots).
    group_booking_auction: bool = False

    # Combined day pass covering Uffizi + partner venues, reducing
    # afternoon Uffizi demand as visitors move to other sites.
    cross_venue_day_pass: bool = False

    # Visitors who leave early get a partial refund, incentivizing
    # Type B visitors to shorten their visits.
    time_spent_rebate: bool = False

    # Real-time occupancy-based pricing: entry price fluctuates with
    # current museum occupancy.
    realtime_occupancy_pricing: bool = False

    # =========================================================================
    # Wave 2: Visitor behavior nudges
    # =========================================================================
    # Low-cost behavioral interventions that nudge visitor choices
    # without restricting freedom.

    # Shows visitors a "museum completion" progress bar, nudging
    # completionists toward undervisited rooms.
    progress_bar: bool = False

    # Gamification: badges and achievements for visiting underused rooms.
    # Target rooms defined in config.ACHIEVEMENT_UNDERUSED_ROOMS.
    achievement_system: bool = False

    # A fraction of visitors start from the west wing (A24) instead
    # of the east, creating counterflow that spreads demand.
    counterflow_incentive: bool = False

    # No tour groups allowed before 11:00, protecting the morning
    # window for crowd-sensitive (Type A) visitors.
    quiet_hours: bool = False

    # Professional lighting in designated rooms for photography,
    # pulling the selfie crowd away from Botticelli.
    designated_photo_spots: bool = False

    # =========================================================================
    # Wave 2: Content and experience
    # =========================================================================
    # Interventions that enrich the visitor experience in specific rooms,
    # making underused galleries more attractive.

    # First-person audio narration in selected rooms, boosting their
    # importance and dwell time. See config.PAINTING_TALKS_ROOMS.
    painting_talks: bool = False

    # Panels linking underused rooms to famous works elsewhere in the
    # museum, making them "satellites" of the magnets.
    comparative_displays: bool = False

    # Touchable replicas of famous works in the specified room.
    # Particularly attractive to families and visually impaired visitors.
    tactile_art_station: str | None = None  # room ID

    # Weekly themed programming that boosts importance of rooms
    # matching the theme. Value is the theme name (e.g., "caravaggio").
    themed_weeks: str | None = None  # theme name

    # A living artist works in the specified room, creating a unique
    # draw to an otherwise underused gallery.
    artist_in_residence: str | None = None  # room ID

    # =========================================================================
    # Wave 2: Data-driven dynamic interventions
    # =========================================================================
    # Interventions that use real-time data to dynamically adjust
    # room attractiveness or visitor routing.

    # Automatically boosts importance of rooms below a density threshold,
    # spreading visitors toward empty galleries.
    realtime_room_boost: bool = False

    # Route suggestions based on forecasted (not just current) density.
    predictive_routing: bool = False

    # "Others are enjoying room X" nudge messages based on real-time
    # occupancy data, leveraging social proof to steer visitors.
    social_proof_nudge: bool = False

    # Real-time route adaptation: the suggested trail updates as
    # the visitor moves through the museum, based on current density.
    adaptive_trail: bool = False

    # Adjusts room lighting based on density: dims overcrowded rooms
    # (subtle "move along" signal), brightens empty rooms.
    crowd_responsive_lighting: bool = False

    # =========================================================================
    # Factory method
    # =========================================================================

    @classmethod
    def from_kwargs(cls, **kwargs) -> InterventionConfig:
        """Construct from old-style keyword arguments (backward compatibility).

        Accepts any superset of kwargs (e.g., a dict that also contains
        simulator parameters like ``daily_total`` or ``seed``) and
        silently discards keys that do not match a field on this class.
        This makes it safe to call with ``**kwargs`` from functions that
        mix intervention and simulator parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments. Only those matching field names on
            InterventionConfig are used; all others are silently dropped.

        Returns
        -------
        InterventionConfig
            A new frozen config instance with the specified fields set.
        """

        # Build the set of valid field names from the dataclass definition.
        valid = {f.name for f in fields(cls)}
        # Filter kwargs to only those that correspond to actual fields.
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)
