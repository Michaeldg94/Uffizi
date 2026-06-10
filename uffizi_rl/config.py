"""Global configuration for the Uffizi RL project.

All constants, seeded defaults, and modeling assumptions live here.
Every parameter is annotated with its source: a published reference,
an official public source, or an explicit [assumption] label.

References
----------
[A22]  Attanasio et al. (2022). "Visitors flow management at Uffizi Gallery
       in Florence, Italy." Information Technology & Tourism, 24(3), 409-434.
       DOI: 10.1007/s40558-022-00231-y
[C21]  Centorrino et al. (2021). "Managing crowded museums: Visitors flow
       measurement, analysis, modeling, and optimization." Journal of
       Computational Science, 53, 101357.
[VL83] Veron & Levasseur (1983). "Ethnographie de l'exposition." BPI,
       Centre Georges Pompidou. Visitor typology: Ant/Butterfly/Fish/Grasshopper.
[UFF]  Official Uffizi website and visitor FAQ (uffizi.it, visituffizi.org).
[MAP]  Official 2023 Uffizi Gallery floor plan (uffizi.it).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

# =============================================================================
# Phase 0 non-negotiables
# =============================================================================

# Two clocks: museum-open duration vs bookable-entry duration.
# The Uffizi opens at 08:15 and closes at 18:30 [UFF].
# Ticket office / last entry at 17:30, one hour before closing [UFF].
MUSEUM_OPEN_MINUTES = 615  # 08:15 -> 18:30 = 10h15m [UFF]
# Grace period after closing for a visitor to finish their current room and
# walk to the exit. A visitor's budget is capped at (closing - entry + this),
# so nobody is stranded inside and late arrivals do shorter visits. [assumption]
CLOSING_GRACE_MINUTES = 30
LAST_ENTRY_MINUTES = 555   # 08:15 -> 17:30 = 9h15m  [UFF]

# [A22] Table 2: 37 time slots of 15 minutes each, from 08:15 to 17:30.
ENTRY_SLOT_MINUTES = 15
N_ENTRY_SLOTS = 37
TIME_STEP_MINUTES = 1

# "The law dictates that given the number of entrances/exits at the Uffizi,
#  no more than 900 people can be inside the museum at any time." [UFF]
# Note: [A22] uses Ct=1000 in their prescriptive model; we use the binding
# legal limit. This is a hard constraint, not a soft penalty.
MAX_MUSEUM_CAPACITY = 900

# ~5M visitors annually in 2023 [UFF]; ~2M in 2018 [A22].
# Peak free-access days reached ~8,000 visitors [A22, Section 3].
# Peak summer season exceeds 8,000/day [UFF].
# Normal-day and peak-day values are modeling parameters calibrated so that
# the simulator produces realistic occupancy pressure under the 900 cap.
DAILY_VISITORS_NORMAL = 5000   # [assumption] typical non-peak weekday
DAILY_VISITORS_PEAK = 12000    # [assumption] high-season Saturday

DEFAULT_SEED = 42

# RL training budgets. Quick defaults for local/smoke execution;
# full values for publication-scale experiments.
TABULAR_EPISODES_DEFAULT = 2500
TABULAR_EPISODES_FULL = 50000
PPO_TIMESTEPS_DEFAULT = 15000
PPO_TIMESTEPS_FULL = 1_000_000
ABLATION_TIMESTEPS_DEFAULT = 10000

# Simulation calibration defaults.
SIM_DAYS_CALIBRATION_DEFAULT = 25
SIM_DAYS_CALIBRATION_FULL = 100

# Observation-space constants.
# Sentinel value for unobserved rooms when visibility masking is active.
UNKNOWN_DENSITY_SENTINEL = -1.0

# Corridor junctions where real-time crowd displays ("kiosks") provide
# full museum density information to the agent. Placed at decision points
# where visitors choose between routes. [assumption]
KIOSK_ROOMS = {"A2", "A15", "A24", "D9"}

# -----------------------------------------------------------------------------
# Visitor type parameters
# -----------------------------------------------------------------------------
# Two-type model inspired by the Veron & Levasseur (1983) taxonomy [VL83].
# Type A ("art lovers") map roughly to VL83 "Butterfly" visitors: heterogeneous
# preferences, crowd-sensitive, willing to deviate from the default path.
# Type B ("checkbox tourists") map roughly to VL83 "Ant" visitors: concentrated
# preferences on canonical masterpieces, route-following, crowd-insensitive.
# The 30/70 split is a modeling assumption; no published source provides an
# exact ratio for the Uffizi. Sensitivity analysis in Phase 7 sweeps this
# fraction from 0% to 100%.
TYPE_A_FRACTION_DEFAULT = 0.10   # Art lovers: spend hours, visit minor galleries
TYPE_B_FRACTION_DEFAULT = 0.90   # Aggregate of Instagram + Standard, see splits below
# Within the Type B aggregate, split between Instagram tourists (selfie-driven,
# 30-60 min visits) and Standard tourists (Fast Itinerary, 90-120 min). The
# Instagram tier is the majority of real Uffizi visitors.
INSTAGRAM_FRACTION = 0.60        # 60% of all visitors
STANDARD_FRACTION  = 0.30        # 30% of all visitors
# (Art Lover takes the remaining 10%, from TYPE_A_FRACTION_DEFAULT.)
#
# Magnet-room sets per tier. The simulator's dwell-feedback logic and the
# personal-importance vector both key off these. Instagram tourists stop
# for selfies at the four masterpiece rooms plus the panoramic terrace
# (cafe + photo spot, where their visit usually ends before Lanzi). Standard
# tourists do the same plus the Tribune and the Caravaggio room.
INSTAGRAM_MAGNET_ROOMS = {"A11", "A12", "A35", "A38", "PANORAMIC_TERRACE"}
STANDARD_MAGNET_ROOMS  = {"A11", "A12", "A16", "A35", "A38", "E4"}

# Crowd sensitivity in the reward function: r_art = importance / (1 + alpha * dens).
# Higher alpha means the visitor loses more utility from crowding.
# Type A is highly crowd-averse; Type B is nearly crowd-indifferent. [assumption]
TYPE_A_CROWD_ALPHA = 6.0   # [assumption] strong crowd penalty
TYPE_B_CROWD_ALPHA = 0.5   # [assumption] weak crowd penalty

# Route-following tendency: probability weight on the "next" room in the
# recommended itinerary when choosing where to move. [assumption]
# Type A deviates more freely; Type B follows the guidebook path.
TYPE_A_ROUTE_BIAS = 0.55   # [assumption]
TYPE_B_ROUTE_BIAS = 0.88   # [assumption]

# NPC dwell-time parameters.
# Base expected dwell in a room with magnetism=1.0. Actual expected dwell
# is base_dwell * room_magnetism * profile.dwell_multiplier. [assumption]
BASE_DWELL_MINUTES = 3.0
# Dwell multiplier by type. Type B visitors spend less time per room
# (quick photo, move on); Type A lingers when uncrowded. [assumption]
TYPE_A_DWELL_MULTIPLIER = 1.0   # [assumption]
TYPE_B_DWELL_MULTIPLIER = 0.3   # [assumption]
# Additive dwell floor for art lovers. Expected dwell becomes
# base_dwell * (room_magnetism * dwell_multiplier + dwell_floor), so an
# art lover lingers even in a low-magnetism minor room (e.g. A28) where a
# standard tourist walks straight through. The gap between the two is
# therefore widest in the quiet galleries and narrowest at the famous
# rooms, where the slot gate caps everyone at the same dwell. [user testimony]
# Kept modest so the art lover can still traverse the whole museum (both
# floors) within an 8-hour budget rather than stalling mid-circuit.
ART_LOVER_DWELL_FLOOR = 0.4

# Probability per step that an NPC reverses direction along the
# recommended route. Prevents perfectly deterministic flow. [assumption]
TYPE_A_BACKTRACK_PROBABILITY = 0.05  # [assumption]
TYPE_B_BACKTRACK_PROBABILITY = 0.02  # [assumption] Type B rarely backtracks

# Anti-crowd bonus: extra weight toward less-crowded neighbors.
# Type A actively avoids crowds; Type B ignores crowd signals. [assumption]
TYPE_A_ANTI_CROWD_BONUS = 1.0   # [assumption]
TYPE_B_ANTI_CROWD_BONUS = 0.0   # [assumption]

# Asymmetric externality: Type A perceives Type B visitors as this factor
# more disruptive than fellow Type A visitors. Reflects the idea that
# large tour groups with cameras and guides impose disproportionate
# negative externalities on contemplative visitors. [assumption]
TYPE_A_CROSS_TYPE_EXTERNALITY = 1.5   # [assumption]

# Type B importance vector: background importance for non-magnet rooms.
# Magnet rooms get 9.5; everything else gets this value. [assumption]
TYPE_B_BACKGROUND_IMPORTANCE = 0.7   # [assumption]
TYPE_B_MAGNET_IMPORTANCE = 9.5       # [assumption]

# -----------------------------------------------------------------------------
# Intervention parameters
# -----------------------------------------------------------------------------

# Dynamic pricing: price multiplier per time window. Visitors have a
# willingness-to-pay drawn from a log-normal distribution; when the
# price exceeds their willingness, they shift to a cheaper slot or
# don't visit. This reshapes the arrival curve. [assumption]
PRICE_WINDOWS = [
    # (start_minute, end_minute, price_multiplier)
    (0, 90, 0.50),       # 08:15-09:45 "dawn" discount
    (90, 165, 0.75),     # 09:45-11:00 "early morning" discount
    (165, 285, 1.00),    # 11:00-13:00 standard rate
    (285, 405, 1.25),    # 13:00-15:00 "peak" surcharge
    (405, 555, 0.85),    # 15:00-17:30 "afternoon" discount
]
# Mean willingness-to-pay (in price-multiplier units). A visitor with
# WTP < price_multiplier for their slot will shift to a cheaper slot. [assumption]
MEAN_WILLINGNESS_TO_PAY = 1.0
WTP_SPREAD = 0.4  # log-normal sigma for WTP distribution [assumption]

# Tour group parameters. A fraction of visitors arrive as synchronized
# groups that move together (correlated movement, same checklist).
# Groups create corridor "shock waves." [assumption]
TOUR_GROUP_FRACTION = 0.15    # [assumption] 15% of visitors are in tour groups
TOUR_GROUP_SIZE_DEFAULT = 15  # [assumption] typical guided tour group
TOUR_GROUP_SIZE_CAPPED = 15   # [assumption] group size under the cap intervention

# Storage rotation: temporary high-importance exhibit in an underused room.
# Simulates placing a headline work from storage in a minor gallery to
# create a new attractor that redistributes demand. [assumption]
TEMPORARY_EXHIBIT_ROOMS = ["A20", "A21", "A30", "A33"]  # underused rooms [assumption]
TEMPORARY_EXHIBIT_IMPORTANCE_BOOST = 7.0   # [assumption] brings room to ~10

# Extended hours / premium evening session parameters.
# A separate evening window after normal closing with limited admission. [assumption]
EVENING_SESSION_START = 630    # 18:45 (15 min after normal close)
EVENING_SESSION_DURATION = 120  # 2 hours
EVENING_SESSION_CAPACITY = 200  # max visitors in evening session [assumption]

# Photography ban: banning photography in bottleneck rooms reduces dwell
# time (~30-40% reduction in magnetism). Visitors look, absorb, and move
# on instead of posing, waiting for clear shots, and taking multiple
# photos. Increases throughput with zero infrastructure cost. [assumption]
PHOTO_BAN_ROOMS = {"A11", "A12"}     # Botticelli rooms [assumption]
PHOTO_BAN_MAGNETISM_FACTOR = 0.6     # 40% reduction in dwell time [assumption]

# Adaptive audio guide: fraction of visitors who carry an audio guide
# that adjusts recommendations based on real-time crowd state. When a
# nearby room is crowded, the guide steers toward quieter alternatives.
# Uses existing audio guide infrastructure. [assumption]
AUDIO_GUIDE_ADOPTION = 0.60          # 60% of visitors have audio guide [assumption]
AUDIO_GUIDE_CROWD_SENSITIVITY = 0.8  # weight on crowd-avoidance signal [assumption]

# Queue-to-content: when Botticelli rooms exceed this density threshold,
# the room before (A10, Pollaiolo) activates contextual content about
# Botticelli, increasing its magnetism and creating a natural absorption
# buffer. Visitors perceive enrichment, not waiting. [assumption]
QUEUE_TO_CONTENT_DENSITY_THRESHOLD = 0.7   # [assumption]
QUEUE_TO_CONTENT_BUFFER_ROOM = "A10"       # room before Botticelli chain
QUEUE_TO_CONTENT_MAGNETISM_BOOST = 3.0     # [assumption] makes A10 a destination

# Crowd forecast at booking: visitors see predicted crowd levels for
# each entry slot and partially shift to less crowded times. The
# forecast reshapes the arrival distribution by dampening peak-hour
# demand. [assumption]
CROWD_FORECAST_RESPONSE_RATE = 0.3   # fraction of visitors who shift slots [assumption]

# Micro-events: scheduled docent talks in underused rooms during peak
# hours (10:00-13:00) create temporary demand attractors. [assumption]
MICRO_EVENT_ROOMS = ["A18", "A27", "A39"]  # Mantegna, Perugino, Niobe [assumption]
MICRO_EVENT_PEAK_WINDOW = (105, 285)       # minutes 105-285 = 10:00-13:00
MICRO_EVENT_IMPORTANCE_BOOST = 4.0         # [assumption]

# Room nobody knows: daily featured underused room announced on social media.
# Creates FOMO and redirects attention to overlooked galleries. [assumption]
ROOM_NOBODY_KNOWS_CANDIDATES = ["A15", "A17", "A19", "A20", "A21", "A22",
                                 "A29", "A30", "A32", "A33"]  # [assumption]
ROOM_NOBODY_KNOWS_IMPORTANCE_BOOST = 5.0   # [assumption]

# Predictive routing strength. Treated as the weight of the projected-
# density anti-crowd term, in the same units as the visitor's regular
# anti_crowd_bonus. 0.5 makes the predictive penalty half as strong as
# baseline Type A crowd avoidance before per-visitor scaling, i.e.
# comparable to the regular crowd term. [assumption]
PREDICTIVE_ROUTING_ALPHA = 0.5

# Conservation theater: a restorer works visibly in an underused room.
# Fascination with the craft creates dwell time exactly where you want it.
CONSERVATION_THEATER_MAGNETISM_BOOST = 4.0  # [assumption]

# Seating strategy: place benches in underused rooms (increase magnetism),
# remove seating from bottleneck rooms (decrease magnetism). Sculpts flow
# with furniture. [assumption]
SEATING_BOOST_ROOMS = ["A5", "A7", "A18", "A27", "A39"]   # underused galleries [assumption]
SEATING_BOOST_FACTOR = 1.5    # 50% more dwell time where benches are added [assumption]
SEATING_REMOVE_ROOMS = {"A11", "A12"}   # Botticelli rooms [assumption]
SEATING_REMOVE_FACTOR = 0.7   # 30% less dwell time without seating [assumption]

# Sound design: period-appropriate music in underused rooms makes them
# destinations. Silence in bottleneck rooms keeps visitors reverent and
# moving. [assumption]
SOUND_DESIGN_ROOMS = ["A5", "A7", "A9", "A18", "A28"]  # rooms with music [assumption]
SOUND_DESIGN_IMPORTANCE_BOOST = 2.0   # [assumption]

# Social media spot: weekly featured "Instagram room." The social media
# crowd follows the designated spot, steering the herd with a hashtag.
SOCIAL_MEDIA_IMPORTANCE_BOOST = 4.0   # [assumption]

# Courtyard programming: temporary installations, performances, or
# exhibitions in the Uffizi courtyard. Every visitor who lingers outdoors
# for 20 minutes is 20 minutes of reduced indoor pressure. [assumption]
COURTYARD_ABSORPTION_FRACTION = 0.08  # 8% of visitors absorbed outdoors [assumption]

# Walk-in premium: walk-in visitors (no reservation) pay more. This
# deters a fraction of walk-ins, effectively reducing unplanned arrivals.
WALK_IN_FRACTION = 0.15     # 15% of visitors are walk-ins [assumption]
WALK_IN_DETERRENCE = 0.40   # 40% of walk-ins deterred by premium [assumption]

# Per-person group surcharge: internalizes the congestion externality
# imposed by tour groups. Higher surcharge reduces group fraction.
# Elasticity: fraction_reduction = surcharge / reference_price. [assumption]
GROUP_SURCHARGE_REFERENCE_PRICE = 25.0  # base ticket price [assumption]

# Vasari narration: alternative chronological route through the museum.
# Giorgio Vasari, the architect, would have visitors follow art history
# chronologically, not topologically. This distributes flow differently
# from the standard route. [assumption]
VASARI_ROUTE = [
    "ENTRY", "GRANDUCAL_STAIRCASE", "A1", "A2", "A3", "A4", "A5", "A6",  # medieval
    "A7", "A4", "A8", "A9",  # early renaissance (loop back through A4)
    "A10", "A11", "A12", "A13",  # Botticelli chain
    "A15", "A16",  # Tribune
    "A17", "A18", "A19", "A20", "A21",  # regional schools (forced chain)
    "A23", "A24",  # corridors
    "A25", "A26", "A27", "A28", "A29",  # U-loop out
    "A31", "A32", "A33", "A34",  # U-loop return
    "A24", "A35", "A36", "A38",  # Leonardo, Raphael
    "A24", "A42", "A24",  # Dürer
    "PANORAMIC_TERRACE", "LANZI_STAIRCASE",  # descend
    "D1", "D6", "D13", "D14", "D15",  # D corridor
    "D19", "D23", "D25", "D26", "D28",  # D key rooms
    "E4", "E5", "E7",  # Caravaggio
    "EXIT",
]
VASARI_ADOPTION = 0.25  # fraction of visitors who use the Vasari route [assumption]

# Weather routing: outdoor/view rooms are weather-dependent.
# Rain: outdoor rooms (terrace) less attractive. Sun: boost them.
OUTDOOR_ROOMS = {"A14", "A23", "PANORAMIC_TERRACE"}  # rooms with outdoor elements [assumption]
WEATHER_RAIN_IMPORTANCE_PENALTY = 3.0   # [assumption]
WEATHER_SUN_IMPORTANCE_BONUS = 2.0      # [assumption]

# Group splitting: fraction of coupled visitors who split into two
# half-budget agents with different starting routes. [assumption]
GROUP_SPLITTING_FRACTION = 0.10  # 10% of visitors are couples who split [assumption]

# Smart defaults: Botticelli window duration assigned at booking. [assumption]
SMART_DEFAULT_BOTTICELLI_WINDOW = 30  # minutes per visitor window [assumption]

# The four canonical masterpiece rooms of the Uffizi. These are the rooms
# that create the peak-hour congestion externality and that all Pigovian
# interventions must target. A11/A12 hold the two Botticellis (Primavera +
# Nascita di Venere); A35 holds Leonardo (Annunciazione, Adorazione dei
# Magi, Battesimo); A38 holds Raphael and Michelangelo (Doni Tondo, etc).
MASTERPIECE_ROOMS = frozenset({"A11", "A12", "A35", "A38"})

# RAMA (Reservation-Anchored Masterpiece Access) parameters.
#
# Booking system: each masterpiece room offers timed entry slots of
# RAMA_SLOT_DURATION_MIN minutes; each slot accommodates up to
# RAMA_SLOT_CAPACITY visitors. When a visitor "books" (= is created at
# simulator init under RAMA), they self-select their three slots
# (Botticelli, Leonardo, Raphael) based on their type. Art lovers
# pick widely spaced slots; Instagram tourists pick back-to-back.
# If their preferred slot is full, they take the next available.
# Visitors with no Botticelli/Leonardo/Raphael slot do not visit
# those rooms at all - their personal importance for those rooms
# is suppressed by the gating logic. Inside the slot, a hard
# RAMA_DWELL_CAP_MIN cap forces turnover.
RAMA_SLOT_DURATION_MIN = 10      # each booked slot is 10 minutes
RAMA_DWELL_CAP_MIN = 10          # Match slot duration so dwell target
                                  # (10 min at Leo/Raph) actually fires
                                  # before the cap. Old 9-min value
                                  # truncated dwell to 9 min, creating
                                  # a 1-min empty gap each slot.
                                  # slot starts, so slot N+1 enters a
                                  # cleared room. Without the gap,
                                  # slot N's stragglers overlap with
                                  # slot N+1's arrivals at the gate.

# Per-room slot capacity. Each masterpiece room has a different size,
# so the booking slot cap matches the room's physical capacity. A11
# and A12 share a single Botticelli ledger because they're a forced
# chain (visitors traverse both in one go); the cap reflects A11's
# Spring room.
RAMA_SLOT_CAPACITY_BY_ROOM = {
    # Botticelli zone (modeled as A11 alone with full 10-min dwell;
    # A12 is a 1-min walk-through). 55 cap matches room cap.
    "A11": 55,
    "A35": 52,   # Leonardo (single room, 10-min dwell)
    "A38": 53,   # Raphael / Michelangelo
}

# Per-segment preferred booking window relative to entry time. A
# visitor of segment S tries to book their Botticelli slot in the
# range (entry + lo, entry + hi). Wider spacing = more lingering
# upstream. Art lovers pick later slots (long museum day);
# Instagram tourists pick early slots (rush in, get the selfie).
RAMA_BOOKING_PREFERENCES = {
    # Minutes-after-anchor buffers. "bott" is anchored on entry time;
    # "leo" and "raph" on the END of the previous masterpiece slot.
    # Each segment's schedule reflects its visit philosophy:
    #
    # - Instagram tourists rush in, do all three masterpieces back-to-
    #   back (slots only minutes apart), and leave. Whole visit under
    #   an hour.
    # - Standard tourists pace moderately - 30-60 min between slots,
    #   typical 2-3 hour visit.
    # - Art lovers spread slots across the whole day. They enter early
    #   and leave just before closing, hours between each slot.
    #
    # The slot-guarantee mechanism (visitor is placed in the room when
    # their slot opens) makes tight Instagram schedules work even
    # though Botticelli to Leonardo is physically far.
    # All buffers are floored by the walking distance:
    #   ENTRY -> A11 (Bot)  = 6 edges
    #   A12 -> A35 (Leo)    = 5 edges  (corridor shortcut)
    #   A35 -> A38 (Raph)   = 2 edges
    "art_lover": {"bott": (60, 240), "leo": (60, 240), "raph": (30, 120)},
    "standard":  {"bott": (15,  60), "leo": (10,  40), "raph": ( 5,  25)},
    # IG: minimal back-to-back. After Botticelli ends, walking to
    # Leonardo through the corridor (fast-transit) is essentially
    # instant; their next slot fires immediately. Same for Leo->Raph.
    # This compresses the masterpiece tour into ~30 min so it fits
    # any 60-min visitor budget.
    "instagram": {"bott": ( 1,  10), "leo": ( 0,   5), "raph": ( 0,   5)},
}

# RAMA pre-day booking model.
#
# Each visitor type draws their preferred slot spacings from a uniform
# distribution. Spacing = minutes from end of previous slot (or from
# entry minute for the first slot). Wider distributions reflect the
# slower museum pace of art lovers vs the back-to-back blitz of IG.
RAMA_SPACING_DISTRIBUTIONS = {
    # (min, max) minutes. The three masterpiece slots are spaced to match the
    # physical distance a visitor walks between them, NOT compressed:
    #   entry_to_first: entry to the Botticelli slot (the rooms before A11).
    #   bott_to_leo:    Botticelli (A12) to Leonardo (A35). The LONG leg. The
    #     thorough profiles walk the Tribune, the 15th-century chain and the
    #     U-loop (~23 rooms) in between, so an art lover needs ~2.5 hours
    #     (~10 min/room) and a standard tourist ~1 hour (~5 min/room).
    #     Instagram tourists take the corridor shortcut and cross in minutes.
    #   leo_to_raph:    Leonardo (A35) to Raphael (A38). SHORT: the rooms are
    #     physically adjacent (A35 -> A36 -> A38), so a few minutes for all.
    # entry_to_first and bott_to_leo stay wide, which is what spreads the
    # per-room masterpiece density across the whole day.
    "art_lover": {
        "entry_to_first":  (30, 120),
        "bott_to_leo":     (90, 165),
        "leo_to_raph":     (10,  25),
        "arrive_early":    ( 5,  15),
    },
    "standard": {
        "entry_to_first":  (10,  60),
        "bott_to_leo":     (15,  90),
        "leo_to_raph":     (10,  20),
        "arrive_early":    ( 2,  10),
    },
    "instagram": {
        "entry_to_first":  ( 1,  15),
        "bott_to_leo":     ( 5,  30),
        "leo_to_raph":     ( 5,  15),
        "arrive_early":    ( 1,   5),
    },
}

# Probability that a given visitor wants to see each masterpiece. Almost
# everyone wants all three (it's why they came to the Uffizi). The
# small skip fraction reflects rare cases: already saw it, niche
# interest, etc.
RAMA_P_WANT_MASTERPIECE = {
    "A11": 0.99,   # Botticelli (Spring + Venus)
    "A35": 0.99,   # Leonardo
    "A38": 0.99,   # Raphael + Michelangelo (same room, same demand)
}

# Walk-in fraction under RAMA: visitors who didn't pre-book any
# masterpiece slot but still buy a general entrance ticket. They know
# at purchase time that they won't see the masterpieces (sold out), so
# they just walk the rest of the museum.
RAMA_WALK_IN_FRACTION = 0.00  # NO walk-ins under RAMA. Visitors who couldn't
                                # book masterpiece slots still book general entry
                                # online and get the discounted ticket.

# Modular RAMA pricing. The Uffizi sells RAMA tickets a la carte:
# a base general-entry ticket plus per-masterpiece booking fees.
# Keeps culture accessible to families and budget travellers while
# recapturing revenue from full-RAMA bookers without crossing the
# political €40+ threshold that would make museum visits feel
# exclusionary.
#   - General entry, no masterpieces: EUR 15
#   - + 1 masterpiece slot:           +EUR 5  -> EUR 20
#   - + 2 masterpiece slots:          +EUR 10 -> EUR 25
#   - + 3 masterpiece slots:          +EUR 15 -> EUR 30
RAMA_BASE_GENERAL_PRICE = 15.0
RAMA_MASTERPIECE_ADDON = 5.0

# Tolerance (minutes) for shifting a preferred slot when the exact
# minute is sold out. The booking website will offer nearby slots
# within this window; outside it, the visitor abandons that masterpiece.
RAMA_BOOKING_TOLERANCE_MIN = 60

# Secondary-attractor curation. The intervention secondary_attractor_enrichment
# applies room-by-room sensory enrichment to a curated set of 14
# pre-masterpiece rooms (the "secondary stars"). Each room has a
# distinct combination of 2-4 sensory channels drawn from
# {visual, audio, tactile, olfactory, live}. No room engages all 5.
#
# Channel effect sizes on the room's importance and magnetism
# attributes. Sums across active channels per room are applied at
# simulator init time. Effect sizes are research-grounded ballparks
# (cite museum-studies literature on dwell-time interventions).
CHANNEL_EFFECTS = {
    "visual":    {"importance": 1.0, "magnetism": 0.5},  # screens, projections
    "audio":     {"importance": 0.5, "magnetism": 0.7},  # narration, music
    "tactile":   {"importance": 0.3, "magnetism": 0.7},  # touch stations
    "olfactory": {"importance": 0.2, "magnetism": 0.3},  # period scents
    "live":      {"importance": 1.5, "magnetism": 1.5},  # curator talks, restoration
}

# Curatorial table for the 14 secondary-attractor pre-masterpiece rooms.
# Each entry maps room_id -> dict with sensory channel keys whose values
# are short human-readable descriptions of the actual intervention. The
# CURATION_DOCUMENT.md file holds the full curatorial reasoning for
# each room; this dict is the executable summary the simulator reads.
# Order: pre-Botticelli, between-Botticelli-and-Leonardo, pre-Leonardo
# (west detour), pre-Raphael.
ROOM_CURATION = {
    # ---- Zone 1: Upstream of Botticelli ----
    "A4":  {  # Giotto & Cimabue: pivotal medieval-to-Renaissance moment
        "visual":    "Dual-screen Cimabue vs Giotto: animated comparison of perspective discovery",
        "audio":     "Notre-Dame School polyphony (Perotin) at low volume",
        "olfactory": "Beeswax and incense (14th-century church smell)",
        "live":      "Wednesday/Saturday Giotto-panel restoration showcase",
    },
    "A7":  {  # Lorenzo Monaco & Gentile (Adoration of the Magi)
        "visual": "Projection-mapping of Gentile's Adoration of the Magi animated",
        "audio":  "Late-medieval Burgundian court music",
    },
    "A8":  {  # Masaccio (perspective revolution) & Beato Angelico
        "visual":  "Interactive vanishing-point demonstration on the floor",
        "tactile": "Alberti perspective-grid device, visitors look through",
        "audio":   "Four-minute triggered narration on Masaccio's invention",
    },
    "A9":  {  # Piero della Francesca, Uccello, F. Lippi
        "visual":  "Animated Uccello battle scenes; projected Piero geometric studies",
        "tactile": "3D-printed Piero perfect-solid models",
    },
    "A10": {  # Pollaiolo: critical pre-Botticelli holdback
        "live":   "Hourly 10-min curator talk 'what Pollaiolo taught Botticelli'",
        "audio":  "Anatomy-study narration linking Pollaiolo's figures to Botticelli's",
        "visual": "Battle of the Nudes engraving animated, anatomy-study video",
    },
    # ---- Zone 2: Between Botticelli and Eastern Corridor ----
    "A13": {  # Hugo van der Goes Portinari Altarpiece: underrated masterpiece-level
        "visual":    "Flemish landscape side-wall projection: Bruges canals, snowy Flanders",
        "audio":     "Burgundian sacred music (Dufay's Nuper rosarum flores)",
        "olfactory": "Cold Northern pine resin",
        "live":      "Wednesday evening 'Flemish in Florence' talk",
    },
    "A14": {  # Uffizi Maps Terrace: natural pause + views
        "visual":    "Animated historical Florence maps overlaid on the actual view",
        "olfactory": "Tuscan herbs: rosemary, sage, citrus from gardens below",
    },
    "A18": {  # Mantegna, Bellini, Antonello: Venetian counterpoint
        "visual":  "Split-screen Florentine line vs Venetian color",
        "audio":   "Gabrieli Venetian brass polychoral (San Marco basilica effect)",
        "tactile": "Replica Mantegna engraving plate, copper texture",
    },
    "A22": {  # Boudoir of Miniatures: intimate curiosity room
        "visual":    "Magnified miniature projections at scale on side walls",
        "olfactory": "Bergamot and iris (Catherine de' Medici's perfume signature)",
        "tactile":   "Replica miniatures under glass for handheld examination",
    },
    # ---- Zone 3: Pre-Leonardo (west detour off Eastern Corridor) ----
    "A25": {  # Ghirlandaio: 'Michelangelo's teacher'
        "visual": "Multimedia on Ghirlandaio's workshop and the young Michelangelo apprenticing",
        "audio":  "Workshop ambient sounds: chisels, brushes, Florentine voices",
    },
    "A27": {  # Perugino: 'Raphael's teacher'
        "visual": "Multimedia on Perugino's clean style and young Raphael's apprenticeship",
        "audio":  "Umbrian folk music (Perugino was Umbrian, not Florentine)",
        "live":   "Weekly curator talk paired with A25 ('the teachers')",
    },
    "A28": {  # Filippino Lippi + Piero di Cosimo: Quattrocento finale
        "visual": "Comparison Filippino Lippi vs his father Filippo Lippi",
        "audio":  "Late-Quattrocento Florentine music",
    },
    "A30": {  # Doryphoros (Greek sculpture): tactile-first
        "tactile":   "Tactile replica visitors run hands over (Greek sculpture is meant to be felt)",
        "olfactory": "Olive, Mediterranean herbs (laurel, thyme)",
        "audio":     "Ancient Greek aulos music",
    },
    # ---- Zone 4: Pre-Raphael holdback ----
    "A36": {  # Hall of Ancient Inscriptions
        "tactile": "Replica inscription for charcoal-rubbing",
        "audio":   "Latin recitation: actor reading Cicero or Res Gestae",
        "live":    "Rotating translator-in-residence: Latin scholar on demand",
    },
}
assert len(ROOM_CURATION) == 14

# Bypass destinations for visitors denied entry by a gating intervention.
# Because the Botticelli and Leonardo chains are one-way, a visitor blocked
# from the gated room would otherwise have no forward edge and would be
# trapped in a corridor or pre-room. The bypass teleports the denied
# visitor to the room immediately AFTER the gated chain so they keep
# moving through the museum. They do not see the gated painting (the
# whole point of the gate) but they also do not stand still. Realistic
# interpretation: "the room is full / I am not in my window, so I walk on
# and go elsewhere (terrace, gift shop, next painting)."
MASTERPIECE_BYPASS = {
    "A11": "A13",  # Botticelli denied -> exit Botticelli chain at Hugo van der Goes
    "A12": "A13",
    "A35": "A24",  # Leonardo denied -> back to Eastern Corridor
    "A38": "A24",  # Raphael denied -> back to Eastern Corridor
}

# Revenue model: ticket prices [UFF, https://www.uffizi.it/en/pages/tickets-fares].
#
# Baseline Uffizi pricing:
#   Walk-in (day of entry): EUR 25
#   Pre-booked (online):    EUR 29   (= walk-in + EUR 4 booking fee)
#   Reduced (EU 18-25):     EUR 2
#   Free (under-18, disabled, etc.): EUR 0
#   Group surcharge (11+):  EUR 70 flat per group (Uffizi only)
#
# The baseline ticket schedule is a FLAT EUR 29 across the day (the
# Uffizi does not currently time-of-day differentiate). The schedule
# below is the SCHEDULE used by the dynamic_pricing intervention,
# which is a proposed counterfactual policy: charge less at dawn and
# late afternoon to spread demand. Values anchored around the EUR 29
# pre-booked base.
BASELINE_TICKET_PRICE = 29.0   # flat pre-booked price [UFF]
TICKET_PRICE_SCHEDULE = [
    # (start_minute, end_minute, price_eur)
    # Used only when dynamic_pricing intervention is on.
    (0, 90, 20.0),       # 08:15-09:45 dawn discount
    (90, 165, 25.0),     # 09:45-11:00 early morning
    (165, 285, 29.0),    # 11:00-13:00 standard (same as baseline)
    (285, 405, 35.0),    # 13:00-15:00 peak surcharge
    (405, 555, 22.0),    # 15:00-17:30 afternoon discount
]
WALK_IN_TICKET_PRICE = 25.0     # day-of price - CHEAPER than pre-booked (no booking fee) [UFF]
FREE_VISITOR_FRACTION = 0.15    # ~10% under-18 + ~5% disabled / school groups [UFF + assumption]
REDUCED_VISITOR_FRACTION = 0.10 # EU 18-25 citizens at EUR 2 [UFF]
REDUCED_TICKET_PRICE = 2.0      # [UFF]

# Pensioner segment (65+). Real Uffizi does NOT offer a senior discount
# (only "Friends of the Uffizi" annual pass). We model PENSIONER_FRACTION
# as a separate paying segment that the simulator can optionally
# discount via a new intervention. Default: pay full price (= adults).
PENSIONER_FRACTION = 0.15       # 15% of daily visitors are 65+ [assumption]
PENSIONER_REDUCED_PRICE = 20.0  # EUR 20 reduced rate if pensioner-pricing intervention active

# Adults remaining: 100% - 15% free - 10% student - 15% pensioner = 60%
# Of those adults, some fraction travel with kids (under 18) and are
# eligible for the family discount intervention.
PARENT_WITH_KIDS_FRACTION = 0.30  # 30% of paying adults travel with a kid [assumption]
PARENT_DISCOUNT_EUR = 5.0         # parent's ticket is EUR 5 below the standard adult price

# Audio guide ancillary revenue. Real Uffizi sells an audio guide at
# the entrance; we estimate ~25% of paying adults rent one. [UFF +
# assumption on take-up rate]
AUDIO_GUIDE_PRICE = 6.0           # EUR per audio guide
AUDIO_GUIDE_TAKE_UP = 0.25        # 25% of paying adults rent one
GROUP_SURCHARGE_FLAT = 70.0     # EUR 70 flat per tour group of 11+ [UFF]

# Annual Pass program (real Uffizi product, not currently in our model).
# EUR 80/year unlimited entry; family pass EUR 120. We model annual-
# pass holders as a separate sub-population that adds ~200 visitors per
# day to the museum at zero per-visit revenue (their EUR 80 is
# amortised across an assumed ~10 visits/year => EUR 8 effective per
# visit on the revenue side). [UFF + assumption]
ANNUAL_PASS_DAILY_VISITORS = 200
ANNUAL_PASS_PRICE_AMORTISED = 8.0
ANNUAL_PASS_TOTAL_PRICE = 80.0

# Experience quality metric: captures what makes a museum visit
# memorable beyond the absence of congestion. [assumption]
INTIMACY_THRESHOLD = 15        # max people in room for intimacy bonus [assumption]
INTIMACY_BONUS_WEIGHT = 2.0    # [assumption]
NARRATIVE_COHERENCE_WEIGHT = 1.5  # bonus for following a coherent trail [assumption]
SURPRISE_WEIGHT = 1.0          # bonus for discovering rooms not on checklist [assumption]
ENGAGEMENT_DEPTH_WEIGHT = 1.5  # bonus for long uncrowded dwell [assumption]

# Competitive landscape: Florence attractions competing for visitor
# time and budget. (name, price_eur, duration_min, experience_score)
FLORENCE_ATTRACTIONS = [
    ("Duomo Dome", 30, 45, 9.0),
    ("Battistero", 12, 30, 7.5),
    ("Accademia (David)", 20, 60, 8.5),
    ("Santa Croce", 8, 60, 8.0),
    ("Palazzo Pitti", 16, 90, 7.0),
    ("Boboli Gardens", 10, 60, 7.0),
    ("Ponte Vecchio", 0, 20, 7.5),
    ("San Lorenzo Market", 0, 45, 6.5),
]
VISITOR_FLORENCE_BUDGET_HOURS = 8.0    # [assumption]
VISITOR_FLORENCE_BUDGET_EUR = 100.0    # [assumption]
UFFIZI_BASE_EXPERIENCE_SCORE = 8.0    # without interventions [assumption]
UFFIZI_VISIT_DURATION_MIN = 150       # average visit [assumption]
# Per-profile visit-length budgets (minutes), sampled uniformly. Instagram
# tourists do the four masterpieces plus a terrace selfie and leave; the
# standard tourist spends a half-to-full afternoon and covers the second
# floor plus a quick pass of the first; the art lover can spend most of
# the day and goes through the whole museum. [user testimony]
INSTAGRAM_VISIT_BUDGET = (60, 90)
STANDARD_VISIT_BUDGET = (120, 180)
ART_LOVER_VISIT_BUDGET = (300, 480)

# Caravaggio (E4/E5) is the must-see of the first floor. A visitor who has
# descended and not yet seen it will trade away any other first-floor room
# to reach it: once their remaining budget drops within this many minutes of
# the walking distance to Caravaggio, they beeline straight there. [user
# testimony]
CARAVAGGIO_PRIORITY_MARGIN = 30

# --- New intervention constants (Wave 2) ---

# Multi-room buffer: rooms that activate content when Botticelli is congested.
BUFFER_CASCADE_ROOMS = ["A9", "A10", "A13"]  # [assumption]
BUFFER_CASCADE_MAGNETISM_BOOST = 2.5  # [assumption]

# Treasure hunt rooms for family visitors (underused, accessible). [assumption]
TREASURE_HUNT_ROOMS = ["A6", "A15", "A19", "A22", "A30", "A34"]

# Magnet room windows: all high-demand rooms get decoupled time windows.
ALL_MAGNET_WINDOW_ROOMS = {"A11", "A12", "A35", "A38", "E4"}  # [assumption]
MAGNET_WINDOW_DURATION = 30  # minutes [assumption]

# Staggered wing opening: west wing delay in minutes after east wing opens.
WEST_WING_DELAY = 75  # 1h15m delay [assumption]
WEST_WING_ROOMS = {"A24", "A25", "A26", "A27", "A28", "A29", "A30", "A31",
                    "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39", "A40", "A42"}

# Lunch free entry: boost arrivals during 12:30-14:00 for locals.
LUNCH_FREE_WINDOW = (255, 345)  # minutes 255-345 = 12:30-14:00
LUNCH_FREE_ARRIVAL_BOOST = 1.3  # 30% more arrivals during this window [assumption]

# Last-hour locals: final 90 minutes at reduced price.
LAST_HOUR_LOCALS_WINDOW = (525, 615)  # minutes 525-615 = 17:00-18:30
LAST_HOUR_LOCALS_BOOST = 1.5  # [assumption]
LAST_HOUR_LOCALS_EXTRA_VISITORS = 180  # additive late-entry local demand [assumption]
LAST_HOUR_LOCALS_PRICE = 10.0  # EUR reduced late-entry local ticket [assumption]

# Seasonal hours.
SUMMER_OPEN_MINUTES = 840  # 7:00-21:00 = 14 hours [assumption]
WINTER_OPEN_MINUTES = 480  # 9:00-17:00 = 8 hours [assumption]

# Breathing pause: 30-min closure at 13:00.
BREATHING_PAUSE_START = 285  # minute 285 = 13:00
BREATHING_PAUSE_DURATION = 30  # minutes

# Resident annual pass: adds off-peak short-visit locals.
RESIDENT_PASS_DAILY_VISITORS = 200  # [assumption]
RESIDENT_PASS_BUDGET_MINUTES = 45  # short visits [assumption]

# Cross-venue day pass: reduces afternoon Uffizi visitors.
CROSS_VENUE_AFTERNOON_REDUCTION = 0.15  # 15% fewer after 13:00 [assumption]

# Time-spent rebate: incentivizes checkbox tourists to leave faster.
REBATE_BUDGET_REDUCTION = 0.8  # Type B budgets reduced by 20% [assumption]

# Counter-flow incentive: fraction starting from west wing.
COUNTERFLOW_FRACTION = 0.15  # [assumption]
COUNTERFLOW_START_ROOM = "A24"

# Quiet hours: no tour groups during specified windows.
QUIET_HOURS_WINDOW = (0, 165)  # minutes 0-165 = 8:15-11:00 [assumption]

# Designated photo spots: rooms with professional lighting for photos.
PHOTO_SPOT_ROOMS = ["A16", "A39", "A9"]  # [assumption]
PHOTO_SPOT_IMPORTANCE_BOOST = 3.0  # [assumption]

# "The Painting Talks": first-person audio narration boosts room importance.
PAINTING_TALKS_ROOMS = ["A4", "A9", "A11", "A35", "E4"]  # rooms with special audio [assumption]
PAINTING_TALKS_IMPORTANCE_BOOST = 2.0  # [assumption]

# Comparative displays: link underused rooms to famous works.
COMPARATIVE_DISPLAY_ROOMS = ["A5", "A7", "A28", "A31"]  # [assumption]
COMPARATIVE_DISPLAY_IMPORTANCE_BOOST = 2.5  # [assumption]

# Tactile art station: magnetism boost for the room with touchable replicas.
TACTILE_STATION_MAGNETISM_BOOST = 3.5  # [assumption]

# Themed weeks: set of rooms that get importance boost for the theme.
THEMED_WEEKS = {
    "caravaggio": ["E4", "E5"],
    "women_of_uffizi": ["A5", "A9", "A12", "D12"],
    "renaissance_masters": ["A4", "A8", "A11", "A35", "A38"],
}
THEMED_WEEK_IMPORTANCE_BOOST = 3.0  # [assumption]

# Artist-in-residence: magnetism boost for the room with a living artist.
ARTIST_RESIDENCE_MAGNETISM_BOOST = 4.0  # [assumption]

# Real-time room boost: importance adjustment factor for empty rooms.
REALTIME_BOOST_DENSITY_THRESHOLD = 0.2  # boost rooms below this density [assumption]
REALTIME_BOOST_FACTOR = 2.0  # [assumption]

# Crowd-responsive lighting: magnetism adjustment based on density.
LIGHTING_OVERCROWDED_FACTOR = 0.8  # dim in crowded rooms [assumption]
LIGHTING_EMPTY_FACTOR = 1.3  # brighten in empty rooms [assumption]
LIGHTING_DENSITY_THRESHOLD = 0.7  # [assumption]

# Progress bar / achievement: boost importance of unvisited rooms.
PROGRESS_BAR_UNVISITED_BOOST = 1.5  # [assumption]
ACHIEVEMENT_UNDERUSED_ROOMS = ["A6", "A15", "A17", "A19", "A20", "A21",
                                "A22", "A29", "A30", "A32", "A33"]  # [assumption]
ACHIEVEMENT_IMPORTANCE_BOOST = 3.0  # [assumption]

# =============================================================================
# Room metadata
# =============================================================================
# Room data derived from the official 2023 Uffizi floor plan [MAP].
# Room names and numbering follow the museum's own labeling.
# A41 does not exist in the floor plan. A37 exists but is unnamed.
#
# Columns: (room_id, name, section, importance, magnetism, capacity)
#   importance: 1-10, cultural significance of the artworks [assumption,
#               informed by guidebook prominence and visitor survey data]
#   magnetism:  dwell-time multiplier (1.0 ~ 3 min base dwell) [assumption]
#   capacity:   comfortable max occupancy (persons) [assumption; no published
#               per-room data exists in [A22] or [C21]]

SECOND_FLOOR_ROOMS: List[Tuple[str, str, str, int, float, int]] = [
    # Capacities derived from pixel-area analysis of the official Uffizi
    # floor plan (proportional to room footprint). Reference: A7 = 13,694 px
    # scaled at ~0.00258 people/px. Corridors use 0.6x density (transit).
    # Entry and Western Corridor
    ("A1", "Lorenese Vestibule", "entry", 2, 0.5, 20),           # 7,587 px
    ("A2", "Western Corridor", "corridor", 6, 1.5, 200),         # 129,817 px (sculpture + Florence views)
    ("A3", "Medieval Painting", "13th-14th", 4, 1.0, 51),        # 19,708 px
    # Giotto loop: A4 -> A5 -> A6 -> A7 -> A4
    ("A4", "Giotto & Cimabue", "13th-14th", 8, 2.5, 65),        # 25,380 px
    ("A5", "Lorenzetti & Simone Martini", "13th-14th", 6, 1.5, 17),  # 6,528 px
    ("A6", "14th Century", "13th-14th", 4, 1.0, 24),             # 9,396 px
    ("A7", "Lorenzo Monaco & Gentile da Fabriano", "early_renaissance", 7, 2.0, 35),  # 13,694 px
    # Sequential off A4
    ("A8", "Masaccio & Beato Angelico", "early_renaissance", 7, 2.0, 26),  # 9,999 px
    ("A9", "Uccello, F. Lippi, Piero della Francesca", "early_renaissance", 8, 2.5, 52),  # 20,115 px
    # Botticelli forced chain: A10 -> A11 -> A12 -> A13
    ("A10", "Pollaiolo", "early_renaissance", 5, 1.5, 22),       # 8,712 px
    ("A11", "Botticelli - The Spring", "early_renaissance", 10, 10.0, 55), # 42,347 px / 2
    ("A12", "Botticelli - Venus", "early_renaissance", 10, 10.0, 55),     # 42,347 px / 2
    ("A13", "Hugo van der Goes", "early_renaissance", 5, 1.0, 43),        # 16,860 px
    ("A14", "Uffizi Maps Terrace", "early_renaissance", 3, 1.0, 14),      # 5,245 px
    ("A15", "Room of Mathematics", "early_renaissance", 2, 0.5, 17),      # 6,547 px
    # Tribune
    ("A16", "Tribune (Medici Venus)", "special", 5, 1.5, 37),    # 14,182 px (octagonal)
    # Forced chain: A17 -> A18 -> A19 -> A20 -> A21 (no corridor exits A18-A21)
    ("A17", "15th Century Siena", "renaissance_other", 4, 1.0, 20),       # 7,740 px
    ("A18", "Mantegna, Bellini, Antonello da Messina", "renaissance_other", 7, 2.0, 25),  # 9,675 px
    ("A19", "15th Century Veneto", "renaissance_other", 4, 1.0, 20),      # 7,740 px
    ("A20", "15th Century Emilia Romagna", "renaissance_other", 3, 0.8, 26),  # 9,933 px
    ("A21", "15th Century Lombardia", "renaissance_other", 3, 0.8, 22),   # 8,651 px
    ("A22", "Boudoir of Miniatures", "renaissance_other", 3, 1.0, 14),    # 5,457 px
    # Corridors
    ("A23", "Southern Corridor", "corridor", 5, 1.2, 53),        # 34,325 px (smaller sculpture passage)
    ("A24", "Eastern Corridor", "corridor", 6, 1.5, 240),        # 154,384 px (Arno/Ponte Vecchio view, Laocoon copy, busts)
    # U-loop: A25 -> A26 -> A27 -> A28 -> A29 -> (A30 dead end) -> A31 -> A32 -> A33 -> A34
    # No cross-connections between rows. Only turn is A29 -> A31.
    ("A25", "Domenico Ghirlandaio", "high_renaissance", 6, 1.5, 39),      # 15,200 px
    ("A26", "Cosimo Rosselli", "high_renaissance", 4, 1.0, 30),           # 11,520 px
    ("A27", "Pietro Perugino", "high_renaissance", 7, 2.0, 31),           # 11,926 px
    ("A28", "Filippino Lippi & Piero di Cosimo", "high_renaissance", 6, 1.5, 36),  # 13,936 px
    ("A29", "Lorenzo di Credi", "high_renaissance", 4, 1.0, 12),          # 4,818 px
    ("A30", "Doryphoros", "high_renaissance", 3, 0.8, 10),                # 4,025 px
    ("A31", "Luca Signorelli", "high_renaissance", 5, 1.2, 14),           # 5,341 px
    ("A32", "Luca Signorelli II", "high_renaissance", 4, 1.0, 22),        # 8,424 px
    ("A33", "Greek Portrait Sculptures", "high_renaissance", 3, 0.8, 22), # 8,589 px
    ("A34", "Antique & Garden of San Marco", "high_renaissance", 4, 1.0, 13),  # 5,029 px
    # Leonardo-Michelangelo group (connected via A36)
    ("A35", "Leonardo da Vinci", "high_renaissance", 10, 9.5, 52),        # 20,297 px
    ("A36", "Hall of Ancient Inscriptions", "high_renaissance", 3, 0.8, 45),  # 17,474 px
    ("A37", "Unnamed Room", "high_renaissance", 2, 0.5, 22),              # 8,501 px
    ("A38", "Raphael & Michelangelo", "high_renaissance", 10, 9.5, 53),   # 20,743 px
    # Dead-end branches off Eastern Corridor
    ("A39", "Niobe Room", "high_renaissance", 5, 1.5, 93),                # 36,224 px
    ("A40", "Hermaphrodite", "high_renaissance", 4, 1.0, 14),             # 5,532 px
    # A41 does not exist in the Uffizi floor plan.
    ("A42", "Dürer & Transalpine Renaissance", "high_renaissance", 6, 1.5, 34),  # 13,060 px
]

# First floor: B block (Contini Bonacossi), C block (Self-Portraits),
# D block (16th-17th century), E block (Caravaggio and Baroque).
# Full topology mapped from the 2023 floor plan [MAP].

FIRST_FLOOR_B_ROOMS: List[Tuple[str, str, str, int, float, int]] = [
    ("B1", "Contini Bonacossi Corridor", "contini_bonacossi", 2, 0.3, 16),  # 6,220 px
    ("B2", "Sassetta & Andrea del Castagno", "contini_bonacossi", 5, 1.2, 20),  # 7,627 px
    ("B3", "Gold Ground Paintings", "contini_bonacossi", 4, 1.0, 13),      # 4,904 px
    ("B4", "Giovanni Bellini", "contini_bonacossi", 6, 1.5, 21),           # 8,143 px
    ("B5", "Bramantino", "contini_bonacossi", 4, 1.0, 19),                # 7,410 px
    ("B6", "Furniture", "contini_bonacossi", 3, 0.8, 17),                 # 6,674 px
    ("B7", "Majolica", "contini_bonacossi", 3, 0.8, 16),                  # 6,068 px
    ("B8", "Bernini", "contini_bonacossi", 7, 2.0, 39),                   # 15,090 px
]

FIRST_FLOOR_C_ROOMS: List[Tuple[str, str, str, int, float, int]] = [
    ("C1", "Self-Portraits: Origins to 17th Century", "self_portraits", 4, 1.0, 85),  # 33,059 px
    ("C2", "Water and Light", "self_portraits", 3, 0.8, 15),              # 5,634 px
    ("C3", "17th Century Self-Portraits", "self_portraits", 3, 0.8, 24),  # 9,139 px
    ("C4", "Showing the Artwork", "self_portraits", 3, 0.8, 22),          # 8,701 px
    ("C5", "Early 18th Century", "self_portraits", 3, 0.8, 21),           # 8,023 px
    ("C6", "Late 18th Century", "self_portraits", 3, 0.8, 38),            # 14,690 px
    ("C7", "Works on Paper I", "self_portraits", 3, 0.8, 25),             # 9,672 px
    ("C8", "Works on Paper II", "self_portraits", 3, 0.8, 29),            # 11,370 px
    ("C9", "19th Century Self-Portraits", "self_portraits", 4, 1.0, 19),  # 7,232 px
    ("C10", "Early 20th Century", "self_portraits", 4, 1.0, 37),          # 14,431 px
    ("C11", "From the War to Our Days", "self_portraits", 4, 1.0, 37),    # 14,336 px
    ("C12", "Contemporary Self-Portraits", "self_portraits", 4, 1.0, 40), # 15,522 px
]

FIRST_FLOOR_D_ROOMS: List[Tuple[str, str, str, int, float, int]] = [
    # Capacities from pixel-area analysis of official floor plan.
    # Corridor spine: D1 -> D6 -> D13 -> D14 -> D15 -> D26 -> D28
    ("D1", "The 16th Century in Lombardia", "16th_century", 4, 1.0, 46),   # 17,744 px
    ("D2", "Dosso Dossi and His Circle", "16th_century", 5, 1.2, 49),     # 18,884 px
    ("D3", "Work in Progress", "16th_century", 1, 0.2, 10),               # 3,198 px (closed)
    ("D4", "Correggio and Parmigianino", "16th_century", 6, 1.5, 21),     # 8,308 px
    ("D5", "Painters from Ferrara", "16th_century", 3, 0.5, 30),          # 11,727 px
    ("D6", "The 16th Century in Bologna", "16th_century", 4, 1.0, 25),    # 9,744 px
    ("D7", "Sebastiano del Piombo and the Influence of Michelangelo", "16th_century", 5, 1.2, 21),  # 8,200 px
    ("D8", "Daniele da Volterra, Francesco Salviati", "16th_century", 4, 1.0, 23),  # 8,800 px
    ("D9", "Andrea del Sarto and Pontormo", "16th_century", 3, 0.5, 45),  # 17,324 px
    ("D10", "Work in Progress", "16th_century", 1, 0.2, 22),              # 8,712 px (closed)
    ("D11", "Work in Progress", "16th_century", 1, 0.2, 29),              # 11,286 px (closed)
    ("D12", "Pontormo, Rosso Fiorentino", "16th_century", 7, 2.0, 21),    # 8,118 px
    ("D13", "Bachiacca, Portraiture in Florence", "16th_century", 4, 1.0, 41),  # 16,016 px
    ("D14", "Bronzino", "16th_century", 6, 1.5, 29),                      # 11,088 px
    ("D15", "Room of the Dynasties", "16th_century", 5, 1.2, 94),         # 36,624 px
    ("D16", "Room of the Pillar", "16th_century", 3, 0.8, 66),            # 25,500 px
    ("D17", "Classic Tradition", "16th_century", 4, 1.0, 22),             # 8,664 px
    ("D18", "Room of the Counter-Reformation", "16th_century", 4, 1.0, 21),  # 8,103 px
    ("D19", "Vasari Corridor Entrance", "16th_century", 2, 0.3, 22),      # 8,520 px (passage)
    ("D20", "Venetian Chapel", "16th_century", 4, 1.0, 33),               # 12,720 px
    ("D21", "El Greco", "16th_century", 6, 1.5, 23),                      # 8,850 px
    ("D22", "Antechamber of Venus", "16th_century", 4, 1.0, 28),          # 10,868 px
    ("D23", "Titian, Venus of Urbino", "16th_century", 9, 3.5, 43),       # 16,733 px
    ("D24", "Naturalismo Veneto", "16th_century", 4, 1.0, 27),            # 10,626 px
    ("D25", "Tintoretto", "16th_century", 6, 1.5, 43),                    # 16,564 px
    ("D26", "Veronese", "16th_century", 6, 1.5, 53),                      # 20,496 px
    ("D27", "Veronese Corridor", "16th_century", 4, 0.8, 120),            # 46,787 px (1st-floor sculpture passage)
    ("D28", "Verone", "16th_century", 4, 1.0, 30),                        # no pixel data, kept
]

FIRST_FLOOR_E_ROOMS: List[Tuple[str, str, str, int, float, int]] = [
    # E1-E3 and E8 removed (cancelled in current floor plan).
    # Chain: D28 -> E4 -> E5 -> E6 -> E7 -> EXIT (Magliabechi).
    ("E4", "Caravaggio The Medusa", "17th_century", 9, 3.5, 30),
    ("E5", "Caravaggio Il Bacco", "17th_century", 8, 3.0, 25),
    ("E6", "Painting by Candlelight", "17th_century", 5, 1.2, 25),
    ("E7", "Rembrandt, Rubens, Van Dyck", "17th_century", 7, 2.0, 30),
]

SPECIAL_NODES: List[Tuple[str, str, str, int, float, int]] = [
    ("ENTRY", "Museum Entrance Hall", "entry_gate", 1, 0.0, 120),
    ("GRANDUCAL_STAIRCASE", "Granducal Staircase", "staircase", 1, 0.0, 30),
    ("PANORAMIC_TERRACE", "Panoramic Terrace", "terrace", 3, 1.0, 60),
    ("LANZI_STAIRCASE", "Lanzi Staircase", "staircase", 1, 0.0, 40),
    ("BUONTALENTI_STAIRCASE", "Buontalenti Staircase", "staircase", 1, 0.0, 30),
    ("EXIT", "Magliabechi Exit", "exit", 1, 0.0, 120),
]

ALL_ROOM_ROWS = (
    SECOND_FLOOR_ROOMS
    + FIRST_FLOOR_B_ROOMS
    + FIRST_FLOOR_C_ROOMS
    + FIRST_FLOOR_D_ROOMS
    + FIRST_FLOOR_E_ROOMS
    + SPECIAL_NODES
)

ROOM_IDS = [row[0] for row in ALL_ROOM_ROWS]
N_ROOMS = len(ROOM_IDS)
ROOM_TO_IDX = {room: i for i, room in enumerate(ROOM_IDS)}
IDX_TO_ROOM = {i: room for room, i in ROOM_TO_IDX.items()}

ROOM_DATA: Dict[str, Dict[str, str | float]] = {
    room_id: {
        "name": name,
        "section": section,
        "importance": float(importance),
        "magnetism": float(magnetism),
        "capacity": float(capacity),
    }
    for room_id, name, section, importance, magnetism, capacity in ALL_ROOM_ROWS
}

# =============================================================================
# Graph topology
# =============================================================================
# Topology mapped from the official 2023 Uffizi floor plan [MAP].
# Key features: A4-A7 loop, A10-A12 Botticelli forced chain, A17-A21
# forced chain, A25-A34 U-loop, A35-A36-A37-A38 Leonardo group,
# D block corridor spine with perpendicular branches, E block parallel rows.

EDGES: List[Tuple[str, str]] = [
    # ===== SECOND FLOOR (A block) =====
    # NOTE: the staircase from ENTRY up to A1 (Granducal) is one-way and
    # lives in DIRECTED_EDGES below, not here. Entries in this list are
    # bidirectional doorways: build_uffizi_graph adds both (a,b) and (b,a).
    ("A1", "A2"),      # vestibule to Western Corridor
    ("A1", "A3"),      # vestibule to Medieval Painting (right turn)
    # Giotto loop: A4 <-> A5 <-> A6 <-> A7 <-> A4
    ("A2", "A4"),      # corridor to Giotto (first entrance on left)
    ("A4", "A5"),
    ("A5", "A6"),
    ("A6", "A7"),
    ("A7", "A4"),      # completes the loop
    # Sequential off A4: A4 -> A8 -> A9.
    # A8 -> A9 is ONE-WAY (lives in DIRECTED_EDGES below) so visitors at
    # A9 cannot back-track to Masaccio; the only exits from A9 are A2
    # (corridor) and A10 (forward into the Botticelli chain).
    ("A4", "A8"),
    # The A2 corridor "return" edges from A13, A15, A16, A17 are ONE-WAY
    # INTO A2 (room -> A2 only). Visitors at A2 cannot use the corridor
    # as a shortcut to skip Botticelli: from A2 their only forward
    # options are A4 (Giotto loop) and A9 (toward Botticelli). The
    # one-way directions are declared in DIRECTED_EDGES below.
    ("A9", "A2"),      # Piero della Francesca back to corridor
    # Botticelli forced chain: enter at A10, exit at A13. The interior
    # transitions (A10->A11, A11->A12, A12->A13) are ONE-WAY in DIRECTED_EDGES
    # so visitors cannot ping-pong between Spring and Venus. A9 <-> A10
    # remains bidirectional (you can step back out to Piero before
    # committing to the Botticelli chain).
    ("A9", "A10"),
    # Post-Botticelli
    # A13 -> A2 is ONE-WAY (in DIRECTED_EDGES). A2 cannot bypass into A13.
    ("A13", "A14"),    # Maps Terrace (dead end)
    ("A13", "A15"),    # Room of Mathematics
    # A15 -> A2 is ONE-WAY (in DIRECTED_EDGES). A2 cannot bypass into A15.
    # Tribune
    ("A15", "A16"),    # Mathematics to Tribune
    # A16 -> A2 is ONE-WAY (in DIRECTED_EDGES). A2 cannot bypass into A16.
    # Forced chain A17-A21 (no corridor exits from A18-A21).
    # The A2 <-> A21 shortcut has been REMOVED. Visitors arriving at A2
    # after Botticelli (exit at A13) must traverse the natural museum
    # itinerary A13 -> A15 -> A16 -> A17 -> A18 -> A19 -> A20 -> A21
    # to reach the Eastern Corridor and Leonardo. This enforces the
    # user-confirmed real-museum constraint that you cannot reach the
    # Leonardo side without first walking past Botticelli.
    ("A16", "A17"),
    # A17 -> A2 is ONE-WAY (in DIRECTED_EDGES). A2 cannot bypass into A17.
    ("A17", "A18"),    # into the forced chain
    ("A18", "A19"),
    ("A19", "A20"),
    ("A20", "A21"),
    ("A21", "A22"),    # Boudoir of Miniatures (dead end)
    # Southern Corridor
    ("A21", "A23"),    # to Southern Corridor
    ("A23", "A24"),    # Southern to Eastern Corridor
    ("A2", "A23"),     # Western Corridor turns into Southern Corridor at the
                       # south end of the museum. This is the real Uffizi
                       # U-shape: A2 (west) -> A23 (south) -> A24 (east). Was
                       # missing entirely; without it the only way from
                       # Botticelli area to Leonardo was through the entire
                       # Tribune chain (11 rooms), which is not how visitors
                       # actually move.
    # U-loop A25-A34 (no cross-connections between rows)
    ("A24", "A25"),    # Eastern Corridor to Ghirlandaio
    ("A25", "A26"),
    ("A26", "A27"),
    ("A27", "A28"),
    ("A28", "A29"),
    ("A29", "A30"),    # Doryphoros (dead end)
    ("A29", "A31"),    # the turn between rows
    ("A31", "A32"),
    ("A32", "A33"),
    ("A33", "A34"),
    ("A34", "A24"),    # back to Eastern Corridor
    # Leonardo-Michelangelo group.
    # All the corridor connections (A24->A35, A35->A36, A36->A38,
    # A36->A24, A38->A24) are ONE-WAY in DIRECTED_EDGES. The Leonardo
    # branch is a single-pass loop: enter via A35, exit via A24. There
    # is no way back to A35 from A24 except by going all the way around.
    # The Hall (A36) has a bidirectional side-trip to A37, the unnamed
    # dead-end room, but everything else is one-way.
    ("A36", "A37"),    # Hall to unnamed dead-end room (bidirectional)
    # Dead-end branches off Eastern Corridor
    ("A24", "A39"),    # Niobe Room
    ("A24", "A40"),    # Hermaphrodite
    ("A24", "A42"),    # Dürer
    # Panoramic Terrace (bidirectional access from A24).
    # The Lanzi and Buontalenti staircases themselves are one-way DOWN
    # and live in DIRECTED_EDGES, not here.
    ("A24", "PANORAMIC_TERRACE"),

    # ===== FIRST FLOOR: B block (Contini Bonacossi) =====
    # All rooms branch off B1 corridor. U-shaped visiting pattern.
    # B1 itself is the foot of Lanzi (one-way DOWN, see DIRECTED_EDGES).
    ("B1", "B2"),      # left side
    ("B1", "B4"),
    ("B1", "B6"),
    ("B1", "B8"),      # turnaround
    ("B1", "B7"),      # right side (return)
    ("B1", "B5"),
    ("B1", "B3"),
    # B and C blocks are on the same 1st floor and physically connected,
    # so visitors can walk between them without going back through Lanzi.
    ("B1", "C1"),

    # ===== FIRST FLOOR: C block (Self-Portraits) =====
    # Forced chain with optional C7-C8 branch.
    # C1 itself is the foot of Lanzi (one-way DOWN, see DIRECTED_EDGES).
    ("C1", "C2"),
    ("C2", "C3"),
    ("C3", "C4"),
    ("C4", "C5"),
    ("C5", "C6"),
    ("C6", "C7"),      # optional branch
    ("C7", "C8"),      # dead end
    ("C6", "C9"),      # continue chain
    ("C9", "C10"),
    ("C10", "C11"),
    ("C11", "C12"),
    ("C12", "D1"),     # into D block

    # ===== FIRST FLOOR: D block (16th-17th century) =====
    # Corridor spine: D1 -> D6 -> D13 -> D14 -> D15 -> D26 -> D28
    ("D1", "D6"),      # corridor
    ("D6", "D13"),
    ("D13", "D14"),
    ("D14", "D15"),
    ("D15", "D26"),
    ("D26", "D28"),
    # Branch: D1 -> D2 -> D3 (dead end), D2 -> D4
    ("D1", "D2"),
    ("D2", "D3"),      # dead end
    ("D2", "D4"),
    # D4 -> D5
    ("D4", "D5"),
    # D5 -> D6 (back to corridor) or D5 -> D7
    ("D5", "D6"),
    ("D5", "D7"),
    # D7 -> D8 (dead end), D7 -> D9
    ("D7", "D8"),      # dead end
    ("D7", "D9"),      # Andrea del Sarto / Buontalenti staircase hub
    # D9 connections
    ("D9", "D13"),     # back to corridor
    ("D9", "D12"),
    ("D9", "D10"),
    # D10 branches
    ("D10", "D11"),    # dead end
    ("D10", "D12"),    # also accessible from D9
    ("D10", "D16"),    # perpendicular to corridor
    # D15 -> D16 (also accessible from D10)
    ("D15", "D16"),
    # D16 branches
    ("D16", "D17"),
    ("D16", "D18"),
    ("D17", "D18"),    # tiny room connection between D17 and D18
    # D16 -> D19, D15 -> D19
    ("D16", "D19"),
    ("D15", "D19"),
    # D19 branches
    ("D19", "D20"),
    ("D19", "D21"),
    ("D20", "D21"),    # loop: D19 -> D20 -> D21
    # D21 connections
    ("D21", "D27"),    # to Veronese Corridor
    ("D27", "D26"),    # connects to corridor
    ("D21", "D22"),
    # D22-D23-D24 triangle
    ("D22", "D23"),    # Titian
    ("D22", "D24"),
    ("D23", "D24"),
    # D24 -> D25 -> D26
    ("D24", "D25"),
    ("D25", "D26"),

    # ===== FIRST FLOOR: E block (Caravaggio) =====
    # Corridor runs from D28 past 4 gallery rooms to EXIT.
    # E4-E7 are side rooms (enter-and-exit spurs) off the corridor.
    # Visitors walk the corridor, popping into rooms as they choose.
    ("D28", "E4"),      # corridor reaches E4 first
    ("D28", "E5"),      # corridor continues past E5
    ("E4", "E5"),       # adjacent along corridor
    ("E5", "E6"),       # adjacent along corridor
    ("E6", "E7"),       # adjacent along corridor
    # ("E7", "EXIT") is one-way OUT, see DIRECTED_EDGES below.
]

# =============================================================================
# DIRECTED EDGES (one-way only)
# =============================================================================
# These edges are NOT bidirectional. They model the real-world directionality
# of the Uffizi's staircases and exits:
#   - Granducal Staircase: one-way UP from the entrance hall to the 2nd floor.
#     This is the only way in. You cannot descend Granducal once you are up.
#   - Lanzi Staircase: one-way DOWN from the Panoramic Terrace. Visitors
#     can stop at the B block (Contini Bonacossi), the C block (Self-
#     Portraits), or skip the 1st floor entirely and continue to EXIT.
#   - Buontalenti Staircase: one-way DOWN from A36 to D9 on the 1st floor.
#     From there, visitors traverse the D and E blocks at their own pace
#     and exit via Magliabechi (E7 -> EXIT).
#   - E7 -> EXIT: the museum's main exit. One-way out.
#
# build_uffizi_graph adds these edges in the listed direction only. Any
# undirected sibling pair would let visitors walk UP the staircases, which
# is physically impossible in the real museum.
DIRECTED_EDGES: List[Tuple[str, str]] = [
    # --- Granducal Staircase (UP only, single entrance) ---
    ("ENTRY", "GRANDUCAL_STAIRCASE"),
    ("GRANDUCAL_STAIRCASE", "A1"),

    # --- Lanzi Staircase (DOWN only, from 2nd floor) ---
    ("PANORAMIC_TERRACE", "LANZI_STAIRCASE"),
    ("LANZI_STAIRCASE", "B1"),      # exit at Contini Bonacossi
    ("LANZI_STAIRCASE", "C1"),      # exit at Self-Portraits chain
    ("LANZI_STAIRCASE", "EXIT"),    # skip the 1st floor entirely, exit to street

    # --- Buontalenti Staircase (DOWN only, from A36 to D9) ---
    ("A36", "BUONTALENTI_STAIRCASE"),
    ("BUONTALENTI_STAIRCASE", "D9"),

    # --- Magliabechi exit (one-way out) ---
    ("E7", "EXIT"),

    # --- Botticelli forced chain (one-way through the masterpieces) ---
    # In the real museum the Botticelli rooms have a forced enter-from-
    # one-side, exit-to-the-other-side layout. Modelling this as
    # bidirectional created a ping-pong loop between Spring and Venus
    # that inflated their dwell time by ~2x and broke the density
    # ranking. Directed edges fix the topology to match reality.
    ("A10", "A11"),    # enter Botticelli Spring
    ("A11", "A12"),    # Spring to Venus
    ("A12", "A13"),    # exit Botticelli to Hugo van der Goes

    # --- Leonardo-Michelangelo forced chain (one-way) ---
    # Single-pass loop: visitors enter Leonardo via A35, walk through the
    # Hall (A36) and Raphael (A38), then exit back to the Eastern Corridor
    # (A24). The corridor connection at the back (A24 -> A36 or A24 ->
    # A38) does NOT exist, so visitors who already left the Leonardo
    # branch cannot ping-pong back through A36 to revisit A38. This
    # matches the user-confirmed real-museum flow.
    ("A24", "A35"),    # corridor to Leonardo (only entry point)
    ("A35", "A36"),    # Leonardo to Hall of Inscriptions
    ("A36", "A38"),    # Hall to Raphael/Michelangelo (only exit from Hall)
    ("A38", "A24"),    # Raphael to corridor (only exit from Raphael)

    # --- A8 -> A9 one-way (no back-track from Piero to Masaccio) ---
    # Prevents visitors at A9 from looping back into the Giotto wing
    # and ensures they continue forward to A10 (and Botticelli). The
    # forward direction is preserved: visitors at A8 can still walk
    # to A9 normally.
    ("A8", "A9"),

    # --- Post-Botticelli return to the Western Corridor ---
    # After Botticelli (A11/A12 -> A13), visitors can exit back to A2
    # (Western Corridor) and take the corridor route A2 -> A23 -> A24
    # to reach Leonardo without traversing every Tribune room. One-way
    # (A13 -> A2 only, not A2 -> A13) so pre-Botticelli visitors at A2
    # still cannot bypass into the Tribune chain - they must go through
    # Botticelli to get there. This matches the real Uffizi U-shape:
    # the Western, Southern and Eastern corridors form one continuous
    # circulation path.
    ("A13", "A2"),
]


# Classic itinerary: the route most visitors follow [MAP].
# A block (full) -> Lanzi staircase -> D block -> E block -> exit.
# Skips B and C blocks.
RECOMMENDED_ROUTE: List[str] = [
    # The LAZY-TOURIST route: minimum rooms to hit Botticelli, Tribune,
    # Leonardo, Raphael/Michelangelo, and Caravaggio, then exit. Everything
    # else (Giotto loop, forced 15th-century chain, U-loop, dead-ends, B
    # wing, C wing, most of D wing) is OFF the route. Type A art lovers
    # still find these via the importance pull on their personal
    # importance_vector; Type B tourists skip them.
    #
    # The route is the same single sequence for every visitor; the
    # spectrum from laziest to most thorough comes from heterogeneous
    # time budgets (sampled per visitor) and per-type importance vectors
    # (Type A is more curious about minor rooms; Type B sticks to the
    # checklist).
    #
    # Granducal Staircase: only way up to the 2nd floor.
    "ENTRY", "GRANDUCAL_STAIRCASE", "A1",
    "A2", "A9",                                # straight toward Botticelli, skip Giotto
    "A10", "A11", "A12", "A13",                # Botticelli chain
    # Botticelli exits to A13. From there visitors walk the natural
    # museum path through Tribune and the 15th-century forced chain to
    # reach the Eastern Corridor. The A2 <-> A21 shortcut has been
    # removed so this is the only available path forward.
    "A15", "A16",                              # Mathematics, Tribune
    "A17", "A18", "A19", "A20", "A21",         # 15th-century forced chain
    "A23", "A24",                              # Southern -> Eastern corridor
    "A35", "A36", "A38", "A36",                # Leonardo, Hall, Raphael
    "A24",                                     # back to corridor
    "PANORAMIC_TERRACE", "LANZI_STAIRCASE",    # descend Lanzi
    "EXIT",                                    # skip 1st floor entirely (laziest exit)
]


# =============================================================================
# Per-profile routes
# =============================================================================
# The three visitor profiles walk the museum differently. The crowd
# simulator selects the matching route map per visitor (by profile.segment)
# and uses it for the route bonus in the movement model. The building is
# fixed; these sequences only express where each profile is steered.
#
#   Instagram (60%): shortest path through the four masterpieces and the
#     terrace, then straight down the Lanzi staircase to the street. Never
#     sees the first floor. The 15th-century Tribune chain is bypassed via
#     the A2 -> A23 -> A24 corridor.
#   Standard (30%):  the full second floor, then down the Lanzi staircase
#     onto the first floor, fast through the Self-Portrait (C) wing and
#     along the 16th-century (D) corridor spine to Caravaggio (E), out via
#     the Magliabechi staircase.
#   Art Lover (10%): the full second floor, then the whole first floor,
#     the Contini Bonacossi (B) wing, the C wing, and a deep sweep of the
#     16th-century (D) rooms (Titian, El Greco, Bronzino, Tintoretto,
#     Veronese), then Caravaggio, out via the Magliabechi staircase. The
#     side rooms not listed are picked up by the art lover's importance
#     pull, which the standard tourist lacks.

INSTAGRAM_ROUTE: List[str] = [
    "ENTRY", "GRANDUCAL_STAIRCASE", "A1",
    "A2", "A9", "A10", "A11", "A12", "A13",    # straight to Botticelli
    "A2", "A23", "A24",                        # corridor, skipping the Tribune chain
    "A35", "A36", "A38", "A24",                # Leonardo, Hall, Raphael, back to corridor
    "PANORAMIC_TERRACE", "LANZI_STAIRCASE",    # terrace selfie, descend Lanzi
    "EXIT",                                    # straight out at floor zero
]
# The set of rooms an Instagram tourist will ever set foot in. For them the
# rest of the museum does not exist: they came only for the masterpiece
# selfies and the terrace, and they leave the moment they are done. The
# simulator filters their movement to this set so they never divert into the
# U-loop, the Tribune chain or the dead-end side rooms.
INSTAGRAM_ROUTE_ROOMS = frozenset(INSTAGRAM_ROUTE)

# Shared second-floor sweep for the two thorough profiles. Covers the
# Giotto loop, Masaccio/Piero, Botticelli, the Tribune and 15th-century
# chain, the U-loop, and the Leonardo group, ending at the Lanzi staircase.
_FLOOR2_SWEEP: List[str] = [
    "ENTRY", "GRANDUCAL_STAIRCASE", "A1",
    "A2", "A4", "A5", "A6", "A7", "A4", "A8", "A9",   # corridor, Giotto loop, Masaccio, Piero
    "A10", "A11", "A12", "A13",                       # Botticelli chain
    "A15", "A16",                                     # Mathematics, Tribune
    "A17", "A18", "A19", "A20", "A21",                # 15th-century forced chain
    "A23", "A24",                                     # Southern -> Eastern corridor
    "A25", "A26", "A27", "A28", "A29", "A31", "A32", "A33", "A34", "A24",  # U-loop
    "A35", "A36", "A38", "A24",                       # Leonardo, Hall, Raphael
    "PANORAMIC_TERRACE", "LANZI_STAIRCASE",           # terrace, descend Lanzi
]

STANDARD_ROUTE: List[str] = _FLOOR2_SWEEP + [
    "C1", "C2", "C3", "C4", "C5", "C6", "C9", "C10", "C11", "C12",  # Self-Portraits spine
    "D1", "D6", "D13", "D14", "D15", "D26", "D28",                  # 16th-century corridor spine
    "E4", "E5", "E6", "E7",                                         # Caravaggio and Baroque
    "EXIT",                                                         # Magliabechi exit
]

ART_LOVER_ROUTE: List[str] = _FLOOR2_SWEEP + [
    "B1",                                                           # Contini Bonacossi (spurs via importance pull)
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C7", "C6", "C9", "C10", "C11", "C12",  # full Self-Portraits
    "D1", "D2", "D4", "D5", "D6", "D13", "D14", "D15",             # 16th century
    "D19", "D21", "D22", "D23", "D24", "D25", "D26", "D28",        # El Greco, Titian, Tintoretto, Veronese
    "E4", "E5", "E6", "E7",                                         # Caravaggio and Baroque
    "EXIT",                                                         # Magliabechi exit
]

# The rooms that Type B ("checkbox tourist") visitors concentrate on.
# These are the canonical "must-see" works that appear in every Uffizi
# guidebook and top-10 list. [assumption]
TYPE_B_MAGNET_ROOMS = {"A11", "A12", "A35", "A38", "E4"}

# Themed alternative itineraries designed to redistribute demand away from
# magnet rooms by reframing lesser-known galleries as coherent narratives.
# These trails are invented for the project (not offered by the Uffizi);
# room selections are based on thematic coherence from the collection
# catalog. [assumption]
# Hidden gem trails are themed alternative itineraries built from rooms
# that hold first-rate work by artists the average Instagram tourist does
# not recognize. The three trails are chronological: Italian Trecento
# (1300s), Quattrocento (1400s), Cinquecento (1500s). No masterpiece
# room (A11/A12 Botticelli, A35 Leonardo, A38 Raphael/Michelangelo) is
# included; the whole point of the trails is to give visitors a reason
# to skip the masterpiece scrum without skipping great art.
#
# Trecento (1300s): the proto-Renaissance, before Botticelli existed.
#   A4  - Giotto & Cimabue (Maesta di Ognissanti, Madonna di Santa Trinita)
#   A5  - Lorenzetti & Simone Martini (Sienese masters)
#
# Quattrocento (1400s): the early Renaissance, where the Florentine
# revolution actually started.
#   A8  - Masaccio & Beato Angelico (Masaccio invented Renaissance
#         perspective; Fra Angelico the spiritual side of the revolution)
#   A9  - Uccello, Filippo Lippi, Piero della Francesca (Piero is one
#         of the greatest mathematician-painters of the Renaissance)
#   A10 - Pollaiolo
#   A13 - Hugo van der Goes (the Portinari Altarpiece, the great
#         Northern Renaissance work commissioned in Florence)
#   A18 - Mantegna, Bellini, Antonello da Messina (Padua + Venice +
#         Sicily, the non-Florentine quattrocento)
#   A25 - Domenico Ghirlandaio (Michelangelo's teacher)
#   A27 - Pietro Perugino (Raphael's teacher)
#   A28 - Filippino Lippi & Piero di Cosimo
#
# Cinquecento (1500s): everyone after Raphael/Michelangelo. Mannerism
# and the Venetian school.
#   D4  - Correggio and Parmigianino
#   D9  - Andrea del Sarto and Pontormo
#   D12 - Pontormo, Rosso Fiorentino (founders of Florentine Mannerism)
#   D14 - Bronzino (the Medici court portraitist)
#   D21 - El Greco
#   D23 - Titian, Venus of Urbino (a masterpiece in everything but the
#         Instagram-tourist sense; not on the canonical "big four" list)
#   D25 - Tintoretto
#   D26 - Veronese
HIDDEN_GEM_TRAILS = {
    "italian_trecento": ["A4", "A5"],
    "italian_quattrocento": ["A8", "A9", "A10", "A13", "A18", "A25", "A27", "A28"],
    "italian_cinquecento": ["D4", "D9", "D12", "D14", "D21", "D23", "D25", "D26"],
}


# =============================================================================
# Utility functions
# =============================================================================

def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return the project's seeded NumPy random generator."""

    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)


def sample_visit_duration(entry_slot_index: int, rng: np.random.Generator) -> int:
    """Slot-dependent visit durations calibrated to the Uffizi's published
    itinerary tiers.

    The Uffizi Gallery's official visitor map defines three itineraries:
      Fast Itinerary       90 to 120 minutes (greatest masterpieces only)
      Classic Itinerary   120+ minutes (all Italian/European art, 13C-17C)
      Complete Itinerary  180+ minutes (every room in the gallery)

    We map entry-slot index to a slot-dependent mean duration that reflects
    how dedicated each cohort is. Early-morning arrivals tend to be the
    "Complete" or "Classic" visitors; late-afternoon arrivals are doing
    "Fast" because they are running out of museum hours.

      Slots  0-6  (08:15-10:00): mean 180 min  (Classic/Complete)
      Slots  7-18 (10:00-13:00): mean 150 min  (Classic)
      Slots 19-30 (13:00-16:00): mean 120 min  (between Fast and Classic)
      Slots 31-36 (16:00-17:30): mean 100 min  (Fast)

    Duration is log-normal distributed. The bias correction sigma^2/2
    ensures E[duration] = mean. The floor of 90 enforces the Uffizi's
    own minimum (no realistic visit takes less than 90 min, per the
    map).
    """

    mean_durations = {
        (0, 7): 180,
        (7, 19): 150,
        (19, 31): 120,
        (31, 37): 100,
    }
    mean = 120
    for (lo, hi), candidate in mean_durations.items():
        if lo <= entry_slot_index < hi:
            mean = candidate
            break

    sigma = 0.3
    # Log-normal: E[X] = exp(mu + sigma^2/2), so mu = log(mean) - sigma^2/2.
    mu = np.log(mean) - 0.5 * sigma ** 2
    duration = int(np.exp(rng.normal(mu, sigma)))
    # Floor at the Uffizi's Fast Itinerary minimum (90 min).
    return max(90, duration)


def normalize_weights(weights: Iterable[float]) -> np.ndarray:
    """Normalize non-negative weights, falling back to uniform when needed."""

    arr = np.asarray(list(weights), dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    s = arr.sum()
    if s <= 0:
        return np.full(n, 1.0 / n, dtype=float)
    return arr / s


def capacity_math_check(daily_visitors: int = DAILY_VISITORS_NORMAL) -> Dict[str, float]:
    """Quick deterministic sanity check for expected occupancy pressure.

    Verifies that daily_visitors * avg_duration / museum_open_minutes
    stays below MAX_MUSEUM_CAPACITY, consistent with [A22] calibration.
    """

    slot_means = [
        120 if i < 7 else 100 if i < 19 else 80 if i < 31 else 65 for i in range(N_ENTRY_SLOTS)
    ]
    avg_duration = float(np.mean(slot_means))
    visitor_minutes = daily_visitors * avg_duration
    avg_occupancy = visitor_minutes / MUSEUM_OPEN_MINUTES
    return {
        "daily_visitors": float(daily_visitors),
        "avg_duration_min": avg_duration,
        "visitor_minutes": visitor_minutes,
        "avg_occupancy": avg_occupancy,
        "capacity": float(MAX_MUSEUM_CAPACITY),
        "within_capacity_on_average": float(avg_occupancy < MAX_MUSEUM_CAPACITY),
    }
