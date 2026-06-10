"""Visitor profile definitions and sampling utilities.

This module defines the two-type visitor model that drives both NPC behavior
in the crowd simulator and the controlled agent's preference structure. The
heterogeneity of visitor profiles is the core mechanism through which the
simulation produces realistic, non-uniform crowd patterns.

Two-type model
--------------
Inspired by the Veron & Levasseur (1983) museum visitor taxonomy [VL83]:

  Type A ("art lovers" / VL83 "Butterfly"):
    Heterogeneous importance vectors (each visitor values rooms differently),
    high crowd sensitivity (alpha=6.0), willing to deviate from the default
    path (route_bias=0.55), and actively seeks less-crowded rooms. These
    visitors linger when uncrowded and flee when a room fills up. They are
    the visitors most helped by interventions that redistribute flow.

  Type B ("checkbox tourists" / VL83 "Ant"):
    Deterministic importance vector (all Type B visitors share the same
    preferences: high on magnet rooms, low background everywhere else),
    low crowd sensitivity (alpha=0.5), strong route-following tendency
    (route_bias=0.88), and no crowd-avoidance behavior. They are the
    source of congestion peaks at Botticelli, Leonardo, and Caravaggio.

Why two types instead of a continuous spectrum:
  The two-type model captures the first-order behavioral split (crowd-
  sensitive vs. crowd-indifferent) with minimal parameters while still
  generating the heterogeneous flow patterns observed in real museums.
  Within Type A, continuous heterogeneity comes from the noisy importance
  vector. The 30/70 default split is a modeling assumption; Phase 7
  sensitivity analysis sweeps this fraction from 0% to 100%.

All behavioral parameters are defined in config.py with source annotations.

References
----------
[VL83] Veron & Levasseur (1983). "Ethnographie de l'exposition." BPI,
       Centre Georges Pompidou.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from uffizi_rl import config

# =============================================================================
# Visitor profile dataclass
# =============================================================================


@dataclass
class VisitorProfile:
    """Per-visitor movement and preference parameters used by the simulator.

    Each NPC and the controlled agent carry a VisitorProfile that determines
    how they move through the museum. The crowd simulator reads these fields
    at every step to compute movement probabilities, dwell times, and
    crowd-avoidance behavior.

    Attributes
    ----------
    name : str
        Visitor type label ("A" or "B"). Used for logging and for the
        cross-type externality calculation in type_a_effective_density.
    crowd_alpha : float
        Crowd sensitivity parameter in the reward/utility formula
        r_art = importance / (1 + alpha * density). Higher alpha means
        the visitor loses more utility from crowding. Type A: 6.0,
        Type B: 0.5. See config.py for source annotations.
    route_bias : float
        Probability weight on the "next" room in the recommended itinerary
        when the NPC chooses where to move. Higher values produce more
        deterministic, guidebook-following behavior. Type A: 0.55 (explores
        freely), Type B: 0.88 (follows the route).
    backtrack_prob : float
        Per-step probability that the NPC reverses direction along the
        recommended route. Prevents perfectly deterministic flow. Type A:
        0.05, Type B: 0.02.
    dwell_multiplier : float
        Multiplier on the base dwell time (BASE_DWELL_MINUTES * magnetism).
        Type A: 1.0 (lingers), Type B: 0.3 (quick photo, moves on).
    anti_crowd_bonus : float
        Extra weight toward less-crowded neighbors when choosing the next
        room. Type A: 1.0 (actively avoids crowds), Type B: 0.0 (ignores
        crowd signals entirely).
    importance_vector : np.ndarray
        Length-N_ROOMS vector of per-room importance scores (0.5 to 10.0).
        This is the core preference heterogeneity mechanism. Type A gets a
        noisy perturbation of base importance; Type B gets a deterministic
        vector with high values only on magnet rooms.
    trail_name : str | None
        If the visitor accepted a Hidden Gem Trail, this is the trail key
        (e.g. "colors_of_florence"). Trail rooms get a +2.0 importance
        boost, steering the visitor toward underused galleries.
    segment : str
        Extended visitor segment label for 6-segment analysis variants.
        Defaults to "A". Not used in the core two-type model.
    time_budget_override : int | None
        If set, overrides the slot-based visit duration from
        config.sample_visit_duration. Used by interventions that modify
        visit length (e.g., resident pass short visits).
    """

    name: str
    crowd_alpha: float
    route_bias: float
    backtrack_prob: float
    dwell_multiplier: float
    anti_crowd_bonus: float
    importance_vector: np.ndarray
    trail_name: str | None = None
    segment: str = "A"  # visitor segment label (A, B, or 6-segment names)
    time_budget_override: int | None = None  # overrides slot-based duration if set
    # Rooms where this visitor "stops" rather than transits. Used by the
    # crowd-feedback dwell adjustment in the simulator. Instagram tourists
    # have a small set (masterpieces + Panoramic Terrace). Standard tourists
    # add the Tribune and Caravaggio. Art lovers stop at any room of
    # personal interest; the field defaults to an empty set since their
    # dwell is driven by the importance vector instead.
    magnet_rooms: frozenset[str] = frozenset()
    # Hard per-room dwell targets (room_id -> minutes). When the visitor
    # enters a target room, the simulator overrides the natural stay
    # decay: stay until time_in_room reaches the target, then move.
    # Empty dict means use the magnetism-based decay only.
    magnet_dwell_target: dict = None
    # Fast-transit: visitor blitzes through non-magnet rooms, performing
    # multiple room transitions per simulated minute. Matches the small
    # physical scale of the Uffizi: a corridor traversal that the
    # discrete simulator would otherwise spend 10+ minutes on.
    fast_transit: bool = False
    # Additive dwell floor (minutes of magnetism-equivalent) applied in the
    # stay-probability decay. Art lovers carry a positive floor so they
    # linger even in low-magnetism minor rooms; Instagram and Standard
    # tourists leave it at zero and walk those rooms fast.
    dwell_floor: float = 0.0


# =============================================================================
# Internal helpers
# =============================================================================


def _base_importance_vector() -> np.ndarray:
    """Return the canonical importance vector from config.ROOM_DATA.

    This is the "ground truth" cultural significance of each room (1-10
    scale), used as the starting point for Type A noisy perturbations and
    as-is for Type B (before the magnet/background override).

    Returns
    -------
    np.ndarray
        Shape (N_ROOMS,) with importance scores in room-index order.
    """
    return np.array([config.ROOM_DATA[r]["importance"] for r in config.ROOM_IDS], dtype=float)


def _room_mask(room_ids: List[str]) -> np.ndarray:
    """Build a binary indicator vector for a subset of rooms.

    Used to construct the Hidden Gem Trail importance boost: trail rooms
    get a mask value of 1.0 so that adding 2.0 * mask selectively boosts
    only the trail rooms.

    Parameters
    ----------
    room_ids : List[str]
        Room identifiers to flag (e.g. ["A5", "A7", "A9"]).

    Returns
    -------
    np.ndarray
        Shape (N_ROOMS,) binary mask; 1.0 for rooms in room_ids, 0.0 elsewhere.
    """
    mask = np.zeros(config.N_ROOMS, dtype=float)
    for room in room_ids:
        if room in config.ROOM_TO_IDX:
            mask[config.ROOM_TO_IDX[room]] = 1.0
    return mask


# =============================================================================
# Type A profile sampling
# =============================================================================


def sample_type_a_profile(
    rng: np.random.Generator,
    heterogeneity_scale: float = 1.0,
    trail_acceptance_prob: float = 0.30,
) -> VisitorProfile:
    """Sample a heterogeneous crowd-sensitive Type-A ("art lover") profile.

    Type A visitors are the core beneficiaries of crowd-management
    interventions. Their heterogeneous importance vectors mean that
    different Type A visitors will be drawn to different rooms, naturally
    distributing flow when given information or incentives.

    The importance vector is constructed in two steps:
      1. Start from the base importance (cultural significance from config).
      2. Add Gaussian noise (sigma=1.2 * heterogeneity_scale) to create
         within-type preference diversity. Each Type A visitor has a unique
         "personal taste" that deviates from the guidebook consensus.

    If the visitor accepts a Hidden Gem Trail (30% probability by default),
    the trail rooms receive an additional +2.0 importance boost, making
    underused galleries more attractive to this visitor.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded RNG for reproducibility. Each visitor draws independent noise.
    heterogeneity_scale : float
        Multiplier on the noise sigma. 1.0 = baseline diversity. Higher
        values spread preferences further from the guidebook consensus.
        Sensitivity analysis in Phase 7 varies this parameter.
    trail_acceptance_prob : float
        Probability that this visitor accepts a Hidden Gem Trail offer.
        [assumption] Default 0.30. Acceptance causes a +2.0 importance
        boost on trail rooms, steering the visitor toward underused galleries.

    Returns
    -------
    VisitorProfile
        A fully parameterized Type A profile with a unique importance vector.
    """

    base = _base_importance_vector()

    # Type A importance vectors are heterogeneous: each visitor values
    # rooms differently, drawn from a noisy perturbation of the base
    # importance scores. Base sigma=1.2 produces ~1-point shifts on the
    # 1-10 importance scale, enough to create meaningful preference
    # diversity without overwhelming the base structure. [assumption]
    noise = rng.normal(loc=0.0, scale=1.2 * heterogeneity_scale, size=config.N_ROOMS)
    imp = np.clip(base + noise, 0.5, 10.0)  # clamp to valid range

    # --- Hidden Gem Trail acceptance mechanism ---
    # With probability trail_acceptance_prob, the visitor picks up one of the
    # themed alternative itineraries (e.g. "colors_of_florence"). This raises
    # the importance of trail rooms by +2.0, making them competitive with
    # magnet rooms. The mechanism redistributes Type A demand toward
    # underused galleries without forcing movement.
    trail_name = None
    if rng.random() < trail_acceptance_prob:
        trail_name = rng.choice(list(config.HIDDEN_GEM_TRAILS.keys())).item()
        trail_rooms = config.HIDDEN_GEM_TRAILS[trail_name]
        trail_mask = _room_mask(trail_rooms)
        # Trail rooms get a +2.0 importance boost. [assumption]
        imp = np.clip(imp + 2.0 * trail_mask, 0.5, 10.0)

    # Masterpiece dwell targets are identical across all profiles because
    # the slot gate caps everyone at the same time there (10 min). The art
    # lover's extra time comes from a long visit budget and a dwell floor
    # that makes them linger in the NON-masterpiece rooms (per their noisy
    # importance vector), where rushed tourists do not. [user testimony]
    art_lover_dwell_target = {
        "A11": 10,
        "A12": 1,
        "A35": 10,
        "A38": 10,
        "PANORAMIC_TERRACE": 15,  # quick view on the way down, not a long selfie stop
    }
    # Art lovers can spend most of the day in the museum. They go through
    # the whole second floor and then the whole first floor (B, C and the
    # 16th-century D rooms) before exiting via the Magliabechi staircase.
    budget = int(rng.integers(config.ART_LOVER_VISIT_BUDGET[0],
                              config.ART_LOVER_VISIT_BUDGET[1] + 1))
    return VisitorProfile(
        name="A",
        crowd_alpha=config.TYPE_A_CROWD_ALPHA,
        route_bias=config.TYPE_A_ROUTE_BIAS,
        backtrack_prob=config.TYPE_A_BACKTRACK_PROBABILITY,
        dwell_multiplier=config.TYPE_A_DWELL_MULTIPLIER,
        anti_crowd_bonus=config.TYPE_A_ANTI_CROWD_BONUS,
        importance_vector=imp,
        trail_name=trail_name,
        segment="art_lover",
        time_budget_override=budget,
        magnet_rooms=frozenset(),  # art lovers stop wherever their personal importance pulls them
        magnet_dwell_target=art_lover_dwell_target,
        dwell_floor=config.ART_LOVER_DWELL_FLOOR,
    )


# =============================================================================
# Type B profile sampling
# =============================================================================


def sample_type_b_profile(rng: np.random.Generator) -> VisitorProfile:
    """Return the deterministic checklist-style Type-B ("checkbox tourist") profile.

    Type B visitors are the primary source of congestion at magnet rooms
    (Botticelli, Leonardo, Caravaggio). Their behavior is modeled as
    deterministic because the "top 10 must-see" checklist is shared across
    guidebooks, social media, and tour operators, producing near-identical
    preferences within this segment.

    Unlike Type A, there is zero within-type heterogeneity: every Type B
    visitor has the same importance vector. This is a deliberate modeling
    choice. It means that without interventions, all Type B visitors
    converge on the same rooms at the same times, creating the realistic
    congestion peaks observed at the Uffizi.

    The importance vector is binary-like:
      - Magnet rooms (A11, A12, A35, A38, A16, E4, E5): importance = 9.5
      - All other rooms: background importance = 0.7
    The 9.5/0.7 ratio ensures that Type B visitors are strongly pulled
    toward magnet rooms but still move through other rooms (they don't
    teleport). The low background value means they traverse non-magnet
    rooms quickly (dwell_multiplier=0.3) without stopping.

    Parameters
    ----------
    rng : np.random.Generator
        Accepted for interface consistency with sample_type_a_profile but
        unused. Type B importance is deterministic.

    Returns
    -------
    VisitorProfile
        The canonical Type B profile (identical for all Type B visitors).
    """

    # rng accepted for interface consistency with sample_type_a_profile
    # but unused: Type B importance is deterministic. High for magnet rooms, low
    # background for everything else. Zero within-type heterogeneity
    # reflects the modeling assumption that checkbox tourists share
    # near-identical preferences driven by guidebook consensus.
    base = np.full(config.N_ROOMS, config.TYPE_B_BACKGROUND_IMPORTANCE, dtype=float)
    for room in config.TYPE_B_MAGNET_ROOMS:
        base[config.ROOM_TO_IDX[room]] = config.TYPE_B_MAGNET_IMPORTANCE  # 9.5 for magnet rooms

    # Masterpiece dwell targets identical to IG so all visitor types
    # spend the SAME time at each masterpiece (5 min A11 + 5 min A12 =
    # 10 min Botticelli, 10 min Leonardo, 10 min Raphael). This makes
    # the slot cycle synchronous across visitor types - Spring and
    # Venus track identically because everyone flushes A11 -> A12 at
    # the same minute. [user testimony]
    standard_dwell_target = {
        "A11": 10,
        "A12": 1,
        "A35": 10,
        "A38": 10,
        "PANORAMIC_TERRACE": 10,  # quick view on the way down, not a long selfie stop
    }
    # The standard tourist spends a half-to-full afternoon: the whole
    # second floor, then a fast pass of the first floor (Self-Portraits,
    # the 16th-century corridor) ending at Caravaggio, out via Magliabechi.
    budget = int(rng.integers(config.STANDARD_VISIT_BUDGET[0],
                              config.STANDARD_VISIT_BUDGET[1] + 1))
    return VisitorProfile(
        name="B",
        crowd_alpha=config.TYPE_B_CROWD_ALPHA,        # 0.5: nearly crowd-indifferent
        route_bias=config.TYPE_B_ROUTE_BIAS,           # 0.88: strongly follows guidebook route
        backtrack_prob=config.TYPE_B_BACKTRACK_PROBABILITY,  # 0.02: rarely backtracks
        dwell_multiplier=config.TYPE_B_DWELL_MULTIPLIER,     # 0.3: quick photo, moves on
        anti_crowd_bonus=config.TYPE_B_ANTI_CROWD_BONUS,     # 0.0: ignores crowd signals
        importance_vector=base,
        trail_name=None,  # Type B never accepts alternative trails
        segment="standard",
        time_budget_override=budget,
        magnet_rooms=frozenset(config.STANDARD_MAGNET_ROOMS),
        magnet_dwell_target=standard_dwell_target,
    )


# =============================================================================
# Instagram tourist profile (selfie-driven, 30-60 min visits)
# =============================================================================


def sample_instagram_profile(rng: np.random.Generator) -> VisitorProfile:
    """Sample an Instagram-tourist profile (30-60 min visit, masterpieces only).

    The majority of real Uffizi visitors. Drives straight from the entrance
    to Botticelli, then Leonardo and Raphael/Michelangelo, then up to the
    panoramic terrace for selfies and a drink, then out via Lanzi. Does not
    visit Tribune, the forced 15th-century chain, the U-loop, the dead-end
    side rooms, or anything on the 1st floor besides what is on the way out.

    Behavioral parameters are tuned for very strict route adherence and
    fast pass-through of non-magnet rooms:
      - route_bias 0.95: the recommended route is followed nearly always
      - dwell_multiplier 0.2: minimal time in non-magnet rooms
      - anti_crowd_bonus 0.0: ignores crowds (they expect them)
      - magnet_rooms = {A11, A12, A35, A38, PANORAMIC_TERRACE}
      - time_budget sampled uniformly in [30, 60] min

    The importance vector is bimodal: 0.3 background, 10.0 on magnets,
    making the importance pull effectively zero everywhere except the
    Instagram destinations.
    """

    # Bimodal importance: nearly zero background, full pull on magnets.
    imp = np.full(config.N_ROOMS, 0.3, dtype=float)
    for room in config.INSTAGRAM_MAGNET_ROOMS:
        if room in config.ROOM_TO_IDX:
            imp[config.ROOM_TO_IDX[room]] = 10.0

    # 60-90 minute visit. The IG tourist blitzes from entrance to the
    # four masterpieces (10 min each at A11+A12, A35, A38) and up to
    # the terrace (30 min). Fast-transit through non-magnet rooms
    # (multiple room transitions per simulated minute) makes the
    # walking essentially free, matching the small physical footprint
    # of the Uffizi. [user testimony]
    budget = int(rng.integers(config.INSTAGRAM_VISIT_BUDGET[0],
                              config.INSTAGRAM_VISIT_BUDGET[1] + 1))

    # Per-room dwell targets. Bott zone is modeled as A11 alone holding
    # the visitor the full 10-min slot (Spring + Venus paintings hang in
    # both rooms, but the experience is one continuous "Botticelli
    # moment"). A12 is a 1-min walk-through. This lets A11 saturate at
    # cap throughout the day rather than oscillate 0->55 in the slot
    # cycle. For chart symmetry, A12's occupancy series is replaced
    # with A11's at plot time.
    dwell_target = {
        "A11": 10,  # Botticelli zone (Spring + Venus together)
        "A12": 1,   # walk-through room
        "A35": 10,  # Leonardo
        "A38": 10,  # Raphael / Michelangelo
        "PANORAMIC_TERRACE": 30,
    }

    return VisitorProfile(
        name="B",                  # treated as non-Type-A for legacy code paths
        crowd_alpha=0.3,           # almost zero crowd sensitivity
        route_bias=0.95,           # strict route adherence
        backtrack_prob=0.01,       # almost never backtracks
        dwell_multiplier=0.2,      # very fast in non-magnet rooms
        anti_crowd_bonus=0.0,      # expects crowds, doesn't avoid
        importance_vector=imp,
        trail_name=None,
        segment="instagram",
        time_budget_override=budget,
        magnet_rooms=frozenset(config.INSTAGRAM_MAGNET_ROOMS),
        magnet_dwell_target=dwell_target,
        fast_transit=True,
    )


# =============================================================================
# Mixture sampling
# =============================================================================


def sample_profile(
    rng: np.random.Generator,
    type_a_fraction: float = config.TYPE_A_FRACTION_DEFAULT,
    heterogeneity_scale: float = 1.0,
    trail_acceptance_prob: float = 0.30,
) -> VisitorProfile:
    """Sample a visitor profile from the Type A / Type B mixture distribution.

    This is the main entry point for creating visitor profiles. The crowd
    simulator calls this once per NPC at arrival time, and the RL environment
    calls it once per episode reset for the controlled agent.

    The mixture weight type_a_fraction determines the expected proportion
    of Type A visitors in the museum. Default: 30% Type A, 70% Type B
    (see config.py). Sensitivity analysis sweeps this parameter.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded RNG. The same RNG is used both for the type coin flip and
        (if Type A) for the importance noise and trail acceptance.
    type_a_fraction : float
        Probability of drawing a Type A profile. Default 0.30. [assumption]
    heterogeneity_scale : float
        Passed through to sample_type_a_profile. Controls within-Type-A
        preference diversity.
    trail_acceptance_prob : float
        Passed through to sample_type_a_profile. Controls the probability
        that a Type A visitor accepts a Hidden Gem Trail.

    Returns
    -------
    VisitorProfile
        Either a Type A (heterogeneous, crowd-sensitive) or Type B
        (deterministic, checklist-style) profile.
    """

    # Three-tier sampling: Art Lover (10%), Standard tourist (30%),
    # Instagram tourist (60%). The Standard / Instagram split sits inside
    # what legacy code treats as "Type B"; both have ``name="B"`` but
    # differ in route_bias, dwell, time budget, and magnet rooms.
    draw = rng.random()
    if draw < type_a_fraction:
        return sample_type_a_profile(
            rng,
            heterogeneity_scale=heterogeneity_scale,
            trail_acceptance_prob=trail_acceptance_prob,
        )
    # Within the non-Art-Lover pool, decide Instagram vs Standard.
    # Default mix: 60% Instagram, 30% Standard out of 90% total Type B.
    # Conditional probability of Instagram given Type B = 0.60 / 0.90.
    nonA_total = max(1e-6, 1.0 - type_a_fraction)
    instagram_conditional = config.INSTAGRAM_FRACTION / nonA_total
    if rng.random() < instagram_conditional:
        return sample_instagram_profile(rng)
    return sample_type_b_profile(rng)


# =============================================================================
# Cross-type externality
# =============================================================================


def type_a_effective_density(occ_a: float, occ_b: float, cap: float) -> float:
    """Compute the effective density perceived by a Type A visitor.

    Type A visitors perceive Type B visitors as disproportionately more
    disruptive than fellow Type A visitors. This captures the real-world
    asymmetry: large tour groups with cameras, selfie sticks, and loud
    guides impose negative externalities on contemplative visitors that
    exceed their mere headcount.

    The cross-type externality factor (default 1.5) means that each Type B
    visitor counts as 1.5 Type A visitors in the density calculation. For
    example, in a room with 10 Type A and 20 Type B visitors at capacity
    100, the raw density is 0.30 but the effective density perceived by
    Type A is (10 + 1.5 * 20) / 100 = 0.40.

    This asymmetry is one-directional: Type B visitors do not perceive
    Type A as more disruptive (they are crowd-indifferent anyway). The
    factor is defined in config.TYPE_A_CROSS_TYPE_EXTERNALITY. [assumption]

    Parameters
    ----------
    occ_a : float
        Number of Type A visitors currently in the room.
    occ_b : float
        Number of Type B visitors currently in the room.
    cap : float
        Room capacity (comfortable maximum occupancy).

    Returns
    -------
    float
        Effective density in [0, inf). Values above 1.0 indicate perceived
        overcrowding beyond comfortable capacity.
    """

    # Weight Type B headcount by the cross-type externality factor (1.5x)
    # before computing density. max(1.0, cap) prevents division by zero
    # for rooms with erroneously zero capacity.
    return (occ_a + config.TYPE_A_CROSS_TYPE_EXTERNALITY * occ_b) / max(1.0, cap)
