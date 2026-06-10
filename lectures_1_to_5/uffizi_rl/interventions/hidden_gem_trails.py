"""Hidden-gem trail intervention: preference reshaping via themed itineraries.

How trails reshape preferences
------------------------------
Each trail (e.g. "Colors of Florence", "Sacred and Profane", "Faces of
Power") is a curated set of 4-6 lesser-known rooms tied together by a
narrative theme (see ``config.HIDDEN_GEM_TRAILS`` for the full map).
When a visitor accepts a trail, two things change in their preference
profile:

1. **Hidden-gem boost**: the importance of each trail room is increased
   by ``boost`` (default 2.0), making the visitor actively seek out rooms
   they would otherwise walk past.  The boost is additive and clipped to
   [0, 10] to stay within the importance scale.

2. **Magnet room dampening**: the importance of canonical magnet rooms
   (A11, A12, A35, A38, A16, E4, E5) is multiplied by
   ``magnet_room_dampen`` (default 0.85), a 15 % reduction.  This does
   not make visitors *avoid* Botticelli or Caravaggio; it simply makes
   the pull slightly weaker, so visitors are more willing to detour
   through trail rooms rather than rushing straight to the magnets.
   The net effect is a redistribution of dwell time from bottleneck
   rooms to underused galleries. [assumption]

Additionally, trail followers have their ``route_bias`` reduced by 0.05,
making them slightly more willing to deviate from the recommended
itinerary.  This reflects the psychological effect of the trail: once
you are "on a mission" to find hidden gems, you pay less attention to
the standard arrows on the wall.

Acceptance model
----------------
Not every visitor accepts a trail offer.  ``assign_trail`` draws a
Bernoulli with probability ``acceptance_prob`` (default 0.30).  This
value is also used in ``visitor_profiles.sample_type_a_profile``, where
trail acceptance is baked into the profile at creation time.  This module
provides a standalone interface for intervention experiments that need
to vary the acceptance rate independently. [assumption]

Note: trail assignment is also partially integrated into visitor_profiles.py
(via trail_acceptance_prob in sample_type_a_profile). This module provides
a standalone, more configurable interface for intervention experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from uffizi_rl import s02_config as config
from uffizi_rl.environment.s04_visitor_profiles import VisitorProfile

# =============================================================================
# Data container
# =============================================================================

@dataclass
class TrailAssignment:
    """Outcome of offering a hidden-gem trail to a visitor.

    Fields
    ------
    trail_name : str
        Name of the assigned trail (e.g. "colors_of_florence"), or "none"
        if the visitor declined.
    accepted : bool
        Whether the visitor accepted the trail offer.
    """

    trail_name: str
    accepted: bool


# =============================================================================
# Trail assignment
# =============================================================================

def assign_trail(
    rng: np.random.Generator,
    acceptance_prob: float = 0.30,
) -> TrailAssignment:
    """Offer a random hidden-gem trail and model the visitor's accept/reject.

    With probability ``acceptance_prob`` (default 0.30 [assumption]), the
    visitor accepts.  If accepted, one of the trails defined in
    ``config.HIDDEN_GEM_TRAILS`` is chosen uniformly at random.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded generator for reproducibility.
    acceptance_prob : float
        Probability that the visitor accepts the trail offer.

    Returns
    -------
    TrailAssignment
        Contains the trail name and acceptance flag.
    """

    accepted = bool(rng.random() < acceptance_prob)
    if not accepted:
        return TrailAssignment(trail_name="none", accepted=False)
    # Uniformly sample one trail from the available set.
    # .item() converts the numpy string to a Python str.
    trail_name = rng.choice(list(config.HIDDEN_GEM_TRAILS.keys())).item()
    return TrailAssignment(trail_name=trail_name, accepted=True)


# =============================================================================
# Preference reshaping
# =============================================================================

def apply_trail_to_profile(
    profile: VisitorProfile,
    trail_name: str,
    boost: float = 2.0,
    magnet_room_dampen: float = 0.85,
) -> VisitorProfile:
    """Apply a trail to a visitor profile, returning a *new* profile.

    This is the core preference-reshaping function.  It modifies two
    components of the visitor's importance vector:

    1. **Hidden-gem boost** (additive): each room on the trail gets
       ``+boost`` importance, clipped to [0, 10].  This makes the visitor
       actively seek out rooms they would otherwise ignore.

    2. **Magnet room dampening** (multiplicative): canonical magnet rooms
       (defined in ``config.TYPE_B_MAGNET_ROOMS``) have their importance
       multiplied by ``magnet_room_dampen`` (0.85 = 15 % reduction).
       This weakens the gravitational pull of Botticelli, Caravaggio,
       etc., making the visitor more willing to detour through trail
       rooms instead of rushing to the magnets.

    Additionally, ``route_bias`` is reduced by 0.05 (floored at 0.2) to
    reflect the trail follower's greater willingness to deviate from the
    standard recommended path.

    Parameters
    ----------
    profile : VisitorProfile
        The original visitor profile (not mutated).
    trail_name : str
        Name of the trail to apply.  "none" is a no-op.
    boost : float
        Additive importance increase for trail rooms. [assumption]
    magnet_room_dampen : float
        Multiplicative factor for magnet room importance. [assumption]

    Returns
    -------
    VisitorProfile
        A new profile with the modified importance vector and route bias.
    """

    # No-op for visitors who did not accept a trail.
    if trail_name == "none":
        return profile

    # --- Step 1: Boost hidden-gem rooms -------------------------------------
    vec = profile.importance_vector.copy()
    for room in config.HIDDEN_GEM_TRAILS.get(trail_name, []):
        # Additive boost, clipped to the [0, 10] importance scale.
        vec[config.ROOM_TO_IDX[room]] = np.clip(vec[config.ROOM_TO_IDX[room]] + boost, 0.0, 10.0)

    # --- Step 2: Dampen canonical magnet rooms ------------------------------
    # Multiplicative reduction (0.85 = 15 % dampening).  This does not make
    # visitors avoid these rooms; it just weakens the pull enough that the
    # trail rooms become competitive alternatives.
    for room in config.TYPE_B_MAGNET_ROOMS:
        idx = config.ROOM_TO_IDX[room]
        vec[idx] *= magnet_room_dampen

    # --- Step 3: Reduce route-following tendency ----------------------------
    # Trail followers are "on a mission" and pay less attention to the
    # standard arrows.  The 0.05 reduction is mild; floored at 0.2 to
    # prevent completely random navigation.
    adjusted_route_bias = max(0.2, profile.route_bias - 0.05)

    return VisitorProfile(
        name=profile.name,
        crowd_alpha=profile.crowd_alpha,
        route_bias=adjusted_route_bias,
        backtrack_prob=profile.backtrack_prob,
        dwell_multiplier=profile.dwell_multiplier,
        anti_crowd_bonus=profile.anti_crowd_bonus,
        importance_vector=vec,
        trail_name=trail_name,
    )


# =============================================================================
# Diagnostic utility
# =============================================================================

def trail_load_vector() -> Dict[str, np.ndarray]:
    """Return one-hot room indicator vectors for each hidden-gem trail.

    Useful for analysis and visualization: multiplying a trail's load
    vector by observed room occupancies gives the aggregate trail-room
    occupancy, showing how much demand the trail redirected.

    Returns
    -------
    Dict[str, np.ndarray]
        Maps trail name to a binary vector of shape (N_ROOMS,) where
        1.0 marks rooms on that trail.
    """

    out: Dict[str, np.ndarray] = {}
    for name, rooms in config.HIDDEN_GEM_TRAILS.items():
        vec = np.zeros(config.N_ROOMS, dtype=float)
        for room in rooms:
            vec[config.ROOM_TO_IDX[room]] = 1.0
        out[name] = vec
    return out
