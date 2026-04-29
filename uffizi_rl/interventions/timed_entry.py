"""Timed-entry intervention: stagger visitor arrivals across the 37 available
15-minute booking slots to reduce peak-hour concentration.

Problem
-------
Without intervention, visitor arrivals at the Uffizi cluster heavily between
10:00 and 13:00, producing a pronounced midday peak that pushes occupancy
toward the 900-person legal capacity [UFF].  The timed-entry system spreads
arrivals more evenly across the day while respecting the empirical preference
for morning visits.

Slot assignment algorithm
-------------------------
1. **Morning-weighted Gaussian**: a truncated Gaussian centered at slot 10
   (~10:45 AM) with width 8 slots (~2 hours).  A floor of 0.35 ensures that
   late-afternoon slots still receive meaningful allocation rather than
   going empty.  The resulting weight vector is normalized to a probability
   distribution over the 37 slots. [assumption]

2. **Deterministic stratification**: each visitor is assigned a base slot by
   inverting the CDF of the weight distribution at the visitor's quantile
   position ``(visitor_index + 1) / (daily_total + 1)``.  This guarantees
   that visitors are spread across the distribution evenly, avoiding the
   random clumping that a purely stochastic draw would produce.

3. **Local jitter**: after stratification, the slot is perturbed by
   ``rng.integers(-1, 2)`` (i.e. -1, 0, or +1 slot) and clipped to valid
   bounds.  This adds realism (visitors do not arrive at exact 15-minute
   marks) without undoing the anti-spike smoothing from step 2.

East/west route alternation
---------------------------
Even-numbered slots suggest an "east_first" route; odd-numbered slots
suggest "west_first".  The idea is that consecutive cohorts of arriving
visitors are steered toward opposite wings of the gallery, halving the
instantaneous load on any single entry corridor.  This is a soft
suggestion (NPCs may deviate based on their profile's ``route_bias``),
not a hard constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from uffizi_rl import config

# =============================================================================
# Data container
# =============================================================================

@dataclass
class TimedEntryAssignment:
    """One visitor's timed-entry slot assignment.

    Fields
    ------
    visitor_index : int
        Ordinal position of this visitor in the day's arrival sequence.
    slot_index : int
        Assigned 15-minute booking slot (0 = 08:15, 36 = 17:15).
    entry_time : int
        Slot start time in minutes after museum opening (slot_index * 15).
    suggested_route : str
        "east_first" or "west_first", alternated by slot parity.
    """

    visitor_index: int
    slot_index: int
    entry_time: int
    suggested_route: str


# =============================================================================
# Slot assignment algorithm
# =============================================================================

def assign_slot(visitor_index: int, daily_total: int, rng: np.random.Generator) -> int:
    """Assign one visitor to a booking slot using stratified Gaussian sampling.

    Steps:
    1. Build a morning-weighted probability vector over 37 slots.
    2. Place the visitor deterministically at its quantile position in
       the CDF of that distribution (stratification).
    3. Apply uniform jitter of +/-1 slot for realism.

    Parameters
    ----------
    visitor_index : int
        This visitor's position in the day (0 to daily_total-1).
    daily_total : int
        Total visitors expected for the day.
    rng : np.random.Generator
        Seeded generator for the jitter draw.

    Returns
    -------
    int
        Slot index in [0, N_ENTRY_SLOTS - 1].
    """

    slots = np.arange(config.N_ENTRY_SLOTS)

    # --- Step 1: Morning-weighted Gaussian distribution ---------------------
    # Gaussian peak at slot 10 (~10:45 AM), sigma = 8 slots (~2 hours).
    # The +0.35 floor prevents late-afternoon slots from having near-zero
    # probability; without it, slots 30+ would be almost empty and the
    # museum would be underutilized in the last two hours.  [assumption]
    weights = np.exp(-0.5 * ((slots - 10.0) / 8.0) ** 2) + 0.35
    weights = weights / weights.sum()

    # --- Step 2: Deterministic stratification -------------------------------
    # Map visitor_index to a quantile in (0, 1) via (index+1)/(total+1).
    # np.searchsorted on the CDF finds the slot whose cumulative weight
    # first exceeds that quantile.  This is the inverse-CDF (quantile)
    # method applied deterministically, guaranteeing that visitors are
    # spread across the distribution without random clumping.
    base = int(np.searchsorted(np.cumsum(weights), (visitor_index + 1) / (daily_total + 1)))

    # --- Step 3: Local jitter -----------------------------------------------
    # Perturb by -1, 0, or +1 slots.  Adds realistic noise without undoing
    # the anti-spike smoothing from stratification.
    jitter = int(rng.integers(-1, 2))
    slot = int(np.clip(base + jitter, 0, config.N_ENTRY_SLOTS - 1))
    return slot


# =============================================================================
# Per-visitor intervention
# =============================================================================

def timed_entry_intervention(
    visitor_index: int,
    daily_total: int,
    rng: np.random.Generator,
) -> Tuple[int, str, int]:
    """Assign an entry time, suggested route, and slot index to one visitor.

    Returns
    -------
    entry_time : int
        Minutes after museum opening when this visitor should enter.
    suggested_route : str
        "east_first" or "west_first", alternated by slot parity to split
        the arriving cohort across both gallery wings.
    slot_index : int
        The assigned slot (for record-keeping and reservation systems).
    """

    slot_index = assign_slot(visitor_index, daily_total, rng)
    # Convert slot index to minutes: each slot spans ENTRY_SLOT_MINUTES (15 min).
    entry_time = slot_index * config.ENTRY_SLOT_MINUTES

    # Alternate route suggestion by slot parity: even slots go east first,
    # odd slots go west first.  This halves the instantaneous load on each
    # wing's entry corridor.
    suggested_route = "east_first" if slot_index % 2 == 0 else "west_first"
    return entry_time, suggested_route, slot_index


# =============================================================================
# Whole-day batch assignment
# =============================================================================

def assign_day(
    daily_total: int,
    seed: int = config.DEFAULT_SEED,
) -> List[TimedEntryAssignment]:
    """Assign timed-entry slots for every visitor in a simulated day.

    Iterates over ``daily_total`` visitors, calling
    ``timed_entry_intervention`` for each one.  The single shared ``rng``
    ensures that jitter draws are sequentially seeded (reproducible).

    Parameters
    ----------
    daily_total : int
        Number of visitors to assign.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[TimedEntryAssignment]
        One assignment per visitor, in arrival order.
    """

    rng = config.get_rng(seed)
    out: List[TimedEntryAssignment] = []
    for i in range(daily_total):
        entry_time, route, slot = timed_entry_intervention(i, daily_total, rng)
        out.append(
            TimedEntryAssignment(
                visitor_index=i,
                slot_index=slot,
                entry_time=entry_time,
                suggested_route=route,
            )
        )
    return out
