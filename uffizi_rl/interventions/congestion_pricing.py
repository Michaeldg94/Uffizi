"""Congestion pricing intervention: reservation-gated access to Botticelli rooms.

Problem
-------
The Botticelli rooms (A11 "The Spring", A12 "Birth of Venus") are the single
largest bottleneck in the Uffizi.  Virtually every visitor wants to see them
(importance = 10, the maximum), and their position on a forced chain
(A10 -> A11 -> A12 -> A13) means visitors cannot bypass them.  The result is
heavy queuing and degraded experience quality for the entire gallery.

Why a quantity instrument, not a price?
---------------------------------------
A Pigouvian tax (congestion surcharge) would set a price and let the market
determine quantity.  But willingness-to-pay for the Botticelli rooms is
extremely inelastic: most visitors traveled to Florence specifically for
these paintings and will pay any surcharge short of their entire trip budget.
A price instrument would need to be implausibly high to deter enough visitors,
and would raise equity concerns (only the wealthy see Botticelli).

Instead, this module uses a *quantity instrument*: the ``ReservationBook``
allocates a fixed number of slots per 15-minute window (default 25).  This
directly controls occupancy, guarantees that no window exceeds comfortable
capacity, and treats all visitors equally on a first-come-first-served basis.
The approach is analogous to timed-entry tickets at the Accademia Gallery
or the Sistine Chapel, which also use quantity rationing rather than price.

Booking decision model
----------------------
Not all visitors accept the friction of reserving a specific window.  The
``booking_decision`` function models this as a simple threshold comparison:
a visitor books if (importance_botticelli - queue_penalty) >= threshold.
Visitors with high intrinsic interest in Botticelli and low aversion to
queuing are more likely to book. [assumption]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import numpy as np

from uffizi_rl import config

# =============================================================================
# Constants
# =============================================================================

# Only the Botticelli rooms are reservation-gated (not all magnet rooms).
# A11 houses "Primavera" (The Spring); A12 houses "Birth of Venus."
# Other high-demand rooms (A35 Leonardo, A38 Raphael, E4 Caravaggio) are
# not gated because they are not on a forced chain and have natural
# alternative routes.
BOTTICELLI_ROOMS = {"A11", "A12"}

# =============================================================================
# Data container
# =============================================================================

@dataclass
class Reservation:
    """One Botticelli reservation assignment.

    Fields
    ------
    visitor_id : int
        Unique visitor identifier.
    slot_index : int
        The 15-minute window this reservation is valid for (0-36).
    """

    visitor_id: int
    slot_index: int


# =============================================================================
# Reservation book (quantity instrument)
# =============================================================================

class ReservationBook:
    """Manages limited per-slot reservation capacity for Botticelli rooms.

    This is the core quantity instrument.  Rather than setting a price and
    letting demand determine occupancy, the book fixes the maximum number
    of visitors per 15-minute window (``per_slot_capacity``) and allocates
    reservations on a first-come-first-served basis.

    With 25 visitors per slot and 37 slots, the daily capacity is
    25 * 37 = 925, close to the museum's 900-person legal limit [UFF].
    This ensures that even if every visitor books, the aggregate flow
    remains below the building's fire-safety constraint. [assumption]
    """

    def __init__(self, per_slot_capacity: int = 25, seed: int = config.DEFAULT_SEED):
        # 25 visitors per 15-min slot = 925 total daily reservation slots,
        # close to the museum's 900 overall capacity. [assumption]
        self.per_slot_capacity = int(per_slot_capacity)
        self.rng = config.get_rng(seed)
        # bookings maps slot_index -> list of visitor_ids who hold a
        # reservation for that window.  defaultdict(list) avoids KeyError
        # when querying unbooked slots.
        self.bookings: Dict[int, list[int]] = defaultdict(list)

    def book(self, visitor_id: int, preferred_slot: int | None = None) -> Reservation | None:
        """Try to book a Botticelli reservation for ``visitor_id``.

        Algorithm:
        1. If no preferred slot is given, draw one uniformly at random.
        2. Sort all 37 slots by distance from the preferred slot (nearest
           first) so the visitor gets the closest available window.
        3. Walk the sorted list; accept the first slot with remaining
           capacity.  If all 925 daily slots are full, return None.

        Returns
        -------
        Reservation or None
            The confirmed reservation, or None if the day is fully booked.
        """
        if preferred_slot is None:
            preferred_slot = int(self.rng.integers(0, config.N_ENTRY_SLOTS))

        # Try preferred slot first, then expand outward to nearest available.
        candidate_slots = sorted(
            range(config.N_ENTRY_SLOTS), key=lambda s: abs(s - preferred_slot)
        )
        for s in candidate_slots:
            if len(self.bookings[s]) < self.per_slot_capacity:
                self.bookings[s].append(visitor_id)
                return Reservation(visitor_id=visitor_id, slot_index=s)
        # All slots full; visitor cannot book Botticelli today.
        return None

    def can_enter(self, visitor_id: int, room_id: str, minute: int, room_occ: int, room_cap: int) -> bool:
        """Check whether ``visitor_id`` may enter ``room_id`` at ``minute``.

        Non-Botticelli rooms always return True (no reservation required).
        For Botticelli rooms, entry is allowed only if:
          1. The room is not at physical capacity (room_occ < room_cap).
          2. The visitor holds a reservation for the current 15-minute slot.
        """
        if room_id not in BOTTICELLI_ROOMS:
            return True
        if room_occ >= room_cap:
            return False

        # Convert continuous minute to discrete slot index.
        slot = minute // config.ENTRY_SLOT_MINUTES
        # Check if this visitor's ID appears in the booking list for this slot.
        return visitor_id in self.bookings.get(slot, [])

    def slot_loads(self) -> np.ndarray:
        """Return an array of shape (N_ENTRY_SLOTS,) with booking counts per slot.

        Useful for diagnostics and visualization: shows how reservations
        are distributed across the day.
        """
        arr = np.zeros(config.N_ENTRY_SLOTS, dtype=int)
        for s, ids in self.bookings.items():
            arr[s] = len(ids)
        return arr


# =============================================================================
# Booking decision model
# =============================================================================

def booking_decision(
    importance_botticelli: float,
    queue_penalty: float,
    willingness_threshold: float = 2.0,
) -> bool:
    """Decide whether a visitor accepts the friction of reserving a Botticelli slot.

    The decision is a simple threshold comparison:
      book = (importance_botticelli - queue_penalty) >= willingness_threshold

    - ``importance_botticelli``: how much this visitor values seeing Botticelli
      (higher = more willing to book).
    - ``queue_penalty``: the visitor's perceived cost of committing to a fixed
      time window (higher = less willing to book).
    - ``willingness_threshold``: net value required for the visitor to bother
      booking.  Default 2.0. [assumption]

    Type B ("checkbox tourist") visitors have importance ~9.5 and low queue
    sensitivity, so they almost always book.  Type A ("art lover") visitors
    have variable importance; some will skip Botticelli rather than commit
    to a rigid schedule.
    """

    return (importance_botticelli - queue_penalty) >= willingness_threshold
