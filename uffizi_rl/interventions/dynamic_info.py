"""Dynamic information display intervention.

Models real-time crowd information kiosks at corridor junctions (A2, A15,
A24, D9). These are the four major decision points in the museum where
visitors choose between routes, so placing kiosks there gives visitors
the information they need exactly when they need it.

Visibility model
----------------
Without kiosks, a visitor can only observe the density of their current
room and its immediate graph neighbors (rooms connected by a door or
corridor). This is realistic: you can see how crowded the room you are
in is, and you can glance into adjacent rooms, but you have no idea
what the Botticelli rooms look like from the other side of the museum.

When a visitor stands at a kiosk room, the kiosk displays real-time
occupancy for ALL rooms in the museum. The visitor's observation vector
switches from partial to full information. Unseen rooms (when not at a
kiosk) receive ``UNKNOWN_DENSITY_SENTINEL`` (-1.0) in the observation
vector, signaling to the RL agent that the value is missing, not zero.

WHY -1.0 and not 0.0: density values are non-negative (0.0 = empty room).
Using 0.0 for "unknown" would be indistinguishable from "empty," causing
the agent to falsely believe unseen rooms are uncrowded. The sentinel
value -1.0 is outside the valid range and can be handled explicitly by
the policy network (e.g., masked out or replaced with a prior).

Oscillation risk
----------------
A known failure mode of information provision is "demand oscillation":
all visitors simultaneously learn that room X is crowded, so they all
avoid it; room X empties; next period everyone rushes back. The
``oscillation_score`` metric detects this pathology by measuring the
mean absolute first difference in density across timesteps. High
oscillation indicates that information provision is creating herd
behavior rather than smooth redistribution.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from uffizi_rl import config


# =============================================================================
# Visibility computation
# =============================================================================


def visible_rooms(current_room: str, neighbors: Iterable[str], kiosk_rooms=None) -> set[str]:
    """Return the rooms visible from the current room under kiosk rules.

    Visibility rules:
      - Always visible: the visitor's current room.
      - Always visible: all graph neighbors (rooms connected by an edge).
      - If the current room is a kiosk room: ALL rooms in the museum
        are visible (the kiosk shows full real-time density data).

    Parameters
    ----------
    current_room : str
        The room the visitor is currently in.
    neighbors : iterable of str
        Rooms directly connected to current_room in the museum graph.
    kiosk_rooms : set of str or None
        Override for the set of kiosk room IDs. If None, uses the
        default config.KIOSK_ROOMS = {"A2", "A15", "A24", "D9"}.

    Returns
    -------
    set of str
        Room IDs that the visitor can observe.
    """

    # Default to the configured kiosk locations if not overridden.
    kiosk_rooms = config.KIOSK_ROOMS if kiosk_rooms is None else set(kiosk_rooms)
    # Base visibility: current room + immediate neighbors.
    vis = {current_room} | set(neighbors)
    # Kiosk upgrade: full museum visibility if standing at a kiosk.
    if current_room in kiosk_rooms:
        vis = set(config.ROOM_IDS)
    return vis


# =============================================================================
# Observation masking
# =============================================================================


def mask_density_and_trend(
    dens_all: np.ndarray,
    trend_all: np.ndarray,
    current_room: str,
    neighbors: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask density and trend arrays to the rooms visible to the agent.

    Constructs the observation vectors that the RL agent actually sees.
    Visible rooms get their true density and trend values; invisible
    rooms get the sentinel value (-1.0). A binary visibility mask is
    also returned so the policy network can distinguish "observed zero"
    from "unobserved."

    Parameters
    ----------
    dens_all : np.ndarray, shape (N_ROOMS,)
        True density (occupancy ratio) for every room.
    trend_all : np.ndarray, shape (N_ROOMS,)
        True density trend (first difference) for every room.
    current_room : str
        The visitor's current room.
    neighbors : iterable of str
        Rooms directly connected to current_room.

    Returns
    -------
    dens_obs : np.ndarray, shape (N_ROOMS,)
        Masked density vector. Invisible rooms have value -1.0.
    trend_obs : np.ndarray, shape (N_ROOMS,)
        Masked trend vector. Invisible rooms have value -1.0.
    visibility : np.ndarray, shape (N_ROOMS,)
        Binary mask: 1.0 for visible rooms, 0.0 for invisible.
    """

    # Initialize all rooms as unobserved (sentinel value).
    dens_obs = np.full(config.N_ROOMS, config.UNKNOWN_DENSITY_SENTINEL, dtype=float)
    trend_obs = np.full(config.N_ROOMS, config.UNKNOWN_DENSITY_SENTINEL, dtype=float)
    visibility = np.zeros(config.N_ROOMS, dtype=float)

    # Compute the set of visible rooms based on current position and kiosks.
    vis = visible_rooms(current_room, neighbors)
    # Unmask visible rooms: copy true values and set visibility flag.
    for room in vis:
        idx = config.ROOM_TO_IDX[room]
        dens_obs[idx] = dens_all[idx]
        trend_obs[idx] = trend_all[idx]
        visibility[idx] = 1.0

    return dens_obs, trend_obs, visibility


# =============================================================================
# Oscillation detection
# =============================================================================


def oscillation_score(density_history: np.ndarray) -> float:
    """Measure demand oscillation severity via mean absolute first difference.

    Oscillation occurs when information provision causes herd behavior:
    all visitors avoid a crowded room simultaneously, then rush back
    when it empties. This creates large timestep-to-timestep density
    swings. The metric is the mean absolute first difference across
    all rooms and timesteps:

      score = mean(|density[t+1] - density[t]|)

    A score near zero indicates smooth, stable crowd distribution.
    A high score indicates demand is "sloshing" between rooms.

    Parameters
    ----------
    density_history : np.ndarray, shape (T, N_ROOMS)
        Density matrix over time (rows = timesteps, columns = rooms).

    Returns
    -------
    float
        Mean absolute first difference. Returns 0.0 if fewer than
        2 timesteps are available.
    """

    # Need at least 2 timesteps to compute a first difference.
    if density_history.shape[0] < 2:
        return 0.0
    # First difference along the time axis.
    diff = np.diff(density_history, axis=0)
    # Mean absolute value across all rooms and timesteps.
    return float(np.mean(np.abs(diff)))
