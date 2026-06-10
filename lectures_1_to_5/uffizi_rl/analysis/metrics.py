"""Core metrics for welfare, congestion, inequality, and experience analysis.

This module provides the quantitative backbone for evaluating museum
management outcomes. The metrics fall into four families:

1. **Inequality metrics** (Gini, Theil): measure how unevenly visitors
   are distributed across rooms. In the museum context these play the
   same role that income-inequality indices play in economics: a Gini
   of 0 means every room has identical occupancy; a Gini near 1 means
   all visitors are crammed into a single room while others sit empty.

2. **Welfare metrics** (welfare proxy, Price of Anarchy, gap closed):
   measure total visitor utility using the same functional form as the
   RL reward, ensuring consistency between training and evaluation.

3. **Congestion metrics** (Botticelli overcrowding fraction): track the
   severity of the worst bottleneck in the museum.

4. **Experience quality** (intimacy, surprise, narrative, engagement):
   captures what makes a visit memorable beyond the absence of crowds.

All functions are pure (no side effects) and operate on NumPy arrays
or simple iterables, so they can be called from sweeps, equilibrium
loops, and portfolio evaluators without import-time overhead.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from uffizi_rl import s02_config as config


# =============================================================================
# Inequality metrics
# =============================================================================
# These are borrowed from the income-inequality literature and repurposed
# to measure spatial inequality of room utilization. High inequality means
# demand is concentrated in a few "magnet" rooms while most galleries are
# underused, exactly the problem interventions aim to fix.


def gini(values: Iterable[float]) -> float:
    """Compute the Gini coefficient for a non-negative sample.

    The Gini coefficient originates in welfare economics as a measure of
    income inequality (Gini, 1912). Here it quantifies how unevenly
    visitors are distributed across rooms:

      Gini = 0  : perfectly uniform occupancy (every room equally used).
      Gini -> 1 : all visitors concentrated in one room.

    WHY this matters for the museum problem: interventions aim to spread
    visitors more evenly. Tracking Gini over a sweep lets us see whether
    a policy actually redistributes demand or merely shifts the bottleneck.

    Formula (discrete, using the sorted-values shortcut):
      G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    where x_i are the sorted values and i is the 1-based rank.

    Parameters
    ----------
    values : iterable of non-negative floats
        Typically mean room densities across a simulated day.

    Returns
    -------
    float
        Gini coefficient in [0, 1]. Returns 0.0 for empty or all-zero input.
    """

    # Materialize to array; clip negatives to zero (density cannot be negative).
    x = np.asarray(list(values), dtype=float)
    if len(x) == 0:
        return 0.0
    # All-zero case: no inequality if nobody is anywhere.
    if np.allclose(x, 0.0):
        return 0.0
    x = np.sort(np.clip(x, 0.0, None))  # sort ascending; clip guards against float noise
    n = len(x)
    idx = np.arange(1, n + 1)  # 1-based rank indices
    # Discrete Gini formula (sorted-values shortcut, avoids O(n^2) pairwise diffs).
    return float((2 * np.sum(idx * x) / (n * np.sum(x))) - (n + 1) / n)


def theil(values: Iterable[float]) -> float:
    """Compute the Theil-T (GE(1)) inequality index.

    The Theil index is an entropy-based inequality measure from the
    generalized entropy family. Unlike Gini, it is additively decomposable:
    total inequality = between-group + within-group. This property makes
    it useful for decomposing inequality across museum wings or floors.

    WHY this matters: when analyzing multi-floor redistribution policies,
    Theil lets us separate "inequality across wings" from "inequality
    within a wing," which Gini cannot do.

    Formula:
      T = (1/n) * sum( (x_i / mu) * ln(x_i / mu) )
    where mu = mean(x).

    Parameters
    ----------
    values : iterable of floats
        Typically mean room densities. Values are clipped to 1e-9 to
        avoid log(0).

    Returns
    -------
    float
        Theil-T index >= 0. Returns 0.0 if mean is non-positive.
    """

    x = np.asarray(list(values), dtype=float)
    x = np.clip(x, 1e-9, None)  # avoid log(0); 1e-9 is effectively zero occupancy
    mean_x = np.mean(x)
    if mean_x <= 0:
        return 0.0
    # Each term is (x_i / mu) * ln(x_i / mu); average over all rooms.
    return float(np.mean((x / mean_x) * np.log(x / mean_x)))


# =============================================================================
# Welfare metrics
# =============================================================================


def price_of_anarchy(social_optimum_welfare: float, equilibrium_welfare: float) -> float:
    """Return the welfare loss ratio between the social optimum and equilibrium.

    The Price of Anarchy (PoA) is a game-theory concept that measures how
    much total welfare is lost when agents act selfishly rather than
    following a socially optimal plan.

      PoA = W*(social optimum) / W(equilibrium)

    A PoA of 1.0 means selfish behavior is costless; PoA > 1 means
    congestion externalities are destroying welfare.

    WHY this matters: the PoA quantifies the *need* for intervention.
    If PoA is close to 1, visitors are already distributing themselves
    efficiently and no policy is needed. A large PoA justifies the
    cost and complexity of interventions.

    Parameters
    ----------
    social_optimum_welfare : float
        Welfare under a hypothetical centralized planner.
    equilibrium_welfare : float
        Welfare under decentralized (selfish) visitor behavior.

    Returns
    -------
    float
        Ratio >= 1.0. Returns inf if equilibrium welfare is effectively zero.
    """

    if equilibrium_welfare <= 1e-9:
        return float("inf")
    return float(social_optimum_welfare / equilibrium_welfare)


# =============================================================================
# Congestion metrics
# =============================================================================


def botticelli_overcrowding_fraction(density_matrix: np.ndarray, idx_a11: int, idx_a12: int) -> float:
    """Measure the share of timesteps where either Botticelli room is overcrowded.

    Rooms A11 (Primavera) and A12 (Birth of Venus) are the museum's worst
    bottleneck: they sit on a forced one-way chain, hold the two most
    famous paintings, and every visitor type wants to visit them. This
    metric tracks the fraction of the day where the worse of the two
    rooms exceeds 80% of capacity.

    WHY this matters: Botticelli overcrowding is the single strongest
    predictor of visitor dissatisfaction. Many interventions specifically
    target this bottleneck, and this fraction is the most direct measure
    of whether they succeed.

    Parameters
    ----------
    density_matrix : np.ndarray, shape (T, N_ROOMS)
        Occupancy ratios (occupancy / capacity) for each room at each
        timestep. Values > 1.0 indicate over-capacity.
    idx_a11, idx_a12 : int
        Column indices for rooms A11 and A12 in the density matrix.

    Returns
    -------
    float
        Fraction of timesteps in [0, 1] where max(A11, A12) density > 0.8.
    """

    if density_matrix.size == 0:
        return 0.0
    # Take the worse of the two Botticelli rooms at each timestep.
    bott = np.maximum(density_matrix[:, idx_a11], density_matrix[:, idx_a12])
    # 0.8 threshold: 80% of room capacity is the "uncomfortable" cutoff. [assumption]
    return float(np.mean(bott > 0.8))


# =============================================================================
# Welfare proxy (density-based)
# =============================================================================


def welfare_proxy_from_density(
    density_matrix: np.ndarray,
    type_a_count: int,
    type_b_count: int,
) -> Dict[str, float]:
    """Proxy welfare decomposition for large-scale sweeps.

    Welfare is computed per-room, per-timestep using the same functional
    form as the RL reward: importance / (1 + alpha * density). This
    ensures the welfare metric is consistent with the agent's objective.

    The two visitor types experience crowds differently:
    - Type A ("art lovers"): high crowd sensitivity (alpha=6.0) and a
      cross-type externality (perceives Type B as 1.5x more disruptive).
    - Type B ("checkbox tourists"): low crowd sensitivity (alpha=0.5),
      no cross-type penalty.

    The cross-type externality produces a tipping-point: beyond a critical
    Type B fraction, magnet-room congestion destroys Type A welfare
    disproportionately, even though Type B visitors themselves are barely
    affected. This asymmetry is what makes the museum problem interesting
    from a mechanism-design perspective.

    Parameters
    ----------
    density_matrix : np.ndarray, shape (T, N_ROOMS)
        Occupancy ratios for each room at each timestep.
    type_a_count : int
        Number of Type A visitors who completed their visit.
    type_b_count : int
        Number of Type B visitors who completed their visit.

    Returns
    -------
    dict with keys: total_welfare, type_a_welfare, type_b_welfare, peak_density.
    """

    if density_matrix.size == 0:
        return {
            "total_welfare": 0.0,
            "type_a_welfare": 0.0,
            "type_b_welfare": 0.0,
            "peak_density": 0.0,
        }

    peak_density = float(np.max(density_matrix))
    n_rooms = density_matrix.shape[1]
    total_visitors = type_a_count + type_b_count
    # Fraction of the population that is Type B; used to scale the externality.
    type_b_fraction = type_b_count / max(1, total_visitors)

    # Room importance vector (aligned with density_matrix columns).
    # Each room's cultural significance on a 1-10 scale from config.ROOM_DATA.
    importance = np.array(
        [config.ROOM_DATA[r]["importance"] for r in config.ROOM_IDS[:n_rooms]],
        dtype=float,
    )

    # --- Type A utility ---
    # Per-room, per-timestep utility: importance / (1 + alpha * density).
    # Type A uses "effective density" that inflates raw density by the
    # cross-type externality factor, scaled by the Type B population share.
    # When type_b_fraction = 0, externality_inflator = 1.0 (no inflation).
    # When type_b_fraction = 1, externality_inflator = TYPE_A_CROSS_TYPE_EXTERNALITY.
    externality_inflator = 1.0 + (config.TYPE_A_CROSS_TYPE_EXTERNALITY - 1.0) * type_b_fraction
    effective_density_a = density_matrix * externality_inflator
    # Broadcasting: importance[None, :] is (1, N_ROOMS), density is (T, N_ROOMS).
    utility_a = importance[None, :] / (1.0 + config.TYPE_A_CROWD_ALPHA * effective_density_a)
    mean_utility_a = float(np.mean(utility_a))

    # --- Type B utility ---
    # Type B uses raw density with low crowd sensitivity (alpha=0.5).
    # No cross-type externality: Type B is indifferent to who else is in the room.
    utility_b = importance[None, :] / (1.0 + config.TYPE_B_CROWD_ALPHA * density_matrix)
    mean_utility_b = float(np.mean(utility_b))

    # --- Aggregate welfare ---
    # Total welfare = sum of per-capita utility weighted by population count.
    # This is a utilitarian social welfare function.
    type_a_welfare = float(type_a_count * mean_utility_a)
    type_b_welfare = float(type_b_count * mean_utility_b)
    total = type_a_welfare + type_b_welfare

    return {
        "total_welfare": float(total),
        "type_a_welfare": float(type_a_welfare),
        "type_b_welfare": float(type_b_welfare),
        "peak_density": peak_density,
    }


# =============================================================================
# Intervention evaluation helpers
# =============================================================================


def intervention_gap_closed(status_quo_welfare: float, intervention_welfare: float, social_optimum: float) -> float:
    """Return the share of the status-quo welfare gap closed by an intervention.

    Measures what fraction of the theoretically achievable improvement
    (from status quo to social optimum) the intervention actually captures:

      gap_closed = (W_intervention - W_status_quo) / (W_optimum - W_status_quo)

    A value of 0.0 means no improvement; 1.0 means the intervention
    fully closes the gap to the social optimum; values > 1.0 are possible
    if the intervention exceeds the estimated optimum.

    WHY this matters: raw welfare numbers are hard to interpret without
    context. This metric normalizes improvement relative to the maximum
    possible, making interventions comparable across different baselines.

    Parameters
    ----------
    status_quo_welfare : float
        Welfare without any intervention.
    intervention_welfare : float
        Welfare with the intervention active.
    social_optimum : float
        Welfare under a hypothetical centralized planner.

    Returns
    -------
    float
        Fraction of the gap closed; 0.0 if the gap is negligible.
    """

    denom = social_optimum - status_quo_welfare
    # Guard against division by zero when status quo already equals the optimum.
    if abs(denom) < 1e-9:
        return 0.0
    return float((intervention_welfare - status_quo_welfare) / denom)


# =============================================================================
# Statistical helpers
# =============================================================================


def confidence_interval(values: Iterable[float], z: float = 1.96) -> tuple[float, float]:
    """Return the mean and normal-approximation confidence interval half-width.

    Uses the standard formula: CI = z * (s / sqrt(n)), where s is the
    sample standard deviation with Bessel's correction (ddof=1).

    Parameters
    ----------
    values : iterable of floats
        Repeated measurements (e.g., welfare across seeds).
    z : float
        Z-score for the desired confidence level. Default 1.96 = 95% CI.

    Returns
    -------
    (mean, half_width) : tuple[float, float]
        The sample mean and the CI half-width. Half-width is 0.0 for
        empty or single-element samples.
    """

    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, 0.0
    # Standard error with Bessel's correction (ddof=1) for unbiased variance.
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean, float(z * se)


# =============================================================================
# Experience quality
# =============================================================================
# Experience quality captures what makes a museum visit memorable beyond
# the absence of crowds. It decomposes into four orthogonal components
# that can be independently targeted by different interventions.


def experience_quality_components(
    visited_rooms: Sequence[str] | set[str],
    checklist_rooms: Sequence[str] | set[str],
    trail_name: str | None,
    room_importance: Mapping[str, float],
    room_mean_occupancy: Mapping[str, float],
) -> Dict[str, float]:
    """Score a visit's experiential quality from explicit component terms.

    The experience quality metric captures four dimensions of visit
    satisfaction that go beyond congestion avoidance:

    1. **Intimacy**: fraction of rooms experienced with few other people.
       A room with <= INTIMACY_THRESHOLD visitors feels "personal."
       WHY: the most memorable museum moments are quiet encounters with
       art; interventions that create pockets of low density improve this.

    2. **Surprise**: fraction of visited rooms that were NOT on the
       visitor's pre-planned checklist. Discovering unexpected galleries
       adds delight. WHY: trail-based interventions (hidden gems, Vasari
       narration) aim to increase this by routing visitors off the beaten path.

    3. **Narrative coherence**: fraction of a thematic trail's rooms that
       the visitor actually covered. Following a story arc (e.g., "Colors
       of Florence") is more satisfying than random wandering. WHY: this
       rewards trail interventions for completing their narrative promise.

    4. **Engagement depth**: fraction of visited rooms that have high
       cultural importance (importance >= 7.0). Spending time with
       masterpieces (not just passing through corridors) indicates
       genuine engagement. WHY: some interventions risk routing visitors
       away from important works; this term penalizes that trade-off.

    Parameters
    ----------
    visited_rooms : sequence or set of room IDs
        All rooms the visitor entered during their visit.
    checklist_rooms : sequence or set of room IDs
        Rooms the visitor planned to see before arriving.
    trail_name : str or None
        The thematic trail followed, if any (e.g., "colors_of_florence").
    room_importance : mapping from room ID to importance score (1-10).
    room_mean_occupancy : mapping from room ID to average occupancy count.

    Returns
    -------
    dict with keys: intimacy, surprise, narrative_coherence,
    engagement_depth, total.
    """

    # Exclude non-gallery nodes (entry, exit, staircases, vestibule).
    # These are transit points, not "rooms" where art appreciation happens.
    terminal_nodes = {"ENTRY", "EXIT", "LANZI_STAIRCASE", "BUONTALENTI_STAIRCASE",
                      "PANORAMIC_TERRACE", "A1"}
    display_rooms = sorted(set(visited_rooms) - terminal_nodes)
    checklist_set = set(checklist_rooms)
    n_rooms = max(1, len(display_rooms))  # avoid division by zero

    # --- Intimacy ---
    # Count rooms where mean occupancy is low enough for a personal experience.
    # Rooms with no occupancy data default to inf (not intimate).
    intimate_rooms = sum(
        1
        for room in display_rooms
        if room_mean_occupancy.get(room, float("inf")) <= config.INTIMACY_THRESHOLD
    )
    # Weighted fraction: INTIMACY_BONUS_WEIGHT scales the contribution.
    intimacy = config.INTIMACY_BONUS_WEIGHT * intimate_rooms / n_rooms

    # --- Surprise ---
    # Rooms visited that were NOT on the pre-planned checklist.
    surprise_rooms = len(set(display_rooms) - checklist_set)
    surprise = config.SURPRISE_WEIGHT * surprise_rooms / n_rooms

    # --- Narrative coherence ---
    # If the visitor followed a named trail, measure completion fraction.
    narrative = 0.0
    if trail_name is not None and trail_name != "none":
        # Look up the trail's room sequence from config.
        if trail_name in config.HIDDEN_GEM_TRAILS:
            trail_rooms = set(config.HIDDEN_GEM_TRAILS[trail_name])
        elif trail_name == "vasari":
            trail_rooms = set(config.VASARI_ROUTE)
        else:
            trail_rooms = set()
        if trail_rooms:
            # Fraction of the trail's rooms that the visitor actually entered.
            narrative = (
                config.NARRATIVE_COHERENCE_WEIGHT
                * len(set(display_rooms) & trail_rooms)
                / len(trail_rooms)
            )

    # --- Engagement depth ---
    # Fraction of visited rooms that hold culturally significant works.
    # importance >= 7.0 marks rooms with major masterpieces (e.g., Botticelli,
    # Leonardo, Caravaggio). [assumption: threshold aligns with config data]
    high_imp_visited = sum(1 for room in display_rooms if room_importance.get(room, 0.0) >= 7.0)
    engagement_depth = config.ENGAGEMENT_DEPTH_WEIGHT * high_imp_visited / n_rooms

    # Aggregate: simple sum (each component is already weighted).
    total = intimacy + surprise + narrative + engagement_depth
    return {
        "intimacy": float(intimacy),
        "surprise": float(surprise),
        "narrative_coherence": float(narrative),
        "engagement_depth": float(engagement_depth),
        "total": float(total),
    }
