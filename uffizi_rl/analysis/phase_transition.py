"""Population sweeps and phase-transition analysis (Phases 5-7).

A "phase transition" in this context borrows the physics metaphor: we
systematically vary a single control parameter (Type B ratio, total
visitor volume, or preference heterogeneity) while holding everything
else constant, and observe where the system's behavior changes
qualitatively, not just quantitatively. For example, there is typically
a critical Type B fraction beyond which Botticelli congestion suddenly
jumps from manageable to catastrophic. Identifying these tipping points
tells us where interventions are most urgently needed and how robust
the current equilibrium is to demographic shifts.

This module provides three sweep functions (one per control parameter),
a tipping-point detector, and a lightweight iterated best-response
equilibrium proxy. All sweeps use common random numbers (same seeds
across parameter values) so that differences are attributable to the
parameter change, not to simulation noise.

Design notes
------------
- Each sweep point runs multiple seeds and aggregates with 95% CIs.
- Parallel execution is supported via ProcessPoolExecutor but defaults
  to serial (n_workers=1) to respect laptop memory constraints.
- The equilibrium proxy is not a full Nash computation; it approximates
  repeated best-response dynamics by adjusting demand-reshaping
  parameters across iterations.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Sequence

import numpy as np

from uffizi_rl import config
from uffizi_rl.analysis.metrics import (
    botticelli_overcrowding_fraction,
    confidence_interval,
    gini,
    theil,
    welfare_proxy_from_density,
)
from uffizi_rl.environment.crowd_simulator import CrowdSimulator


# =============================================================================
# Single-day simulation kernel
# =============================================================================


def _simulate_population_day(
    type_b_fraction: float,
    daily_total: int,
    seed: int,
    heterogeneity_scale: float = 1.0,
    trail_acceptance_prob: float = 0.30,
    **kwargs,
) -> Dict[str, float]:
    """Run one simulator day and return the sweep metrics used downstream.

    This is the atomic unit of work for all sweeps. It instantiates a
    CrowdSimulator with the given parameters, runs a full simulated day,
    exports the density matrix, and computes every metric that downstream
    analysis needs (welfare, inequality, congestion, experience quality).

    Parameters
    ----------
    type_b_fraction : float
        Share of visitors that are Type B (0.0 = all art lovers, 1.0 = all
        checkbox tourists).
    daily_total : int
        Total visitor arrivals for the day.
    seed : int
        Random seed for reproducibility.
    heterogeneity_scale : float
        Multiplier on preference heterogeneity. Values > 1.0 make visitors
        more diverse in their room preferences; values < 1.0 make them more
        homogeneous (and thus more likely to crowd the same rooms).
    trail_acceptance_prob : float
        Probability that a visitor accepts a suggested alternative trail
        instead of the default recommended route.
    **kwargs
        Forwarded to CrowdSimulator; typically intervention flags from
        InterventionConfig.

    Returns
    -------
    dict
        Flat dictionary of metric name -> float, suitable for aggregation.
    """

    # Derive Type A fraction; clamp to [0, 1] for safety.
    type_a_fraction = max(0.0, min(1.0, 1.0 - type_b_fraction))

    # Instantiate and run the simulator for one full museum day.
    sim = CrowdSimulator(
        daily_total=daily_total,
        seed=seed,
        type_a_fraction=type_a_fraction,
        heterogeneity_scale=heterogeneity_scale,
        trail_acceptance_prob=trail_acceptance_prob,
        **kwargs,
    )
    day = sim.run_day()
    # Density matrix: shape (T, N_ROOMS), values are occupancy / capacity.
    dens = sim.export_density_matrix()

    # Compute welfare decomposition using the density-based proxy.
    welfare = welfare_proxy_from_density(
        density_matrix=dens,
        type_a_count=int(day["type_a_completed"]),
        type_b_count=int(day["type_b_completed"]),
    )

    # Mean density per room across all timesteps (for inequality metrics).
    room_density_means = dens.mean(axis=0) if dens.size else np.zeros(config.N_ROOMS)
    # Fraction of the day where Botticelli rooms exceed 80% capacity.
    bott_frac = botticelli_overcrowding_fraction(
        dens,
        idx_a11=config.ROOM_TO_IDX["A11"],
        idx_a12=config.ROOM_TO_IDX["A12"],
    )

    return {
        "total_welfare": welfare["total_welfare"],
        "type_a_welfare": welfare["type_a_welfare"],
        "type_b_welfare": welfare["type_b_welfare"],
        "peak_botticelli_density": day["peak_botticelli_density"],
        "botticelli_over80_frac": bott_frac,
        # Inequality metrics applied to room-level mean densities.
        "occupancy_gini": gini(room_density_means),
        "occupancy_theil": theil(room_density_means),
        "peak_inside": day["peak_inside"],
        "mean_inside": day["mean_inside"],
        # Revenue and experience quality (only populated when those models are active).
        "revenue": day.get("revenue", 0.0),
        "experience_quality": day.get("experience_quality", 0.0),
        "experience_intimacy": day.get("experience_intimacy", 0.0),
        "experience_surprise": day.get("experience_surprise", 0.0),
        "experience_narrative_coherence": day.get("experience_narrative_coherence", 0.0),
        "experience_engagement_depth": day.get("experience_engagement_depth", 0.0),
    }


# =============================================================================
# Aggregation and parallelism helpers
# =============================================================================


def _aggregate(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate repeated simulation records into means and confidence intervals.

    For each metric key, computes the sample mean and 95% CI half-width
    across seeds. Output keys are ``{metric}_mean`` and ``{metric}_ci``.
    """

    keys = records[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in records]
        m, ci = confidence_interval(vals)
        out[f"{k}_mean"] = m
        out[f"{k}_ci"] = ci
    return out


def _run_single(kwargs: Dict) -> Dict[str, float]:
    """Picklable wrapper for ProcessPoolExecutor.

    ProcessPoolExecutor requires that the callable passed to pool.map()
    accept a single argument. This wrapper unpacks the kwargs dictionary
    and forwards to _simulate_population_day.
    """

    return _simulate_population_day(**kwargs)


def _parallel_map(tasks: List[Dict], n_workers: int) -> List[Dict[str, float]]:
    """Run simulation tasks in parallel or serial depending on n_workers.

    When n_workers <= 1, runs sequentially (no subprocess overhead).
    Otherwise, uses ProcessPoolExecutor with the specified worker count.

    Parameters
    ----------
    tasks : list of dicts
        Each dict contains kwargs for _simulate_population_day.
    n_workers : int
        Number of parallel worker processes. 1 = serial execution.
    """

    if n_workers <= 1:
        return [_simulate_population_day(**t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_run_single, tasks))


# =============================================================================
# Sweep type 1: Type B ratio sweep
# =============================================================================
# Varies the fraction of Type B ("checkbox tourist") visitors from 0%
# to 100% while holding total volume and heterogeneity constant.
#
# WHY: the Type B ratio is the primary driver of Botticelli congestion.
# Type B visitors are route-following and crowd-insensitive, so they
# pile into magnet rooms without self-regulating. Sweeping this ratio
# reveals the critical fraction where congestion becomes catastrophic
# (the "tipping point") and quantifies the welfare transfer from
# Type A to Type B as the ratio increases.


def run_type_b_ratio_sweep(
    ratios: Iterable[float] = tuple(np.linspace(0.0, 1.0, 11)),
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    daily_total: int = config.DAILY_VISITORS_NORMAL,
    heterogeneity_scale: float = 1.0,
    n_workers: int = 1,
) -> List[Dict[str, float]]:
    """Sweep the Type-B share to identify heterogeneous-crowd tipping points.

    Parameters
    ----------
    ratios : iterable of float
        Type B fractions to test. Default: 0%, 10%, ..., 100%.
    seeds : iterable of int
        Random seeds for repeated runs per ratio value.
    daily_total : int
        Total daily visitors (held constant across the sweep).
    heterogeneity_scale : float
        Preference heterogeneity multiplier (held constant).
    n_workers : int
        Parallel workers. Default 1 (serial) for laptop safety.

    Returns
    -------
    list of dicts
        One aggregated record per ratio value, with ``_mean`` and ``_ci``
        suffixes plus a ``type_b_ratio`` key.
    """

    seeds_list = list(seeds)
    output = []
    for ratio in ratios:
        # Build one task per seed for this ratio value.
        tasks = [
            {
                "type_b_fraction": float(ratio),
                "daily_total": daily_total,
                "seed": int(s),
                "heterogeneity_scale": heterogeneity_scale,
            }
            for s in seeds_list
        ]
        runs = _parallel_map(tasks, n_workers)
        agg = _aggregate(runs)
        agg["type_b_ratio"] = float(ratio)  # tag the record with its sweep value
        output.append(agg)
    return output


# =============================================================================
# Sweep type 2: Volume sweep
# =============================================================================
# Varies total daily visitor count while holding the Type B ratio and
# heterogeneity constant.
#
# WHY: capacity constraints are at the heart of the museum problem.
# This sweep reveals how welfare degrades as volume increases, and
# identifies the volume threshold where the 900-person legal capacity
# constraint starts binding and congestion becomes severe.


def run_volume_sweep(
    volumes: Iterable[int] = (1000, 2000, 3000, 4000, 5000, 7000, 9000, 12000),
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    type_b_fraction: float = config.TYPE_B_FRACTION_DEFAULT,
    n_workers: int = 1,
) -> List[Dict[str, float]]:
    """Sweep daily visitor volume while holding composition fixed.

    Parameters
    ----------
    volumes : iterable of int
        Daily visitor counts to test. Default range spans low-season
        (1000) to peak free-admission days (12000).
    seeds : iterable of int
        Random seeds for repeated runs per volume value.
    type_b_fraction : float
        Type B share (held constant across the sweep).
    n_workers : int
        Parallel workers.

    Returns
    -------
    list of dicts
        One aggregated record per volume, tagged with ``daily_total``.
    """

    seeds_list = list(seeds)
    output = []
    for vol in volumes:
        tasks = [
            {
                "type_b_fraction": type_b_fraction,
                "daily_total": int(vol),
                "seed": int(s),
            }
            for s in seeds_list
        ]
        runs = _parallel_map(tasks, n_workers)
        agg = _aggregate(runs)
        agg["daily_total"] = float(vol)
        output.append(agg)
    return output


# =============================================================================
# Sweep type 3: Heterogeneity sweep
# =============================================================================
# Varies the preference heterogeneity scale while holding volume and
# composition constant.
#
# WHY: when all visitors want the same rooms (low heterogeneity),
# congestion is inevitable no matter what interventions do. When
# preferences are diverse (high heterogeneity), visitors naturally
# spread out and congestion is less severe. This sweep quantifies
# how much "natural dispersion" helps, and whether interventions
# are still needed when heterogeneity is high.


def run_heterogeneity_sweep(
    scales: Iterable[float] = (0.4, 0.7, 1.0, 1.4, 1.8),
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    daily_total: int = config.DAILY_VISITORS_NORMAL,
    type_b_fraction: float = config.TYPE_B_FRACTION_DEFAULT,
    n_workers: int = 1,
) -> List[Dict[str, float]]:
    """Sweep preference heterogeneity while holding composition fixed.

    Parameters
    ----------
    scales : iterable of float
        Heterogeneity multipliers. 1.0 = baseline; < 1.0 = more
        homogeneous (worse congestion); > 1.0 = more diverse (less
        congestion).
    seeds : iterable of int
        Random seeds for repeated runs.
    daily_total : int
        Total daily visitors (held constant).
    type_b_fraction : float
        Type B share (held constant).
    n_workers : int
        Parallel workers.

    Returns
    -------
    list of dicts
        One aggregated record per scale, tagged with ``heterogeneity_scale``.
    """

    seeds_list = list(seeds)
    output = []
    for h in scales:
        tasks = [
            {
                "type_b_fraction": type_b_fraction,
                "daily_total": daily_total,
                "seed": int(s),
                "heterogeneity_scale": float(h),
            }
            for s in seeds_list
        ]
        runs = _parallel_map(tasks, n_workers)
        agg = _aggregate(runs)
        agg["heterogeneity_scale"] = float(h)
        output.append(agg)
    return output


# =============================================================================
# Public entry point for single-day evaluation
# =============================================================================


def simulate_day_metrics(
    type_b_fraction: float,
    daily_total: int,
    seed: int,
    heterogeneity_scale: float = 1.0,
    trail_acceptance_prob: float = 0.30,
    **kwargs,
) -> Dict[str, float]:
    """Public helper used by intervention and equilibrium workflows.

    Thin wrapper around _simulate_population_day that provides a stable
    public API. All intervention parameters are forwarded to
    CrowdSimulator via **kwargs; see InterventionConfig for the full
    list of available intervention flags.

    Parameters
    ----------
    type_b_fraction : float
        Share of Type B visitors.
    daily_total : int
        Total visitor arrivals.
    seed : int
        Random seed.
    heterogeneity_scale : float
        Preference heterogeneity multiplier.
    trail_acceptance_prob : float
        Probability a visitor accepts an alternative trail.
    **kwargs
        Intervention flags forwarded to CrowdSimulator.

    Returns
    -------
    dict
        Flat dictionary of metric name -> float.
    """

    return _simulate_population_day(
        type_b_fraction=type_b_fraction,
        daily_total=daily_total,
        seed=seed,
        heterogeneity_scale=heterogeneity_scale,
        trail_acceptance_prob=trail_acceptance_prob,
        **kwargs,
    )


# =============================================================================
# Tipping point detection
# =============================================================================


def infer_tipping_point(records: Sequence[Dict[str, float]], key: str = "total_welfare_mean") -> float | None:
    """Infer the first sweep point where welfare falls materially below baseline.

    Scans the sweep records in order and returns the parameter value at
    which welfare drops to 65% or less of the first record's value. This
    is a simple threshold-based detector, not a statistical changepoint
    test; it works well for the sharp transitions observed in Type B
    ratio sweeps but may miss gradual degradation in volume sweeps.

    The 35% drop threshold is a modeling choice: it marks the point where
    congestion externalities have destroyed roughly a third of total
    welfare, which corresponds in practice to Botticelli rooms being
    over 80% capacity for most of the day.

    Parameters
    ----------
    records : sequence of dicts
        Sweep output from one of the run_*_sweep functions.
    key : str
        Metric key to monitor. Default: ``total_welfare_mean``.

    Returns
    -------
    float or None
        The sweep parameter value at the tipping point, or None if no
        tipping point is detected within the sweep range.
    """

    if not records:
        return None

    vals = np.array([rec[key] for rec in records], dtype=float)
    # Need at least 3 points to meaningfully detect a transition.
    if len(vals) < 3:
        return None

    baseline = vals[0]  # first sweep point is the reference
    # Tipping defined as first point with >= 35% welfare drop from baseline.
    for rec, v in zip(records, vals):
        if v <= 0.65 * baseline:
            # Return whichever sweep parameter is present in the record.
            return float(rec.get("type_b_ratio", rec.get("daily_total", np.nan)))
    return None


# =============================================================================
# Iterated best-response equilibrium proxy
# =============================================================================
# A full Nash equilibrium computation for 5000+ heterogeneous agents is
# intractable. Instead, we approximate equilibrium dynamics through
# iterated best-response: after each simulated day, we observe whether
# Botticelli congestion exceeded a threshold, and if so, we assume
# visitors would adapt their behavior in the next iteration:
#
# - Type A visitors (crowd-sensitive) increase their trail acceptance
#   and preference diversity, dispersing more aggressively.
# - Type B visitors (route-sticky) adapt slowly, slightly increasing
#   their willingness to try alternative trails.
#
# This produces a convergence trajectory: welfare typically stabilizes
# after 3-4 iterations, giving us a "behavioral equilibrium" that
# accounts for partial visitor adaptation without requiring explicit
# game-theoretic solution concepts.
#
# Limitation: this is a heuristic, not a formal equilibrium. The
# adaptation rules are assumed, not derived from utility maximization.


def population_rollout_equilibrium(
    iterations: int = 4,
    seed: int = config.DEFAULT_SEED,
    daily_total: int = config.DAILY_VISITORS_NORMAL,
    type_b_fraction: float = config.TYPE_B_FRACTION_DEFAULT,
) -> List[Dict[str, float]]:
    """Lightweight iterated best-response proxy.

    Simulates repeated days where visitors gradually adapt their behavior
    based on experienced congestion. Each iteration adjusts two demand-
    reshaping parameters:

    - trail_acceptance: how willing visitors are to follow alternative
      trails (increases when Botticelli is congested).
    - heterogeneity_scale: how diverse visitor preferences are (increases
      when congestion is severe, representing learning to explore).

    Parameters
    ----------
    iterations : int
        Number of best-response rounds. Default 4 (convergence is
        typically reached by iteration 3).
    seed : int
        Base random seed; each iteration uses seed + i.
    daily_total : int
        Total daily visitors.
    type_b_fraction : float
        Type B share (fixed across iterations; only behavior changes).

    Returns
    -------
    list of dicts
        One record per iteration, including the adaptation parameters
        (trail_acceptance, heterogeneity_scale) alongside all standard
        sweep metrics.
    """

    records: List[Dict[str, float]] = []
    # Initial conditions: low trail acceptance (visitors start naive).
    trail_acceptance = 0.15
    heterogeneity_scale = 1.0

    for i in range(iterations):
        # Simulate one day with current adaptation parameters.
        # Each iteration uses a different seed (seed + i) to avoid
        # confounding adaptation effects with random variation.
        rec = _simulate_population_day(
            type_b_fraction=type_b_fraction,
            daily_total=daily_total,
            seed=seed + i,
            heterogeneity_scale=heterogeneity_scale,
            trail_acceptance_prob=trail_acceptance,
        )
        # Tag the record with iteration metadata.
        rec["iteration"] = float(i)
        rec["trail_acceptance"] = float(trail_acceptance)
        rec["heterogeneity_scale"] = float(heterogeneity_scale)
        records.append(rec)

        # --- Best-response update heuristics ---
        # If Botticelli was overcrowded for > 25% of the day, visitors
        # react strongly: larger jump in trail acceptance and heterogeneity.
        # Otherwise, mild adaptation (word-of-mouth, gradual learning).
        if rec["botticelli_over80_frac"] > 0.25:
            # Strong response: visitors heard about the crowds and actively
            # seek alternatives. Trail acceptance jumps by 0.08 (capped at 0.55).
            trail_acceptance = min(0.55, trail_acceptance + 0.08)
            # Preference diversity also increases (visitors discover new rooms).
            heterogeneity_scale = min(2.0, heterogeneity_scale + 0.10)
        else:
            # Mild response: gradual learning, smaller trail acceptance increase.
            trail_acceptance = min(0.40, trail_acceptance + 0.03)

    return records
