"""Intervention portfolio optimizer for the curated publication set.

This module implements a three-stage pipeline for selecting the best
combination of museum management interventions:

  Stage 1 (screen): evaluate each curated intervention in
    isolation to measure its marginal welfare and revenue impact.

  Stage 2 (greedy forward selection): starting from the empty portfolio,
    iteratively add the intervention that most improves the scalarized
    objective (welfare + lambda * revenue). Stop when no addition helps.

  Stage 3 (1-opt local search): starting from the greedy solution, try
    adding or removing each intervention one at a time and keep any
    improving move. Repeat until no single add/drop improves the
    objective. This escapes greedy sub-optimality without full
    combinatorial search (2^30 ~ 1 billion portfolios).

All evaluations use common random numbers (same seed) so that
differences between portfolios are attributable to the intervention
mix, not to simulation noise.

The curated interventions span several categories:
  - Access control (timed entry, gating, caps)
  - Demand redistribution (trails, audio guides, nudges)
  - Pricing (surcharges, annual passes, free windows)
  - Content enrichment (themed weeks, painting talks, exhibits)
  - Physical design (seating, sound, photo spots)
  - Information provision (forecasts, progress bars, social proof)

Each was chosen because it maps to a distinct mechanism for reducing
congestion, improving equity, or increasing revenue, and because it
can be modeled within the CrowdSimulator without unrealistic assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from uffizi_rl import config
from uffizi_rl.analysis.phase_transition import simulate_day_metrics


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class InterventionSpec:
    """Canonical definition for one curated intervention.

    Attributes
    ----------
    name : str
        Machine-readable identifier (matches InterventionConfig field names).
    label : str
        Human-readable label for plots and tables.
    kwargs : dict
        Keyword arguments to pass to CrowdSimulator to activate this
        intervention. May contain multiple keys if the intervention
        requires setting several parameters simultaneously (e.g.,
        hidden_gem_trails sets both trail_acceptance_prob and
        heterogeneity_scale).
    """

    name: str
    label: str
    kwargs: Dict


@dataclass
class PortfolioResult:
    """Portfolio optimization result with an explicit deployment order.

    Attributes
    ----------
    active_interventions : list of str
        Intervention names in the final portfolio (unordered set).
    deployment_order : list of str
        The order in which interventions were added during greedy
        selection. Useful for understanding which interventions
        provide the largest marginal gains.
    welfare : float
        Total welfare (Type A + Type B) under this portfolio.
    revenue : float
        Total ticket revenue under this portfolio.
    experience_quality : float
        Aggregate experience quality score.
    objective : float
        Scalarized objective: welfare + lambda * revenue.
    kwargs : dict
        Merged keyword arguments for the full portfolio.
    """

    active_interventions: List[str]
    deployment_order: List[str]
    welfare: float
    revenue: float
    experience_quality: float
    objective: float
    kwargs: Dict


# =============================================================================
# The curated interventions
# =============================================================================
# Each InterventionSpec bundles a name, a display label, and the kwargs
# that activate it in CrowdSimulator. The curated interventions are grouped
# by mechanism:
#
# ACCESS CONTROL (items 1-3): directly limit or schedule entry.
#   - decoupled_botticelli_gating: time-window access to Botticelli rooms,
#     chosen because the Botticelli chain is the single worst bottleneck.
#   - tour_group_cap: limits group size to 15 (from default 30), reducing
#     the "shock waves" that large groups create in corridors.
#   - timed_entry: assigns visitors to entry time slots, smoothing the
#     arrival curve and preventing the 10:00-12:00 crush.
#
# DEMAND REDISTRIBUTION: PRICING (items 4, 29):
#   - resident_annual_pass: adds 200 off-peak short-visit locals per day,
#     filling quiet periods without adding peak-hour pressure.
#   - per_person_group_surcharge (EUR12): internalizes the congestion
#     externality of tour groups by making organizers pay per head,
#     reducing group fraction via price elasticity.
#
# DEMAND REDISTRIBUTION: CONTENT (items 5-7, 16-17, 28):
#   - temporary_exhibit_room (A20): places a headline work in an underused
#     gallery, creating a new attractor that pulls visitors away from magnets.
#   - skip_the_famous (20%): persuades 20% of visitors to skip Botticelli/
#     Leonardo, directly reducing magnet-room pressure.
#   - themed_weeks (Caravaggio): boosts importance of Caravaggio rooms,
#     redirecting demand toward the E-block.
#   - designated_photo_spots: creates "Instagram rooms" in underused
#     galleries, leveraging social media to redistribute foot traffic.
#   - comparative_displays: links underused rooms to famous works,
#     making them "satellites" of the magnets.
#   - painting_talks: first-person audio narration boosts room importance
#     in selected galleries, enriching experience and redistributing dwell.
#
# DEMAND REDISTRIBUTION: ROUTING (items 14, 18, 20-22, 30):
#   - vasari_narration: chronological alternative route through the museum,
#     chosen because it produces a fundamentally different flow pattern.
#   - weather_routing (rain): adjusts outdoor room attractiveness based on
#     weather, chosen because rain days naturally shift demand indoors.
#   - adaptive_trail: real-time route suggestions based on current density.
#   - predictive_routing: like adaptive_trail but uses forecasted density.
#   - room_nobody_knows (A22): daily social-media feature of an underused
#     room, leveraging FOMO to redirect a fraction of visitors.
#   - hidden_gem_trails: curated alternative itineraries that route
#     visitors through lesser-known galleries.
#
# INFORMATION PROVISION (items 9, 24-26):
#   - crowd_forecast: shows predicted crowd levels at booking time,
#     shifting demand to less congested entry slots.
#   - progress_bar: shows visitors how much of the museum they have covered,
#     nudging completionists toward undervisited rooms.
#   - social_proof_nudge: displays real-time "others are enjoying room X"
#     messages to steer visitors toward uncrowded galleries.
#   - queue_to_content: when Botticelli is congested, activates contextual
#     content in the room before it (A10), creating a natural buffer.
#
# TIME ARCHITECTURE (items 11, 13, 15):
#   - last_hour_locals: reduced-price entry in the final 90 minutes,
#     chosen because evening visitors fill otherwise-empty capacity.
#   - lunch_free_entry: free entry during 12:30-14:00 for locals,
#     filling the lunch-hour dip with short visits.
#   - quiet_hours: no tour groups before 11:00, protecting the morning
#     window for crowd-sensitive visitors.
#
# PHYSICAL DESIGN (items 8, 19, 23, 27):
#   - multi_room_buffer: activates content in rooms A9/A10/A13 when
#     Botticelli is congested, creating an absorption cascade.
#   - sound_design: period-appropriate music in underused rooms makes
#     them destinations; silence in bottleneck rooms keeps flow moving.
#   - seating_strategy: adds benches in underused rooms (increases dwell)
#     and removes them from bottleneck rooms (decreases dwell).
#   - social_media_spot (A16): designates the Tribune as the weekly
#     Instagram spot, leveraging its existing magnetism.
#
# BEHAVIORAL NUDGES (items 10, 12):
#   - achievement_system: gamification badges for visiting underused rooms.
#   - adaptive_audio_guide: audio guide that steers visitors away from
#     congested rooms in real time.

CURATED_INTERVENTIONS: List[InterventionSpec] = [
    # The catalog has been filtered down to Pigovian / common-pool
    # governance interventions only. Every entry below either taxes a
    # congestion externality directly (group surcharge), caps the
    # behavior that creates the externality (group cap, slot cap, timed
    # entry, quiet hours), or redistributes demand away from the four
    # masterpiece rooms during peak hours (multi-room buffer, queue-to-
    # content, alternative attractors, predictive routing).
    #
    # Interventions that boost the naive welfare metric by adding
    # off-peak visitors (locals, free lunch entry, etc.) without
    # reducing peak masterpiece density have been REMOVED from the
    # portfolio candidate set because they do not address the actual
    # externality. Their implementations remain in the simulator for
    # ablation/sensitivity studies but they are not considered in the
    # optimal portfolio search.
    #
    # --- Access control ---
    # Tour group cap = 10 visitors. Half of the median gallery room
    # capacity (median ~25), so no single group occupies more than
    # half of any typical room.
    InterventionSpec("tour_group_cap", "Tour Group Cap (10)", {"tour_group_cap": 10}),
    # Timed entry = visitors are admitted in fixed slots and comply
    # exactly. Useful as a baseline alternative to RAMA when only the
    # entrance is booked (not the masterpieces).
    InterventionSpec("timed_entry", "Timed Entry", {"timed_entry": True}),
    # Quiet Hours: no tour groups before 11:00 (real, defensible) +
    # 15% reduction in arrival rate during the morning quiet window
    # (assumption: "premium quiet" marketing suppresses casual demand).
    InterventionSpec("quiet_hours", "Quiet Hours", {"quiet_hours": True}),
    # --- Pricing ---
    # Pigovian group surcharge on top of the EUR 70 baseline group fee.
    # An extra EUR 70 doubles the group price (to EUR 140), suppressing
    # tour group demand via standard price elasticity.
    # Aggressive Pigovian surcharge: +EUR150 on top of baseline EUR 70
    # = EUR 220 per group of 15. Designed to nearly eliminate tour-group
    # demand by pricing the externality of coordinated 15-person
    # movement through narrow corridors.
    InterventionSpec("per_person_group_surcharge", "Group Surcharge (+EUR150 flat)", {"per_person_group_surcharge": 150.0}),
    # --- Headline interventions ---
    InterventionSpec("rama", "RAMA (Reservation-Anchored Masterpiece Access)", {"rama": True}),
    InterventionSpec(
        "secondary_attractor_enrichment",
        "Secondary-Attractor Enrichment",
        {"secondary_attractor_enrichment": True},
    ),
    InterventionSpec(
        "rama_plus_enrichment",
        "RAMA + Secondary-Attractor Enrichment",
        {"rama": True, "secondary_attractor_enrichment": True},
    ),
    # --- Capacity / pricing levers ---
    # Extended hours: 7:00-19:00 (12-hour day, vs 8:15-18:30 baseline).
    # More entrance slots, more masterpiece slots, more total visits.
    InterventionSpec("extended_hours", "Extended Hours (7:00-19:00)", {"extended_hours": True}),
    # Time-of-day price differentiation: dawn slot 15 EUR, mid-morning
    # 20, midday standard 25, peak afternoon 30, late afternoon
    # discount 18. Price elasticity = 0.5 (research-grounded
    # ballpark for museum demand, e.g. Cellini & Cuccia 2018).
    InterventionSpec("dynamic_pricing", "Dynamic Pricing (peak surcharge)", {"dynamic_pricing": True}),
    # Expanded annual pass program: doubles the take-up of the EUR 80
    # Florentine annual pass [UFF]. Adds another 200 short-budget
    # residents to the daily mix at amortised cost (EUR 8/visit).
    InterventionSpec(
        "resident_annual_pass",
        "Expanded Annual Pass Program",
        {"resident_annual_pass": True},
    ),
    # Combo: extended hours + dynamic pricing + RAMA + enrichment.
    InterventionSpec(
        "rama_enrich_extended_priced",
        "RAMA + Enrichment + Extended Hours + Dynamic Pricing",
        {"rama": True, "secondary_attractor_enrichment": True,
         "extended_hours": True, "dynamic_pricing": True},
    ),
]
# Catalog is now 11 specs: 9 distinct levers plus 2 pre-bundled combos
# (RAMA+Enrichment, and RAMA+Enrichment+Extended+Pricing). The earlier
# brainstormed catalog had far more; the ones below did NOT survive a
# brutally honest audit and were cut:
#   - Decoupled Masterpiece Gating  (superseded by RAMA)
#   - Multi-Room Buffer, Queue-to-Content  (magic magnetism boost on
#     A10 with no realistic trigger mechanism)
#   - Predictive Routing  (under RAMA, masterpieces are always at cap
#     so predicting their density is tautological)
#   - Adaptive Trail, Hidden Gem Trails  (under RAMA, art lovers have
#     pre-booked itineraries; re-routing them at entry is incoherent)
#   - Room Nobody Knows, Temporary Exhibition  (assumed importance
#     boosts of +5 / +1.75 are wildly unrealistic for a small mini-
#     exhibit; real effect is closer to +0.5-1.0)
#   - Weather Routing  (rain effect modelled as -3 importance is the
#     wrong mechanic; would be a multiplicative visit-rate effect)
#   - Quiet Hours Tour-Ban-Only / Arrival-Reduction-Only ablations
#     (combined version retained)
assert len(CURATED_INTERVENTIONS) == 11
# Lookup table for O(1) access by name.
_INTERVENTION_BY_NAME = {spec.name: spec for spec in CURATED_INTERVENTIONS}


# =============================================================================
# Evaluation helpers
# =============================================================================


def _evaluate(kwargs: Dict, seed: int, daily_total: int) -> Dict[str, float]:
    """Run one deterministic portfolio evaluation on common random numbers.

    Uses the same seed for every evaluation so that differences between
    portfolios are attributable to the intervention mix, not simulation
    noise. Revenue model is always active during portfolio optimization
    because the scalarized objective can include a revenue term.

    Parameters
    ----------
    kwargs : dict
        Merged intervention kwargs for the portfolio being evaluated.
    seed : int
        Random seed (shared across all evaluations for fairness).
    daily_total : int
        Total daily visitors.

    Returns
    -------
    dict
        Flat dictionary of metric name -> float.
    """

    return simulate_day_metrics(
        type_b_fraction=config.TYPE_B_FRACTION_DEFAULT,
        daily_total=daily_total,
        seed=seed,
        revenue_model=True,  # always active during portfolio evaluation
        **kwargs,
    )


def _merge_kwargs(intervention_names: List[str]) -> Dict:
    """Merge kwargs for the ordered intervention list.

    Later interventions in the list overwrite earlier ones if they share
    the same kwargs key. This is intentional: it models the fact that
    interventions with conflicting parameter settings cannot be deployed
    simultaneously (last-write-wins semantics).
    """

    merged = {}
    for name in intervention_names:
        merged.update(_INTERVENTION_BY_NAME[name].kwargs)
    return merged


def _objective(metrics: Dict[str, float], lambda_revenue: float) -> float:
    """Scalarized portfolio objective used in greedy and local search.

    Combines welfare and revenue into a single scalar:
      objective = total_welfare + lambda * revenue

    When lambda = 0, the optimizer maximizes welfare only.
    When lambda > 0, the optimizer trades off welfare for revenue,
    tracing the welfare-revenue Pareto frontier as lambda varies.

    Parameters
    ----------
    metrics : dict
        Simulation output from _evaluate().
    lambda_revenue : float
        Weight on the revenue term. 0.0 = welfare only.

    Returns
    -------
    float
        Scalarized objective value.
    """

    return float(metrics["total_welfare"] + lambda_revenue * metrics.get("revenue", 0.0))


# =============================================================================
# Public accessors
# =============================================================================


def curated_intervention_specs() -> List[InterventionSpec]:
    """Return the canonical curated intervention list in publication order.

    Returns a copy to prevent accidental mutation of the module-level list.
    """

    return list(CURATED_INTERVENTIONS)


def combined_intervention_kwargs() -> Dict:
    """Return the kwargs for the full curated-intervention bundle.

    Turns on every curated intervention at once (the "kitchen sink"
    portfolio). Note: later interventions overwrite conflicting keys from
    earlier ones (last-write-wins).
    """

    return _merge_kwargs([spec.name for spec in CURATED_INTERVENTIONS])


def _result_from_order(
    order: List[str],
    metrics: Dict[str, float],
    lambda_revenue: float,
) -> PortfolioResult:
    """Build a ``PortfolioResult`` from an ordered intervention set.

    Encapsulates the construction logic so that greedy_portfolio and
    local_search do not need to duplicate it.
    """

    merged = _merge_kwargs(order)
    return PortfolioResult(
        active_interventions=list(order),
        deployment_order=list(order),
        welfare=float(metrics["total_welfare"]),
        revenue=float(metrics.get("revenue", 0.0)),
        experience_quality=float(metrics.get("experience_quality", 0.0)),
        objective=_objective(metrics, lambda_revenue),
        kwargs=merged,
    )


# =============================================================================
# Stage 1: Individual screening
# =============================================================================


def screen_interventions(
    seed: int = config.DEFAULT_SEED,
    daily_total: int = config.DAILY_VISITORS_NORMAL,
) -> List[Dict[str, float]]:
    """Stage 1: evaluate each curated intervention individually and rank it.

    Runs the baseline (no interventions) once, then activates each
    intervention in isolation and measures its marginal welfare gain
    (delta_w) and marginal revenue change (delta_r) relative to the
    baseline. Results are sorted by welfare gain (descending), then
    by revenue gain, then alphabetically for deterministic ordering.

    Parameters
    ----------
    seed : int
        Common random number seed.
    daily_total : int
        Total daily visitors.

    Returns
    -------
    list of dicts
        One record per intervention (plus baseline), sorted by impact.
    """

    # Evaluate the status quo (no interventions active).
    baseline = _evaluate({}, seed, daily_total)
    results = [{
        "name": "baseline",
        "label": "Status Quo (baseline)",
        "welfare": baseline["total_welfare"],
        "revenue": baseline["revenue"],
        "experience_quality": baseline.get("experience_quality", 0.0),
        "delta_w": 0.0,
        "delta_r": 0.0,
    }]

    # Evaluate each intervention in isolation.
    for spec in CURATED_INTERVENTIONS:
        m = _evaluate(spec.kwargs, seed, daily_total)
        results.append({
            "name": spec.name,
            "label": spec.label,
            "welfare": m["total_welfare"],
            "revenue": m["revenue"],
            "experience_quality": m.get("experience_quality", 0.0),
            "delta_w": m["total_welfare"] - baseline["total_welfare"],
            "delta_r": m["revenue"] - baseline["revenue"],
        })

    # Sort: highest welfare gain first; ties broken by revenue, then name.
    return sorted(
        results,
        key=lambda row: (-row["delta_w"], -row["delta_r"], row["name"]),
    )


# =============================================================================
# Stage 2: Greedy forward selection
# =============================================================================
# Greedy forward selection is a classic combinatorial optimization
# heuristic. At each step, it evaluates every remaining intervention,
# picks the one that most improves the objective, and locks it in.
# The process stops when no candidate improves the objective.
#
# WHY greedy: with this many interventions, the full combinatorial space has
# 2^30 ~ 1 billion subsets. Greedy narrows this to at most 30 rounds
# of 30 evaluations = 900 simulator calls, which is tractable.
#
# LIMITATION: greedy can get stuck in sub-optimal solutions because
# it never removes an intervention once added. Stage 3 (local search)
# addresses this by allowing removals.
#
# INTERACTION EFFECTS: when two interventions are deployed together,
# their combined effect may be more or less than the sum of their
# individual effects. Greedy naturally captures positive interactions
# (synergies) because it evaluates each candidate in the context of
# already-selected interventions. Negative interactions (redundancies)
# cause an intervention to be skipped even if it was beneficial in
# isolation.


def greedy_portfolio(
    seed: int = config.DEFAULT_SEED,
    daily_total: int = config.DAILY_VISITORS_NORMAL,
    lambda_revenue: float = 0.0,
) -> PortfolioResult:
    """Stage 2: greedily add interventions while the scalarized objective improves.

    Parameters
    ----------
    seed : int
        Common random number seed.
    daily_total : int
        Total daily visitors.
    lambda_revenue : float
        Weight on the revenue term in the scalarized objective.

    Returns
    -------
    PortfolioResult
        The greedy-optimal portfolio with deployment order.
    """

    baseline = _evaluate({}, seed, daily_total)
    best_metrics = baseline
    best_obj = _objective(best_metrics, lambda_revenue)

    selected: List[str] = []  # interventions locked in so far
    remaining = [spec.name for spec in CURATED_INTERVENTIONS]

    # At most 30 rounds (one per intervention).
    for _ in range(len(CURATED_INTERVENTIONS)):
        best_candidate_name = None
        best_candidate_metrics = None
        best_candidate_obj = best_obj

        # Try adding each remaining intervention to the current portfolio.
        for name in remaining:
            trial = selected + [name]
            merged = _merge_kwargs(trial)
            m = _evaluate(merged, seed, daily_total)
            obj = _objective(m, lambda_revenue)
            # Accept if strictly better (1e-9 tolerance for float noise),
            # or if tied, prefer the alphabetically first candidate for
            # deterministic tie-breaking.
            if obj > best_candidate_obj + 1e-9 or (
                abs(obj - best_candidate_obj) <= 1e-9 and best_candidate_name is not None and name < best_candidate_name
            ):
                best_candidate_name = name
                best_candidate_metrics = m
                best_candidate_obj = obj

        # Stop if no candidate improved the objective.
        if best_candidate_name is None or best_candidate_metrics is None:
            break

        # Lock in the best candidate.
        selected.append(best_candidate_name)
        remaining.remove(best_candidate_name)
        best_metrics = best_candidate_metrics
        best_obj = best_candidate_obj

    return _result_from_order(selected, best_metrics, lambda_revenue)


# =============================================================================
# Stage 3: 1-opt local search
# =============================================================================
# 1-opt local search explores the "neighborhood" of the current
# portfolio by trying every single add-or-drop move:
#   - For each intervention IN the portfolio: try removing it.
#   - For each intervention NOT in the portfolio: try adding it.
# If any move improves the objective, accept it and repeat.
# Stop when a full pass over all curated interventions finds no improvement
# (local optimum), or after n_iterations passes.
#
# WHY 1-opt after greedy: greedy may include an intervention that was
# beneficial when added early but becomes redundant after later
# additions. 1-opt can remove it. Conversely, an intervention that was
# not beneficial in isolation may become valuable in combination with
# the greedy selection; 1-opt can add it.
#
# Interaction effect formula (implicit):
#   interaction(A, B) = objective({A, B}) - objective({A}) - objective({B}) + objective({})
# A positive interaction means A and B are synergistic; negative means
# redundant. The greedy + 1-opt pipeline captures these effects without
# explicitly computing the formula, because every evaluation runs the
# full portfolio through the simulator.


def local_search(
    portfolio: PortfolioResult,
    seed: int = config.DEFAULT_SEED,
    daily_total: int = config.DAILY_VISITORS_NORMAL,
    lambda_revenue: float = 0.0,
    n_iterations: int = 3,
) -> PortfolioResult:
    """Stage 3: deterministic 1-opt local search around the greedy portfolio.

    Parameters
    ----------
    portfolio : PortfolioResult
        Starting portfolio (typically from greedy_portfolio).
    seed : int
        Common random number seed.
    daily_total : int
        Total daily visitors.
    lambda_revenue : float
        Revenue weight in the scalarized objective.
    n_iterations : int
        Maximum number of full passes. Default 3 (convergence is
        typically reached in 1-2 passes).

    Returns
    -------
    PortfolioResult
        Locally optimal portfolio after 1-opt search.
    """

    current_order = list(portfolio.deployment_order)
    # Re-evaluate the starting portfolio to establish the baseline objective.
    best = _result_from_order(
        current_order,
        _evaluate(_merge_kwargs(current_order), seed, daily_total),
        lambda_revenue,
    )

    for _ in range(n_iterations):
        improved = False
        for spec in CURATED_INTERVENTIONS:
            # Toggle: if the intervention is in the portfolio, try removing it;
            # if it is not, try adding it.
            if spec.name in current_order:
                trial_order = [name for name in current_order if name != spec.name]
            else:
                trial_order = current_order + [spec.name]

            merged = _merge_kwargs(trial_order)
            m = _evaluate(merged, seed, daily_total)
            obj = _objective(m, lambda_revenue)

            # Accept if strictly better (1e-9 tolerance).
            if obj > best.objective + 1e-9:
                best = _result_from_order(trial_order, m, lambda_revenue)
                current_order = trial_order
                improved = True

        # If a full pass found no improvement, we are at a local optimum.
        if not improved:
            break

    return best


# =============================================================================
# Combined pipeline
# =============================================================================


def optimize_portfolio(
    seed: int = config.DEFAULT_SEED,
    daily_total: int = config.DAILY_VISITORS_NORMAL,
    lambda_revenue: float = 0.0,
) -> PortfolioResult:
    """Run greedy selection followed by deterministic 1-opt local search.

    This is the main entry point for portfolio optimization. It chains
    Stage 2 (greedy) and Stage 3 (1-opt) into a single call.

    Parameters
    ----------
    seed : int
        Common random number seed.
    daily_total : int
        Total daily visitors.
    lambda_revenue : float
        Revenue weight. 0.0 = welfare only.

    Returns
    -------
    PortfolioResult
        Locally optimal portfolio.
    """

    portfolio = greedy_portfolio(seed, daily_total, lambda_revenue)
    return local_search(portfolio, seed, daily_total, lambda_revenue)


# =============================================================================
# Pareto frontier sweep
# =============================================================================


def pareto_sweep(
    seed: int = config.DEFAULT_SEED,
    daily_total: int = config.DAILY_VISITORS_NORMAL,
    lambda_values: List[float] | None = None,
) -> List[PortfolioResult]:
    """Trace the welfare-revenue Pareto frontier by varying the revenue weight.

    For each lambda value, runs the full optimize_portfolio pipeline.
    As lambda increases, the optimizer shifts from pure welfare
    maximization toward portfolios that also generate revenue, tracing
    out the Pareto frontier of non-dominated (welfare, revenue) outcomes.

    Parameters
    ----------
    seed : int
        Common random number seed.
    daily_total : int
        Total daily visitors.
    lambda_values : list of float or None
        Revenue weights to sweep. Default covers the range from
        pure welfare (0.0) to revenue-dominated (2.0).

    Returns
    -------
    list of PortfolioResult
        One optimized portfolio per lambda value.
    """

    if lambda_values is None:
        lambda_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    results = []
    for lam in lambda_values:
        r = optimize_portfolio(seed, daily_total, lam)
        results.append(r)

    return results
