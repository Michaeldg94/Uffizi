"""Pareto frontier computation for multi-objective optimization.

A Pareto frontier (also called the Pareto front or efficient frontier)
is the set of solutions where improving one objective necessarily
worsens another. In our museum context, the two objectives are:

  1. Total visitor welfare (we want to maximize).
  2. Total ticket revenue (the museum also wants to maximize).

A solution (portfolio of interventions) is **Pareto-optimal** if no
other solution is strictly better in BOTH welfare and revenue. The
Pareto frontier is the set of all such non-dominated solutions.

WHY this matters: welfare and revenue can conflict. For example,
reducing visitor volume improves welfare (less congestion) but reduces
revenue. The Pareto frontier shows decision-makers the exact trade-off:
for any desired revenue level, what is the maximum achievable welfare,
and vice versa. Points below the frontier are strictly sub-optimal
(there exists a portfolio that is better in at least one dimension
without sacrificing the other).

The algorithm used here is the classic O(n log n) sweep:
  1. Sort solutions by obj1 descending.
  2. Scan through the sorted list, keeping track of the best obj2
     seen so far.
  3. A solution is on the frontier if and only if its obj2 value
     exceeds the best obj2 of all solutions with higher obj1.

This works because after sorting by obj1 descending, a solution can
only be dominated by one that appeared earlier in the sorted order
(which has higher or equal obj1). Such a dominator would also need
higher obj2. So if the current solution's obj2 exceeds the running
maximum, it is non-dominated.

Assumption: both objectives are to be maximized. If one objective
should be minimized, negate it before calling this function.
"""

from __future__ import annotations

from typing import Dict, List


def compute_pareto_frontier(
    results: List[Dict[str, float]],
    obj1_key: str = "total_welfare",
    obj2_key: str = "revenue",
) -> List[Dict[str, float]]:
    """Filter results to the Pareto frontier (both objectives maximized).

    Returns the subset of results where no other result dominates
    (i.e., is strictly better in both objectives simultaneously).

    Parameters
    ----------
    results : list of dicts
        Each dict represents one evaluated portfolio and must contain
        at least the keys ``obj1_key`` and ``obj2_key``.
    obj1_key : str
        Dictionary key for the first objective. Default: "total_welfare".
    obj2_key : str
        Dictionary key for the second objective. Default: "revenue".

    Returns
    -------
    list of dicts
        The Pareto-optimal subset, ordered by obj1 descending (and
        therefore by obj2 ascending along the frontier).
    """

    if not results:
        return []

    # Step 1: sort by obj1 descending so that earlier entries have higher obj1.
    sorted_results = sorted(results, key=lambda r: r.get(obj1_key, 0.0), reverse=True)

    # Step 2: sweep through, keeping only solutions whose obj2 exceeds the
    # running maximum. Such solutions cannot be dominated by any earlier
    # entry (which has higher obj1 but lower obj2).
    frontier = []
    best_obj2 = float("-inf")
    for r in sorted_results:
        obj2_val = r.get(obj2_key, 0.0)
        if obj2_val > best_obj2:
            frontier.append(r)
            best_obj2 = obj2_val

    return frontier
