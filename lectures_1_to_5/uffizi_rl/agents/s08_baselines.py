"""Baseline policies for Phase 2 benchmarking on the toy museum graph.

[READING ORDER: file 8 of 12 - read after s07_q_learning.py]

This module defines five handcrafted baseline policies against which the
trained Q-learning agent is compared. Each baseline captures a different
visitor strategy, from naive to moderately intelligent:

  1. default_path: follows the canonical museum route (what most tourists do).
  2. random: uniform random walk (theoretical lower bound on sensible behavior).
  3. greedy_least_crowded: myopic crowd avoidance (no value awareness).
  4. greedy_value_ratio: myopic value/crowd trade-off (one-step lookahead).
  5. peak_avoidance: targeted Botticelli avoidance + greedy elsewhere
     (closest handcrafted approximation to the temporal-arbitrage idea).

The baselines serve two purposes:
  - They establish the performance range for the toy graph, letting us
    quantify how much the Q-learning agent improves over each strategy.
  - They test specific hypotheses: does avoiding Botticelli at peak times
    help (peak_avoidance vs. default_path)? Does crowd awareness alone
    suffice (greedy_least_crowded vs. greedy_value_ratio)?

All baselines share the same function signature so they can be wrapped by
_wrap_policy() and passed to evaluate_policy() from q_learning.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from uffizi_rl import s02_config as config
from uffizi_rl.agents.s07_q_learning import ToyTabularEnv, evaluate_policy


# =============================================================================
# Type alias and default route
# =============================================================================

# Baseline policy signature: (env, state, valid_mask, rng) -> action int.
# The env parameter gives baselines access to graph structure and crowd
# levels, which a pure RL agent would not have (it only sees the state).
# This is intentional: baselines are allowed to "cheat" with oracle info,
# making them a stronger comparison point.
PolicyFn = Callable[[ToyTabularEnv, tuple, np.ndarray, np.random.Generator], int]

# The default museum route through the toy graph, mirroring the real Uffizi's
# recommended itinerary compressed to 12 rooms. This is the path that Type B
# ("checkbox tourist") visitors follow in the full simulator.
#
# ENTRY -> C1 -> A11 -> A12 -> C2 -> A35 -> A38 -> EXIT
#
# This route hits all the "must-see" rooms (Botticelli, Leonardo, Raphael)
# in topological order but makes no effort to avoid crowd peaks. The
# default_path_policy implements this route, and its performance is the
# primary benchmark: the Q-learning agent must beat it to demonstrate that
# temporal arbitrage adds value.
DEFAULT_ROUTE = ["ENTRY", "C1", "A11", "A12", "C2", "A35", "A38", "EXIT"]

# Precomputed next-room lookup for the default route.
# Maps each room to the room that follows it on the route.
# e.g., DEFAULT_NEXT["A11"] = "A12", DEFAULT_NEXT["A38"] = "EXIT".
DEFAULT_NEXT = {DEFAULT_ROUTE[i]: DEFAULT_ROUTE[i + 1] for i in range(len(DEFAULT_ROUTE) - 1)}


# =============================================================================
# Baseline result container
# =============================================================================


@dataclass
class BaselineResult:
    """Named benchmark result for a handcrafted toy-graph policy.

    Attributes
    ----------
    name : str
        Human-readable policy name (e.g., "default_path", "random").
    metrics : Dict[str, float]
        Dictionary of evaluation metrics returned by evaluate_policy():
        mean_return, std_return, crowded_steps, offpeak_botticelli_visits.
    """

    name: str
    metrics: Dict[str, float]


# =============================================================================
# Baseline policy 1: default path
# =============================================================================


def default_path_policy(env: ToyTabularEnv, state, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Follow the default museum route, one room at a time.

    WHY this baseline exists: it represents the typical tourist who follows
    the guidebook path without any crowd awareness. This is the behavior
    the Q-learning agent must outperform. If the RL agent cannot beat a
    simple follow-the-signs strategy, temporal arbitrage has no value.

    The policy looks up the next room on DEFAULT_ROUTE and moves there if
    it is a valid neighbor. If the current room is not on the route (e.g.,
    after a detour), or the next room is not adjacent, the agent stays put
    (action 0).
    """

    room = env.current_room
    target = DEFAULT_NEXT.get(room)
    if target is None:
        return 0                                       # at EXIT or off-route: stay
    neighbors = sorted(env.g.neighbors(room))
    if target in neighbors:
        return neighbors.index(target) + 1             # +1 because action 0 = stay
    return 0                                           # target not adjacent: stay


# =============================================================================
# Baseline policy 2: random walk
# =============================================================================


def random_policy(env: ToyTabularEnv, state, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Choose uniformly at random among currently valid actions.

    WHY this baseline exists: it establishes the theoretical lower bound on
    expected return for any policy that respects the action mask. Any
    strategy that cannot beat uniform random is not learning anything useful.
    It also provides a variance benchmark: random walks have high return
    variance, so a good agent should have lower std_return.
    """

    valid = np.flatnonzero(mask)
    return int(rng.choice(valid)) if len(valid) else 0


# =============================================================================
# Baseline policy 3: greedy least crowded
# =============================================================================


def greedy_least_crowded_policy(env: ToyTabularEnv, state, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Move to whichever neighboring room currently has the lowest crowd density.

    WHY this baseline exists: it tests the hypothesis that pure crowd
    avoidance (ignoring room value entirely) is a viable strategy. If this
    policy performs well, it suggests that congestion is the dominant factor
    in reward; if it performs poorly, it shows that value awareness matters.

    This policy is purely myopic: it considers only the current crowd state,
    not how crowds will evolve in future steps. The Q-learning agent should
    beat it by planning ahead.

    Note: accesses env._crowd_level() directly (oracle information).
    """

    neighbors = sorted(env.g.neighbors(env.current_room))
    if not neighbors:
        return 0
    # Query the exact crowd density in each neighbor at the current time.
    densities = [env._crowd_level(n, env.t) for n in neighbors]  # noqa: SLF001
    # Move to the least crowded neighbor (argmin + 1 for action offset).
    return int(np.argmin(densities) + 1)


# =============================================================================
# Baseline policy 4: greedy value ratio
# =============================================================================


def greedy_value_ratio_policy(env: ToyTabularEnv, state, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Choose the neighbor that maximizes importance / (1 + alpha * density).

    WHY this baseline exists: it combines value awareness with crowd
    sensitivity in a one-step greedy fashion, using the same reward formula
    as the environment. This is the strongest myopic baseline: it makes the
    locally optimal choice at every step. The Q-learning agent can only beat
    it by planning multi-step sequences (e.g., delaying a visit to wait for
    a crowd peak to pass), which is exactly the temporal-arbitrage behavior
    we want to demonstrate.

    Uses TYPE_A_CROWD_ALPHA (6.0) from config, matching the environment's
    reward function.
    """

    neighbors = sorted(env.g.neighbors(env.current_room))
    if not neighbors:
        return 0
    scores = []
    for n in neighbors:
        imp = env.g.nodes[n]["importance"]
        dens = env._crowd_level(n, env.t)  # noqa: SLF001
        # Same formula as the environment reward (without novelty or step cost).
        scores.append(imp / (1.0 + config.TYPE_A_CROWD_ALPHA * dens))
    return int(np.argmax(scores) + 1)


# =============================================================================
# Baseline policy 5: peak avoidance
# =============================================================================


def peak_avoidance_policy(env: ToyTabularEnv, state, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Avoid Botticelli rooms when crowded, then greedily pick the best neighbor.

    WHY this baseline exists: it is the closest handcrafted approximation
    to the temporal-arbitrage strategy the Q-learning agent should discover.
    It explicitly avoids A11/A12 when their density >= 0.6, and among the
    remaining neighbors, picks the one with the best value/crowd ratio.

    If the Q-learning agent only matches this baseline, it has merely
    rediscovered a simple heuristic. The agent should outperform it because:
      - This policy only avoids Botticelli; the agent can learn timing
        strategies for Leonardo, Raphael, and Tribune as well.
      - This policy is myopic (one-step); the agent can plan multi-step
        detours through minor rooms to wait out a crowd peak.

    Uses a softer alpha (2.5 vs. 6.0) to test whether a gentler crowd
    penalty combined with explicit avoidance can compete with the learned
    policy.
    """

    neighbors = sorted(env.g.neighbors(env.current_room))
    if not neighbors:
        return 0

    # Step 1: filter out Botticelli rooms if they are currently crowded.
    filtered = []
    for n in neighbors:
        if n in {"A11", "A12"} and env._crowd_level(n, env.t) >= 0.6:  # noqa: SLF001
            continue                                   # skip crowded Botticelli
        filtered.append(n)

    # Step 2: among remaining neighbors, pick the best value/crowd ratio.
    if filtered:
        scores = []
        for n in filtered:
            imp = env.g.nodes[n]["importance"]
            dens = env._crowd_level(n, env.t)  # noqa: SLF001
            # Deliberately uses a softer alpha (2.5) than the RL agent (6.0)
            # to test whether a simple avoid-and-greedily-pick heuristic
            # can compete with the learned policy.
            scores.append(imp / (1.0 + 2.5 * dens))
        best = filtered[int(np.argmax(scores))]
    else:
        # All neighbors are crowded Botticelli; fall back to the first.
        best = neighbors[0]

    return neighbors.index(best) + 1                   # +1 for action offset


# =============================================================================
# Policy wrapper and evaluation harness
# =============================================================================


def _wrap_policy(env: ToyTabularEnv, fn: PolicyFn, seed: int):
    """Adapt a 4-argument baseline policy to the 2-argument evaluate_policy API.

    evaluate_policy() (in q_learning.py) expects a function with signature
    (state, mask) -> int. Baseline policies have a richer signature
    (env, state, mask, rng) -> int because they need access to the graph
    and crowd oracle. This wrapper closes over env and rng, producing a
    function that evaluate_policy() can call directly.
    """
    rng = config.get_rng(seed)

    def _policy(state, mask):
        return fn(env, state, mask, rng)

    return _policy


def evaluate_baselines(
    episodes: int = 200,
    seed: int = config.DEFAULT_SEED,
) -> list[BaselineResult]:
    """Evaluate the full handcrafted baseline suite on the toy graph.

    Creates one ToyTabularEnv instance and runs each of the five baseline
    policies for ``episodes`` episodes, collecting mean return, crowded
    steps, and off-peak Botticelli visit counts.

    Each baseline gets a different seed offset (seed + i) to ensure
    independent randomness across policies while remaining reproducible.

    Parameters
    ----------
    episodes : int
        Number of evaluation episodes per baseline. Default 200.
    seed : int
        Base random seed. Each baseline uses seed + i for i in 0..4.

    Returns
    -------
    list[BaselineResult]
        One BaselineResult per policy, in the order: default_path, random,
        greedy_least_crowded, greedy_value_ratio, peak_avoidance.
    """

    env = ToyTabularEnv(seed=seed)

    # All five baselines, in evaluation order.
    baselines: Dict[str, PolicyFn] = {
        "default_path": default_path_policy,
        "random": random_policy,
        "greedy_least_crowded": greedy_least_crowded_policy,
        "greedy_value_ratio": greedy_value_ratio_policy,
        "peak_avoidance": peak_avoidance_policy,
    }

    results: list[BaselineResult] = []
    for i, (name, fn) in enumerate(baselines.items()):
        # Wrap the 4-arg baseline into the 2-arg signature evaluate_policy expects.
        metrics = evaluate_policy(env, _wrap_policy(env, fn, seed + i), episodes=episodes, seed=seed + i)
        results.append(BaselineResult(name=name, metrics=metrics))

    return results
