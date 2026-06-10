"""Tabular Q-learning on a toy museum graph (Phase 2).

[READING ORDER: file 7 of 12 - read after s06_uffizi_env.py]

This module implements the Phase 2 proof-of-concept for the Uffizi RL project:
a tabular Q-learning agent that learns to route a single visitor through a
simplified 12-room museum graph while avoiding peak crowd times.

The key insight this module demonstrates is *temporal arbitrage*: the agent
learns that visiting high-value rooms (Botticelli, Leonardo) during off-peak
windows yields higher reward than following the default path into peak-hour
congestion. This validates the core hypothesis before scaling to the full
museum graph with deep RL in later phases.

Architecture overview
---------------------
1. ToyTabularEnv: a lightweight environment wrapping the 12-node toy graph
   from museum_graph.toy_graph(), with synthetic (deterministic) crowd waves
   that simulate time-varying congestion patterns.
2. QLearningAgent: a standard tabular Q-learning agent with epsilon-greedy
   exploration, storing Q(s, a) in a defaultdict keyed by discrete state
   tuples.
3. train_q_learning(): the training loop that runs episodes, collects
   training curves, and tracks off-peak Botticelli visits as the primary
   metric for temporal arbitrage.
4. evaluate_policy(): a rollout evaluator that measures mean return,
   crowded-step count, and off-peak Botticelli visits for any policy
   function.

References
----------
[VL83]  Veron & Levasseur (1983). Visitor typology.
[A22]   Attanasio et al. (2022). Visitor flow management at Uffizi.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from uffizi_rl import s02_config as config
from uffizi_rl.environment.s03_museum_graph import toy_graph


# =============================================================================
# Constants
# =============================================================================

# The six gallery rooms tracked in the visited-mask portion of the state.
# These are the rooms with the highest importance scores in the toy graph
# (the "must-see" masterpieces). The visited mask lets the agent distinguish
# "I have already seen Botticelli" from "I still need to see Botticelli,"
# which is critical for learning not to revisit rooms unnecessarily.
KEY_ROOMS = ["A11", "A12", "A16", "A35", "A38", "E4"]


# =============================================================================
# Toy tabular environment
# =============================================================================


class ToyTabularEnv:
    """Toy graph environment with deterministic, non-stationary crowd waves.

    The 12-room toy graph preserves the essential structure of the real Uffizi:
      - ENTRY: visitor starting point.
      - C1, C2: corridor hubs connecting gallery wings (high degree).
      - A11, A12: Botticelli rooms (forced chain, highest importance = 10).
      - A16: Tribune (Medici Venus, importance = 8).
      - A35: Leonardo da Vinci (importance = 9).
      - A38: Raphael and Michelangelo (importance = 9).
      - E4: Caravaggio (importance = 9).
      - M1, M2: minor rooms (importance = 4) providing an alternative path
        to EXIT that bypasses the high-value but congested galleries.
      - EXIT: terminal node.

    The graph is small enough for tabular Q-learning (finite state space) but
    rich enough to exhibit the core planning challenge: the agent must decide
    WHEN to visit high-value rooms to avoid synthetic crowd peaks, trading off
    immediate reward against future congestion.

    Crowd model
    -----------
    Crowd levels are deterministic Gaussian peaks in time, not stochastic.
    Each major room has a peak at a different normalized time, simulating the
    empirical pattern where Botticelli rooms fill up in mid-morning and
    Leonardo rooms peak around midday. This lets the agent learn a repeatable
    timing strategy without needing to handle observation noise (that comes
    in Phase 4 with the full environment).

    State representation
    --------------------
    The state is a 5-tuple: (room, time_bin, crowd_bott, crowd_leo, visited_mask).
    See get_state() for details on each component.

    Action space
    ------------
    Action 0 = stay in current room. Actions 1..max_degree = move to the i-th
    neighbor (sorted alphabetically). Invalid actions (index beyond the
    current room's neighbor count) are silently converted to "stay" with a
    penalty.
    """

    def __init__(self, seed: int = config.DEFAULT_SEED, horizon: int = 120):
        self.g = toy_graph()                           # 12-node NetworkX graph
        self.horizon = int(horizon)                    # max steps per episode

        # The action space must be uniform across all states for the Q-table.
        # We size it to 1 (stay) + max degree across all nodes, so that every
        # node can index its neighbors without out-of-bounds.
        self.max_degree = max(dict(self.g.degree()).values())
        self.action_size = 1 + self.max_degree         # 0=stay, 1..max_degree=move

        # Episode state (reset in reset()).
        self.current_room = "ENTRY"
        self.t = 0                                     # discrete time step
        self.visited = set()                           # rooms visited this episode

    # -----------------------------------------------------------------
    # Action helpers
    # -----------------------------------------------------------------

    def _neighbors(self) -> List[str]:
        """Return the alphabetically sorted neighbors of the current room.

        Sorting guarantees a deterministic mapping from action index to room
        across episodes and seeds.
        """
        return sorted(self.g.neighbors(self.current_room))

    def _action_to_room(self, action: int) -> str:
        """Map an integer action to the target room.

        Action 0 always means "stay." Actions 1..N map to the sorted neighbor
        list. If the action index exceeds the number of neighbors (possible
        because action_size is uniform), the agent stays in place.
        """
        if action == 0:
            return self.current_room
        nbrs = self._neighbors()
        idx = action - 1
        if 0 <= idx < len(nbrs):
            return nbrs[idx]
        return self.current_room                       # out-of-range -> stay

    def valid_actions(self) -> np.ndarray:
        """Return a binary mask of valid actions for the current room.

        mask[0] = 1 always (staying is always valid).
        mask[1 + i] = 1 for each neighbor i in the sorted neighbor list.
        All other entries are 0 (invalid).
        """
        mask = np.zeros(self.action_size, dtype=np.int8)
        mask[0] = 1                                    # "stay" always valid
        for i, _ in enumerate(self._neighbors()):
            mask[1 + i] = 1
        return mask

    # -----------------------------------------------------------------
    # Synthetic crowd model
    # -----------------------------------------------------------------

    def _crowd_level(self, room: str, t: int) -> float:
        """Return the synthetic crowd density in [0, ~1.5] for a room at time t.

        The crowd model uses Gaussian peaks centered at different normalized
        times for each major room, approximating the empirical observation
        that different galleries fill up at different hours [A22]:

          Room(s)     Peak time (normalized)   Interpretation
          ---------   ----------------------   ----------------------------------
          A11, A12    0.35 (mid-morning)       Botticelli rooms fill first because
                                               they are near the entrance and on
                                               every guidebook's "see first" list.
          A16         0.40 (late morning)      Tribune fills slightly later; it is
                                               deeper in the east wing.
          A35         0.50 (midday)            Leonardo peaks at noon as the wave
                                               of visitors progresses through the
                                               second-floor corridor.
          A38         0.55 (early afternoon)   Raphael/Michelangelo peak last
                                               among the major galleries.

        All other rooms (corridors, minor galleries, EXIT) get a low sinusoidal
        background density that gently oscillates, ensuring they are never
        completely empty but also never critically congested.

        The Gaussian shape means crowds rise, peak, and fall smoothly.
        The agent can exploit the valleys (off-peak windows) for higher reward.

        Parameters
        ----------
        room : str
            Room identifier in the toy graph.
        t : int
            Current discrete time step.

        Returns
        -------
        float
            Crowd density, clipped to a room-specific maximum.
        """
        # Normalize time to [0, 1] so peak positions are horizon-independent.
        x = t / max(1, self.horizon)

        # --- Botticelli rooms: peak at 35% of the visit horizon ---
        # Amplitude 1.2, narrow variance 0.03 -> sharp, intense peak.
        if room in {"A11", "A12"}:
            return float(np.clip(1.2 * np.exp(-((x - 0.35) ** 2) / 0.03), 0.0, 1.5))

        # --- Leonardo: peak at 50% (midday) ---
        if room == "A35":
            return float(np.clip(1.0 * np.exp(-((x - 0.50) ** 2) / 0.04), 0.0, 1.2))

        # --- Raphael/Michelangelo: peak at 55% (early afternoon) ---
        if room == "A38":
            return float(np.clip(1.1 * np.exp(-((x - 0.55) ** 2) / 0.05), 0.0, 1.2))

        # --- Tribune: peak at 40% (late morning) ---
        if room == "A16":
            return float(np.clip(0.9 * np.exp(-((x - 0.40) ** 2) / 0.04), 0.0, 1.0))

        # --- All other rooms: low sinusoidal background ---
        # Oscillates between 0.1 and 0.5; never triggers the congestion penalty.
        return float(np.clip(0.3 + 0.2 * np.sin(2 * np.pi * x), 0.0, 0.8))

    # -----------------------------------------------------------------
    # State discretisation helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _crowd_bin(d: float) -> int:
        """Quantise a continuous crowd density into one of 4 discrete bins.

        Bin boundaries are chosen so that the agent can distinguish:
          0: empty/quiet   (density < 0.25)
          1: moderate       (0.25 <= density < 0.50)
          2: busy           (0.50 <= density < 0.80)
          3: congested      (density >= 0.80, triggers congestion penalty)
        """
        if d < 0.25:
            return 0
        if d < 0.50:
            return 1
        if d < 0.80:
            return 2
        return 3

    def _visited_mask(self) -> int:
        """Encode which KEY_ROOMS have been visited as a bitmask integer.

        Bit i is set if KEY_ROOMS[i] has been visited this episode. With 6
        key rooms this gives 2^6 = 64 possible values, keeping the state
        space tractable for a Q-table while letting the agent distinguish
        "still need to see Botticelli" from "already visited."
        """
        mask = 0
        for i, room in enumerate(KEY_ROOMS):
            if room in self.visited:
                mask |= 1 << i
        return mask

    def get_state(self) -> Tuple[str, int, int, int, int]:
        """Build the discrete state tuple observed by the Q-learning agent.

        Returns
        -------
        (room, time_bin, crowd_bott, crowd_leo, visited_mask)
            room : str
                Current room identifier (12 possible values).
            time_bin : int
                Discretised progress through the episode, 0..9 (10 bins).
                Computed as floor(10 * t / horizon), capped at 9.
            crowd_bott : int
                Crowd bin (0..3) for the Botticelli rooms. Takes the max of
                A11 and A12 because they form a forced chain; if either is
                crowded the agent should avoid the pair.
            crowd_leo : int
                Crowd bin (0..3) for the Leonardo room (A35).
            visited_mask : int
                Bitmask encoding which of the 6 KEY_ROOMS have been visited.
                Range 0..63.

        The total state space is: 12 rooms * 10 time bins * 4 crowd_bott *
        4 crowd_leo * 64 visited masks = 122,880 states (upper bound; many
        combinations are unreachable). This is large but tractable for
        tabular Q-learning with ~2500 episodes.
        """
        time_bin = min(9, int(10 * self.t / max(1, self.horizon)))
        # Botticelli crowd: max of A11 and A12 (forced chain, treat as one).
        crowd_bott = self._crowd_bin(max(self._crowd_level("A11", self.t), self._crowd_level("A12", self.t)))
        # Leonardo crowd: just A35.
        crowd_leo = self._crowd_bin(self._crowd_level("A35", self.t))
        visited_mask = self._visited_mask()
        return (self.current_room, time_bin, crowd_bott, crowd_leo, visited_mask)

    # -----------------------------------------------------------------
    # Episode lifecycle
    # -----------------------------------------------------------------

    def reset(self) -> Tuple[str, int, int, int, int]:
        """Reset the environment to the start of a new episode.

        The visitor begins at ENTRY (already counted as visited) at time 0.
        """
        self.current_room = "ENTRY"
        self.t = 0
        self.visited = {"ENTRY"}                       # ENTRY is always "visited"
        return self.get_state()

    def step(self, action: int):
        """Execute one time step: move (or stay), compute reward, advance clock.

        Reward design
        -------------
        The reward function encodes the Type A ("art lover") visitor's utility:

          r = importance * novelty / (1 + alpha * density) - step_cost

        where:
          - importance: room's cultural significance (1-10 from the graph).
          - novelty: 1.0 for the first visit to a room, 0.2 for revisits.
            This incentivises covering all key rooms rather than looping.
          - alpha: TYPE_A_CROWD_ALPHA (6.0) from config. High alpha makes
            the agent strongly prefer low-density visits.
          - density: synthetic crowd level from _crowd_level().
          - step_cost: 0.01 per step, encouraging efficient routes.

        Additional penalties:
          - Congestion penalty: -0.5 * density when density > 0.8 in a
            gallery room (not corridors). This creates a hard disincentive
            for entering rooms during peak crowding.
          - Invalid action penalty: -0.2 if the agent picks an action that
            is out of range or masked as invalid.

        The episode ends when the agent reaches EXIT or the time horizon
        is exceeded.

        Parameters
        ----------
        action : int
            Action index (0 = stay, 1..max_degree = move to neighbor).

        Returns
        -------
        (state, reward, done, info)
            state : tuple
                The new discrete state after the transition.
            reward : float
                Scalar reward for this transition.
            done : bool
                True if the episode has terminated.
            info : dict
                Diagnostic info: room name, crowd density, invalid flag.
        """
        # Validate the action; invalid actions are coerced to "stay."
        valid = self.valid_actions()
        invalid = action < 0 or action >= len(valid) or valid[action] == 0
        if invalid:
            action = 0

        next_room = self._action_to_room(action)
        visited_before = next_room in self.visited

        self.current_room = next_room
        self.visited.add(next_room)

        dens = self._crowd_level(next_room, self.t)
        imp = self.g.nodes[next_room]["importance"]

        # --- Reward computation ---
        # Novelty bonus: full credit on first visit, 20% on revisits.
        novelty = 1.0 if not visited_before else 0.2
        # Core reward: importance discounted by crowd density, minus step cost.
        reward = imp * novelty / (1.0 + config.TYPE_A_CROWD_ALPHA * dens) - 0.01
        # Congestion penalty for gallery rooms only (corridors are transient).
        if next_room not in {"ENTRY", "C1", "C2", "EXIT"} and dens > 0.8:
            reward -= 0.5 * dens
        # Penalty for attempting an invalid action (teaches valid-action masking).
        if invalid:
            reward -= 0.2

        self.t += 1

        # Terminal conditions: reached the exit, or time budget exhausted.
        done = next_room == "EXIT" or self.t >= self.horizon
        return self.get_state(), float(reward), bool(done), {
            "room": next_room,
            "density": dens,
            "invalid": invalid,
        }


# =============================================================================
# Q-learning agent
# =============================================================================


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration.

    The Q-table is a defaultdict mapping state tuples to numpy arrays of
    shape (action_size,). On first access for any state, the Q-values are
    initialised to zeros (optimistic initialisation is not used here; the
    epsilon schedule handles exploration).

    Hyperparameters
    ---------------
    lr : float
        Learning rate (alpha). Controls how much each TD update shifts
        the Q-value toward the new estimate. 0.1 is standard for tabular
        settings with moderate state-space coverage.
    gamma : float
        Discount factor. 0.99 makes the agent far-sighted, valuing future
        room visits almost as much as immediate ones. This is important
        because the best strategy often involves delaying a high-value
        visit until the crowd peak has passed.
    epsilon : float
        Initial exploration rate. Starts at 1.0 (fully random) and decays
        toward epsilon_min via multiplicative decay after each episode.
    epsilon_decay : float
        Per-episode multiplicative decay for epsilon. 0.9995 gives a slow
        annealing: after 2500 episodes, epsilon ~ 0.29; after 5000, ~0.08.
    epsilon_min : float
        Floor for epsilon. Ensures a small amount of exploration persists
        even late in training, preventing the policy from getting stuck
        in a local optimum.
    """

    def __init__(self, action_size: int, lr: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0):
        # Q-table: maps state tuples -> np.ndarray of shape (action_size,).
        # defaultdict auto-initialises unseen states to all-zero Q-values.
        self.q_table = defaultdict(lambda: np.zeros(action_size, dtype=float))
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = 0.9995                    # multiplicative per episode
        self.epsilon_min = 0.01                        # exploration floor
        self.action_size = int(action_size)

    def select_action(self, state, valid_mask: np.ndarray, rng: np.random.Generator) -> int:
        """Choose an action using epsilon-greedy over valid actions only.

        With probability epsilon, pick uniformly among valid actions (explore).
        Otherwise, pick the valid action with the highest Q-value (exploit).
        Invalid actions are masked to -1e9 before argmax so they are never
        selected during exploitation.
        """
        valid_actions = np.flatnonzero(valid_mask)
        if len(valid_actions) == 0:
            return 0                                   # safety fallback

        # Exploration: uniform random over valid actions.
        if rng.random() < self.epsilon:
            return int(rng.choice(valid_actions))

        # Exploitation: greedy over Q-values, masking invalid actions.
        q = self.q_table[state].copy()
        q[valid_mask == 0] = -1e9                      # mask invalid to -inf
        return int(np.argmax(q))

    def update(self, s, a: int, r: float, ns, done: bool, next_valid_mask: np.ndarray) -> None:
        """Apply the one-step Q-learning (off-policy TD(0)) update rule.

        Q(s, a) <- Q(s, a) + lr * [r + gamma * max_a' Q(s', a') - Q(s, a)]

        The max over next-state Q-values is restricted to valid actions in
        the next state (masked Q-learning), ensuring the agent does not
        overestimate by considering unreachable transitions.

        At terminal states (done=True), the TD target is just the immediate
        reward r (no future value).
        """
        q_sa = self.q_table[s][a]
        # Mask invalid next-state actions before taking max.
        next_q = self.q_table[ns].copy()
        next_q[next_valid_mask == 0] = -1e9
        # TD target: r (terminal) or r + gamma * max Q(s', a').
        td_target = r if done else r + self.gamma * np.max(next_q)
        # Incremental update toward the TD target.
        self.q_table[s][a] = q_sa + self.lr * (td_target - q_sa)

    def decay_epsilon(self) -> None:
        """Multiplicatively decay epsilon, floored at epsilon_min.

        Called once per episode (not per step) to gradually shift the
        agent from exploration to exploitation.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# Training history
# =============================================================================


@dataclass
class TrainingHistory:
    """Training curves collected during tabular Q-learning.

    Attributes
    ----------
    episode_rewards : List[float]
        Total undiscounted return per episode. The learning curve: should
        trend upward as the agent discovers temporal arbitrage.
    episode_lengths : List[int]
        Number of steps per episode. Shorter episodes (reaching EXIT
        quickly) are not necessarily better; the agent should take enough
        steps to visit all key rooms.
    botticelli_offpeak_visits : List[int]
        Per-episode count of steps where the agent was in A11 or A12 AND
        the crowd density was below 0.6. This is the headline metric for
        temporal arbitrage: a trained agent should have more off-peak
        Botticelli visits than the default-path baseline.
    """

    episode_rewards: List[float]
    episode_lengths: List[int]
    botticelli_offpeak_visits: List[int]


# Type alias for a policy function: takes (state_tuple, valid_action_mask)
# and returns an integer action. Used by evaluate_policy and the baseline
# wrapper in baselines.py.
PolicyFn = Callable[[Tuple[str, int, int, int, int], np.ndarray], int]


# =============================================================================
# Policy evaluation
# =============================================================================


def evaluate_policy(
    env: ToyTabularEnv,
    policy_fn: PolicyFn,
    episodes: int = 100,
    seed: int = config.DEFAULT_SEED,  # noqa: ARG001
) -> Dict[str, float]:
    """Evaluate a toy-graph policy over repeated episodes.

    Rolls out the policy for ``episodes`` episodes and collects aggregate
    metrics. Because the environment is deterministic (no observation noise,
    no stochastic transitions), variance across episodes comes only from
    stochastic policies (e.g., random baseline) or epsilon exploration.

    Metrics tracked
    ---------------
    mean_return : float
        Average total undiscounted return across episodes.
    std_return : float
        Standard deviation of per-episode returns (Bessel-corrected).
    crowded_steps : float
        Total number of steps (across all episodes) where the agent was
        in a room with density > 0.8. Lower is better: the agent should
        avoid peak congestion.
    offpeak_botticelli_visits : float
        Total number of steps where the agent was in A11 or A12 with
        density < 0.6. Higher is better: this is the temporal arbitrage
        signal, measuring whether the agent learned to time its
        Botticelli visit to avoid the crowd peak.
    """

    returns = []
    crowded_steps = 0
    bott_visits = 0

    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            mask = env.valid_actions()
            a = policy_fn(s, mask)
            s, r, done, info = env.step(a)
            ep_ret += r
            # Track steps in congested rooms (any room, density > 0.8).
            if info["density"] > 0.8:
                crowded_steps += 1
            # Track off-peak Botticelli visits (the temporal-arbitrage metric).
            if info["room"] in {"A11", "A12"} and info["density"] < 0.6:
                bott_visits += 1
        returns.append(ep_ret)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns, ddof=1) if len(returns) > 1 else 0.0),
        "crowded_steps": float(crowded_steps),
        "offpeak_botticelli_visits": float(bott_visits),
    }


# =============================================================================
# Training loop
# =============================================================================


def train_q_learning(
    episodes: int = config.TABULAR_EPISODES_DEFAULT,
    seed: int = config.DEFAULT_SEED,
) -> tuple[QLearningAgent, TrainingHistory, ToyTabularEnv]:
    """Train tabular Q-learning on the toy graph and return the history.

    This is the main entry point for Phase 2 experiments. It runs the full
    training loop, collecting per-episode metrics into a TrainingHistory
    object that the plotting pipeline consumes.

    What "temporal arbitrage" means
    -------------------------------
    The default museum path pushes visitors into Botticelli (A11/A12) at
    exactly the time when those rooms are most crowded (peak at ~35% of the
    visit horizon). The Q-learning agent learns to deviate from this path:
    it visits minor rooms or corridors during the Botticelli peak, then
    enters A11/A12 after the crowd wave has passed. This timing strategy
    is "temporal arbitrage": exploiting the predictable crowd schedule to
    obtain higher reward from the same set of rooms by choosing WHEN to
    visit them. The botticelli_offpeak_visits metric in TrainingHistory
    directly measures whether the agent has learned this behavior.

    Parameters
    ----------
    episodes : int
        Number of training episodes. Default 2500 (config.TABULAR_EPISODES_DEFAULT)
        is enough for convergence on the toy graph; 50,000 is used for
        publication-quality learning curves.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (agent, history, env)
        agent : QLearningAgent
            The trained agent with a populated Q-table.
        history : TrainingHistory
            Per-episode rewards, lengths, and off-peak Botticelli visit counts.
        env : ToyTabularEnv
            The environment instance (useful for subsequent evaluation calls).
    """

    rng = config.get_rng(seed)
    env = ToyTabularEnv(seed=seed)
    agent = QLearningAgent(action_size=env.action_size)

    # Per-episode accumulators for the training history.
    rewards: List[float] = []
    lengths: List[int] = []
    offpeak_bott: List[int] = []

    for _ in range(int(episodes)):
        s = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_offpeak = 0

        # --- Inner loop: one episode ---
        while not done:
            mask = env.valid_actions()
            a = agent.select_action(s, mask, rng)      # epsilon-greedy
            ns, r, done, info = env.step(a)
            next_mask = env.valid_actions()             # mask for s' (needed by update)
            agent.update(s, a, r, ns, done, next_mask)  # Q-learning TD update

            s = ns
            ep_reward += r
            ep_len += 1
            # Track off-peak Botticelli visits for the temporal-arbitrage metric.
            if info["room"] in {"A11", "A12"} and info["density"] < 0.6:
                ep_offpeak += 1

        # Decay exploration rate after each episode (not after each step).
        agent.decay_epsilon()
        rewards.append(float(ep_reward))
        lengths.append(ep_len)
        offpeak_bott.append(ep_offpeak)

    history = TrainingHistory(
        episode_rewards=rewards,
        episode_lengths=lengths,
        botticelli_offpeak_visits=offpeak_bott,
    )
    return agent, history, env


# =============================================================================
# Greedy policy extraction
# =============================================================================


def greedy_q_policy(agent: QLearningAgent):
    """Build a deterministic greedy policy callable from a trained Q-table.

    Returns a function with signature (state, valid_mask) -> int that always
    picks the valid action with the highest Q-value (no exploration). Used
    at evaluation time after training is complete.
    """

    def _policy(state, mask: np.ndarray) -> int:
        q = agent.q_table[state].copy()
        q[mask == 0] = -1e9                            # mask invalid actions
        return int(np.argmax(q))

    return _policy
