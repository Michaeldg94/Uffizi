"""Single-agent Gymnasium environment for Uffizi Gallery navigation.

This module defines the RL environment for a single controlled visitor
navigating the Uffizi while thousands of NPC visitors (simulated by
CrowdSimulator) create realistic crowd conditions. The agent chooses
which adjacent room to move to at each minute-step, seeking to maximize
cumulative art enjoyment while avoiding overcrowded galleries.

Architecture
------------
The environment wraps a CrowdSimulator instance that runs the NPC crowd
model in lockstep. At each step:
  1. The agent selects a movement action (stay or move to a neighbor).
  2. Invalid actions are caught by the action mask and penalized.
  3. The NPC simulator advances one minute.
  4. The reward is computed from art importance, crowd density, and time.
  5. Optional potential-based reward shaping provides a learning signal.

Observation space (394 or 394 + N_ROOMS dimensions)
----------------------------------------------------
Without visibility masking (default, obs_dim = 4 * N_ROOMS + 2 = 394):
  [0 : N_ROOMS]              One-hot encoding of the agent's current room.
  [N_ROOMS : 2*N_ROOMS]      Density of each room (occupancy / capacity),
                              clipped to [0.0, 1.0]. Full observability.
  [2*N_ROOMS : 3*N_ROOMS]    Density trend (current minus 5-step-ago baseline),
                              clipped to [-1.0, 1.0]. Tells the agent whether
                              rooms are filling or emptying.
  [3*N_ROOMS]                 Normalized time elapsed (0.0 at start, 1.0 at
                              episode end). Encodes urgency.
  [3*N_ROOMS+1 : 4*N_ROOMS+1] Binary visited flags per room (1.0 if visited).
  [4*N_ROOMS+1]               Fatigue (linearly increasing from 0.0 to 1.0).

With visibility masking (obs_dim = 5 * N_ROOMS + 2):
  Density and trend are replaced by partial-observability versions where
  unobserved rooms carry the sentinel value -1.0. An additional N_ROOMS
  binary visibility mask is inserted after the density block. Rooms are
  visible if they are the current room, a neighbor, or the agent is at a
  kiosk room (KIOSK_ROOMS) which provides full museum information.

Action space
------------
Discrete(1 + max_degree), where max_degree is the highest node degree in
the museum graph. Action 0 = stay in current room. Actions 1..max_degree
map to sorted neighbors. Invalid actions (index beyond actual neighbor
count) are caught, replaced with "stay", and penalized.

Reward components
-----------------
  r_art:        importance * crowd_factor * novelty. The core art-enjoyment
                signal. crowd_factor = 1/(1 + alpha * density) penalizes
                crowded rooms. novelty = 1.0 if first visit, 0.2 if revisit.
  r_time:       -0.01 per step. Constant time pressure that discourages
                aimless wandering.
  r_congestion: -0.5 * density when gallery density > 0.8. Corridors and
                transit rooms are exempt (see _compute_reward docstring).
  r_shaping:    Potential-based reward shaping (Ng et al., 1999). Preserves
                optimal policy while accelerating learning.
  Terminal:     +0.1 * |visited_rooms| when the agent reaches EXIT.

References
----------
[Ng99] Ng, Harada, Russell (1999). "Policy invariance under reward
       transformations." ICML. Potential-based shaping theorem.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from uffizi_rl import config
from uffizi_rl.environment.crowd_simulator import CrowdSimulator
from uffizi_rl.environment.museum_graph import all_pairs_shortest_paths, build_uffizi_graph
from uffizi_rl.environment.visitor_profiles import VisitorProfile, sample_type_a_profile

# Try to import Gymnasium; fall back to lightweight stubs if unavailable.
# The stubs allow the environment to be instantiated and tested without
# a full Gymnasium installation (e.g. in CI or analysis-only workflows).
try:
    import gymnasium as gym
    from gymnasium import spaces
    EnvBase = gym.Env
except Exception:  # pragma: no cover
    gym = object

    class _DummyBox:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _DummyDiscrete:
        def __init__(self, n):
            self.n = n

    class _DummySpaces:
        Box = _DummyBox
        Discrete = _DummyDiscrete

    spaces = _DummySpaces()

    class EnvBase:  # pragma: no cover
        pass


# =============================================================================
# Reward decomposition
# =============================================================================


@dataclass
class EnvStepInfo:
    """Structured reward decomposition for one environment step.

    Stored per step so that training logs and analysis scripts can
    disaggregate the total reward into its components. This is essential
    for diagnosing whether the agent is optimizing art enjoyment vs.
    gaming the shaping signal vs. avoiding congestion penalties.

    Attributes
    ----------
    reward_art : float
        Art enjoyment: importance * crowd_factor * novelty.
    reward_time : float
        Constant per-step time penalty (-0.01).
    reward_congestion : float
        Extra penalty for severely overcrowded gallery rooms (>80% density).
        Zero for corridors and transit rooms.
    reward_shaping : float
        Potential-based reward shaping difference (gamma * Phi' - Phi).
        Zero if potential_shaping is disabled.
    density_current : float
        Raw density (occupancy / capacity) at the agent's current room.
        Logged for analysis; not part of the reward calculation.
    """

    reward_art: float
    reward_time: float
    reward_congestion: float
    reward_shaping: float
    density_current: float


# =============================================================================
# Main environment class
# =============================================================================


class UffiziEnv(EnvBase):  # type: ignore[misc]
    """Single-agent Gymnasium environment for museum navigation.

    The agent controls one Type A visitor navigating the Uffizi Gallery
    floor plan. At each minute-step, it chooses to stay or move to an
    adjacent room. The crowd simulator runs thousands of NPC visitors
    in parallel, creating realistic and dynamic congestion patterns.

    The action space is fixed-width Discrete(1 + max_degree) with an
    action mask to handle variable-degree nodes. This avoids the need
    for a multi-discrete or graph-aware action space, which simplifies
    integration with standard RL libraries (SB3, CleanRL).

    Parameters
    ----------
    seed : int
        Base random seed. Each episode uses seed + episode_counter for
        reproducible but non-identical episodes.
    daily_total : int
        Total NPC visitors per simulated day. Controls crowd pressure.
    with_visibility_mask : bool
        If True, density observations are partially masked (only current
        room + neighbors visible, unless at a kiosk room). Adds N_ROOMS
        dimensions to the observation for the visibility mask itself.
    invalid_action_penalty : float
        Reward penalty applied when the agent selects an invalid action
        (index beyond actual neighbor count). Default -0.25. [assumption]
    type_a_fraction : float
        Fraction of NPC visitors that are Type A. Passed to CrowdSimulator.
    episode_minutes : int
        Maximum episode length in minutes. Default: MUSEUM_OPEN_MINUTES (615).
    potential_shaping : bool
        If True, add potential-based reward shaping (Ng et al., 1999).
        Preserves optimal policy while providing denser learning signal.
    shaping_gamma : float
        Discount factor used in the shaping term: gamma * Phi(s') - Phi(s).
        Must match the discount factor used by the RL algorithm for the
        shaping to be policy-invariant.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        seed: int = config.DEFAULT_SEED,
        daily_total: int = config.DAILY_VISITORS_NORMAL,
        with_visibility_mask: bool = False,
        invalid_action_penalty: float = -0.25,
        type_a_fraction: float = config.TYPE_A_FRACTION_DEFAULT,
        episode_minutes: int = config.MUSEUM_OPEN_MINUTES,
        potential_shaping: bool = True,
        shaping_gamma: float = 0.99,
        random_start: bool = False,
        interventions=None,
    ) -> None:
        super().__init__()

        # --- Seeding and graph construction ---
        self.seed_value = seed
        self.rng = config.get_rng(seed)
        self.g = build_uffizi_graph()                   # museum topology (NetworkX graph)
        self._distances = all_pairs_shortest_paths(self.g)  # precomputed hop distances
        self.max_degree = max(dict(self.g.degree()).values())  # determines action space width
        self._episode_counter = 0  # incremented each reset for per-episode seeding

        # --- Mutable episode state ---
        self.current_room = "A1"   # agent starts in the Lorenese Vestibule
        self.time_elapsed = 0      # minutes since episode start
        self.fatigue = 0.0         # linear ramp from 0.0 to 1.0 over the episode
        self.visited = set()       # set of room IDs visited this episode

        # --- Configuration flags ---
        self.with_visibility_mask = bool(with_visibility_mask)
        # Exploring starts: when True, TRAINING episodes begin from a random
        # gallery room (not A1), so the agent gathers experience across the
        # whole museum and learns the reactive dwell-then-move rule everywhere,
        # rather than overfitting to the rooms near the entrance and getting
        # trapped in a lazy local optimum. Evaluation keeps random_start=False
        # (honest A1 start). This is a standard exploration technique, not
        # demonstration/warm-start: the policy is still learned end-to-end.
        self.random_start = bool(random_start)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.episode_minutes = int(episode_minutes)
        # Intervention bundle the agent lives inside (None = baseline). Passed to
        # the NPC crowd simulator so the agent experiences the intervened crowd;
        # extended hours lengthen the agent's own day too.
        self.interventions = interventions
        if interventions is not None and getattr(interventions, "extended_hours", False):
            self.episode_minutes = max(self.episode_minutes, 720)  # 7:00-19:00
        # RAMA booking gate. When RAMA is active the three floor-2 masterpieces
        # are reservation-only: the agent may appreciate one only after checking
        # in during the 10-min entry window it booked (set by the booking
        # wrapper via set_secured_windows). A group with no booking, or first
        # reached after its window has closed, yields nothing.
        self._rama_active = bool(
            interventions is not None and getattr(interventions, "rama", False))
        self._secured_windows: Dict[str, tuple[int, int]] = {}  # group -> (start,end) min-from-open
        self._rama_checkin: Dict[str, bool] = {}                # group -> entered within window
        # Sub-goal reward for keeping a booked appointment (checking in during
        # the window). Bridges the long gap between the pre-visit booking and
        # the masterpiece payoff; set by the booking wrapper (0 = off).
        self.rama_checkin_bonus = 0.0
        # Per-minute penalty for idling inside a booked masterpiece before its
        # window opens (parking). The opportunity cost of a room the agent could
        # be seeing instead; kills the "book one late slot and wait all day"
        # degenerate optimum. Set by the booking wrapper (0 = off).
        self.rama_wait_penalty = 0.0
        self.potential_shaping = bool(potential_shaping)
        self.shaping_gamma = float(shaping_gamma)

        # --- Importance-scaled dwell + slack-based egress (knobs) ---
        # dwell_per_importance: an art lover lingers in proportion to value.
        #   Marginal appreciation in a room declines linearly to zero over
        #   desired_dwell = dwell_per_importance * personal_importance minutes,
        #   so an importance-10 masterpiece earns ~30 min of appreciation and a
        #   minor room only a few. Accumulated across visits, so backtracking
        #   into a satiated room pays ~0.
        # step_cost: small per-minute cost. Small enough never to cut short a
        #   legitimate 30-min masterpiece dwell (art ~10/min dominates), large
        #   enough that once a room is satiated every extra minute is a net loss
        #   -> the agent moves on, and eventually leaves.
        # exit_bonus: positive reward for a voluntary exit (the goal to reach).
        # closing_pressure / egress_buffer: slack-based egress. Zero while the
        #   agent can still reach an exit comfortably (hops + buffer <= minutes
        #   left); escalates per minute of shortfall once behind schedule.
        # noexit_penalty: terminal penalty if still inside at closing (strong,
        #   not a full wipe, which gave no learning gradient).
        self.dwell_per_importance = 3.0      # ideal (uncrowded) appreciation minutes per importance point (imp 10 -> ~30 min)
        self.dwell_importance_floor = 0.0    # no floor: every room's value scales with its encoded importance (minor room = less, not zero)
        self.satiation_shoulder = 5.0        # plateau: full value over the whole dwell, ramping to zero only in the last this-many minutes
        self.dwell_crowd_beta = 1.0          # crowd-driven dwell: progress/min slows with local density (fight-to-the-front), so a packed room takes longer to get through. 0 = fixed dwell.
        self.art_crowd_alpha = 0.5           # GENTLE crowd penalty for art value (profile crowd_alpha=6 is too harsh): crowd diminishes value but mildly, so a crowded masterpiece still far outweighs any random room
        self.step_cost = 0.15
        self.exit_bonus = 50.0
        self.closing_pressure = 4.0
        self.egress_buffer = 5
        self.noexit_penalty = -50.0
        self.completion_k = 15.0             # completion bonus for finishing a must-see's full dwell: makes "stay ~30 min, then move on" the dominant strategy and advances the tour to the next masterpiece
        self.boredom_k = 0.0                 # OFF: penalizing stay-in-satiated backfired -- the agent learned to never stay and LOOP instead (revisit rooms to evade it), the opposite of dwelling. The strong tour pull below is the "leave" signal instead.
        # Tour-shaping potential: a must-see room is one whose PERSONAL importance
        # is >= tour_threshold. The potential pulls the agent toward the nearest
        # must-see it has not yet fully appreciated, so greedy walks a complete
        # masterpiece tour instead of ping-ponging among the local minor rooms.
        # Policy-invariant (Ng et al. 1999): guides learning, not the optimum.
        self.tour_weight = 1.0               # moderate pull toward the nearest uncompleted must-see (w=4 destabilized into ping-pong). Tested here with boredom OFF + exploring starts (the one untested combination).
        self.tour_threshold = 7.0
        self._tour_targets: list[str] = []       # must-see room ids, set per-episode in reset()
        self._extracted: dict[str, float] = {}   # crowd-weighted appreciation progress per room
        self._completed: set = set()             # rooms already fully appreciated (one-time completion bonus)

        # Which visitor profile the controlled agent samples each episode.
        # Default Type A (art lover); subclasses (e.g. the normal tourist) swap
        # in a different sampler via this hook without touching reset().
        self._profile_sampler = sample_type_a_profile
        self.agent_profile: VisitorProfile = self._profile_sampler(self.rng)

        # NPC crowd simulator runs in lockstep with the environment.
        self.sim = CrowdSimulator(
            daily_total=daily_total,
            seed=seed,
            type_a_fraction=type_a_fraction,
            interventions=interventions,
        )

        # Rolling window of density snapshots for computing density trends.
        # Window size = 5 minutes. Trend = current_density - baseline_density.
        self._density_window: deque[np.ndarray] = deque(maxlen=5)

        # --- Action space ---
        # Discrete(1 + max_degree): action 0 = stay, actions 1..max_degree
        # map to sorted neighbors. Not all actions are valid at every room
        # (rooms with fewer neighbors leave high-index actions invalid).
        # The action mask (get_action_mask) communicates validity to the agent.
        self.action_space = spaces.Discrete(1 + self.max_degree)

        # --- Observation space ---
        # Without visibility mask: 5 * N_ROOMS + 3
        #   (one-hot + density + trend + visited + appreciation-progress
        #    + time + egress-slack + fatigue)
        # With visibility mask: 6 * N_ROOMS + 3 (adds the visibility mask vector)
        # The appreciation-progress vector is essential: the art reward's value
        # per minute decays with how long the agent has already dwelled in a
        # room, so the policy must observe that progress to know when to stop
        # dwelling and move on. The binary visited flag is not enough (it flips
        # after the first minute), which made "dwell, then move on" unlearnable.
        obs_dim = 5 * config.N_ROOMS + 3  # +1 for the egress-slack feature
        if self.with_visibility_mask:
            obs_dim += config.N_ROOMS  # extra N_ROOMS dims for visibility mask

        self.observation_space = spaces.Box(
            low=-1.0,    # -1.0 needed for UNKNOWN_DENSITY_SENTINEL
            high=2.0,    # densities can slightly exceed 1.0 in edge cases
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # =========================================================================
    # Seeding and neighbor utilities
    # =========================================================================

    def seed(self, seed: int | None = None) -> None:
        """Reset the environment RNG to a reproducible base seed.

        Also resets the episode counter so that subsequent reset() calls
        produce the same sequence of episode seeds as the original run.

        Parameters
        ----------
        seed : int or None
            New base seed. If None, re-uses the existing seed_value.
        """

        if seed is None:
            seed = self.seed_value
        self.seed_value = int(seed)
        self.rng = config.get_rng(seed)
        self._episode_counter = 0  # reset counter for reproducible episode sequence

    def _sorted_neighbors(self, room: str) -> list[str]:
        """Return the neighbors of a room in deterministic sorted order.

        Sorting ensures that action index i always maps to the same neighbor
        for a given room, regardless of graph construction order. This is
        critical for the action mask and action_to_room mapping to be
        consistent across episodes.

        Returns
        -------
        list[str]
            Sorted neighbor room IDs.
        """
        return sorted(self.g.neighbors(room))

    # =========================================================================
    # Action masking
    # =========================================================================

    def get_action_mask(self) -> np.ndarray:
        """Return a fixed-width binary mask indicating valid movement actions.

        The action space is Discrete(1 + max_degree) for ALL rooms, but most
        rooms have fewer neighbors than max_degree. The mask communicates
        which actions are valid at the current room:

          mask[0] = 1 always (action 0 = stay in current room, always valid).
          mask[1 + i] = 1 if sorted_neighbors[i] exists, else 0.

        The sb3-contrib MaskablePPO algorithm reads this mask to zero out
        the logits of invalid actions before sampling, ensuring the agent
        never selects an impossible move during training. If an unmasked
        algorithm selects an invalid action, step() catches it, substitutes
        "stay", and applies invalid_action_penalty.

        Returns
        -------
        np.ndarray
            Shape (1 + max_degree,), dtype int8. 1 = valid, 0 = invalid.
        """

        neighbors = self._sorted_neighbors(self.current_room)
        mask = np.zeros(1 + self.max_degree, dtype=np.int8)
        mask[0] = 1  # "stay" is always valid
        for i in range(len(neighbors)):
            mask[1 + i] = 1  # valid neighbor actions
        return mask

    def action_to_room(self, action: int) -> str:
        """Map an action index to the corresponding target room.

        Action 0 always means "stay in the current room." Actions 1 through
        max_degree map to sorted neighbors. If the action index is out of
        range (invalid), defaults to "stay" as a safety fallback.

        Parameters
        ----------
        action : int
            Action index from the agent's policy.

        Returns
        -------
        str
            Target room ID.
        """

        if action == 0:
            return self.current_room  # stay action
        neighbors = self._sorted_neighbors(self.current_room)
        idx = action - 1
        if 0 <= idx < len(neighbors):
            return neighbors[idx]  # valid neighbor
        return self.current_room  # fallback: invalid action treated as "stay"

    def _eject_action(self) -> int:
        """Action index that moves one hop toward the nearest exit.

        Used by the closing-time ejection: the guards override the agent's
        choice and walk it to the exit. Picks the neighbor with the smallest
        hop-distance to EXIT, so the distance strictly decreases each step and
        the visitor reaches a door in exactly ``_distance_to_exit`` steps.
        """

        if self.current_room == "EXIT":
            return 0
        neighbors = self._sorted_neighbors(self.current_room)
        if not neighbors:
            return 0
        best_i, best_d = 0, None
        for i, nb in enumerate(neighbors):
            d = self._distance_to_exit(nb)
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        return best_i + 1  # action 0 = stay; neighbors map to 1..max_degree

    # =========================================================================
    # Density and observation construction
    # =========================================================================

    def _density_all_rooms(self, include_agent: bool = True) -> np.ndarray:
        """Compute current room densities (occupancy / capacity) for all rooms.

        Density is the ratio of current occupancy to comfortable capacity.
        Values above 1.0 indicate overcrowding beyond the designed limit.
        The controlled agent's presence is included by default so the
        observation reflects what the agent actually experiences.

        Parameters
        ----------
        include_agent : bool
            If True, add 1 to the occupancy of the agent's current room.
            Set to False when computing density for reward without double-
            counting (not currently used, but available for extensions).

        Returns
        -------
        np.ndarray
            Shape (N_ROOMS,), density per room. Unbounded above but
            typically in [0.0, 2.0].
        """

        dens = np.zeros(config.N_ROOMS, dtype=float)
        for room, idx in config.ROOM_TO_IDX.items():
            cap = self.g.nodes[room]["capacity"]
            occ = self.sim.occ.get(room, 0)  # NPC occupancy from simulator
            if include_agent and room == self.current_room:
                occ += 1  # count the controlled agent
            dens[idx] = occ / max(1.0, cap)  # max(1.0, cap) prevents division by zero
        return dens

    def _trend_all_rooms(self, current: np.ndarray) -> np.ndarray:
        """Compute the short-window density trend for all rooms.

        Trend = current_density - density_5_minutes_ago, clipped to [-1, 1].
        Positive trend means the room is filling up; negative means it is
        emptying. This gives the agent a temporal signal: it can learn to
        avoid rooms that are trending upward (about to become crowded) and
        move toward rooms that are trending downward.

        Parameters
        ----------
        current : np.ndarray
            Current density snapshot, shape (N_ROOMS,).

        Returns
        -------
        np.ndarray
            Shape (N_ROOMS,), trend values in [-1.0, 1.0].
        """

        if not self._density_window:
            return np.zeros_like(current)  # no history yet: trend = 0
        baseline = self._density_window[0]  # oldest snapshot in the window (5 min ago)
        return np.clip(current - baseline, -1.0, 1.0)

    def _visible_rooms(self) -> set[str]:
        """Return the set of rooms whose density is observable by the agent.

        Used only when with_visibility_mask=True. Implements a partial
        observability model:
          - The agent always sees its own room and direct neighbors.
          - If the agent is at a kiosk room (A2, A15, A24, D9), it sees
            all rooms (the kiosk displays real-time density for the entire
            museum). [assumption]

        This creates a natural information-gathering incentive: the agent
        can move to a kiosk room to observe the full museum state before
        deciding which wing to explore.

        Returns
        -------
        set[str]
            Room IDs whose density is visible to the agent.
        """

        visible = {self.current_room} | set(self.g.neighbors(self.current_room))
        if self.current_room in config.KIOSK_ROOMS:
            visible = set(self.g.nodes)  # kiosk provides full museum information
        return visible

    def _appreciation_progress(self) -> np.ndarray:
        """Per-room appreciation progress in [0, 1].

        progress = extracted / ideal_dwell, clipped to 1. 0 = fresh (full art
        value per minute still available), 1 = fully appreciated (staying pays
        nothing). This is the exact variable the art reward's novelty term
        depends on, so the policy must be able to observe it to decide when to
        stop dwelling in the current room and head for the next masterpiece.
        The binary visited flag cannot carry this (it flips after one minute).
        """
        iv = self.agent_profile.importance_vector
        ideal = np.maximum(
            1.0,
            self.dwell_per_importance * np.maximum(0.0, iv - self.dwell_importance_floor),
        )
        progress = np.zeros(config.N_ROOMS, dtype=float)
        for room, extracted in self._extracted.items():
            idx = config.ROOM_TO_IDX.get(room)
            if idx is not None:
                progress[idx] = min(1.0, extracted / ideal[idx])
        return progress

    def _build_observation(self) -> np.ndarray:
        """Assemble the observation vector for the current state.

        Layout WITHOUT visibility masking (5 * N_ROOMS + 3 dims):
          one-hot current room       (which room am I in?)
          density per room [0, 1]     (how crowded is each room?)
          density trend [-1, 1]       (filling or emptying?)
          normalized time [0, 1]      (how much time is left?)
          egress slack [-1, 1]        (can I still afford to sightsee?)
          visited flags {0, 1}        (which rooms have I seen?)
          appreciation progress [0,1] (how much of each room's dwell I have used)
          fatigue [0, 1]              (how tired am I?)

        Layout WITH visibility masking (492 dimensions):
          dims [0:98]      one-hot current room
          dims [98:196]    density (visible rooms only; sentinel -1 elsewhere)
          dims [196:294]   visibility mask {0, 1}       (which rooms can I see?)
          dims [294:392]   density trend (visible only; sentinel -1 elsewhere)
          dims [392]       normalized time
          dims [393:491]   visited flags
          dims [491]       fatigue

        Returns
        -------
        np.ndarray
            Float32 observation vector, shape (obs_dim,).
        """

        # --- Current room: one-hot encoding ---
        one_hot = np.zeros(config.N_ROOMS, dtype=float)
        one_hot[config.ROOM_TO_IDX[self.current_room]] = 1.0

        # --- Density and trend for all rooms ---
        dens_all = self._density_all_rooms()
        trend_all = self._trend_all_rooms(dens_all)

        # --- Visited flags: binary indicator of rooms already seen ---
        visited_bin = np.zeros(config.N_ROOMS, dtype=float)
        for room in self.visited:
            visited_bin[config.ROOM_TO_IDX[room]] = 1.0

        # --- Scalar features ---
        time_norm = np.array([self.time_elapsed / max(1, self.episode_minutes)], dtype=float)
        fatigue = np.array([self.fatigue], dtype=float)
        # Egress slack: minutes left minus hops to the nearest exit, normalized.
        # Positive = the visitor can still afford to sightsee; <= 0 = must
        # already be heading out (closing-time ejection is imminent). This is
        # the feature that lets the policy plan its exit around closing.
        t_remaining = self.episode_minutes - self.time_elapsed
        egress_slack = np.array([np.clip(
            (t_remaining - self._distance_to_exit(self.current_room)) / max(1, self.episode_minutes),
            -1.0, 1.0)], dtype=float)

        # Per-room appreciation progress: lets the policy tell a fresh room
        # from one it has already drained, so it can dwell then move on.
        progress = self._appreciation_progress()

        if not self.with_visibility_mask:
            # Full observability: agent sees all room densities, clipped to [0, 1]
            dens_clipped = np.clip(dens_all, 0.0, 1.0)
            obs = np.concatenate([one_hot, dens_clipped, trend_all, time_norm, egress_slack, visited_bin, progress, fatigue])
            return obs.astype(np.float32)

        # --- Partial observability mode ---
        # Unobserved rooms get the sentinel value -1.0 so the agent can
        # distinguish "I don't know this room's density" from "this room
        # is empty." The visibility mask is included as a separate feature
        # block so the network can learn to weight known vs. unknown rooms.
        vis_rooms = self._visible_rooms()
        dens_obs = np.full(config.N_ROOMS, config.UNKNOWN_DENSITY_SENTINEL, dtype=float)
        trend_obs = np.full(config.N_ROOMS, config.UNKNOWN_DENSITY_SENTINEL, dtype=float)
        vis_mask = np.zeros(config.N_ROOMS, dtype=float)

        for room in vis_rooms:
            idx = config.ROOM_TO_IDX[room]
            dens_obs[idx] = np.clip(dens_all[idx], 0.0, 1.0)  # reveal density
            trend_obs[idx] = trend_all[idx]                     # reveal trend
            vis_mask[idx] = 1.0                                 # mark as visible

        obs = np.concatenate([one_hot, dens_obs, vis_mask, trend_obs, time_norm, egress_slack, visited_bin, progress, fatigue])
        return obs.astype(np.float32)

    # =========================================================================
    # Reward computation
    # =========================================================================

    def compute_reward(self, room_id: str, visited_before: bool, is_stay: bool = False) -> Tuple[float, EnvStepInfo]:
        """Compute the task reward for occupying ``room_id`` this minute.

        The reward has three additive components:

        1. Art enjoyment (r_art):
           importance * crowd_factor * novelty
           - importance: cultural significance of the room (1-10 from config).
           - crowd_factor: 1 / (1 + alpha * density). Penalizes crowded rooms.
             At density=0, crowd_factor=1 (full enjoyment). At density=1 with
             alpha=6 (Type A), crowd_factor=0.14 (86% enjoyment lost).
           - novelty: 1.0 on first visit, 0.2 on revisits. Encourages
             exploration over camping in one high-importance room.

        2. Time penalty (r_time):
           -0.01 per step. Creates urgency: the agent cannot afford to
           stand still indefinitely. Over a 615-step episode, this totals
           -6.15, roughly comparable to visiting 2-3 medium-importance rooms.

        3. Congestion penalty (r_congestion):
           -0.5 * density when a gallery room exceeds 80% capacity.
           Corridors and transit rooms are exempt (see below).

        Corridor exemption rationale
        ----------------------------
        Corridors (A1, A2, A23, A24, B1, D5, D9, D27), entry/exit points,
        staircases, and the terrace are exempt from the congestion penalty.
        Their high density is a topological artifact: they are mandatory
        transit nodes that every visitor passes through. Penalizing the
        agent for corridor congestion would punish it for a condition it
        cannot avoid, distorting the learning signal. The penalty targets
        only gallery rooms where the agent has a meaningful choice to
        enter or avoid.

        Parameters
        ----------
        room_id : str
            Room the agent is currently occupying.
        visited_before : bool
            Whether the agent has visited this room in the current episode.

        Returns
        -------
        tuple[float, EnvStepInfo]
            (total_reward, decomposition). The decomposition excludes the
            shaping term (added later in step()).
        """

        capacity = self.g.nodes[room_id]["capacity"]

        # Occupancy includes NPC visitors + the controlled agent if present
        occ = self.sim.occ[room_id] + (1 if room_id == self.current_room else 0)
        dens = occ / max(1.0, capacity)

        # crowd_factor: the crowd diminishes a room's value, but GENTLY
        # (art_crowd_alpha ~0.5, far below the profile's crowd_alpha=6). So a
        # crowded masterpiece keeps most of its value (at density 1.5,
        # crowd_factor ~0.47, not ~0.07), and a packed Botticelli still far
        # outweighs any random uncrowded room. Crowd genuinely reduces value
        # (no fake fixed value) and an off-peak visit is worth more, but it is
        # never so punishing that the agent would rather see a side gallery.
        crowd_factor = 1.0 / (1.0 + self.art_crowd_alpha * dens * dens)

        # Personal importance: the agent's preference vector, not the room's
        # generic cultural importance -- what THIS visitor actually cares about
        # (concentrated on the masterpieces for a Type A art lover).
        if room_id in config.ROOM_TO_IDX:
            idx = config.ROOM_TO_IDX[room_id]
            importance = float(self.agent_profile.importance_vector[idx])
        else:
            importance = 0.0

        # RAMA gate. A booked masterpiece pays nothing until the agent has
        # checked in during its 10-min entry window; before the window it is
        # merely waiting (no value yet, no dwell consumed), and arriving after
        # the window has closed without a prior check-in forfeits the slot for
        # good. Unbooked/declined masterpieces are inaccessible. Once checked
        # in, appreciation proceeds normally (the full ~30-min calm visit).
        rama_gated = False
        rama_just_checkin = False
        rama_waiting = False
        if self._rama_active and room_id in config.MASTERPIECE_ROOMS:
            key = self._rama_group(room_id)
            win = self._secured_windows.get(key)
            if win is None:
                rama_gated = True
            else:
                w_start, w_end = win
                if (not self._rama_checkin.get(key, False)
                        and w_start <= self.time_elapsed <= w_end):
                    self._rama_checkin[key] = True
                    rama_just_checkin = True
                rama_gated = not self._rama_checkin.get(key, False)
                rama_waiting = rama_gated and self.time_elapsed < w_start  # parked, window not yet open

        # Importance-scaled dwell, FIXED length. The agent appreciates a room
        # for up to ideal_dwell minutes (scaled by importance, ~30 for a
        # masterpiece), value declining to zero over that span. The dwell
        # length is fixed; crowd does NOT stretch it. Instead the gentle
        # crowd_factor above reduces the value PER MINUTE. So a masterpiece's
        # total value does fall with crowd (no fake fixed value), but only
        # mildly, and stays far above any random room. The art lover, dwelling
        # the full span, loses more total value to the crowd than a quick
        # glance would: penalized, but still goes and still gets a big payoff.
        # Every room's dwell scales with its encoded importance (a minor room
        # earns a short dwell and small value, not zero), so the whole museum
        # has value, with the masterpieces dominating.
        ideal_dwell = max(1.0, self.dwell_per_importance * max(0.0, importance - self.dwell_importance_floor))
        extracted = self._extracted.get(room_id, 0.0)
        # Plateau satiation: the room pays its full per-minute value for the
        # whole ideal-dwell window, winding down only over the final
        # satiation_shoulder minutes, then zero. A glance captures only a small
        # fraction of the room's worth, so "stay until satiated, then move on"
        # beats drifting away early. A triangular decay (the previous shape)
        # made late minutes nearly worthless, so the agent grabbed the cheap
        # first minutes and left: the masterpiece under-dwell we kept seeing.
        novelty = max(0.0, min(1.0, (ideal_dwell - extracted) / self.satiation_shoulder))
        # Crowd-driven dwell: you make less appreciation progress per minute the
        # more crowded the room (fighting to the front), so a packed room takes
        # more minutes to get through. With a fixed closing time this means high
        # crowd -> longer dwells -> fewer rooms fit -> fewer total points.
        if rama_gated:
            # Waiting for the window or slot forfeit: no value, no progress, so
            # the dwell budget is preserved for when the window actually opens.
            r_art = 0.0
        else:
            self._extracted[room_id] = extracted + 1.0 / (1.0 + self.dwell_crowd_beta * dens)
            r_art = float(importance * crowd_factor * novelty)
        r_time = -self.step_cost  # per-minute cost; lingering after satiation is a net loss

        # One-time completion bonus for FULLY appreciating a must-see room.
        # This is the discrete, learnable target that makes the agent dwell a
        # masterpiece to the end (~30 min) instead of glancing, and -- because
        # the tour potential advances only when a must-see is completed -- it is
        # what unsticks the tour so the agent moves on to the next masterpiece.
        # Restricted to must-sees (importance >= tour_threshold) so the agent
        # does not grind out trivial rooms for the bonus. Crowd-independent: the
        # satisfaction of having properly seen a great work does not depend on
        # how crowded it was, so the agent completes the crowded Leonardo and
        # Michelangelo too rather than substituting calm-room time.
        r_completion = 0.0
        if (not rama_gated and ideal_dwell > 1.0 and room_id not in self._completed
                and self._extracted.get(room_id, 0.0) >= ideal_dwell):
            self._completed.add(room_id)
            if importance >= self.tour_threshold:
                r_completion = float(self.completion_k * importance)

        # Crowd is modeled via the stretched dwell above, not a separate
        # congestion penalty (which double-counted aversion and made the agent
        # skip crowded masterpieces instead of braving them).
        r_congestion = 0.0

        # Slack-based egress pressure: zero while the agent can still reach an
        # exit comfortably (hops + buffer <= minutes left), then escalating in
        # proportion to the shortfall once it is behind schedule. This forces
        # it to turn toward a door in time without nagging during normal
        # touring (so returns are not crushed). It is the learnable egress
        # signal, and the shortfall grows the deeper and later it is.
        time_remaining = self.episode_minutes - self.time_elapsed
        deficit = self._distance_to_exit(room_id) + self.egress_buffer - time_remaining
        r_closing = float(-self.closing_pressure * max(0.0, deficit))

        # Boredom: choosing to STAY in a room already fully appreciated is a net
        # loss. With the completion bonus pulling the agent to stay until
        # satiated and boredom pushing it to leave once satiated, the optimal
        # dwell is a sharp ~ideal_dwell, and the degenerate "camp one room
        # forever" policy is killed. Only the stay action triggers it, so
        # walking THROUGH a satiated room (necessary transit) is never penalized.
        r_boredom = 0.0
        if is_stay and extracted >= ideal_dwell:
            r_boredom = -self.boredom_k

        r_checkin = self.rama_checkin_bonus if rama_just_checkin else 0.0
        r_wait = -self.rama_wait_penalty if (rama_waiting and is_stay) else 0.0
        reward = r_art + r_time + r_congestion + r_closing + r_completion + r_boredom + r_checkin + r_wait
        info = EnvStepInfo(
            reward_art=r_art,
            reward_time=r_time,
            reward_congestion=r_congestion,
            reward_shaping=0.0,      # filled in by step() after potential computation
            density_current=float(dens),
        )
        return reward, info

    # =========================================================================
    # RAMA booking gate (reservation-anchored masterpiece access)
    # =========================================================================

    @staticmethod
    def _rama_group(room_id: str) -> str | None:
        """Map a masterpiece room to its booking group. A11 and A12 share the
        single Botticelli ledger (one forced traversal); A35 is Leonardo, A38
        is Raphael."""
        if room_id in ("A11", "A12"):
            return "bott"
        if room_id == "A35":
            return "leo"
        if room_id == "A38":
            return "raph"
        return None

    def set_secured_windows(self, windows: Dict[str, tuple[int, int]]) -> None:
        """Install the agent's booked masterpiece entry windows, mapping booking
        group ('bott'/'leo'/'raph') to (entry_start, entry_end) in minutes from
        museum open. Called by the booking wrapper after the pre-visit booking
        phase; a missing group means that masterpiece was declined or sold out."""
        self._secured_windows = dict(windows)
        self._rama_checkin = {}

    def rama_access_features(self) -> tuple[float, float, float, float]:
        """Perception of the room the agent is standing in, as a reservation
        problem. Lets the policy connect 'I get nothing here' to 'I have no (or
        not-yet-open) reservation for this room'. Returns:
          is_masterpiece : 1 if the current room is a RAMA-gated masterpiece
          unbooked       : 1 if it is a masterpiece with no reservation (no access)
          access_now     : 1 if I may appreciate it right now (checked in, or my
                           window is open) -- i.e. standing here pays off now
          time_to_open   : normalized minutes until my window opens (>0 = wait,
                           <=0 = open or past); 0 when not applicable
        All zero when the current room is not a gated masterpiece."""
        cur = self.current_room
        if not (self._rama_active and cur in config.MASTERPIECE_ROOMS):
            return (0.0, 0.0, 0.0, 0.0)
        key = self._rama_group(cur)
        win = self._secured_windows.get(key)
        if win is None:
            return (1.0, 1.0, 0.0, 0.0)  # masterpiece I never reserved: no access
        w_start, w_end = win
        em = max(1, self.episode_minutes)
        checked = self._rama_checkin.get(key, False)
        access_now = 1.0 if (checked or w_start <= self.time_elapsed <= w_end) else 0.0
        ttl = float(np.clip((w_start - self.time_elapsed) / em, -1.0, 1.0))
        return (1.0, 0.0, access_now, ttl)

    # =========================================================================
    # Potential-based reward shaping
    # =========================================================================

    def _distance_to_exit(self, room_id: str) -> int:
        """Return the shortest hop distance from a room to the EXIT node.

        Used by the potential function to encode exit urgency. Rooms
        closer to EXIT have lower distance, reducing the exit penalty
        component of the potential.

        Parameters
        ----------
        room_id : str
            Room to query.

        Returns
        -------
        int
            Hop count to EXIT. Returns 999 if the room is unreachable
            (should not happen in a connected graph).
        """

        return self._distances.get(room_id, {}).get("EXIT", 999)

    def _potential(self) -> float:
        """Potential function Phi(s) for potential-based reward shaping.

        The shaping reward at each step is: F(s, s') = gamma * Phi(s') - Phi(s).
        By the theorem of Ng, Harada & Russell (1999), this additive shaping
        preserves the set of optimal policies under the original (unshaped)
        reward, while providing a denser learning signal that accelerates
        convergence.

        The potential combines three signals:

        1. Visited importance (weight 0.03):
           Sum of importance scores of all gallery rooms visited so far.
           This rewards exploration: visiting more (and higher-importance)
           rooms increases the potential, producing a positive shaping
           reward when the agent enters a new gallery.

        2. Tour pull (weight tour_weight):
           negative hop distance to the nearest must-see room not yet fully
           appreciated. Moving toward the next uncompleted masterpiece raises
           the potential (positive shaping reward), giving the agent a clear
           gradient to walk the corridor to it rather than ping-ponging among
           the local minor rooms. Zero once every must-see is completed, so
           the exit term then leads the agent to a door. This replaces an
           earlier current-room-value term that used the profile's steep
           crowd_alpha (6) and so penalized being in a crowded masterpiece,
           steering the agent away from exactly the rooms it should brave.

        3. Exit urgency penalty (weight 0.05):
           (time_elapsed / episode_length) * distance_to_exit.
           As time runs out, the potential drops if the agent is far from
           the exit. This gently steers the agent toward EXIT in the
           final phase of the episode without overriding the art-seeking
           objective early on.

        Returns
        -------
        float
            Potential value for the current state.
        """

        # Exclude non-gallery nodes from the importance sum; they have
        # importance 0-1 and are not meaningful art-viewing locations.
        visited_gallery_rooms = [
            room
            for room in self.visited
            if room not in {"ENTRY", "EXIT", "LANZI_STAIRCASE", "BUONTALENTI_STAIRCASE", "PANORAMIC_TERRACE"}
        ]
        # Visited importance uses the agent's PERSONAL importance vector
        # for consistency with the reward (the metric the agent should
        # optimize is its own preference satisfaction, not a generic
        # room-cultural-importance score).
        visited_importance = sum(
            float(self.agent_profile.importance_vector[config.ROOM_TO_IDX[room]])
            for room in visited_gallery_rooms
            if room in config.ROOM_TO_IDX
        )
        # Tour pull: hop distance to the nearest must-see room not yet fully
        # appreciated. Moving toward it raises the potential; once every
        # must-see is completed this is zero and the exit term leads.
        remaining = [t for t in self._tour_targets if t not in self._completed]
        if remaining:
            dist_here = self._distances.get(self.current_room, {})
            tour_distance = float(min(dist_here.get(t, 999) for t in remaining))
        else:
            tour_distance = 0.0
        # Exit urgency: grows from 0 to 1 over the episode, penalizing
        # distance from EXIT more heavily as time runs out
        exit_urgency = self.time_elapsed / max(1, self.episode_minutes)
        exit_penalty = exit_urgency * self._distance_to_exit(self.current_room)
        # Weighted combination (policy-invariant potential shaping). Coefficients
        # keep the shaping magnitude comparable to, but below, the task reward.
        return float(
            0.03 * visited_importance
            - self.tour_weight * tour_distance
            - 0.05 * exit_penalty
        )

    # =========================================================================
    # Episode lifecycle
    # =========================================================================

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """Reset the episode and return the initial observation.

        Each episode uses a deterministic seed (base_seed + episode_counter),
        producing a unique but reproducible crowd scenario. The agent starts
        in A1 (Lorenese Vestibule) with a fresh Type A profile.

        Parameters
        ----------
        seed : int or None
            If provided, resets the base seed and episode counter.
        options : dict or None
            Unused. Present for Gymnasium API compatibility.

        Returns
        -------
        tuple[np.ndarray, dict]
            (observation, info). The info dict is empty at reset.
        """

        if seed is not None:
            self.seed(seed)

        # Per-episode seed: base + counter gives each episode a unique but
        # reproducible RNG state. Counter increments ensure non-identical episodes.
        episode_seed = self.seed_value + self._episode_counter
        self._episode_counter += 1
        self.rng = config.get_rng(episode_seed)

        # Reset mutable episode state
        if self.random_start:
            # Exploring starts (training only): wake up in a random gallery room
            # so the agent learns to act well everywhere, not just near A1.
            non_gallery = {"ENTRY", "EXIT", "LANZI_STAIRCASE", "BUONTALENTI_STAIRCASE", "PANORAMIC_TERRACE"}
            gallery = [r for r in config.ROOM_TO_IDX if r not in non_gallery]
            self.current_room = str(self.rng.choice(gallery))
        else:
            self.current_room = "A1"         # start in the Lorenese Vestibule (honest eval start)
        self.time_elapsed = 0
        self.fatigue = 0.0
        self.visited = {self.current_room}
        self.agent_profile = self._profile_sampler(self.rng)  # fresh visitor profile
        # Secondary-attractor enrichment: the agent perceives the same per-room
        # importance lift the curatorial table applies to the NPC crowd, so the
        # enriched secondary stars become genuinely worth its time (and may cross
        # the tour threshold below, drawing the visit outward from the 3-4
        # masterpieces). Identical delta + 10.0 cap as crowd_simulator init.
        if self.interventions is not None and getattr(
                self.interventions, "secondary_attractor_enrichment", False):
            iv = self.agent_profile.importance_vector
            for rid, channels in config.ROOM_CURATION.items():
                idx = config.ROOM_TO_IDX.get(rid)
                if idx is None:
                    continue
                imp_delta = sum(
                    config.CHANNEL_EFFECTS.get(c, {}).get("importance", 0.0)
                    for c in channels)
                iv[idx] = min(10.0, float(iv[idx]) + imp_delta)
        self._extracted = {}         # reset crowd-weighted appreciation progress
        self._completed = set()      # reset completed-room set
        self._secured_windows = {}   # RAMA bookings, installed by the wrapper post-reset
        self._rama_checkin = {}      # RAMA per-group check-in flags
        # Must-see targets for the tour-shaping potential: the rooms this
        # visitor most cares about (personal importance >= tour_threshold).
        self._tour_targets = [
            r for r, i in config.ROOM_TO_IDX.items()
            if float(self.agent_profile.importance_vector[i]) >= self.tour_threshold
        ]

        # Rebuild the NPC crowd simulator with the episode-specific seed.
        # This ensures that the NPC arrival pattern differs across episodes
        # but is reproducible for a given seed.
        self.sim = CrowdSimulator(
            daily_total=self.sim.daily_total,
            seed=episode_seed,
            type_a_fraction=self.sim.type_a_fraction,
            heterogeneity_scale=self.sim.heterogeneity_scale,
            trail_acceptance_prob=self.sim.trail_acceptance_prob,
            interventions=self.interventions,
        )
        self.sim.step_arrivals()  # process minute-0 arrivals

        # Pre-fill the density window so that trend features are defined
        # from the first step (all trends start at zero).
        self._density_window.clear()
        dens = self._density_all_rooms()
        for _ in range(5):
            self._density_window.append(dens.copy())

        return self._build_observation(), {}

    def step(self, action: int):
        """Advance the environment by one minute.

        Execution order:
          1. Validate the action against the action mask.
          2. Record pre-move potential (for shaping).
          3. Move the agent to the target room.
          4. Advance the NPC simulator by one minute.
          5. Compute task reward (art + time + congestion).
          6. Compute potential-based shaping reward.
          7. Add terminal bonus and invalid action penalty.
          8. Check termination (reached EXIT) and truncation (time limit).

        Parameters
        ----------
        action : int
            Action index from the agent's policy. 0 = stay, 1..max_degree
            = move to sorted neighbor.

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict]
            (observation, reward, terminated, truncated, info).
            terminated = True when the agent reaches EXIT.
            truncated = True when time_elapsed >= episode_minutes.
        """

        # --- Step 1: action validation ---
        # If the action is invalid (out of range or masked), substitute
        # "stay" (action 0) and apply the invalid_action_penalty. This
        # teaches unmasked algorithms to respect the graph structure.
        valid_mask = self.get_action_mask()
        valid_action = 0 <= action < len(valid_mask) and valid_mask[action] == 1
        penalty_invalid = 0.0
        if not valid_action:
            action = 0  # fall back to "stay"
            penalty_invalid = self.invalid_action_penalty  # -0.25 by default

        # --- Step 2: record pre-move potential ---
        prev_potential = self._potential() if self.potential_shaping else 0.0

        # --- Step 3: move the agent ---
        target_room = self.action_to_room(action)
        self.current_room = target_room

        visited_before = target_room in self.visited
        self.visited.add(target_room)

        # --- Step 4: advance NPC simulator ---
        # One minute of NPC movement, arrivals, departures, and dwell ticks.
        sim_stats = self.sim.step()

        # Update density window for trend computation
        dens = self._density_all_rooms()
        self._density_window.append(dens.copy())

        # --- Step 5: compute task reward ---
        reward, reward_info = self.compute_reward(target_room, visited_before, is_stay=(action == 0))

        # --- Step 6: advance time and fatigue ---
        self.time_elapsed += 1
        self.fatigue = min(1.0, self.fatigue + 1.0 / max(1, self.episode_minutes))

        # --- Step 7: termination and truncation ---
        terminated = target_room in {"EXIT"}   # agent voluntarily reached an exit
        truncated = (self.time_elapsed >= self.episode_minutes) and not terminated  # closing time

        # --- Step 8: potential-based reward shaping ---
        # F(s, s') = gamma * Phi(s') - Phi(s). Provably policy-invariant
        # (Ng et al., 1999). The shaping reward is positive when the agent
        # moves to a higher-potential state (more art seen, less crowded,
        # closer to exit when time is short).
        shaping_reward = 0.0
        if self.potential_shaping:
            shaping_reward = self.shaping_gamma * self._potential() - prev_potential
            reward += shaping_reward

        # Apply invalid action penalty (not part of the task reward).
        reward += penalty_invalid
        reward_info.reward_shaping = shaping_reward

        # Voluntary exit is the goal: a clear positive bonus for reaching a door.
        if terminated:
            reward += self.exit_bonus

        # Failing to leave before closing is strictly bad: a strong terminal
        # penalty (not a full wipe -- a wipe destroys the learning signal). The
        # dense closing pressure in compute_reward is what actually teaches the
        # agent to avoid ending up here.
        if truncated:
            reward += self.noexit_penalty

        # --- Build info dict for logging ---
        info = {
            "invalid_action": float(not valid_action),
            "reward_art": reward_info.reward_art,
            "reward_time": reward_info.reward_time,
            "reward_congestion": reward_info.reward_congestion,
            "reward_shaping": reward_info.reward_shaping,
            "density_current": reward_info.density_current,
            "npc_inside": sim_stats["inside"],        # total NPCs inside the museum
            "queue": sim_stats["queue"],               # NPCs waiting outside (capacity gate)
            "botticelli_density": sim_stats["botticelli_density"],  # A11+A12 density for monitoring
        }

        return self._build_observation(), float(reward), bool(terminated), bool(truncated), info

    # =========================================================================
    # Rendering and cleanup
    # =========================================================================

    def render(self) -> None:
        """Print a compact human-readable state summary.

        Shows current time, room, total NPCs inside, and queue length.
        Useful for debugging and human-in-the-loop evaluation.
        """

        print(
            f"t={self.time_elapsed:03d} room={self.current_room:>7} "
            f"npc_inside={self.sim.total_inside:4d} queue={len(self.sim.outside_queue):4d}"
        )

    def close(self) -> None:
        """Release environment resources (no-op; present for Gymnasium API)."""

        return


# =============================================================================
# SB3-contrib helper
# =============================================================================


def get_action_mask(env: UffiziEnv) -> np.ndarray:
    """Module-level helper for sb3-contrib ActionMasker wrapper.

    sb3-contrib's MaskablePPO requires a callable that takes the environment
    and returns the action mask. This function provides that interface without
    requiring a lambda or partial, keeping the training script clean.

    Parameters
    ----------
    env : UffiziEnv
        The environment instance.

    Returns
    -------
    np.ndarray
        Binary action mask, shape (1 + max_degree,).
    """

    return env.get_action_mask()
