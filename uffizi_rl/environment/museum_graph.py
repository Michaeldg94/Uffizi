"""Museum graph construction and validation helpers.

This module builds the NetworkX graph that represents the physical layout of
the Uffizi Gallery.  Every room is a node (with attributes: name, section,
importance, magnetism, capacity) and every doorway or passage between rooms
is an undirected edge.  The graph is the core data structure consumed by:

  - The crowd simulator (crowd_simulator.py), which moves NPC visitors
    along edges according to route-following and crowd-avoidance rules.
  - The Gymnasium environment (uffizi_env.py), which uses shortest-path
    distances for reward shaping and observation construction.
  - The tabular Q-learning experiments (Phase 2), which use a reduced
    "toy graph" preserving the essential bottleneck structure.

The topology is hand-curated from the official 2023 Uffizi floor plan [MAP].
Validation checks ensure that key structural invariants hold after any edit
(connectivity, corridor hub degrees, Botticelli forced chain, reachability).

References
----------
[MAP]  Official 2023 Uffizi Gallery floor plan (uffizi.it).
[A22]  Attanasio et al. (2022). "Visitors flow management at Uffizi Gallery."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np

from uffizi_rl import config


# =============================================================================
# Graph validation dataclass
# =============================================================================


@dataclass
class GraphValidation:
    """Named graph-validation checks for the curated museum topology.

    Each boolean field corresponds to one structural invariant that the graph
    must satisfy for the simulator to behave correctly.  The ``as_dict``
    method is used by the calibration pipeline to log validation results.

    Attributes
    ----------
    connected : bool
        True if the graph is a single connected component. A disconnected
        graph would strand visitors in unreachable rooms.
    corridors_have_high_degree : bool
        True if the three main corridor hubs (A2 Western, A24 Eastern,
        B1 Contini Bonacossi) have degree >= 80th percentile. These hubs
        are the backbone of visitor flow; low degree would mean the floor
        plan was mis-encoded.
    botticelli_forced_chain_ok : bool
        True if rooms A10-A11-A12-A13 form a strict linear chain with no
        side exits from A11 or A12. This forced chain is the single biggest
        bottleneck in the real museum, and its topology drives the core
        congestion dynamics that the RL agent must learn to manage.
    all_rooms_reachable_from_entry : bool
        True if every node is reachable from ENTRY via BFS. Redundant with
        ``connected`` for undirected graphs, but kept as an explicit sanity
        check against accidental directed-edge bugs.
    """

    connected: bool
    corridors_have_high_degree: bool
    botticelli_forced_chain_ok: bool
    all_rooms_reachable_from_entry: bool

    def as_dict(self) -> Dict[str, bool]:
        """Return all checks as a flat dict, including a summary ``all_pass`` key.

        Returns
        -------
        Dict[str, bool]
            Keys are check names; ``all_pass`` is True only when every
            individual check passes.
        """
        return {
            "connected": self.connected,
            "corridors_have_high_degree": self.corridors_have_high_degree,
            "botticelli_forced_chain_ok": self.botticelli_forced_chain_ok,
            "all_rooms_reachable_from_entry": self.all_rooms_reachable_from_entry,
            # Aggregate: one False anywhere fails the whole validation
            "all_pass": all((
                self.connected,
                self.corridors_have_high_degree,
                self.botticelli_forced_chain_ok,
                self.all_rooms_reachable_from_entry,
            )),
        }


# =============================================================================
# Graph construction
# =============================================================================


def build_uffizi_graph() -> nx.Graph:
    """Construct the full Uffizi museum graph with room attributes.

    Reads room metadata from ``config.ALL_ROOM_ROWS`` (which concatenates
    second-floor A-block, first-floor B/C/D/E blocks, and special nodes)
    and edge definitions from ``config.EDGES`` (hand-mapped from the 2023
    floor plan [MAP]).

    Each node receives the following attributes (from ``config.ROOM_DATA``):
      - name (str): human-readable gallery name
      - section (str): thematic grouping (e.g. "early_renaissance")
      - importance (float): 1-10 cultural significance [assumption]
      - magnetism (float): dwell-time multiplier, 1.0 ~ 3 min base [assumption]
      - capacity (float): comfortable max occupancy from pixel-area
        analysis of the floor plan [assumption]

    Returns
    -------
    nx.Graph
        Undirected graph with ~100 nodes and ~130 edges. Called once at
        environment init; the resulting object is reused for the entire
        training run.
    """

    g = nx.Graph()

    # Iterate ALL_ROOM_ROWS to preserve the canonical ordering defined in
    # config.py (second floor first, then B, C, D, E, special nodes).
    # We unpack only room_id; the remaining 5 fields (name, section,
    # importance, magnetism, capacity) are looked up from the pre-built
    # ROOM_DATA dict so attribute types are already cast to float.
    for room_id, _, _, _, _, _ in config.ALL_ROOM_ROWS:
        g.add_node(room_id, **config.ROOM_DATA[room_id])

    # Edges are undirected: visitors can walk in either direction through
    # every doorway. The topology is defined in config.EDGES [MAP].
    g.add_edges_from(config.EDGES)

    return g


# =============================================================================
# Route helpers
# =============================================================================


def recommended_next_map(route: List[str] | None = None) -> Dict[str, str]:
    """Build a lookup from each room on a route to its successor.

    Used by the crowd simulator to implement route-following behavior:
    when an NPC must choose among neighbors, it checks this map to see
    which neighbor is the "guidebook next" room and applies a route-bias
    weight (TYPE_A_ROUTE_BIAS or TYPE_B_ROUTE_BIAS from config).

    If a room appears multiple times in the route (e.g. A4 appears twice
    in the recommended route because the Giotto loop returns to it), only
    the *last* occurrence's successor is stored. This is intentional:
    after completing the loop, the visitor should proceed forward, not
    re-enter the loop. [assumption]

    Parameters
    ----------
    route : List[str] or None
        Ordered list of room IDs. Defaults to ``config.RECOMMENDED_ROUTE``
        (the classic itinerary from the floor plan [MAP]).

    Returns
    -------
    Dict[str, str]
        Mapping ``{room_id: next_room_id}`` for all rooms except the last.
    """

    seq = config.RECOMMENDED_ROUTE if route is None else route
    nxt: Dict[str, str] = {}
    for i in range(len(seq) - 1):
        # Later occurrences overwrite earlier ones (last-write-wins)
        nxt[seq[i]] = seq[i + 1]
    return nxt


def all_pairs_shortest_paths(g: nx.Graph) -> Dict[str, Dict[str, int]]:
    """Precompute shortest-path distances (hop counts) between all node pairs.

    Returns a nested dict: ``distances[src][dst] = hop_count``.

    This is called once at simulator initialization and cached. The reward
    function uses these distances for shaping (penalizing moves away from
    high-value rooms, rewarding progress toward the exit). Computing BFS
    on every step would be O(V+E) per call; precomputing is O(V*(V+E))
    once but makes the inner loop O(1) per distance lookup.

    For the full Uffizi graph (~100 nodes, ~130 edges), the precomputation
    takes <1 ms and the resulting dict uses negligible memory.

    Parameters
    ----------
    g : nx.Graph
        The museum graph (from ``build_uffizi_graph`` or ``toy_graph``).

    Returns
    -------
    Dict[str, Dict[str, int]]
        ``distances[src][dst]`` gives the minimum number of edges on any
        path from ``src`` to ``dst``.
    """

    # nx.all_pairs_shortest_path_length returns an iterator of
    # (node, {target: length}) pairs; wrapping in dict() materializes it.
    return dict(nx.all_pairs_shortest_path_length(g))


# =============================================================================
# Toy graph for tabular experiments
# =============================================================================


def toy_graph() -> nx.Graph:
    """Small 12-node graph for tabular Q-learning experiments (Phase 2).

    Preserves the essential Uffizi topology in miniature so that tabular
    methods (which scale poorly beyond ~100 states) can converge quickly
    and serve as a sanity check before scaling to PPO on the full graph.

    Key structural features retained:
      - Two corridor hubs (C1, C2) with high degree, mirroring A2/A24.
      - A forced Botticelli chain (A11-A12) with no bypass, reproducing
        the bottleneck that drives congestion in the real museum.
      - High-importance "magnet" rooms (A11, A12, A35, A38, E4) that
        attract visitors and create uneven demand.
      - Two minor rooms (M1, M2) that represent underused galleries
        the RL agent could learn to redirect visitors toward.
      - A single entry and single exit, matching the real museum's
        one-way flow constraint.

    Node attributes use the same schema as the full graph (name, section,
    importance, magnetism, capacity) so that the environment code does not
    need to branch on graph size. [assumption: attribute values are
    simplified but preserve relative ordering of importance/magnetism]

    Returns
    -------
    nx.Graph
        12-node, 12-edge undirected graph.
    """

    g = nx.Graph()

    # --- Node definitions ---
    # Each node carries the same 5 attributes as the full graph.
    # Capacities are scaled down proportionally; importance and magnetism
    # preserve the relative ranking from config.ROOM_DATA. [assumption]
    nodes = {
        "ENTRY": {"name": "Entry", "section": "entry_gate", "importance": 1, "capacity": 50, "magnetism": 0.2},
        # C1 and C2 are high-degree corridor hubs (analogous to A2 and A24)
        "C1": {"name": "Corridor 1", "section": "corridor", "importance": 2, "capacity": 60, "magnetism": 0.4},
        "C2": {"name": "Corridor 2", "section": "corridor", "importance": 2, "capacity": 60, "magnetism": 0.4},
        # Botticelli forced chain: must pass through both, no bypass
        "A11": {"name": "Botticelli - Spring", "section": "early_renaissance", "importance": 10, "capacity": 25, "magnetism": 5.0},
        "A12": {"name": "Botticelli - Venus", "section": "early_renaissance", "importance": 10, "capacity": 25, "magnetism": 5.0},
        # Tribune: high-importance, small capacity (bottleneck potential)
        "A16": {"name": "Tribune", "section": "special", "importance": 8, "capacity": 15, "magnetism": 3.0},
        # High Renaissance magnets
        "A35": {"name": "Leonardo", "section": "high_renaissance", "importance": 9, "capacity": 30, "magnetism": 4.0},
        "A38": {"name": "Raphael & Michelangelo", "section": "high_renaissance", "importance": 9, "capacity": 30, "magnetism": 4.0},
        # Caravaggio: the last major magnet before exit
        "E4": {"name": "Caravaggio - Medusa", "section": "17th_century", "importance": 9, "capacity": 25, "magnetism": 3.5},
        # Minor rooms: underused, high capacity; redistribution targets
        "M1": {"name": "Minor Room 1", "section": "renaissance_other", "importance": 4, "capacity": 30, "magnetism": 1.0},
        "M2": {"name": "Minor Room 2", "section": "renaissance_other", "importance": 4, "capacity": 30, "magnetism": 1.0},
        "EXIT": {"name": "Exit", "section": "exit", "importance": 1, "capacity": 60, "magnetism": 0.1},
    }

    for n, attrs in nodes.items():
        g.add_node(n, **attrs)

    # --- Edge definitions ---
    # The topology creates two paths from ENTRY to EXIT:
    #   1. ENTRY -> C1 -> A11 -> A12 -> C2 -> EXIT  (through Botticelli)
    #   2. ENTRY -> C1 -> M1 -> M2 -> EXIT          (minor-room bypass)
    # The agent can learn to steer visitors toward path 2 when path 1 is
    # congested, trading off importance for reduced crowding.
    edges = [
        ("ENTRY", "C1"),       # single entry point
        ("C1", "A11"),         # corridor 1 to Botticelli chain start
        ("A11", "A12"),        # forced chain (no bypass)
        ("A12", "C2"),         # chain exits to corridor 2
        ("C1", "A16"),         # corridor 1 to Tribune (side branch)
        ("C2", "A35"),         # corridor 2 to Leonardo
        ("C2", "A38"),         # corridor 2 to Raphael/Michelangelo
        ("C2", "E4"),          # corridor 2 to Caravaggio
        ("C1", "M1"),          # corridor 1 to minor rooms (bypass route)
        ("M1", "M2"),          # minor room chain
        ("M2", "EXIT"),        # bypass exit
        ("C2", "EXIT"),        # main exit from corridor 2
    ]
    g.add_edges_from(edges)
    return g


# =============================================================================
# Graph validation
# =============================================================================


def validate_uffizi_graph(g: nx.Graph | None = None) -> GraphValidation:
    """Validate the key topological invariants of the museum graph.

    These checks catch common errors when editing the edge list in config.py:
    accidentally deleting an edge that disconnects a wing, adding a shortcut
    around the Botticelli chain, or misnaming a corridor hub. Running this
    after any topology change gives immediate feedback.

    Parameters
    ----------
    g : nx.Graph or None
        Graph to validate. If None, builds a fresh graph via
        ``build_uffizi_graph()``. Passing an explicit graph allows
        validating modified topologies (e.g. with intervention edges).

    Returns
    -------
    GraphValidation
        Dataclass with one boolean per check plus an ``as_dict()`` method
        for logging.
    """

    g = build_uffizi_graph() if g is None else g

    # --- Check 1: global connectivity ---
    # The museum is a single building; every room must be reachable from
    # every other room. A disconnected graph would strand NPC visitors.
    connected = nx.is_connected(g)

    # --- Check 2: corridor hub degrees ---
    # The three main corridor hubs should be among the highest-degree nodes
    # (>= 80th percentile of all node degrees). This ensures they serve as
    # the backbone of visitor flow, connecting many galleries.
    degrees = dict(g.degree())
    # A2 (Western Corridor), A24 (Eastern Corridor), and B1 (Contini
    # Bonacossi Corridor) are the true hubs. A23 (Southern Corridor) is
    # just a passage with degree 2, so it is NOT checked here.
    corridor_degrees = [degrees.get("A2", 0), degrees.get("A24", 0), degrees.get("B1", 0)]
    all_degrees = np.array(list(degrees.values()), dtype=float)
    degree_threshold = float(np.quantile(all_degrees, 0.80))  # 80th percentile
    corridors_have_high_degree = all(d >= degree_threshold for d in corridor_degrees)

    # --- Check 3: Botticelli forced chain ---
    # The Botticelli rooms (A10-A11-A12-A13) must form a strict linear chain
    # with no shortcuts. This is the museum's most important bottleneck:
    # every visitor who wants to see The Spring (A11) or The Birth of Venus
    # (A12) must traverse the chain sequentially. If A11 or A12 had any
    # neighbor besides its chain predecessor and successor, visitors could
    # bypass the queue, and the congestion dynamics that our RL agent must
    # manage would be unrealistically mild.
    botticelli_forced_chain_ok = (
        g.has_edge("A10", "A11")                         # chain entry
        and g.has_edge("A11", "A12")                     # Spring to Venus
        and g.has_edge("A12", "A13")                     # chain exit
        and set(g.neighbors("A11")) == {"A10", "A12"}    # exactly degree 2
        and set(g.neighbors("A12")) == {"A11", "A13"}    # exactly degree 2
    )

    # --- Check 4: reachability from ENTRY ---
    # In a connected undirected graph, all nodes are trivially reachable
    # from any node. We verify reachability from ENTRY explicitly as a
    # guard against accidentally introducing directed edges or other bugs
    # that break the undirected assumption.
    all_rooms_reachable_from_entry = (
        len(nx.single_source_shortest_path(g, "ENTRY")) == g.number_of_nodes()
    )

    return GraphValidation(
        connected=connected,
        corridors_have_high_degree=corridors_have_high_degree,
        botticelli_forced_chain_ok=botticelli_forced_chain_ok,
        all_rooms_reachable_from_entry=all_rooms_reachable_from_entry,
    )
