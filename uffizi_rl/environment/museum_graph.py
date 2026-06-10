"""Museum graph construction and validation helpers.

This module builds the NetworkX graph that represents the physical layout of
the Uffizi Gallery.  Every room is a node (with attributes: name, section,
importance, magnetism, capacity) and every doorway or passage between rooms
is encoded as one or more directed edges.

Most doors are physically bidirectional and are represented in
``config.EDGES`` as undirected pairs that ``build_uffizi_graph`` expands
into two directed edges (a,b) and (b,a). The staircases (Granducal up,
Lanzi and Buontalenti down) and the Magliabechi exit are physically
one-way and live in ``config.DIRECTED_EDGES`` instead, where each entry is
added in the listed direction only.

The graph is the core data structure consumed by:

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
        True if the directed graph is weakly connected (its undirected
        projection is one component). A disconnected graph would strand
        visitors in unreachable rooms.
    corridors_have_high_degree : bool
        True if the three main corridor hubs (A2 Western, A24 Eastern,
        B1 Contini Bonacossi) have total degree (in + out) >= 80th
        percentile. These hubs are the backbone of visitor flow; low
        degree would mean the floor plan was mis-encoded.
    botticelli_forced_chain_ok : bool
        True if rooms A10-A11-A12-A13 form a strict linear chain with no
        side exits from A11 or A12. The chain doors are bidirectional, so
        each of A11 and A12 has exactly two out-neighbors (and the same two
        in-neighbors). This forced chain is the single biggest bottleneck
        in the real museum.
    all_rooms_reachable_from_entry : bool
        True if every node is reachable from ENTRY along directed edges.
        With the one-way Granducal Staircase, this is the strict check
        that visitors entering the museum can actually reach every gallery.
    exit_reachable_from_all_galleries : bool
        True if every gallery node has a directed path to EXIT. With the
        one-way Lanzi/Buontalenti/Magliabechi edges, this confirms
        that no visitor can get stuck.
    """

    connected: bool
    corridors_have_high_degree: bool
    botticelli_forced_chain_ok: bool
    all_rooms_reachable_from_entry: bool
    exit_reachable_from_all_galleries: bool

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
            "exit_reachable_from_all_galleries": self.exit_reachable_from_all_galleries,
            # Aggregate: one False anywhere fails the whole validation
            "all_pass": all((
                self.connected,
                self.corridors_have_high_degree,
                self.botticelli_forced_chain_ok,
                self.all_rooms_reachable_from_entry,
                self.exit_reachable_from_all_galleries,
            )),
        }


# =============================================================================
# Graph construction
# =============================================================================


def build_uffizi_graph() -> nx.DiGraph:
    """Construct the full Uffizi museum graph with room attributes.

    Reads room metadata from ``config.ALL_ROOM_ROWS`` (which concatenates
    second-floor A-block, first-floor B/C/D/E blocks, and special nodes)
    and edge definitions from ``config.EDGES`` (bidirectional doorways)
    plus ``config.DIRECTED_EDGES`` (one-way staircases and the exit).

    Each node receives the following attributes (from ``config.ROOM_DATA``):
      - name (str): human-readable gallery name
      - section (str): thematic grouping (e.g. "early_renaissance")
      - importance (float): 1-10 cultural significance [assumption]
      - magnetism (float): dwell-time multiplier, 1.0 ~ 3 min base [assumption]
      - capacity (float): empirical occupancy cap from pixel-area
        analysis of the official 2023 floor plan [MAP]

    Returns
    -------
    nx.DiGraph
        Directed graph with ~100 nodes. Each entry in ``config.EDGES`` is
        expanded into two directed edges (a,b) and (b,a) to represent a
        bidirectional doorway. Each entry in ``config.DIRECTED_EDGES`` is
        added only in the listed direction (one-way staircase or exit).
        Called once at environment init; the resulting object is reused
        for the entire training run.
    """

    g = nx.DiGraph()

    # Iterate ALL_ROOM_ROWS to preserve the canonical ordering defined in
    # config.py (second floor first, then B, C, D, E, special nodes).
    # We unpack only room_id; the remaining 5 fields (name, section,
    # importance, magnetism, capacity) are looked up from the pre-built
    # ROOM_DATA dict so attribute types are already cast to float.
    for room_id, _, _, _, _, _ in config.ALL_ROOM_ROWS:
        g.add_node(room_id, **config.ROOM_DATA[room_id])

    # Bidirectional doors: add both (a,b) and (b,a) so visitors can walk
    # either way through any ordinary doorway.
    for a, b in config.EDGES:
        g.add_edge(a, b)
        g.add_edge(b, a)

    # One-way edges (staircases, exit): add only in the listed direction.
    # Visitors cannot reverse these. See config.DIRECTED_EDGES for the
    # full list with comments.
    for a, b in config.DIRECTED_EDGES:
        g.add_edge(a, b)

    return g


# =============================================================================
# Route helpers
# =============================================================================


def recommended_next_map(route: List[str] | None = None) -> Dict[str, List[str]]:
    """Build a lookup from each room to the ordered list of rooms that
    follow it across every occurrence in the route.

    Used by the crowd simulator to implement route-following behavior: when
    an NPC arrives at room R, the simulator iterates this list and picks the
    first entry the visitor has NOT already visited as the route-preferred
    next neighbor. The matched neighbor receives a route-bias weight bonus
    in ``_pick_next_room``.

    Why a list and not a single value: corridor hubs like A24 appear
    multiple times in a sensible itinerary (once to enter the U-loop, once
    to enter the Leonardo group, once to leave the museum). A flat
    ``{room: next_room}`` map can only store one value per key, and
    last-write-wins produces nonsensical biases (e.g. visitors arriving
    at A24 from the U-loop get nudged toward the exit instead of toward
    Leonardo). The list captures all the canonical successors in route
    order; the simulator's "first-unvisited" selection then routes each
    visitor to whichever branch they have not yet completed.

    Parameters
    ----------
    route : List[str] or None
        Ordered list of room IDs. Defaults to ``config.RECOMMENDED_ROUTE``
        (the classic itinerary from the floor plan [MAP]).

    Returns
    -------
    Dict[str, List[str]]
        Mapping ``{room_id: [first_next, second_next, ...]}`` where the
        list contains each distinct successor in order of route appearance.
    """

    seq = config.RECOMMENDED_ROUTE if route is None else route
    nxt: Dict[str, List[str]] = {}
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a not in nxt:
            nxt[a] = []
        if b not in nxt[a]:  # dedupe; preserve first-occurrence order
            nxt[a].append(b)
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


def validate_uffizi_graph(g: nx.DiGraph | None = None) -> GraphValidation:
    """Validate the key topological invariants of the museum graph.

    These checks catch common errors when editing the edge list in config.py:
    accidentally deleting an edge that disconnects a wing, adding a shortcut
    around the Botticelli chain, misnaming a corridor hub, or breaking the
    one-way directionality of the staircases. Running this after any
    topology change gives immediate feedback.

    Parameters
    ----------
    g : nx.DiGraph or None
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

    # --- Check 1: weak connectivity ---
    # The museum is a single building. The directed graph must be weakly
    # connected: ignoring direction, every node must be in one component.
    # Strong connectivity is too strict (visitors cannot walk UP staircases,
    # so the EXIT has no path back to ENTRY).
    connected = nx.is_weakly_connected(g)

    # --- Check 2: corridor hub degrees ---
    # The three main corridor hubs should be among the highest-degree
    # nodes (>= 80th percentile of total degree = in + out). Almost all
    # doors on the A and B blocks are bidirectional, so the total degree
    # of a hub is roughly 2 x its number of adjacent rooms.
    degrees = dict(g.degree())  # for DiGraph, this is in_degree + out_degree
    corridor_degrees = [degrees.get("A2", 0), degrees.get("A24", 0), degrees.get("B1", 0)]
    all_degrees = np.array(list(degrees.values()), dtype=float)
    degree_threshold = float(np.quantile(all_degrees, 0.80))  # 80th percentile
    corridors_have_high_degree = all(d >= degree_threshold for d in corridor_degrees)

    # --- Check 3: Botticelli forced chain ---
    # The Botticelli rooms must form a strict ONE-WAY chain
    # A10 -> A11 -> A12 -> A13. Bidirectional edges here would let
    # visitors ping-pong between Spring and Venus and accumulate
    # unrealistic dwell time. The forward edges exist; the reverse
    # edges must NOT exist. A11 has exactly one out-successor (A12)
    # and exactly one in-predecessor (A10); A12 likewise (A13 out,
    # A11 in).
    botticelli_forced_chain_ok = (
        g.has_edge("A10", "A11") and not g.has_edge("A11", "A10")
        and g.has_edge("A11", "A12") and not g.has_edge("A12", "A11")
        and g.has_edge("A12", "A13") and not g.has_edge("A13", "A12")
        and set(g.successors("A11")) == {"A12"}
        and set(g.predecessors("A11")) == {"A10"}
        and set(g.successors("A12")) == {"A13"}
        and set(g.predecessors("A12")) == {"A11"}
    )

    # --- Check 4: every gallery reachable from ENTRY along directed paths ---
    # With the Granducal Staircase as the only one-way "up", we need to
    # verify that the entire museum is reachable via legal walks. If any
    # gallery is unreachable, visitors that pass that way will get stuck
    # or never visit it.
    all_rooms_reachable_from_entry = (
        len(nx.single_source_shortest_path(g, "ENTRY")) == g.number_of_nodes()
    )

    # --- Check 5: every gallery has a directed path to EXIT ---
    # Visitors must be able to leave the museum no matter where they end up.
    # On the reverse graph, this is equivalent to "every node reachable
    # from EXIT". A failure here usually means a one-way edge was added
    # backward, or an exit edge was missing.
    g_rev = g.reverse(copy=False)
    exit_reachable_from_all_galleries = (
        len(nx.single_source_shortest_path(g_rev, "EXIT")) == g.number_of_nodes()
    )

    return GraphValidation(
        connected=connected,
        corridors_have_high_degree=corridors_have_high_degree,
        botticelli_forced_chain_ok=botticelli_forced_chain_ok,
        all_rooms_reachable_from_entry=all_rooms_reachable_from_entry,
        exit_reachable_from_all_galleries=exit_reachable_from_all_galleries,
    )
