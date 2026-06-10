"""Tests for museum graph construction, validation, and topology."""

import networkx as nx
import numpy as np
import pytest

from uffizi_rl import config
from uffizi_rl.environment.museum_graph import (
    all_pairs_shortest_paths,
    build_uffizi_graph,
    recommended_next_map,
    toy_graph,
    validate_uffizi_graph,
)


class TestUffiziGraph:
    @pytest.fixture
    def g(self):
        return build_uffizi_graph()

    def test_node_count(self, g):
        assert g.number_of_nodes() == config.N_ROOMS

    def test_edge_count(self, g):
        # Each bidirectional door in config.EDGES contributes two directed
        # edges; each entry in config.DIRECTED_EDGES contributes one.
        expected = 2 * len(config.EDGES) + len(config.DIRECTED_EDGES)
        assert g.number_of_edges() == expected

    def test_weakly_connected(self, g):
        assert nx.is_weakly_connected(g)

    def test_is_directed(self, g):
        assert g.is_directed()

    def test_staircases_one_way(self, g):
        # Granducal: ENTRY -> GRANDUCAL -> A1 only; no reverse.
        assert g.has_edge("ENTRY", "GRANDUCAL_STAIRCASE")
        assert not g.has_edge("GRANDUCAL_STAIRCASE", "ENTRY")
        assert g.has_edge("GRANDUCAL_STAIRCASE", "A1")
        assert not g.has_edge("A1", "GRANDUCAL_STAIRCASE")
        # Buontalenti: A36 -> BUONTALENTI -> D9 only; no reverse.
        assert g.has_edge("A36", "BUONTALENTI_STAIRCASE")
        assert not g.has_edge("BUONTALENTI_STAIRCASE", "A36")
        assert g.has_edge("BUONTALENTI_STAIRCASE", "D9")
        assert not g.has_edge("D9", "BUONTALENTI_STAIRCASE")
        # Lanzi: down only.
        assert g.has_edge("PANORAMIC_TERRACE", "LANZI_STAIRCASE")
        assert not g.has_edge("LANZI_STAIRCASE", "PANORAMIC_TERRACE")
        assert g.has_edge("LANZI_STAIRCASE", "EXIT")
        assert not g.has_edge("EXIT", "LANZI_STAIRCASE")
        # E7 exit is one-way out.
        assert g.has_edge("E7", "EXIT")
        assert not g.has_edge("EXIT", "E7")

    def test_all_nodes_have_attributes(self, g):
        required = {"name", "section", "importance", "magnetism", "capacity"}
        for node in g.nodes:
            attrs = set(g.nodes[node].keys())
            missing = required - attrs
            assert not missing, f"Node {node} missing attributes: {missing}"

    def test_corridors_high_degree(self, g):
        # A2 (Western Corridor) and A24 (Eastern Corridor) are the main
        # hubs of the 2nd floor. Degree counts in_degree + out_degree
        # for DiGraphs. A2 has degree 6: bidirectional doorways to A1,
        # A4, A9 only; the post-Botticelli rooms (A13, A15, A16, A17,
        # A21) used to connect back to A2 but those return edges were
        # removed to enforce a one-way flow through the Botticelli
        # chain. A24 has degree 16 from the eight Leonardo/west-wing
        # doorways.
        degrees = dict(g.degree())
        assert degrees["A2"] >= 6, f"A2 degree too low: {degrees['A2']}"
        assert degrees["A24"] >= 8, f"A24 degree too low: {degrees['A24']}"
        # A23 (Southern Corridor) is a passage, lower degree is expected.
        assert degrees["A23"] >= 2, f"A23 degree too low: {degrees['A23']}"

    def test_botticelli_chain_one_way(self, g):
        # The Botticelli chain is ONE-WAY: A10 -> A11 -> A12 -> A13.
        # A11's only out-successor is A12; A12's only out-successor is A13.
        assert set(g.successors("A11")) == {"A12"}
        assert set(g.successors("A12")) == {"A13"}
        # And the reverse edges must not exist.
        assert not g.has_edge("A11", "A10")
        assert not g.has_edge("A12", "A11")
        assert not g.has_edge("A13", "A12")

    def test_botticelli_no_bypass(self, g):
        # Removing A11 and A12 should make A10 a dead end:
        # A10 only connects to A9 (since A11 is removed and A10->A11 was
        # the only out-edge from A10 that the chain provided).
        g2 = g.copy()
        g2.remove_node("A11")
        g2.remove_node("A12")
        assert set(g2.successors("A10")) == {"A9"}

    def test_entry_to_exit_path_exists(self, g):
        assert nx.has_path(g, "ENTRY", "EXIT")

    def test_all_rooms_reachable_from_entry(self, g):
        reachable = nx.single_source_shortest_path(g, "ENTRY")
        assert len(reachable) == g.number_of_nodes()

    def test_exit_reachable_from_all_rooms(self, g):
        # Every gallery must have a directed path to EXIT, otherwise
        # visitors get trapped.
        g_rev = g.reverse(copy=False)
        reachable_to_exit = nx.single_source_shortest_path(g_rev, "EXIT")
        assert len(reachable_to_exit) == g.number_of_nodes()


class TestValidation:
    def test_all_pass(self):
        v = validate_uffizi_graph()
        d = v.as_dict()
        assert d["all_pass"], f"Validation failed: {d}"

    def test_each_field_true(self):
        v = validate_uffizi_graph()
        assert v.connected
        assert v.corridors_have_high_degree
        assert v.botticelli_forced_chain_ok
        assert v.all_rooms_reachable_from_entry
        assert v.exit_reachable_from_all_galleries


class TestToyGraph:
    @pytest.fixture
    def tg(self):
        return toy_graph()

    def test_node_count(self, tg):
        assert tg.number_of_nodes() == 12

    def test_connected(self, tg):
        assert nx.is_connected(tg)

    def test_has_required_rooms(self, tg):
        required = {"ENTRY", "EXIT", "A11", "A12", "A35", "A38", "A16", "E4"}
        assert required.issubset(set(tg.nodes))

    def test_botticelli_chain(self, tg):
        assert tg.has_edge("A11", "A12")

    def test_attributes_match_full_schema(self, tg):
        required = {"name", "section", "importance", "magnetism", "capacity"}
        for node in tg.nodes:
            attrs = set(tg.nodes[node].keys())
            missing = required - attrs
            assert not missing, f"Toy node {node} missing: {missing}"


class TestAllPairsShortestPaths:
    def test_symmetric_outside_oneway_edges(self):
        # The graph is directed; one-way edges (Granducal staircase up,
        # Lanzi/Buontalenti staircases down, Botticelli forced chain
        # forward, A8 -> A9 one-way feed into Botticelli, Magliabechi
        # exit) cannot be expected to be symmetric. We sanity-check
        # symmetry only between pairs of bidirectional-door rooms in
        # the Giotto loop (A2, A4, A5, A6, A7).
        g = build_uffizi_graph()
        dists = all_pairs_shortest_paths(g)
        bidirectional_only = ["A2", "A4", "A5", "A6", "A7"]
        for u in bidirectional_only:
            for v in bidirectional_only:
                if u == v:
                    continue
                assert dists[u][v] == dists[v][u], f"Asymmetry: {u}-{v}"

    def test_self_distance_zero(self):
        g = build_uffizi_graph()
        dists = all_pairs_shortest_paths(g)
        for node in g.nodes:
            assert dists[node][node] == 0

    def test_neighbor_distance_one(self):
        g = build_uffizi_graph()
        dists = all_pairs_shortest_paths(g)
        for u, v in g.edges:
            assert dists[u][v] == 1


class TestRecommendedRoute:
    def test_starts_at_entry(self):
        assert config.RECOMMENDED_ROUTE[0] == "ENTRY"

    def test_ends_at_exit(self):
        assert config.RECOMMENDED_ROUTE[-1] == "EXIT"

    def test_next_map_covers_route(self):
        nxt = recommended_next_map()
        # The recommended route defines a visiting order, not a walk.
        # Consecutive rooms in the route may not share an edge (e.g.,
        # A3->A4 are both spokes off A2, visited sequentially via A2).
        # Verify that every route room appears as a key (except the last).
        for room in config.RECOMMENDED_ROUTE[:-1]:
            assert room in nxt, f"Route room {room} missing from next_map"

    def test_route_covers_key_rooms(self):
        # The recommended route is now the lazy-tourist path (masterpieces
        # + Lanzi exit). Caravaggio (E4) is on the 1st floor and is not on
        # the lazy path: only the most thorough visitors reach it via the
        # importance pull. The Botticelli, Leonardo, and Raphael rooms
        # remain mandatory.
        route_set = set(config.RECOMMENDED_ROUTE)
        for key_room in ["A11", "A12", "A35", "A38"]:
            assert key_room in route_set, f"Key room {key_room} missing from route"
