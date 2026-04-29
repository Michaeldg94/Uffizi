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
        assert g.number_of_edges() == len(config.EDGES)

    def test_connected(self, g):
        assert nx.is_connected(g)

    def test_undirected(self, g):
        assert not g.is_directed()

    def test_all_nodes_have_attributes(self, g):
        required = {"name", "section", "importance", "magnetism", "capacity"}
        for node in g.nodes:
            attrs = set(g.nodes[node].keys())
            missing = required - attrs
            assert not missing, f"Node {node} missing attributes: {missing}"

    def test_corridors_high_degree(self, g):
        degrees = dict(g.degree())
        for corridor in ["A2", "A24"]:
            assert degrees[corridor] >= 8, f"{corridor} degree too low: {degrees[corridor]}"
        # A23 (Southern Corridor) is a passage, lower degree is expected.
        assert degrees["A23"] >= 2, f"A23 degree too low: {degrees['A23']}"

    def test_botticelli_chain_forced(self, g):
        assert set(g.neighbors("A11")) == {"A10", "A12"}
        assert set(g.neighbors("A12")) == {"A11", "A13"}

    def test_botticelli_no_bypass(self, g):
        # Removing A11 and A12 should make A10 a dead end:
        # A10 only connects to A9 (since A11 is removed).
        g2 = g.copy()
        g2.remove_node("A11")
        g2.remove_node("A12")
        assert set(g2.neighbors("A10")) == {"A9"}

    def test_entry_to_exit_path_exists(self, g):
        assert nx.has_path(g, "ENTRY", "EXIT")

    def test_all_rooms_reachable_from_entry(self, g):
        reachable = nx.single_source_shortest_path(g, "ENTRY")
        assert len(reachable) == g.number_of_nodes()


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
    def test_symmetric(self):
        g = build_uffizi_graph()
        dists = all_pairs_shortest_paths(g)
        for u in list(dists.keys())[:10]:
            for v in list(dists[u].keys())[:10]:
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
        # The recommended route may revisit rooms (e.g., A4 in the Giotto
        # loop, A24 when returning from the U-loop). What matters is that
        # key rooms appear at least once.
        route_set = set(config.RECOMMENDED_ROUTE)
        for key_room in ["A11", "A12", "A35", "A38", "E4"]:
            assert key_room in route_set, f"Key room {key_room} missing from route"
