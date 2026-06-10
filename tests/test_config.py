"""Tests for config.py: constants, constraints, and utility functions."""

import numpy as np
import pytest

from uffizi_rl import config


class TestTimeConstants:
    def test_museum_open_minutes(self):
        # 08:15 to 18:30 = 10h15m = 615 minutes.
        assert config.MUSEUM_OPEN_MINUTES == 615

    def test_last_entry_minutes(self):
        # 08:15 to 17:30 = 9h15m = 555 minutes.
        assert config.LAST_ENTRY_MINUTES == 555

    def test_entry_slots_consistent(self):
        assert config.LAST_ENTRY_MINUTES == config.N_ENTRY_SLOTS * config.ENTRY_SLOT_MINUTES

    def test_last_entry_before_closing(self):
        assert config.LAST_ENTRY_MINUTES < config.MUSEUM_OPEN_MINUTES


class TestCapacity:
    def test_capacity_math_within_bounds(self):
        check = config.capacity_math_check()
        assert check["avg_occupancy"] < config.MAX_MUSEUM_CAPACITY

    def test_peak_day_pressure(self):
        check = config.capacity_math_check(daily_visitors=config.DAILY_VISITORS_PEAK)
        # Peak days SHOULD exceed average capacity (that's the point).
        assert check["avg_occupancy"] > config.MAX_MUSEUM_CAPACITY


class TestVisitorTypes:
    def test_fractions_sum_to_one(self):
        assert abs(config.TYPE_A_FRACTION_DEFAULT + config.TYPE_B_FRACTION_DEFAULT - 1.0) < 1e-9

    def test_type_a_more_crowd_sensitive(self):
        assert config.TYPE_A_CROWD_ALPHA > config.TYPE_B_CROWD_ALPHA

    def test_type_b_more_route_following(self):
        assert config.TYPE_B_ROUTE_BIAS > config.TYPE_A_ROUTE_BIAS

    def test_type_a_longer_dwell(self):
        assert config.TYPE_A_DWELL_MULTIPLIER > config.TYPE_B_DWELL_MULTIPLIER

    def test_type_a_avoids_crowds(self):
        assert config.TYPE_A_ANTI_CROWD_BONUS > config.TYPE_B_ANTI_CROWD_BONUS

    def test_cross_type_externality_positive(self):
        assert config.TYPE_A_CROSS_TYPE_EXTERNALITY > 1.0


class TestRoomData:
    def test_all_rooms_have_data(self):
        for rid in config.ROOM_IDS:
            assert rid in config.ROOM_DATA

    def test_room_ids_unique(self):
        assert len(config.ROOM_IDS) == len(set(config.ROOM_IDS))

    def test_n_rooms_matches(self):
        assert config.N_ROOMS == len(config.ROOM_IDS)

    def test_idx_bijection(self):
        for rid in config.ROOM_IDS:
            idx = config.ROOM_TO_IDX[rid]
            assert config.IDX_TO_ROOM[idx] == rid

    def test_importance_range(self):
        for rid, data in config.ROOM_DATA.items():
            assert 0 <= data["importance"] <= 10, f"{rid} importance out of range"

    def test_capacity_positive(self):
        for rid, data in config.ROOM_DATA.items():
            assert data["capacity"] > 0, f"{rid} has non-positive capacity"

    def test_magnetism_non_negative(self):
        for rid, data in config.ROOM_DATA.items():
            assert data["magnetism"] >= 0, f"{rid} has negative magnetism"

    def test_botticelli_highest_importance(self):
        bott_imp = max(
            config.ROOM_DATA["A11"]["importance"],
            config.ROOM_DATA["A12"]["importance"],
        )
        for rid, data in config.ROOM_DATA.items():
            assert data["importance"] <= bott_imp, (
                f"{rid} has higher importance than Botticelli"
            )

    def test_magnet_rooms_exist(self):
        for rid in config.TYPE_B_MAGNET_ROOMS:
            assert rid in config.ROOM_TO_IDX, f"Magnet room {rid} not in graph"

    def test_hidden_gem_trails_rooms_exist(self):
        for trail, rooms in config.HIDDEN_GEM_TRAILS.items():
            for rid in rooms:
                assert rid in config.ROOM_TO_IDX, (
                    f"Trail '{trail}' references nonexistent room {rid}"
                )

    def test_kiosk_rooms_exist(self):
        for rid in config.KIOSK_ROOMS:
            assert rid in config.ROOM_TO_IDX, f"Kiosk room {rid} not in graph"


class TestEdges:
    def test_all_edge_nodes_exist(self):
        room_set = set(config.ROOM_IDS)
        for u, v in config.EDGES:
            assert u in room_set, f"Edge references unknown room {u}"
            assert v in room_set, f"Edge references unknown room {v}"

    def test_no_self_loops(self):
        for u, v in config.EDGES:
            assert u != v, f"Self-loop on {u}"

    def test_no_duplicate_edges(self):
        edge_set = set()
        for u, v in config.EDGES:
            key = (min(u, v), max(u, v))
            assert key not in edge_set, f"Duplicate edge {u}-{v}"
            edge_set.add(key)

    def test_recommended_route_rooms_exist(self):
        room_set = set(config.ROOM_IDS)
        for rid in config.RECOMMENDED_ROUTE:
            assert rid in room_set, f"Route references unknown room {rid}"


class TestSampleVisitDuration:
    def test_returns_positive(self):
        rng = config.get_rng(42)
        for slot in range(config.N_ENTRY_SLOTS):
            d = config.sample_visit_duration(slot, rng)
            assert d >= 20

    def test_early_longer_than_late(self):
        rng = config.get_rng(42)
        early = [config.sample_visit_duration(2, config.get_rng(i)) for i in range(1000)]
        late = [config.sample_visit_duration(35, config.get_rng(i + 1000)) for i in range(1000)]
        assert np.mean(early) > np.mean(late)

    def test_mean_close_to_target(self):
        # Means updated to match the Uffizi's official itinerary tiers:
        # Fast 90-120 min, Classic 120+, Complete 180+. Early-slot
        # visitors are calibrated as Classic; late-slot as Fast.
        # The 90-min floor on the log-normal sampler slightly inflates
        # the late-slot mean (the lower tail is clipped), so the
        # tolerance is 10% rather than 5% for the late slot.
        targets = [(3, 180, 0.05), (12, 150, 0.05), (25, 120, 0.05), (34, 100, 0.10)]
        for slot, expected, tol in targets:
            samples = [config.sample_visit_duration(slot, config.get_rng(i)) for i in range(5000)]
            actual = np.mean(samples)
            assert abs(actual - expected) / expected < tol, (
                f"Slot {slot}: expected ~{expected}, got {actual:.1f}"
            )


class TestNormalizeWeights:
    def test_sums_to_one(self):
        w = config.normalize_weights([1.0, 2.0, 3.0])
        assert abs(w.sum() - 1.0) < 1e-9

    def test_uniform_on_zeros(self):
        w = config.normalize_weights([0.0, 0.0, 0.0])
        np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3])

    def test_empty_returns_empty(self):
        w = config.normalize_weights([])
        assert len(w) == 0

    def test_single_element(self):
        w = config.normalize_weights([5.0])
        np.testing.assert_allclose(w, [1.0])


class TestReproducibility:
    def test_rng_deterministic(self):
        rng1 = config.get_rng(123)
        rng2 = config.get_rng(123)
        assert rng1.random() == rng2.random()

    def test_visit_duration_deterministic(self):
        d1 = config.sample_visit_duration(10, config.get_rng(42))
        d2 = config.sample_visit_duration(10, config.get_rng(42))
        assert d1 == d2
