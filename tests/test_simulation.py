"""Tests for crowd simulator, visitor profiles, and environment."""

import numpy as np
import pytest

from uffizi_rl import config
from uffizi_rl.environment.crowd_simulator import CrowdSimulator
from uffizi_rl.environment.museum_graph import build_uffizi_graph
from uffizi_rl.environment.uffizi_env import UffiziEnv
from uffizi_rl.environment.visitor_profiles import (
    sample_profile,
    sample_type_a_profile,
    sample_type_b_profile,
    type_a_effective_density,
)


class TestVisitorProfiles:
    def test_type_a_crowd_alpha(self):
        rng = config.get_rng(42)
        p = sample_type_a_profile(rng)
        assert p.crowd_alpha == config.TYPE_A_CROWD_ALPHA

    def test_type_b_crowd_alpha(self):
        rng = config.get_rng(42)
        p = sample_type_b_profile(rng)
        assert p.crowd_alpha == config.TYPE_B_CROWD_ALPHA

    def test_type_a_heterogeneous(self):
        profiles = [sample_type_a_profile(config.get_rng(i)) for i in range(20)]
        vecs = np.stack([p.importance_vector for p in profiles])
        # Variance across visitors should be non-trivial.
        assert vecs.std(axis=0).mean() > 0.5

    def test_type_b_deterministic(self):
        p1 = sample_type_b_profile(config.get_rng(1))
        p2 = sample_type_b_profile(config.get_rng(2))
        np.testing.assert_array_equal(p1.importance_vector, p2.importance_vector)

    def test_type_b_magnet_rooms_high(self):
        p = sample_type_b_profile(config.get_rng(42))
        for room in config.TYPE_B_MAGNET_ROOMS:
            idx = config.ROOM_TO_IDX[room]
            assert p.importance_vector[idx] == config.TYPE_B_MAGNET_IMPORTANCE

    def test_sample_profile_respects_fraction(self):
        rng = config.get_rng(42)
        types = [sample_profile(rng, type_a_fraction=0.5).name for _ in range(1000)]
        frac_a = sum(1 for t in types if t == "A") / len(types)
        assert 0.4 < frac_a < 0.6

    def test_effective_density_asymmetric(self):
        # With equal counts, effective density for Type A should be higher
        # due to cross-type externality.
        symmetric = type_a_effective_density(10, 10, 50)
        raw = 20 / 50
        assert symmetric > raw

    def test_importance_vector_length(self):
        p = sample_type_a_profile(config.get_rng(42))
        assert len(p.importance_vector) == config.N_ROOMS


class TestCrowdSimulator:
    @pytest.fixture
    def sim(self):
        return CrowdSimulator(daily_total=5000, seed=42)

    def test_capacity_never_exceeded(self, sim):
        sim.run_day()
        assert sim.capacity_violations == 0
        assert sim.max_total_inside <= config.MAX_MUSEUM_CAPACITY

    def test_botticelli_is_bottleneck(self, sim):
        day = sim.run_day()
        dens = sim.export_density_matrix()
        room_means = dens.mean(axis=0)
        bott_idx = [config.ROOM_TO_IDX["A11"], config.ROOM_TO_IDX["A12"]]
        bott_mean = max(room_means[bott_idx[0]], room_means[bott_idx[1]])
        overall_mean = room_means.mean()
        assert bott_mean > overall_mean * 2, "Botticelli should be significantly above average"

    def test_visitors_complete(self, sim):
        day = sim.run_day()
        total = day["type_a_completed"] + day["type_b_completed"]
        assert total > 0
        assert total == day["completed_visitors"]

    def test_arrival_rate_zero_after_last_entry(self, sim):
        rate = sim.desired_arrival_rate(config.LAST_ENTRY_MINUTES + 1)
        assert rate == 0.0

    def test_arrival_rate_positive_during_hours(self, sim):
        rate = sim.desired_arrival_rate(100)
        assert rate > 0.0

    def test_density_matrix_shape(self, sim):
        sim.run_day()
        dens = sim.export_density_matrix()
        assert dens.shape[1] == config.N_ROOMS
        assert dens.shape[0] > 0

    def test_reproducible(self):
        sim1 = CrowdSimulator(daily_total=5000, seed=42)
        day1 = sim1.run_day()
        sim2 = CrowdSimulator(daily_total=5000, seed=42)
        day2 = sim2.run_day()
        assert day1 == day2

    def test_stay_probability_decreases_with_time(self):
        p0 = CrowdSimulator.npc_stay_probability(3.0, 0, 1.0)
        p5 = CrowdSimulator.npc_stay_probability(3.0, 5, 1.0)
        p20 = CrowdSimulator.npc_stay_probability(3.0, 20, 1.0)
        assert p0 > p5 > p20

    def test_stay_probability_bounded(self):
        for mag in [0.1, 1.0, 5.0]:
            for t in [0, 1, 10, 100]:
                p = CrowdSimulator.npc_stay_probability(mag, t, 1.0)
                assert 0.05 <= p <= 0.95


class TestInterventions:
    def test_timed_entry_reduces_peak(self):
        sq = CrowdSimulator(daily_total=5000, seed=42)
        sq_day = sq.run_day()

        te = CrowdSimulator(daily_total=5000, seed=42, timed_entry=True)
        te_day = te.run_day()

        assert te_day["peak_inside"] <= sq_day["peak_inside"]

    def test_congestion_pricing_reduces_botticelli(self):
        sq = CrowdSimulator(daily_total=5000, seed=42)
        sq_day = sq.run_day()

        cp = CrowdSimulator(daily_total=5000, seed=42, botticelli_slot_cap=3)
        cp_day = cp.run_day()

        assert cp_day["peak_botticelli_density"] < sq_day["peak_botticelli_density"]

    def test_dynamic_info_changes_behavior(self):
        sq = CrowdSimulator(daily_total=5000, seed=42)
        sq_day = sq.run_day()
        sq_dens = sq.export_density_matrix()

        di = CrowdSimulator(daily_total=5000, seed=42, dynamic_info=True)
        di_day = di.run_day()
        di_dens = di.export_density_matrix()

        # Density matrices should differ (info changes movement decisions).
        assert not np.allclose(sq_dens, di_dens)


class TestUffiziEnv:
    @pytest.fixture
    def env(self):
        return UffiziEnv(seed=42, episode_minutes=60)

    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset()
        expected_dim = 4 * config.N_ROOMS + 2
        assert obs.shape == (expected_dim,)

    def test_reset_deterministic(self):
        env1 = UffiziEnv(seed=42, episode_minutes=60)
        obs1, _ = env1.reset()
        env2 = UffiziEnv(seed=42, episode_minutes=60)
        obs2, _ = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)

    def test_action_mask_valid(self, env):
        env.reset()
        mask = env.get_action_mask()
        assert mask[0] == 1  # stay always valid
        assert mask.sum() >= 2  # at least stay + one neighbor

    def test_step_returns_five_tuple(self, env):
        env.reset()
        mask = env.get_action_mask()
        action = int(np.flatnonzero(mask)[0])
        result = env.step(action)
        assert len(result) == 5

    def test_agent_uses_type_a_alpha(self, env):
        env.reset()
        assert env.agent_profile.crowd_alpha == config.TYPE_A_CROWD_ALPHA

    def test_episode_terminates_at_exit(self, env):
        env.reset()
        # Force the agent to an exit.
        env.current_room = "EXIT"
        _, _, terminated, _, _ = env.step(0)
        assert terminated

    def test_episode_truncates_at_time_limit(self, env):
        env.reset()
        env.time_elapsed = env.episode_minutes - 1
        _, _, _, truncated, _ = env.step(0)
        assert truncated

    def test_invalid_action_penalized(self, env):
        env.reset()
        mask = env.get_action_mask()
        # Find an invalid action.
        invalid = None
        for i in range(len(mask)):
            if mask[i] == 0:
                invalid = i
                break
        if invalid is not None:
            _, reward_invalid, _, _, info = env.step(invalid)
            assert info["invalid_action"] == 1.0

    def test_observation_dtype_float32(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_visibility_mask_increases_obs_dim(self):
        env = UffiziEnv(seed=42, episode_minutes=60, with_visibility_mask=True)
        obs, _ = env.reset()
        expected_dim = 5 * config.N_ROOMS + 2
        assert obs.shape == (expected_dim,)
