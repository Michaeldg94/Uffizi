"""Tests for metrics, phase transitions, and welfare computations."""

import numpy as np
import pytest

from uffizi_rl import config
from uffizi_rl.analysis.metrics import (
    botticelli_overcrowding_fraction,
    confidence_interval,
    gini,
    intervention_gap_closed,
    price_of_anarchy,
    theil,
    welfare_proxy_from_density,
)


class TestGini:
    def test_perfect_equality(self):
        assert gini([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-6)

    def test_maximal_inequality(self):
        g = gini([0.0, 0.0, 0.0, 100.0])
        assert g > 0.7

    def test_empty(self):
        assert gini([]) == 0.0

    def test_all_zeros(self):
        assert gini([0.0, 0.0]) == 0.0

    def test_range_zero_to_one(self):
        g = gini([1.0, 2.0, 3.0, 4.0, 5.0])
        assert 0.0 <= g <= 1.0


class TestTheil:
    def test_perfect_equality(self):
        assert theil([1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-6)

    def test_inequality_positive(self):
        t = theil([1.0, 1.0, 10.0])
        assert t > 0.0


class TestPriceOfAnarchy:
    def test_optimal_equals_equilibrium(self):
        assert price_of_anarchy(100.0, 100.0) == pytest.approx(1.0)

    def test_suboptimal_equilibrium(self):
        poa = price_of_anarchy(100.0, 50.0)
        assert poa == pytest.approx(2.0)

    def test_zero_equilibrium(self):
        assert price_of_anarchy(100.0, 0.0) == float("inf")


class TestBotticelliOvercrowding:
    def test_no_overcrowding(self):
        idx_a11 = config.ROOM_TO_IDX["A11"]
        idx_a12 = config.ROOM_TO_IDX["A12"]
        dens = np.full((100, config.N_ROOMS), 0.3)
        frac = botticelli_overcrowding_fraction(dens, idx_a11, idx_a12)
        assert frac == 0.0

    def test_full_overcrowding(self):
        idx_a11 = config.ROOM_TO_IDX["A11"]
        idx_a12 = config.ROOM_TO_IDX["A12"]
        dens = np.full((100, config.N_ROOMS), 0.9)
        frac = botticelli_overcrowding_fraction(dens, idx_a11, idx_a12)
        assert frac == 1.0

    def test_empty_matrix(self):
        dens = np.zeros((0, config.N_ROOMS))
        assert botticelli_overcrowding_fraction(dens, 10, 11) == 0.0


class TestWelfareProxy:
    def test_empty_density(self):
        w = welfare_proxy_from_density(np.zeros((0, config.N_ROOMS)), 100, 200)
        assert w["total_welfare"] == 0.0

    def test_type_a_hurt_more_by_congestion(self):
        dens = np.full((100, config.N_ROOMS), 0.5)
        w = welfare_proxy_from_density(dens, 100, 100)
        # Per-capita welfare for A should be lower than B at moderate density
        # because Type A has higher crowd sensitivity (alpha=6 vs 0.5).
        per_a = w["type_a_welfare"] / 100
        per_b = w["type_b_welfare"] / 100
        assert per_a < per_b

    def test_zero_density_positive_welfare(self):
        dens = np.zeros((100, config.N_ROOMS))
        w = welfare_proxy_from_density(dens, 100, 100)
        assert w["total_welfare"] > 0


class TestInterventionGapClosed:
    def test_status_quo_zero(self):
        assert intervention_gap_closed(50.0, 50.0, 100.0) == pytest.approx(0.0)

    def test_social_optimum_one(self):
        assert intervention_gap_closed(50.0, 100.0, 100.0) == pytest.approx(1.0)

    def test_halfway(self):
        assert intervention_gap_closed(0.0, 50.0, 100.0) == pytest.approx(0.5)


class TestConfidenceInterval:
    def test_single_value(self):
        mean, ci = confidence_interval([5.0])
        assert mean == 5.0
        assert ci == 0.0

    def test_empty(self):
        mean, ci = confidence_interval([])
        assert mean == 0.0

    def test_ci_narrows_with_samples(self):
        _, ci_small = confidence_interval([1.0, 2.0, 3.0])
        _, ci_large = confidence_interval([1.0, 2.0, 3.0] * 100)
        assert ci_large < ci_small
