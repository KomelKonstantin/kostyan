import pytest

from cs2_analyst.probability import (calibrate_rounds, log5, map_round_distribution, map_win_prob,
                                     series_distribution)


def test_series_bo3_iid():
    d = series_distribution([0.6, 0.6, 0.6], 3)
    assert d[(2, 0)] == pytest.approx(0.36)
    assert d[(2, 1)] == pytest.approx(2 * 0.6 * 0.4 * 0.6)
    assert sum(d.values()) == pytest.approx(1.0)


def test_series_bo5_sums_to_one_and_bo1():
    d = series_distribution([0.55, 0.7, 0.4, 0.5, 0.6], 5)
    assert sum(d.values()) == pytest.approx(1.0)
    assert set(d) == {(3, 0), (3, 1), (3, 2), (0, 3), (1, 3), (2, 3)}
    assert series_distribution([0.7], 1) == {(1, 0): pytest.approx(0.7), (0, 1): pytest.approx(0.3)}


def test_round_distribution_valid_scores():
    d = map_round_distribution(0.55, 0.48)
    assert sum(d.values()) == pytest.approx(1.0, abs=1e-9)
    for (a, b), p in d.items():
        assert p >= 0
        if max(a, b) == 13:
            assert min(a, b) <= 11
        else:  # овертайм
            assert max(a, b) >= 16 and abs(a - b) in (1, 2, 3, 4) or (a - b) % 1 == 0


def test_symmetric_map_is_coin_flip():
    assert map_win_prob(0.5, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("p", [0.3, 0.5, 0.62, 0.8])
def test_calibration_hits_target(p):
    r_ct, r_t = calibrate_rounds(p, side_bias=0.1, ct_edge=0.05)
    assert map_win_prob(r_ct, r_t) == pytest.approx(p, abs=1e-4)
    assert r_ct > r_t


def test_log5():
    assert log5(0.5, 0.5) == pytest.approx(0.5)
    assert log5(0.6, 0.5) == pytest.approx(0.6)
