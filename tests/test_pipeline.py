import numpy as np
import pytest

from anomaly.pipeline import adaptive_threshold, label_free_threshold


def test_label_free_threshold_is_quantile():
    assert label_free_threshold(np.arange(101, dtype=float), 0.9) == pytest.approx(90.0)


def test_adaptive_threshold_is_causal():
    rng = np.random.default_rng(0)
    val, test = rng.random(50), rng.random(40)
    thr = adaptive_threshold(val, test, window=20, quantile=0.9)
    changed = test.copy()
    changed[25:] += 100.0  # change the present and future
    thr2 = adaptive_threshold(val, changed, window=20, quantile=0.9)
    # Thresholds up to and including t=25 depend only on scores before t.
    np.testing.assert_allclose(thr[:26], thr2[:26])
    assert thr2[30] > thr[30]


def test_adaptive_threshold_starts_from_validation_history():
    val, test = np.zeros(10), np.ones(5)
    thr = adaptive_threshold(val, test, window=10, quantile=0.5)
    assert thr.shape == (5,)
    assert np.all(np.isfinite(thr))
    assert thr[0] == pytest.approx(0.0)


def test_adaptive_threshold_follows_level_shift():
    # After a sustained level shift (drift), the threshold rises to the new normal.
    rng = np.random.default_rng(1)
    val = rng.normal(0, 0.1, 100)
    test = np.r_[rng.normal(0, 0.1, 50), rng.normal(5, 0.1, 300)]
    thr = adaptive_threshold(val, test, window=100, quantile=0.99)
    alarms = test >= thr
    # The onset is flagged, but a rolling 99th percentile absorbs a sustained shift after
    # ~1% of the window (here 1-2 steps), so later points at the new level stop alarming.
    assert alarms[50:52].all()
    assert alarms[-100:].mean() < 0.05
