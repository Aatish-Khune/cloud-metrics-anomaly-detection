import numpy as np
import pandas as pd
import pytest

from anomaly.data import (
    MinMaxScaler,
    align_window_scores,
    make_windows,
    nab_labels,
    split_train_val,
)


def test_make_windows_shape_and_content():
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    w = make_windows(x, window=4)
    assert w.shape == (7, 4, 2)
    np.testing.assert_array_equal(w[0], x[:4])
    np.testing.assert_array_equal(w[-1], x[-4:])


def test_make_windows_stride():
    x = np.zeros((10, 3), dtype=np.float32)
    assert make_windows(x, window=4, stride=2).shape == (4, 4, 3)


def test_make_windows_too_short():
    with pytest.raises(ValueError):
        make_windows(np.zeros((3, 1)), window=5)


def test_align_window_scores_uses_window_end():
    scores = align_window_scores(np.array([1.0, 2.0, 3.0]), length=5, window=3)
    # Score of the window ending at t lands on t; first window-1 steps reuse the first score.
    np.testing.assert_array_equal(scores, [1.0, 1.0, 1.0, 2.0, 3.0])


def test_scaler_fit_on_train_only_no_leakage():
    train = np.array([[0.0], [10.0]], dtype=np.float32)
    test = np.array([[5.0], [1000.0]], dtype=np.float32)
    scaler = MinMaxScaler().fit(train)
    out = scaler.transform(test)
    assert out[0, 0] == pytest.approx(0.5)
    # A huge test value must not change the scaling; it is clipped instead.
    assert out[1, 0] == pytest.approx(2.0)
    assert scaler.min_[0] == 0.0 and scaler.range_[0] == 10.0


def test_scaler_constant_feature_does_not_divide_by_zero():
    train = np.ones((5, 1), dtype=np.float32)
    out = MinMaxScaler().fit(train).transform(train)
    assert np.all(np.isfinite(out))


def test_scaler_roundtrip_state():
    s = MinMaxScaler().fit(np.random.rand(20, 3).astype(np.float32))
    s2 = MinMaxScaler.from_state(s.state())
    x = np.random.rand(4, 3).astype(np.float32)
    np.testing.assert_allclose(s.transform(x), s2.transform(x))


def test_split_train_val_is_time_ordered():
    x = np.arange(10)
    fit, val = split_train_val(x, 0.2)
    np.testing.assert_array_equal(fit, np.arange(8))
    np.testing.assert_array_equal(val, [8, 9])


def test_nab_labels_inclusive_windows():
    ts = pd.Series(pd.date_range("2024-01-01", periods=6, freq="5min"))
    labels = nab_labels(ts, [["2024-01-01 00:05:00", "2024-01-01 00:15:00"]])
    np.testing.assert_array_equal(labels, [0, 1, 1, 1, 0, 0])
