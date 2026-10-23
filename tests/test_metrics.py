import math

import numpy as np
import pytest

from anomaly.metrics import (
    best_f1_threshold,
    evaluate,
    event_recall_and_delay,
    point_prf,
    segments,
)


def test_segments():
    assert segments(np.array([0, 1, 1, 0, 0, 1, 0, 1])) == [(1, 2), (5, 5), (7, 7)]
    assert segments(np.zeros(4)) == []


def test_point_prf_hand_example():
    y_true = np.array([0, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    p, r, f1 = point_prf(y_true, y_pred)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(1 / 3)
    assert f1 == pytest.approx(0.4)


def test_point_prf_no_predictions_is_zero_not_nan():
    assert point_prf(np.array([1, 0]), np.array([0, 0])) == (0.0, 0.0, 0.0)


def test_event_recall_and_delay():
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])  # first segment caught 2 steps late
    recall, delay = event_recall_and_delay(y_true, y_pred)
    assert recall == pytest.approx(0.5)
    assert delay == pytest.approx(2.0)


def test_event_recall_no_segments_is_nan():
    recall, delay = event_recall_and_delay(np.zeros(5), np.ones(5))
    assert math.isnan(recall) and math.isnan(delay)


def test_best_f1_threshold_perfect_separation():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    thr, f1 = best_f1_threshold(y, scores)
    assert f1 == pytest.approx(1.0)
    assert thr == pytest.approx(0.8)


def test_best_f1_threshold_handles_ties():
    y = np.array([1, 0, 0])
    scores = np.array([0.5, 0.5, 0.1])
    # Cannot split the tie: flagging both 0.5s gives P=0.5, R=1 -> F1=2/3.
    _, f1 = best_f1_threshold(y, scores)
    assert f1 == pytest.approx(2 / 3)


def test_evaluate_keys_and_alarm_rate():
    y = np.array([0, 0, 1, 1, 0])
    scores = np.array([0.0, 0.1, 0.9, 0.8, 0.2])
    m = evaluate(y, scores, threshold=0.5)
    assert m["f1"] == pytest.approx(1.0)
    assert m["alarm_rate"] == pytest.approx(0.4)
    assert m["auc_pr"] == pytest.approx(1.0)
    assert m["event_recall"] == pytest.approx(1.0)
