"""Evaluation metrics for time-series anomaly detection.

We deliberately report plain point-wise metrics plus event-level recall and detection delay.
We do NOT use "point-adjusted" F1 (marking a whole anomaly segment as detected if any single
point is flagged), because it is known to inflate scores, even for random detectors.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous anomaly segments as inclusive (start, end) index pairs."""
    labels = np.asarray(labels).astype(int)
    padded = np.concatenate([[0], labels, [0]])
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def point_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Point-wise precision, recall, F1 (0.0 when undefined)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def event_recall_and_delay(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Share of anomaly segments with at least one alarm inside them, and the mean delay
    (in time steps from segment start to first alarm) over the detected segments.

    Delay is NaN when no segment is detected; recall is NaN when there are no segments.
    """
    segs = segments(y_true)
    if not segs:
        return float("nan"), float("nan")
    y_pred = np.asarray(y_pred).astype(bool)
    delays = []
    for start, end in segs:
        hits = np.flatnonzero(y_pred[start : end + 1])
        if hits.size:
            delays.append(int(hits[0]))
    recall = len(delays) / len(segs)
    delay = float(np.mean(delays)) if delays else float("nan")
    return recall, delay


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Oracle threshold that maximizes point-wise F1 on the given labels.

    This uses test labels, so it is an UPPER BOUND for reference only, never a deployable result.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    tp = np.cumsum(y_true[order])
    fp = np.cumsum(1 - y_true[order])
    total_pos = y_true.sum()
    if total_pos == 0:
        return float("nan"), 0.0
    precision = tp / (tp + fp)
    recall = tp / total_pos
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    # Only consider cut points where the score changes, so ties are handled correctly.
    last_of_tie = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    f1 = np.where(last_of_tie, f1, -1.0)
    i = int(np.argmax(f1))
    return float(sorted_scores[i]), float(f1[i])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """All metrics for one detector on one series."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = scores >= threshold
    precision, recall, f1 = point_prf(y_true, y_pred)
    ev_recall, delay = event_recall_and_delay(y_true, y_pred)
    has_both = 0 < y_true.sum() < len(y_true)
    auc_pr = float(average_precision_score(y_true, scores)) if has_both else float("nan")
    _, oracle_f1 = best_f1_threshold(y_true, scores)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "event_recall": ev_recall,
        "mean_delay_steps": delay,
        "auc_pr": auc_pr,
        "oracle_best_f1": oracle_f1,
        "alarm_rate": float(np.mean(y_pred)),
        "anomaly_rate": float(np.mean(y_true)),
    }
