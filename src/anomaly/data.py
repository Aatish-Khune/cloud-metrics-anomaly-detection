"""Loading, scaling, splitting, and windowing for SMD and NAB time series."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaly.config import NAB_TRAIN_FRACTION, RAW_DIR


@dataclass
class Series:
    """One benchmark series: unlabeled training data plus labeled test data.

    Arrays are shaped (time, features). `test_labels` is 1 for anomalous time steps.
    """

    name: str
    train: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    feature_names: list[str]
    test_index: pd.Index | None = None  # timestamps for NAB, None for SMD


def load_smd(machine: str) -> Series:
    """Load one SMD machine. Train is assumed mostly normal; labels exist only for test."""
    base = RAW_DIR / "smd"
    train = np.loadtxt(base / "train" / f"{machine}.txt", delimiter=",", dtype=np.float32)
    test = np.loadtxt(base / "test" / f"{machine}.txt", delimiter=",", dtype=np.float32)
    labels = np.loadtxt(base / "test_label" / f"{machine}.txt", dtype=np.int64)
    names = [f"metric_{i:02d}" for i in range(train.shape[1])]
    return Series(machine, train, test, labels, names)


def nab_labels(timestamps: pd.Series, windows: list[list[str]]) -> np.ndarray:
    """Turn NAB [start, end] timestamp windows into a 0/1 label per time step."""
    labels = np.zeros(len(timestamps), dtype=np.int64)
    for start, end in windows:
        mask = (timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(end))
        labels[mask.to_numpy()] = 1
    return labels


def load_nab(series: str, train_fraction: float = NAB_TRAIN_FRACTION) -> Series:
    """Load one NAB AWS series. The first `train_fraction` is the (probationary) training part."""
    base = RAW_DIR / "nab"
    df = pd.read_csv(base / f"{series}.csv", parse_dates=["timestamp"])
    with open(base / "combined_windows.json", encoding="utf-8") as fh:
        windows = json.load(fh)[f"realAWSCloudwatch/{series}.csv"]
    labels = nab_labels(df["timestamp"], windows)
    values = df[["value"]].to_numpy(dtype=np.float32)
    cut = int(len(df) * train_fraction)
    return Series(
        name=series,
        train=values[:cut],
        test=values[cut:],
        test_labels=labels[cut:],
        feature_names=["value"],
        test_index=pd.Index(df["timestamp"].iloc[cut:]),
    )


class MinMaxScaler:
    """Min-max scaler fit on training data only (prevents test-set leakage)."""

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> MinMaxScaler:
        self.min_ = x.min(axis=0)
        rng = x.max(axis=0) - self.min_
        self.range_ = np.where(rng == 0, 1.0, rng).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("Scaler must be fit before transform.")
        # Test values can fall outside the training range; clip to keep scores bounded.
        return np.clip((x - self.min_) / self.range_, -1.0, 2.0).astype(np.float32)

    def state(self) -> dict[str, list[float]]:
        return {"min": self.min_.tolist(), "range": self.range_.tolist()}

    @classmethod
    def from_state(cls, state: dict[str, list[float]]) -> MinMaxScaler:
        s = cls()
        s.min_ = np.asarray(state["min"], dtype=np.float32)
        s.range_ = np.asarray(state["range"], dtype=np.float32)
        return s


def split_train_val(x: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered split: the last `val_fraction` of the training data becomes validation."""
    cut = int(len(x) * (1 - val_fraction))
    return x[:cut], x[cut:]


def make_windows(x: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """Sliding windows over time: (T, D) -> (N, window, D), N = (T - window) // stride + 1."""
    if len(x) < window:
        raise ValueError(f"Series length {len(x)} is shorter than window {window}.")
    starts = np.arange(0, len(x) - window + 1, stride)
    return np.stack([x[s : s + window] for s in starts])


def align_window_scores(window_scores: np.ndarray, length: int, window: int) -> np.ndarray:
    """Map one score per window (stride 1) onto time steps.

    Each window's score is assigned to its LAST time step, so a score at time t only uses
    data up to t (as a streaming detector would). The first window-1 steps have no full
    window and get the first available score.
    """
    if len(window_scores) != length - window + 1:
        raise ValueError("Expected one score per stride-1 window.")
    head = np.full(window - 1, window_scores[0], dtype=np.float64)
    return np.concatenate([head, window_scores.astype(np.float64)])
