"""Two detectors with the same interface: fit on (mostly normal) training data, then
return one anomaly score per time step. Higher score = more anomalous.

- IsolationForestDetector: classical baseline on short flattened lag windows.
- LSTMAutoencoderDetector: PyTorch sequence autoencoder; score = reconstruction error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch import nn

from anomaly.config import SEED, TrainConfig
from anomaly.data import align_window_scores, make_windows


def set_seed(seed: int = SEED) -> None:
    """Seed NumPy and PyTorch for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)


class IsolationForestDetector:
    """Isolation Forest on flattened windows of the last `if_window` time steps."""

    name = "isolation_forest"

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.model = IsolationForest(
            n_estimators=cfg.if_estimators, random_state=SEED, n_jobs=-1
        )

    def _features(self, x: np.ndarray) -> np.ndarray:
        w = make_windows(x, self.cfg.if_window)
        return w.reshape(len(w), -1)

    def fit(self, train: np.ndarray) -> IsolationForestDetector:
        self.model.fit(self._features(train))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        # score_samples is higher for normal points, so negate it.
        window_scores = -self.model.score_samples(self._features(x))
        return align_window_scores(window_scores, len(x), self.cfg.if_window)


class LSTMAutoencoder(nn.Module):
    """Encoder LSTM compresses a window into its last hidden state; the decoder LSTM
    reconstructs the whole window from that state repeated over time."""

    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)  # h: (1, batch, hidden)
        z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)  # (batch, window, hidden)
        out, _ = self.decoder(z)
        return self.output(out)  # (batch, window, n_features)


class LSTMAutoencoderDetector:
    """Train an LSTM autoencoder on normal windows. The anomaly score at time t is the
    reconstruction error of the window's last step (it only uses data up to t)."""

    name = "lstm_autoencoder"

    def __init__(self, cfg: TrainConfig, n_features: int) -> None:
        self.cfg = cfg
        self.n_features = n_features
        self.model = LSTMAutoencoder(n_features, cfg.hidden)
        self.history: list[dict[str, float]] = []

    def fit(self, train: np.ndarray, val: np.ndarray | None = None) -> LSTMAutoencoderDetector:
        set_seed()
        x = torch.from_numpy(make_windows(train, self.cfg.window, self.cfg.train_stride))
        x_val = torch.from_numpy(make_windows(val, self.cfg.window)) if val is not None else None
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(SEED)
        for epoch in range(self.cfg.epochs):
            self.model.train()
            perm = torch.randperm(len(x), generator=gen)
            total = 0.0
            for i in range(0, len(x), self.cfg.batch_size):
                batch = x[perm[i : i + self.cfg.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.model(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            record = {"epoch": epoch + 1, "train_loss": total / len(x)}
            if x_val is not None:
                record["val_loss"] = self._mean_loss(x_val)
            self.history.append(record)
        return self

    @torch.no_grad()
    def _mean_loss(self, x: torch.Tensor) -> float:
        self.model.eval()
        losses = [
            nn.functional.mse_loss(self.model(x[i : i + 1024]), x[i : i + 1024], reduction="sum")
            for i in range(0, len(x), 1024)
        ]
        return float(sum(losses) / x.numel())

    @torch.no_grad()
    def score(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        windows = torch.from_numpy(make_windows(x, self.cfg.window))
        errors = []
        for i in range(0, len(windows), 1024):
            batch = windows[i : i + 1024]
            recon = self.model(batch)
            errors.append(((recon[:, -1, :] - batch[:, -1, :]) ** 2).mean(dim=1))
        window_scores = torch.cat(errors).numpy()
        return align_window_scores(window_scores, len(x), self.cfg.window)

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_features": self.n_features,
                "cfg": self.cfg.__dict__,
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> tuple[LSTMAutoencoderDetector, dict]:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        det = cls(TrainConfig(**ckpt["cfg"]), ckpt["n_features"])
        det.model.load_state_dict(ckpt["state_dict"])
        det.model.eval()
        return det, ckpt["extra"]
