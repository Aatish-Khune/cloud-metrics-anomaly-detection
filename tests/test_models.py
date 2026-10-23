import numpy as np
import torch

from anomaly.config import TrainConfig
from anomaly.models import IsolationForestDetector, LSTMAutoencoder, LSTMAutoencoderDetector

CFG = TrainConfig(window=8, if_window=3, hidden=8, epochs=3, batch_size=16, if_estimators=20)


def _sine(n: int, d: int = 2) -> np.ndarray:
    t = np.arange(n, dtype=np.float32)[:, None]
    return (0.5 + 0.4 * np.sin(t / 5 + np.arange(d))).astype(np.float32)


def test_autoencoder_forward_shape():
    model = LSTMAutoencoder(n_features=3, hidden=8)
    out = model(torch.zeros(4, 10, 3))
    assert out.shape == (4, 10, 3)


def test_detectors_return_one_score_per_time_step():
    train, test = _sine(200), _sine(60)
    for det in (IsolationForestDetector(CFG), LSTMAutoencoderDetector(CFG, n_features=2)):
        det.fit(train)
        scores = det.score(test)
        assert scores.shape == (60,)
        assert np.all(np.isfinite(scores))


def test_lstm_scores_injected_spike_higher():
    train = _sine(300)
    det = LSTMAutoencoderDetector(
        TrainConfig(window=8, hidden=16, epochs=15, batch_size=32), n_features=2
    ).fit(train)
    test = _sine(80)
    test[50] = 1.8  # an out-of-pattern spike
    scores = det.score(test)
    assert scores[50] > np.percentile(np.delete(scores, 50), 95)


def test_lstm_training_is_reproducible():
    train = _sine(120)
    a = LSTMAutoencoderDetector(CFG, n_features=2).fit(train).score(train)
    b = LSTMAutoencoderDetector(CFG, n_features=2).fit(train).score(train)
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_save_and_load_roundtrip(tmp_path):
    train = _sine(100)
    det = LSTMAutoencoderDetector(CFG, n_features=2).fit(train)
    path = tmp_path / "m.pt"
    det.save(path, extra={"threshold": 0.1})
    loaded, extra = LSTMAutoencoderDetector.load(path)
    assert extra["threshold"] == 0.1
    np.testing.assert_allclose(det.score(train), loaded.score(train), rtol=1e-6)
