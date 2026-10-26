"""Streamlit demo: replay a server-metric stream and watch the detectors raise alerts.

Run locally with `python run.py app`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly.config import (  # noqa: E402
    MODELS_DIR,
    NAB_AWS_SERIES,
    NAB_CONFIG,
    REPORTS_DIR,
    SMD_CONFIG,
    SMD_MACHINES,
)
from anomaly.data import (  # noqa: E402
    MinMaxScaler,
    informative_features,
    load_nab,
    load_smd,
    split_train_val,
)
from anomaly.download import download, file_list  # noqa: E402
from anomaly.metrics import evaluate, segments  # noqa: E402
from anomaly.models import IsolationForestDetector, LSTMAutoencoderDetector  # noqa: E402
from anomaly.pipeline import adaptive_threshold, label_free_threshold  # noqa: E402

MODEL_LABELS = {"lstm_autoencoder": "LSTM autoencoder (PyTorch)",
                "isolation_forest": "Isolation Forest (baseline)"}

st.set_page_config(page_title="Cloud Metrics Anomaly Detection", layout="wide")


def ensure_files(dataset: str, name: str) -> None:
    """Download only the raw files needed for one series (skips files already present)."""
    for url, dest in file_list():
        rel = dest.as_posix()
        if (dataset == "smd" and "/smd/" in rel and dest.stem == name) or (
            dataset == "nab" and (dest.stem == name or dest.name == "combined_windows.json")
        ):
            download(url, dest)


@st.cache_data(show_spinner="Loading data and scoring the stream...")
def score_series(dataset: str, name: str, model: str, mode: str) -> dict:
    ensure_files(dataset, name)
    series = load_smd(name) if dataset == "smd" else load_nab(name)
    cfg = SMD_CONFIG if dataset == "smd" else NAB_CONFIG
    fit_raw, val_raw = split_train_val(series.train, cfg.val_fraction)
    keep = informative_features(fit_raw)

    if model == "lstm_autoencoder":
        det, extra = LSTMAutoencoderDetector.load(MODELS_DIR / dataset / f"{name}.pt")
        scaler = MinMaxScaler.from_state(extra["scaler"])
        keep = np.asarray(extra["keep"], dtype=bool)
    else:
        scaler = MinMaxScaler().fit(fit_raw[:, keep])
        det = IsolationForestDetector(cfg).fit(scaler.transform(fit_raw[:, keep]))

    val_scores = det.score(scaler.transform(val_raw[:, keep]))
    scores = det.score(scaler.transform(series.test[:, keep]))
    if mode == "adaptive":
        threshold = adaptive_threshold(val_scores, scores, cfg.adaptive_window,
                                       cfg.threshold_quantile)
    else:
        fixed = label_free_threshold(val_scores, cfg.threshold_quantile)
        threshold = np.full(len(scores), fixed)
    return {
        "test": series.test,
        "labels": series.test_labels,
        "scores": scores,
        "threshold": threshold,
        "features": series.feature_names,
        "index": (series.test_index if series.test_index is not None
                  else pd.RangeIndex(len(series.test))),
    }


def make_figure(res: dict, feature: int, upto: int) -> go.Figure:
    x = res["index"][:upto]
    values = res["test"][:upto, feature]
    scores = res["scores"][:upto]
    threshold = res["threshold"][:upto]
    alerts = scores >= threshold
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.55, 0.45],
                        subplot_titles=(f"Metric: {res['features'][feature]}", "Anomaly score"))
    fig.add_trace(go.Scatter(x=x, y=values, name="metric", line=dict(color="#1f77b4", width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x[alerts], y=values[alerts], mode="markers", name="alert",
                             marker=dict(color="#ff7f0e", size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=scores, name="score", line=dict(color="#2ca02c", width=1)),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=threshold, name="threshold",
                             line=dict(color="#7f7f7f", width=1, dash="dash")), row=2, col=1)
    # Shapes must be added after the traces: Plotly skips shapes on empty subplots.
    for start, end in segments(res["labels"][:upto]):
        for row in (1, 2):
            fig.add_vrect(x0=x[start], x1=x[end], fillcolor="#d62728", opacity=0.25,
                          line_width=0, row=row, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.08))
    return fig


st.title("Real-time anomaly detection for cloud server metrics")
st.caption("Red bands = labeled anomalies. Orange dots = alerts raised by the selected model. "
           "Thresholds never use labels: either fixed from held-out training data, or "
           "adaptive from the stream's own recent scores.")

with st.sidebar:
    dataset = st.radio("Dataset", ["smd", "nab"], format_func=lambda d: {
        "smd": "Server Machine Dataset (38 metrics)",
        "nab": "NAB AWS CloudWatch (1 metric)"}[d])
    options = SMD_MACHINES if dataset == "smd" else NAB_AWS_SERIES
    name = st.selectbox("Series", options)
    model = st.radio("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
    mode = st.radio("Alarm threshold", ["fixed", "adaptive"], format_func={
        "fixed": "Fixed (99th pct of held-out training scores)",
        "adaptive": "Adaptive (rolling 99th pct of the last day)"}.get)
    speed = st.slider("Replay speed (steps per frame)", 20, 500, 150, step=10)

res = score_series(dataset, name, model, mode)
default_feature = int(np.argmax(res["test"].std(axis=0)))
feature = st.selectbox("Metric to plot", range(len(res["features"])), index=default_feature,
                       format_func=lambda i: res["features"][i])

m = evaluate(res["labels"], res["scores"], res["threshold"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Point F1", f"{m['f1']:.3f}")
c2.metric("AUC-PR", f"{m['auc_pr']:.3f}")
c3.metric("Anomaly events caught", f"{m['event_recall']:.0%}")
c4.metric("Alarm rate", f"{m['alarm_rate']:.1%}")

chart = st.empty()
status = st.empty()
if st.button("▶ Replay stream", type="primary"):
    n = len(res["scores"])
    for upto in range(speed, n + speed, speed):
        upto = min(upto, n)
        chart.plotly_chart(make_figure(res, feature, upto), use_container_width=True)
        n_alerts = int((res["scores"][:upto] >= res["threshold"][:upto]).sum())
        status.info(f"t = {upto:,} / {n:,} steps · alerts so far: {n_alerts:,}")
        time.sleep(0.05)
else:
    chart.plotly_chart(make_figure(res, feature, len(res["scores"])), use_container_width=True)

results_path = REPORTS_DIR / "summary.md"
with st.expander("Benchmark results (all series, from `python run.py train`)"):
    if results_path.exists():
        st.markdown(results_path.read_text(encoding="utf-8"))
    else:
        st.write("Run `python run.py train` to generate results.")
