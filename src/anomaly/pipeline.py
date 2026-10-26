"""Train and evaluate both detectors on every series, log to MLflow, write reports."""

from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd

from anomaly.config import (
    MLFLOW_DB,
    MODELS_DIR,
    NAB_AWS_SERIES,
    NAB_CONFIG,
    REPORTS_DIR,
    SMD_CONFIG,
    SMD_MACHINES,
    TrainConfig,
)
from anomaly.data import (
    MinMaxScaler,
    Series,
    informative_features,
    load_nab,
    load_smd,
    split_train_val,
)
from anomaly.metrics import evaluate
from anomaly.models import IsolationForestDetector, LSTMAutoencoderDetector, set_seed


def label_free_threshold(val_scores: np.ndarray, quantile: float) -> float:
    """Pick the alarm threshold from validation scores only (no labels used)."""
    return float(np.quantile(val_scores, quantile))


def adaptive_threshold(
    val_scores: np.ndarray, test_scores: np.ndarray, window: int, quantile: float
) -> np.ndarray:
    """Causal rolling threshold: at time t, the `quantile` of the previous `window` scores.

    History starts with the validation scores, so the first test steps already have a
    threshold. Only past scores are used (no labels, no future data), so it works in a
    live stream and adapts when the metrics drift to a new normal level.
    """
    history = pd.Series(np.concatenate([val_scores, test_scores]))
    min_periods = min(window, len(val_scores))
    rolling = history.rolling(window, min_periods=min_periods).quantile(quantile).shift(1)
    return rolling.to_numpy()[len(val_scores):]


def run_series(series: Series, cfg: TrainConfig, dataset: str) -> list[dict]:
    """Fit both detectors on one series and return one result row per detector and
    threshold mode (fixed vs. adaptive)."""
    set_seed()
    fit_raw, val_raw = split_train_val(series.train, cfg.val_fraction)
    keep = informative_features(fit_raw)
    scaler = MinMaxScaler().fit(fit_raw[:, keep])
    fit_x, val_x, test_x = (
        scaler.transform(a[:, keep]) for a in (fit_raw, val_raw, series.test)
    )

    detectors = [
        IsolationForestDetector(cfg),
        LSTMAutoencoderDetector(cfg, n_features=int(keep.sum())),
    ]
    rows = []
    for det in detectors:
        start = time.perf_counter()
        if isinstance(det, LSTMAutoencoderDetector):
            det.fit(fit_x, val_x)
        else:
            det.fit(fit_x)
        fit_seconds = time.perf_counter() - start
        val_scores, test_scores = det.score(val_x), det.score(test_x)
        threshold = label_free_threshold(val_scores, cfg.threshold_quantile)
        results = {
            "fixed": evaluate(series.test_labels, test_scores, threshold),
            "adaptive": evaluate(
                series.test_labels,
                test_scores,
                adaptive_threshold(val_scores, test_scores, cfg.adaptive_window,
                                   cfg.threshold_quantile),
            ),
        }

        with mlflow.start_run(run_name=f"{dataset}/{series.name}/{det.name}"):
            mlflow.set_tags({"dataset": dataset, "series": series.name, "model": det.name})
            mlflow.log_params({**cfg.__dict__, "n_features": int(keep.sum()),
                               "n_features_dropped": int((~keep).sum())})
            for mode, metrics in results.items():
                mlflow.log_metrics(
                    {f"{mode}_{k}": v for k, v in metrics.items() if np.isfinite(v)}
                )
            mlflow.log_metrics({"threshold": threshold, "fit_seconds": fit_seconds})
            if isinstance(det, LSTMAutoencoderDetector):
                for rec in det.history:
                    mlflow.log_metric("train_loss", rec["train_loss"], step=rec["epoch"])
                    if "val_loss" in rec:
                        mlflow.log_metric("val_loss", rec["val_loss"], step=rec["epoch"])

        if isinstance(det, LSTMAutoencoderDetector):
            det.save(
                MODELS_DIR / dataset / f"{series.name}.pt",
                extra={"scaler": scaler.state(), "threshold": threshold,
                       "keep": keep.tolist()},
            )
        for mode, metrics in results.items():
            rows.append(
                {
                    "dataset": dataset,
                    "series": series.name,
                    "model": det.name,
                    "threshold_mode": mode,
                    "fit_seconds": round(fit_seconds, 1),
                    **metrics,
                }
            )
            print(
                f"{dataset:>4} {series.name:<38} {det.name:<17} {mode:<8} "
                f"F1={metrics['f1']:.3f} AUC-PR={metrics['auc_pr']:.3f} "
                f"alarms={metrics['alarm_rate']:.1%} ({fit_seconds:.0f}s)"
            )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric per dataset and model (the README headline table).

    Series with no labeled anomalies in the test part (one NAB series) are excluded, since
    recall, F1, and AUC-PR are undefined there. They remain in results.csv.
    """
    cols = ["auc_pr", "oracle_best_f1", "precision", "recall", "f1", "alarm_rate",
            "event_recall", "mean_delay_steps"]
    keys = ["dataset", "model", "threshold_mode"]
    labeled = results[results["anomaly_rate"] > 0]
    summary = labeled.groupby(keys)[cols].mean().round(3)
    summary.insert(0, "n_series", labeled.groupby(keys).size())
    return summary.reset_index()


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (avoids an extra dependency)."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body]) + "\n"


def run_all(include_nab: bool = True) -> pd.DataFrame:
    """Run the full benchmark and write reports/results.csv and reports/summary.md."""
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("cloud-metrics-anomaly-detection")
    rows: list[dict] = []
    for machine in SMD_MACHINES:
        rows += run_series(load_smd(machine), SMD_CONFIG, "smd")
    if include_nab:
        for name in NAB_AWS_SERIES:
            rows += run_series(load_nab(name), NAB_CONFIG, "nab")

    results = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(REPORTS_DIR / "results.csv", index=False)
    summary = summarize(results)
    (REPORTS_DIR / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    with open(REPORTS_DIR / "configs.json", "w", encoding="utf-8") as fh:
        json.dump({"smd": SMD_CONFIG.__dict__, "nab": NAB_CONFIG.__dict__}, fh, indent=2)
    print("\n" + summary.to_string(index=False))
    return results
