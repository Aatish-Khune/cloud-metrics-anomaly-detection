"""Train and evaluate both detectors on every series, log to MLflow, write reports."""

from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd

from anomaly.config import (
    MLFLOW_DIR,
    MODELS_DIR,
    NAB_AWS_SERIES,
    NAB_CONFIG,
    REPORTS_DIR,
    SMD_CONFIG,
    SMD_MACHINES,
    TrainConfig,
)
from anomaly.data import MinMaxScaler, Series, load_nab, load_smd, split_train_val
from anomaly.metrics import evaluate
from anomaly.models import IsolationForestDetector, LSTMAutoencoderDetector, set_seed


def label_free_threshold(val_scores: np.ndarray, quantile: float) -> float:
    """Pick the alarm threshold from validation scores only (no labels used)."""
    return float(np.quantile(val_scores, quantile))


def run_series(series: Series, cfg: TrainConfig, dataset: str) -> list[dict]:
    """Fit both detectors on one series and return one result row per detector."""
    set_seed()
    fit_raw, val_raw = split_train_val(series.train, cfg.val_fraction)
    scaler = MinMaxScaler().fit(fit_raw)
    fit_x, val_x, test_x = (scaler.transform(a) for a in (fit_raw, val_raw, series.test))

    detectors = [
        IsolationForestDetector(cfg),
        LSTMAutoencoderDetector(cfg, n_features=series.train.shape[1]),
    ]
    rows = []
    for det in detectors:
        start = time.perf_counter()
        if isinstance(det, LSTMAutoencoderDetector):
            det.fit(fit_x, val_x)
        else:
            det.fit(fit_x)
        fit_seconds = time.perf_counter() - start
        threshold = label_free_threshold(det.score(val_x), cfg.threshold_quantile)
        metrics = evaluate(series.test_labels, det.score(test_x), threshold)

        with mlflow.start_run(run_name=f"{dataset}/{series.name}/{det.name}"):
            mlflow.set_tags({"dataset": dataset, "series": series.name, "model": det.name})
            mlflow.log_params({**cfg.__dict__, "n_features": series.train.shape[1]})
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if np.isfinite(v)}
                | {"threshold": threshold, "fit_seconds": fit_seconds}
            )
            if isinstance(det, LSTMAutoencoderDetector):
                for rec in det.history:
                    mlflow.log_metric("train_loss", rec["train_loss"], step=rec["epoch"])
                    if "val_loss" in rec:
                        mlflow.log_metric("val_loss", rec["val_loss"], step=rec["epoch"])

        if isinstance(det, LSTMAutoencoderDetector):
            det.save(
                MODELS_DIR / dataset / f"{series.name}.pt",
                extra={"scaler": scaler.state(), "threshold": threshold},
            )
        rows.append(
            {
                "dataset": dataset,
                "series": series.name,
                "model": det.name,
                "threshold": threshold,
                "fit_seconds": round(fit_seconds, 1),
                **metrics,
            }
        )
        print(
            f"{dataset:>4} {series.name:<38} {det.name:<17} "
            f"F1={metrics['f1']:.3f} AUC-PR={metrics['auc_pr']:.3f} "
            f"event_recall={metrics['event_recall']:.2f} ({fit_seconds:.0f}s)"
        )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric per dataset and model (the README headline table).

    Series with no labeled anomalies in the test part (one NAB series) are excluded, since
    recall, F1, and AUC-PR are undefined there. They remain in results.csv.
    """
    cols = ["precision", "recall", "f1", "auc_pr", "event_recall", "mean_delay_steps",
            "oracle_best_f1"]
    labeled = results[results["anomaly_rate"] > 0]
    summary = labeled.groupby(["dataset", "model"])[cols].mean().round(3)
    summary.insert(0, "n_series", labeled.groupby(["dataset", "model"]).size())
    return summary.reset_index()


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (avoids an extra dependency)."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body]) + "\n"


def run_all(include_nab: bool = True) -> pd.DataFrame:
    """Run the full benchmark and write reports/results.csv and reports/summary.md."""
    mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
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
