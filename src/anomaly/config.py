"""Project-wide paths, dataset lists, and default hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MLFLOW_DB = ROOT / "mlflow.db"  # local SQLite tracking store (MLflow 3 default backend)

SEED = 42

# Server Machine Dataset (SMD) subset: one machine from each of the 3 machine groups.
SMD_MACHINES = ["machine-1-1", "machine-2-1", "machine-3-1"]
SMD_BASE_URL = (
    "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
)

# Numenta Anomaly Benchmark (NAB): all 17 AWS CloudWatch series.
NAB_BASE_URL = "https://raw.githubusercontent.com/numenta/NAB/master"
NAB_AWS_SERIES = [
    "ec2_cpu_utilization_24ae8d",
    "ec2_cpu_utilization_53ea38",
    "ec2_cpu_utilization_5f5533",
    "ec2_cpu_utilization_77c1ca",
    "ec2_cpu_utilization_825cc2",
    "ec2_cpu_utilization_ac20cd",
    "ec2_cpu_utilization_c6585a",
    "ec2_cpu_utilization_fe7f93",
    "ec2_disk_write_bytes_1ef3de",
    "ec2_disk_write_bytes_c0d644",
    "ec2_network_in_257a54",
    "ec2_network_in_5abac7",
    "elb_request_count_8c0756",
    "grok_asg_anomaly",
    "iio_us-east-1_i-a2eb1cd9_NetworkIn",
    "rds_cpu_utilization_cc0c53",
    "rds_cpu_utilization_e47b3b",
]


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters shared by both models. Sensible defaults, no tuning search."""

    window: int = 30  # LSTM autoencoder input length (time steps)
    if_window: int = 5  # Isolation Forest input length (flattened lag window)
    hidden: int = 64
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    train_stride: int = 2  # stride between training windows (speed)
    val_fraction: float = 0.2  # last part of the training split, used only to pick thresholds
    threshold_quantile: float = 0.99  # label-free threshold: quantile of validation scores
    adaptive_window: int = 1440  # rolling-threshold history (SMD: 1-minute steps = 1 day)
    if_estimators: int = 200


SMD_CONFIG = TrainConfig()
# NAB series are short (~4k points) and univariate, so use a smaller model.
# NAB is sampled every 5 minutes, so 288 steps = 1 day of history for the rolling threshold.
NAB_CONFIG = TrainConfig(
    window=24, hidden=32, epochs=30, batch_size=64, train_stride=1, adaptive_window=288
)
# NAB convention: the first 15% of each series is a "probationary" period used for learning.
NAB_TRAIN_FRACTION = 0.15
