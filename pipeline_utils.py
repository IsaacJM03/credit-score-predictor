from __future__ import annotations

import json
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "results" / "synthetic" / "dataset_5m" / "synthetic_dataset_5000000.csv"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models"

TARGET_COLUMN = "Repayment_Status"
DURATION_COLUMN = "Duration_Months"

BASE_FEATURES: List[str] = [
    "Age",
    "Monthly_Income",
    "Credit_Score",
    "Loan_Amount_Requested",
    "Existing_Debts",
    "Num_Previous_Loans",
    "Payment_History_Encoded",
    "Employment_Status_Encoded",
    "Debt_to_Income_Ratio",
    "Loan_to_Income_Ratio",
    "Repayment_Consistency",
    "Application_Velocity",
    "Credit_Utilization_Score",
]


@dataclass
class TopicArtifact:
    name: str
    path: str
    kind: str


def ensure_directory(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_dataset(dataset_path: Path | str = DEFAULT_DATASET_PATH, max_rows: Optional[int] = None) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, nrows=max_rows)


def select_features(frame: pd.DataFrame, features: Sequence[str] = BASE_FEATURES) -> Tuple[pd.DataFrame, List[str]]:
    available = [column for column in features if column in frame.columns]
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Required feature columns missing from dataset: {missing}")
    return frame[available].copy(), available


def prepare_classification_frame(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    frame = load_dataset(dataset_path=dataset_path, max_rows=max_rows).dropna().copy()
    feature_frame, features = select_features(frame)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column missing from dataset: {TARGET_COLUMN}")
    target = frame[TARGET_COLUMN].astype(int)
    return feature_frame, target, features, frame


def prepare_survival_frame(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    frame = load_dataset(dataset_path=dataset_path, max_rows=max_rows).dropna().copy()
    feature_frame, features = select_features(frame)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column missing from dataset: {TARGET_COLUMN}")
    if DURATION_COLUMN not in frame.columns:
        base = (
            24.0
            + 0.00005 * frame["Monthly_Income"].astype(float)
            + 0.03 * frame["Credit_Score"].astype(float)
            - 0.004 * frame["Existing_Debts"].astype(float)
        )
        noise = np.random.default_rng(42).normal(0.0, 4.0, size=len(frame))
        frame[DURATION_COLUMN] = np.clip(np.round(base + noise), 1, 72).astype(int)
    return pd.concat([feature_frame, frame[[TARGET_COLUMN, DURATION_COLUMN]]], axis=1), features


def prepare_unsupervised_frame(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    frame = load_dataset(dataset_path=dataset_path, max_rows=max_rows).dropna().copy()
    feature_frame, features = select_features(frame)
    return feature_frame, features


def make_topic_root(topic: str) -> Path:
    return ensure_directory(DEFAULT_MODEL_ROOT / topic)


def make_topic_visualization_root(topic: str) -> Path:
    return ensure_directory(DEFAULT_MODEL_ROOT / topic / "visualizations")


def save_json(payload: Dict[str, Any], path: Path | str) -> None:
    output = Path(path)
    ensure_directory(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def save_joblib_artifact(obj: Any, path: Path | str) -> None:
    output = Path(path)
    ensure_directory(output.parent)
    joblib.dump(obj, output)


def save_figure(figure, path: Path | str, dpi: int = 300) -> str:
    output = Path(path)
    ensure_directory(output.parent)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    return str(output)


def log_message(topic: str, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{topic}] {message}", flush=True)


def elapsed_seconds(start_time: float) -> float:
    return float(time.perf_counter() - start_time)


def safe_roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def safe_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> Optional[float]:
    try:
        from sklearn.metrics import f1_score

        return float(f1_score(y_true, y_pred))
    except Exception:
        return None


def safe_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> Optional[float]:
    try:
        from sklearn.metrics import accuracy_score

        return float(accuracy_score(y_true, y_pred))
    except Exception:
        return None


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
):
    from sklearn.model_selection import train_test_split

    return train_test_split(features, target, test_size=test_size, stratify=target, random_state=seed)


def common_report(
    dataset_path: Path | str,
    feature_names: Sequence[str],
    artifact_paths: Sequence[TopicArtifact],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset_path": str(dataset_path),
        "feature_count": len(feature_names),
        "features": list(feature_names),
        "artifacts": [asdict(item) for item in artifact_paths],
        "metrics": metrics,
    }