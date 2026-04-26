from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from pipeline_utils import (
    TopicArtifact,
    common_report,
    elapsed_seconds,
    log_message,
    make_topic_root,
    make_topic_visualization_root,
    prepare_classification_frame,
    save_figure,
    save_json,
    save_joblib_artifact,
    safe_accuracy,
    safe_f1,
    safe_roc_auc,
    stratified_split,
)

SEED = 42
DEFAULT_DATASET = Path("results/synthetic/dataset_5m/synthetic_dataset_5000000.csv")

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False


class TabularLSTM(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.feature_count = feature_count

    def forward(self, x):
        sequence = x.unsqueeze(-1)
        outputs, _ = self.lstm(sequence)
        return self.head(outputs[:, -1, :]).squeeze(-1)


def _build_models(seed: int):
    rf = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier

        xgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, random_state=seed)
    if HAS_LIGHTGBM:
        lgbm = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier

        lgbm = GradientBoostingClassifier(random_state=seed)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft",
        weights=[2, 2, 1],
        n_jobs=-1,
    )
    return rf, xgb, lgbm, ensemble


def _fit_lstm(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int, output_dir: Path):
    log_message("post_loan", "Training model: lstm")
    lstm_start = time.perf_counter()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = TabularLSTM(feature_count=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    losses: List[float] = []

    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train.astype(np.float32), dtype=torch.float32, device=device)
    batch_size = min(2048, max(256, len(x_tensor) // 128))

    model.train()
    for _ in range(3):
        permutation = torch.randperm(len(x_tensor), device=device)
        for start in range(0, len(x_tensor), batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_x = x_tensor[batch_idx]
            batch_y = y_tensor[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32, device=device))).cpu().numpy()
    artifact_path = output_dir / "lstm_state.pt"
    torch.save({"state_dict": model.state_dict(), "feature_count": x_train.shape[1]}, artifact_path)
    log_message("post_loan", f"Completed lstm in {elapsed_seconds(lstm_start):.1f}s")
    return model, scores, artifact_path, losses


def _save_scores_plot(y_true, scores, path: Path, title: str) -> Optional[str]:
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_true, scores, ax=axes[0], name=title)
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=axes[1], name=title)
    return save_figure(fig, path, dpi=300)


def _save_training_loss(losses: List[float], path: Path) -> Optional[str]:
    if not losses:
        return None
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(losses, color="#d62728", linewidth=1.5)
    ax.set_title("LSTM Training Loss")
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Loss")
    return save_figure(fig, path, dpi=300)


def run(
    dataset_path: Path | str = DEFAULT_DATASET,
    output_root: Path | str = Path("models/post_loan_monitoring"),
    seed: int = SEED,
    selected_models: Optional[Sequence[str]] = None,
):
    pipeline_start = time.perf_counter()
    log_message("post_loan", f"Loading dataset from {dataset_path}")
    features, target, feature_names, _ = prepare_classification_frame(dataset_path=dataset_path)
    x_train, x_test, y_train, y_test = stratified_split(features, target, seed=seed)
    log_message("post_loan", f"Prepared train/test split | train={len(x_train)} test={len(x_test)}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    output_dir = make_topic_root("post_loan_monitoring") if output_root is None else Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = make_topic_visualization_root("post_loan_monitoring")

    rf, xgb, lgbm, ensemble = _build_models(seed)
    model_payloads = {
        "random_forest": rf,
        "xgboost": xgb,
        "lightgbm": lgbm,
        "weighted_soft_voting": ensemble,
    }
    all_model_names: Set[str] = set(model_payloads.keys()) | {"lstm"}
    selected: Set[str] = set(selected_models or all_model_names)
    unknown = sorted(selected - all_model_names)
    if unknown:
        raise ValueError(f"Unknown post-loan models requested: {unknown}")

    artifact_paths: List[TopicArtifact] = [TopicArtifact(name="scaler", path=str(output_dir / "scaler.pkl"), kind="preprocessor")]
    save_joblib_artifact(scaler, output_dir / "scaler.pkl")

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    visualizations: Dict[str, str] = {}
    for name, model in model_payloads.items():
        if name not in selected:
            continue
        start = time.perf_counter()
        log_message("post_loan", f"Training model: {name}")
        model.fit(x_train_scaled if name != "random_forest" else x_train, y_train)
        probability_input = x_test_scaled if name != "random_forest" else x_test
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(probability_input)[:, 1]
        else:
            scores = model.decision_function(probability_input)
            scores = 1.0 / (1.0 + np.exp(-scores))
        predictions = (scores >= 0.5).astype(int)
        metrics[name] = {
            "roc_auc": safe_roc_auc(y_test, scores),
            "f1": safe_f1(y_test, predictions),
            "accuracy": safe_accuracy(y_test, predictions),
        }
        artifact_path = output_dir / f"{name}.pkl"
        save_joblib_artifact(model, artifact_path)
        artifact_paths.append(TopicArtifact(name=name, path=str(artifact_path), kind="model"))
        visualizations[name] = _save_scores_plot(y_test, scores, viz_dir / f"{name}_roc_pr.png", f"{name} ROC / PR") or ""
        log_message("post_loan", f"Completed {name} in {elapsed_seconds(start):.1f}s | roc_auc={metrics[name]['roc_auc']} f1={metrics[name]['f1']}")

    if "lstm" in selected:
        _, lstm_scores, lstm_path, losses = _fit_lstm(
            x_train_scaled,
            y_train.to_numpy(),
            x_test_scaled,
            y_test.to_numpy(),
            seed,
            output_dir,
        )
        metrics["lstm"] = {
            "roc_auc": safe_roc_auc(y_test, lstm_scores),
            "f1": safe_f1(y_test, (lstm_scores >= 0.5).astype(int)),
            "accuracy": safe_accuracy(y_test, (lstm_scores >= 0.5).astype(int)),
        }
        artifact_paths.append(TopicArtifact(name="lstm", path=str(lstm_path), kind="model"))
        visualizations["lstm"] = _save_scores_plot(y_test, lstm_scores, viz_dir / "lstm_roc_pr.png", "LSTM ROC / PR") or ""
        visualizations["lstm_loss"] = _save_training_loss(losses, viz_dir / "lstm_training_loss.png") or ""

    report = common_report(dataset_path, feature_names, artifact_paths, metrics)
    if "weighted_soft_voting" in selected:
        report["classification_report"] = classification_report(
            y_test,
            (ensemble.predict_proba(x_test)[:, 1] >= 0.5).astype(int),
            output_dict=True,
        )
    report["visualizations"] = {key: value for key, value in visualizations.items() if value}
    report["visualization_dir"] = str(viz_dir)
    save_json(report, output_dir / "post_loan_monitoring_report.json")
    log_message("post_loan", f"Finished post-loan monitoring pipeline in {elapsed_seconds(pipeline_start):.1f}s")
    return report


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Train post-loan monitoring models on the 5M synthetic dataset.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default="models/post_loan_monitoring")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    run(dataset_path=args.dataset_path, output_root=args.output_root, seed=args.seed)


if __name__ == "__main__":
    main()