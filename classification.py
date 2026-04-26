from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_utils import (
    TopicArtifact,
    common_report,
    make_topic_root,
    make_topic_visualization_root,
    prepare_classification_frame,
    elapsed_seconds,
    log_message,
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


def _build_estimators(seed: int):
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=250,
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
    dnn = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    max_iter=60,
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=seed,
                ),
            ),
        ]
    )
    stacking = StackingClassifier(
        estimators=[("logreg", logreg), ("rf", rf), ("xgb", xgb)],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    return {
        "logistic_regression": logreg,
        "random_forest": rf,
        "xgboost": xgb,
        "stacked_ensemble": stacking,
        "dnn": dnn,
    }


def _evaluate(model, x_test, y_test) -> Dict[str, Optional[float]]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": safe_roc_auc(y_test, probabilities),
        "f1": safe_f1(y_test, predictions),
        "accuracy": safe_accuracy(y_test, predictions),
    }


def _save_curve(y_true, y_score, path: Path, title: str) -> Optional[str]:
    if not HAS_MATPLOTLIB:
        return None
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[0], name=title)
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axes[1], name=title)
    fig.tight_layout()
    result = save_figure(fig, path, dpi=300)
    plt.close(fig)
    return result


def _save_confusion_matrix(y_true, y_pred, path: Path, title: str) -> Optional[str]:
    if not HAS_MATPLOTLIB:
        return None
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    result = save_figure(fig, path, dpi=300)
    plt.close(fig)
    return result


def _save_feature_importance(model, feature_names, path: Path, title: str) -> Optional[str]:
    if not HAS_MATPLOTLIB:
        return None
    importance = None
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]
    else:
        estimator = model
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "estimators_") and len(getattr(estimator, "estimators_", [])):
        try:
            stacked_importances = []
            for fitted in estimator.estimators_:
                base = fitted.named_steps["model"] if hasattr(fitted, "named_steps") and "model" in fitted.named_steps else fitted
                if hasattr(base, "feature_importances_"):
                    stacked_importances.append(np.asarray(base.feature_importances_))
            if stacked_importances:
                importance = np.mean(stacked_importances, axis=0)
        except Exception:
            importance = None
    if importance is None:
        return None
    order = np.argsort(importance)[::-1][: min(12, len(feature_names))]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.asarray(feature_names)[order][::-1], importance[order][::-1], color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    result = save_figure(fig, path, dpi=300)
    plt.close(fig)
    return result


def run(
    dataset_path: Path | str = DEFAULT_DATASET,
    output_root: Path | str = Path("models/classification"),
    seed: int = SEED,
    selected_models: Optional[Sequence[str]] = None,
):
    pipeline_start = time.perf_counter()
    log_message("classification", f"Loading dataset from {dataset_path}")
    features, target, feature_names, _ = prepare_classification_frame(dataset_path=dataset_path)
    x_train, x_test, y_train, y_test = stratified_split(features, target, seed=seed)
    log_message("classification", f"Prepared train/test split | train={len(x_train)} test={len(x_test)}")

    output_dir = make_topic_root("classification") if output_root is None else Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = make_topic_visualization_root("classification")

    estimators = _build_estimators(seed)
    selected: Set[str] = set(selected_models or estimators.keys())
    unknown = sorted(selected - set(estimators.keys()))
    if unknown:
        raise ValueError(f"Unknown classification models requested: {unknown}")

    artifact_paths: List[TopicArtifact] = []
    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    curves: Dict[str, str] = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, model in estimators.items():
        if name not in selected:
            continue
        start = time.perf_counter()
        log_message("classification", f"Training model: {name}")
        model.fit(x_train, y_train)
        evaluation = _evaluate(model, x_test, y_test)
        metrics[name] = evaluation
        predictions[name] = model.predict(x_test)
        artifact_path = output_dir / f"{name}.pkl"
        save_joblib_artifact(model, artifact_path)
        artifact_paths.append(TopicArtifact(name=name, path=str(artifact_path), kind="model"))
        curves[name] = _save_curve(y_test, model.predict_proba(x_test)[:, 1], viz_dir / f"{name}_roc_pr.png", f"{name} ROC / PR") or ""
        log_message(
            "classification",
            f"Completed {name} in {elapsed_seconds(start):.1f}s | roc_auc={evaluation['roc_auc']} f1={evaluation['f1']}",
        )

    report = common_report(dataset_path, feature_names, artifact_paths, metrics)
    report["visualizations"] = {
        key: value for key, value in curves.items() if value
    }
    report["visualization_dir"] = str(viz_dir)
    if "stacked_ensemble" in selected:
        report["classification_report"] = classification_report(
            y_test,
            estimators["stacked_ensemble"].predict(x_test),
            output_dict=True,
        )
    save_json(report, output_dir / "classification_report.json")

    if "stacked_ensemble" in selected and "stacked_ensemble" in predictions:
        _save_confusion_matrix(
            y_test,
            predictions["stacked_ensemble"],
            viz_dir / "stacked_ensemble_confusion_matrix.png",
            "Stacked Ensemble Confusion Matrix",
        )
    if "random_forest" in selected:
        _save_feature_importance(
            estimators["random_forest"],
            feature_names,
            viz_dir / "random_forest_feature_importance.png",
            "Random Forest Feature Importance",
        )
    if "xgboost" in selected:
        _save_feature_importance(
            estimators["xgboost"],
            feature_names,
            viz_dir / "xgboost_feature_importance.png",
            "XGBoost Feature Importance",
        )
    log_message("classification", f"Finished classification pipeline in {elapsed_seconds(pipeline_start):.1f}s")
    return report


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Train classification models on the 5M synthetic dataset.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default="models/classification")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    run(dataset_path=args.dataset_path, output_root=args.output_root, seed=args.seed)


if __name__ == "__main__":
    main()