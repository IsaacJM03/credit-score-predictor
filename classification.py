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

STACKING_SUBSET_SIZE = 500_000  # 👈 KEY SPEED BOOST

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False


def _build_estimators(seed: int):
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])

    rf = RandomForestClassifier(
        n_estimators=100,              # 👈 reduced from 300
        max_depth=15,                  # 👈 limit depth
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )

    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=100,          # 👈 reduced from 250
            max_depth=4,               # 👈 reduced depth
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",        # keep stable (gpu_hist unreliable on M1)
            random_state=seed,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        xgb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=seed)

    dnn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64),   # 👈 reduced size
            max_iter=40,                    # 👈 faster training
            early_stopping=True,
            n_iter_no_change=5,
            random_state=seed,
        )),
    ])

    # 👇 LIGHTER STACK (important)
    stacking = StackingClassifier(
        estimators=[
            ("logreg", logreg),
            ("rf", rf),
            ("xgb", xgb),
        ],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        stack_method="predict_proba",
        n_jobs=-1,
        cv=3,                 # 👈 BIG speed improvement
        passthrough=True,     # 👈 helps convergence
    )

    return {
        "logistic_regression": logreg,
        "random_forest": rf,
        "xgboost": xgb,
        "stacked_ensemble": stacking,
        "dnn": dnn,
    }


def _evaluate(model, x_test, y_test):
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": safe_roc_auc(y_test, probabilities),
        "f1": safe_f1(y_test, predictions),
        "accuracy": safe_accuracy(y_test, predictions),
    }


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

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    estimators = _build_estimators(seed)
    selected: Set[str] = set(selected_models or estimators.keys())

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    predictions: Dict[str, np.ndarray] = {}
    trained_models: Set[str] = set()
    artifact_paths: List[TopicArtifact] = []

    for name, model in estimators.items():
        if name not in selected:
            continue

        artifact_path = output_dir / f"{name}.pkl"
        if artifact_path.exists():
            log_message("classification", f"Skipping model (artifact exists): {name}")
            artifact_paths.append(TopicArtifact(name=name, path=str(artifact_path), kind="model"))
            metrics[name] = {"status": "skipped_existing_artifact"}
            continue

        start = time.perf_counter()
        log_message("classification", f"Training model: {name}")

        # 👇 SPECIAL HANDLING FOR STACKING
        if name == "stacked_ensemble":
            subset_size = min(STACKING_SUBSET_SIZE, len(x_train))
            indices = np.random.choice(len(x_train), subset_size, replace=False)

            x_sub = x_train.iloc[indices]
            y_sub = y_train.iloc[indices]

            log_message("classification", f"Stacking using subset: {subset_size}")
            model.fit(x_sub, y_sub)
        else:
            model.fit(x_train, y_train)

        evaluation = _evaluate(model, x_test, y_test)
        metrics[name] = evaluation
        predictions[name] = model.predict(x_test)
        trained_models.add(name)

        save_joblib_artifact(model, artifact_path)
        artifact_paths.append(TopicArtifact(name=name, path=str(artifact_path), kind="model"))

        log_message(
            "classification",
            f"{name} done in {elapsed_seconds(start):.1f}s | roc_auc={evaluation['roc_auc']}",
        )

    report = common_report(dataset_path, feature_names, artifact_paths, metrics)

    if "stacked_ensemble" in trained_models:
        report["classification_report"] = classification_report(
            y_test,
            estimators["stacked_ensemble"].predict(x_test),
            output_dict=True,
        )

    save_json(report, output_dir / "classification_report.json")

    log_message("classification", f"Finished in {elapsed_seconds(pipeline_start):.1f}s")

    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default="models/classification")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    run(dataset_path=args.dataset_path, output_root=args.output_root, seed=args.seed)


if __name__ == "__main__":
    main()