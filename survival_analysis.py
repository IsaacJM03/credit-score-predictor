from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from pipeline_utils import (
    DURATION_COLUMN,
    TARGET_COLUMN,
    TopicArtifact,
    common_report,
    elapsed_seconds,
    log_message,
    make_topic_root,
    make_topic_visualization_root,
    prepare_survival_frame,
    save_figure,
    save_json,
    save_joblib_artifact,
)

SEED = 42
DEFAULT_DATASET = Path("results/synthetic/dataset_5m/synthetic_dataset_5000000.csv")

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter

    HAS_LIFELINES = True
except Exception:
    CoxPHFitter = None
    KaplanMeierFitter = None
    HAS_LIFELINES = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


def _save_km_plot(frame: pd.DataFrame, output_path: Path) -> Optional[str]:
    if not (HAS_LIFELINES and HAS_MATPLOTLIB):
        return None
    fitter = KaplanMeierFitter()
    fitter.fit(frame[DURATION_COLUMN], event_observed=frame[TARGET_COLUMN])
    ax = fitter.plot()
    ax.set_title("Kaplan-Meier Survival Curve")
    save_figure(ax.figure, output_path, dpi=300)
    plt.close(ax.figure)
    return str(output_path)


def _save_cox_coefficients(cox, output_path: Path) -> Optional[str]:
    if not HAS_MATPLOTLIB:
        return None
    summary = cox.summary.sort_values("coef", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(summary.index.astype(str), summary["coef"], color="#9467bd")
    ax.set_title("Cox PH Coefficients")
    ax.set_xlabel("Coefficient")
    return save_figure(fig, output_path, dpi=300)


def run(
    dataset_path: Path | str = DEFAULT_DATASET,
    output_root: Path | str = Path("models/survival_analysis"),
    seed: int = SEED,
    selected_models: Optional[Sequence[str]] = None,
):
    pipeline_start = time.perf_counter()
    all_model_names: Set[str] = {"kaplan_meier", "cox_ph"}
    selected: Set[str] = set(selected_models or all_model_names)
    unknown = sorted(selected - all_model_names)
    if unknown:
        raise ValueError(f"Unknown survival models requested: {unknown}")

    log_message("survival", f"Loading dataset from {dataset_path}")
    frame, feature_names = prepare_survival_frame(dataset_path=dataset_path)
    output_dir = make_topic_root("survival_analysis") if output_root is None else Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = make_topic_visualization_root("survival_analysis")

    artifact_paths: List[TopicArtifact] = []
    metrics = {}
    curves = {}
    skipped_kaplan_meier = False

    if "kaplan_meier" in selected:
        km_path = output_dir / "kaplan_meier_curve.png"
        if km_path.exists():
            log_message("survival", "Skipping model (artifact exists): kaplan_meier")
            curves["kaplan_meier"] = str(km_path)
            skipped_kaplan_meier = True
        else:
            log_message("survival", "Generating Kaplan-Meier curve")
            curves["kaplan_meier"] = _save_km_plot(frame, km_path)
        artifact_paths.append(TopicArtifact(name="kaplan_meier_curve", path=str(km_path), kind="plot"))
        if skipped_kaplan_meier:
            metrics["kaplan_meier"] = {"status": "skipped_existing_artifact"}

    if "cox_ph" in selected and HAS_LIFELINES:
        cox_artifact = output_dir / "cox_ph.pkl"
        if cox_artifact.exists():
            log_message("survival", "Skipping model (artifact exists): cox_ph")
            artifact_paths.append(TopicArtifact(name="cox_ph", path=str(cox_artifact), kind="model"))
            metrics["cox_ph"] = {"status": "skipped_existing_artifact"}
            cox_plot = str(viz_dir / "cox_ph_coefficients.png") if (viz_dir / "cox_ph_coefficients.png").exists() else None
        else:
            start = time.perf_counter()
            log_message("survival", "Training model: cox_ph")
            cox = CoxPHFitter()
            cox.fit(frame[feature_names + [DURATION_COLUMN, TARGET_COLUMN]], duration_col=DURATION_COLUMN, event_col=TARGET_COLUMN)
            save_joblib_artifact(cox, cox_artifact)
            artifact_paths.append(TopicArtifact(name="cox_ph", path=str(cox_artifact), kind="model"))
            metrics["cox_ph"] = {
                "partial_log_likelihood": float(getattr(cox, "log_likelihood_", np.nan)),
                "concordance_index": float(getattr(cox, "concordance_index_", np.nan)),
            }
            cox_plot = _save_cox_coefficients(cox, viz_dir / "cox_ph_coefficients.png")
            log_message("survival", f"Completed cox_ph in {elapsed_seconds(start):.1f}s | concordance={metrics['cox_ph']['concordance_index']}")
    elif "cox_ph" in selected:
        metrics["cox_ph"] = {"status": "lifelines unavailable"}
        cox_plot = None
        log_message("survival", "Skipped cox_ph because lifelines is unavailable")
    else:
        cox_plot = None

    report = common_report(dataset_path, feature_names, artifact_paths, metrics)
    report["duration_column"] = DURATION_COLUMN
    report["event_column"] = TARGET_COLUMN
    report["curves"] = {key: value for key, value in curves.items() if value}
    report["visualization_dir"] = str(viz_dir)
    report["visualizations"] = {
        "kaplan_meier": curves.get("kaplan_meier"),
        "cox_ph_coefficients": cox_plot,
    }
    save_json(report, output_dir / "survival_analysis_report.json")
    log_message("survival", f"Finished survival pipeline in {elapsed_seconds(pipeline_start):.1f}s")
    return report


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Train survival analysis models on the 5M synthetic dataset.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default="models/survival_analysis")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    run(dataset_path=args.dataset_path, output_root=args.output_root, seed=args.seed)


if __name__ == "__main__":
    main()