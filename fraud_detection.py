from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from pipeline_utils import (
    TopicArtifact,
    common_report,
    elapsed_seconds,
    log_message,
    make_topic_root,
    make_topic_visualization_root,
    prepare_unsupervised_frame,
    save_figure,
    save_json,
    save_joblib_artifact,
)

SEED = 42
DEFAULT_DATASET = Path("results/synthetic/dataset_5m/synthetic_dataset_5000000.csv")
AUTOENCODER_MAX_ROWS = 150_000
AUTOENCODER_BATCH_SIZE = 1024
AUTOENCODER_EPOCHS = 4
VIS_SAMPLE_ROWS = 40_000


class Autoencoder(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        hidden = max(16, feature_count // 2)
        bottleneck = max(8, feature_count // 4)
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_count),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _fit_autoencoder(x_train: np.ndarray, seed: int, output_dir: Path):
    torch.manual_seed(seed)
    # Keep autoencoder training CPU-only for stability on macOS/PyTorch MPS.
    # Pin PyTorch to a single thread to avoid conflicts with any OMP runtime
    # loaded by sklearn models that ran earlier in the same process.
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = Autoencoder(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    batch_size = min(AUTOENCODER_BATCH_SIZE, max(256, len(x_tensor) // 128))
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    epoch_losses: List[float] = []
    for _ in range(AUTOENCODER_EPOCHS):
        batch_losses: List[float] = []
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        epoch_losses.append(float(np.mean(batch_losses)))

    model.eval()
    reconstructed_batches: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            batch = x_tensor[start : start + batch_size].to(device)
            reconstructed_batches.append(model(batch).cpu().numpy())
    reconstruction = np.vstack(reconstructed_batches) if reconstructed_batches else np.empty_like(x_train)
    error = np.mean((x_train - reconstruction) ** 2, axis=1)
    feature_mae = np.mean(np.abs(x_train - reconstruction), axis=0)
    artifact_path = output_dir / "autoencoder_state.pt"
    torch.save({"state_dict": model.state_dict(), "feature_count": x_train.shape[1]}, artifact_path)
    return model, error, artifact_path, feature_mae, epoch_losses


def _save_ae_training_loss(epoch_losses: List[float], path: Path) -> Optional[str]:
    if not epoch_losses:
        return None
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", color="#e07a5f", linewidth=2.0)
    ax.set_title("Autoencoder Training Loss (per Epoch)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean MSE Loss")
    ax.set_xticks(range(1, len(epoch_losses) + 1))
    return save_figure(fig, path, dpi=300)


def _save_score_histogram(scores: np.ndarray, threshold: float, path: Path, title: str, xlabel: str, color: str) -> Optional[str]:
    if scores.size == 0:
        return None
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores, bins=80, color=color, alpha=0.85)
    ax.axvline(threshold, color="#111111", linestyle="--", linewidth=1.5, label="95th percentile")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    return save_figure(fig, path, dpi=300)


def run(
    dataset_path: Path | str = DEFAULT_DATASET,
    output_root: Path | str = Path("models/fraud_detection"),
    seed: int = SEED,
    selected_models: Optional[Sequence[str]] = None,
):
    pipeline_start = time.perf_counter()
    all_model_names: Set[str] = {"isolation_forest", "one_class_svm", "autoencoder"}
    selected: Set[str] = set(selected_models or all_model_names)
    unknown = sorted(selected - all_model_names)
    if unknown:
        raise ValueError(f"Unknown fraud models requested: {unknown}")

    log_message("fraud", f"Loading dataset from {dataset_path}")
    frame, feature_names = prepare_unsupervised_frame(dataset_path=dataset_path, max_rows=200_000)
    output_dir = make_topic_root("fraud_detection") if output_root is None else Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = make_topic_visualization_root("fraud_detection")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(frame)
    save_joblib_artifact(scaler, output_dir / "scaler.pkl")

    fit_rows = min(len(x_scaled), 200_000)
    fit_sample = x_scaled[:fit_rows]

    iforest = None
    ocsvm = None
    iforest_scores = np.array([], dtype=float)
    ocsvm_scores = np.array([], dtype=float)
    ae_error = np.array([], dtype=float)
    ae_feature_mae = np.array([], dtype=float)
    ae_path = None
    ae_epoch_losses: List[float] = []

    if "isolation_forest" in selected:
        iforest_artifact = output_dir / "isolation_forest.pkl"
        if iforest_artifact.exists():
            log_message("fraud", "Skipping model (artifact exists): isolation_forest")
        else:
            iforest = IsolationForest(
                n_estimators=300,
                contamination=0.05,
                random_state=seed,
                # n_jobs=1: loky workers leave an OpenMP runtime alive that segfaults
                # PyTorch's CPU thread pool when it initialises immediately after.
                n_jobs=1,
            )
            start = time.perf_counter()
            log_message("fraud", "Training model: isolation_forest")
            iforest.fit(fit_sample)
            iforest_scores = -iforest.score_samples(fit_sample)
            log_message("fraud", f"Completed isolation_forest in {elapsed_seconds(start):.1f}s")

    if "one_class_svm" in selected:
        ocsvm_artifact = output_dir / "one_class_svm.pkl"
        if ocsvm_artifact.exists():
            log_message("fraud", "Skipping model (artifact exists): one_class_svm")
        else:
            ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
            start = time.perf_counter()
            log_message("fraud", "Training model: one_class_svm")
            ocsvm.fit(fit_sample)
            ocsvm_scores = -ocsvm.decision_function(fit_sample)
            log_message("fraud", f"Completed one_class_svm in {elapsed_seconds(start):.1f}s")

    if "autoencoder" in selected:
        ae_artifact = output_dir / "autoencoder_state.pt"
        if ae_artifact.exists():
            log_message("fraud", "Skipping model (artifact exists): autoencoder")
            ae_path = ae_artifact
        else:
            start = time.perf_counter()
            log_message("fraud", "Training model: autoencoder")
            ae_train_sample = fit_sample
            if len(ae_train_sample) > AUTOENCODER_MAX_ROWS:
                ae_train_sample = ae_train_sample[:AUTOENCODER_MAX_ROWS]
                log_message("fraud", f"Autoencoder training sample capped at {AUTOENCODER_MAX_ROWS:,} rows")
            _, ae_error, ae_path, ae_feature_mae, ae_epoch_losses = _fit_autoencoder(ae_train_sample, seed, output_dir)
            log_message("fraud", f"Completed autoencoder in {elapsed_seconds(start):.1f}s")

    pca = PCA(n_components=2, random_state=seed)
    reduced = pca.fit_transform(x_scaled[:fit_rows])

    import matplotlib.pyplot as plt
    scatter_path = None
    iforest_hist_path = None
    ocsvm_hist_path = None
    ae_hist_path = None
    ae_percentile_path = None
    ae_ranked_path = None
    ae_top_feature_path = None
    ae_pca_overlay_path = None
    ae_loss_path = None
    score_correlation_path = None
    anomaly_rate_path = None

    iforest_threshold = None
    ae_threshold = None
    ocsvm_threshold = None

    if iforest is not None:
        iforest_threshold = float(np.percentile(iforest_scores, 95))
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(reduced[:, 0], reduced[:, 1], c=iforest.predict(fit_sample), cmap="coolwarm", s=8, alpha=0.55)
        ax.set_title("Fraud / Anomaly PCA Projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        scatter_path = save_figure(fig, viz_dir / "anomaly_pca_projection.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(iforest_scores, bins=80, color="#1f77b4", alpha=0.85)
        ax.axvline(iforest_threshold, color="#111111", linestyle="--", linewidth=1.5, label="95th percentile")
        ax.set_title("Isolation Forest Anomaly Score Distribution")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")
        iforest_hist_path = save_figure(fig, viz_dir / "iforest_score_distribution.png", dpi=300)
        plt.close(fig)

    if ocsvm is not None:
        ocsvm_threshold = float(np.percentile(ocsvm_scores, 95))
        ocsvm_hist_path = _save_score_histogram(
            ocsvm_scores, ocsvm_threshold,
            viz_dir / "ocsvm_score_distribution.png",
            "One-Class SVM Anomaly Score Distribution", "Anomaly Score", "#8d99ae",
        )
        if ocsvm_hist_path:
            plt.close("all")

    if ae_path is not None and ae_error.size > 0:
        ae_threshold = float(np.percentile(ae_error, 95))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(ae_error, bins=80, color="#d62728", alpha=0.85)
        ax.axvline(ae_threshold, color="#111111", linestyle="--", linewidth=1.5, label="95th percentile")
        ax.set_title("Autoencoder Reconstruction Error Distribution")
        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")
        ae_hist_path = save_figure(fig, viz_dir / "autoencoder_reconstruction_error.png", dpi=300)
        plt.close(fig)

        percentiles = np.arange(1, 100)
        percentile_values = np.percentile(ae_error, percentiles)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(percentiles, percentile_values, color="#bc4749", linewidth=2.0)
        ax.axhline(ae_threshold, color="#111111", linestyle="--", linewidth=1.5, label="95th percentile threshold")
        ax.set_title("Autoencoder Error Percentile Curve")
        ax.set_xlabel("Percentile")
        ax.set_ylabel("Reconstruction Error")
        ax.legend(loc="upper left")
        ae_percentile_path = save_figure(fig, viz_dir / "autoencoder_error_percentile_curve.png", dpi=300)
        plt.close(fig)

        ranked_errors = np.sort(ae_error)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ranked_errors, color="#e07a5f", linewidth=1.8)
        ax.axhline(ae_threshold, color="#111111", linestyle="--", linewidth=1.5, label="95th percentile threshold")
        ax.set_title("Ranked Autoencoder Reconstruction Error")
        ax.set_xlabel("Sorted Sample Index")
        ax.set_ylabel("Reconstruction Error")
        ax.legend(loc="upper left")
        ae_ranked_path = save_figure(fig, viz_dir / "autoencoder_ranked_reconstruction_error.png", dpi=300)
        plt.close(fig)

        if ae_feature_mae.size > 0:
            top_k = min(10, len(feature_names))
            order = np.argsort(ae_feature_mae)[-top_k:]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(np.array(feature_names)[order], ae_feature_mae[order], color="#2a9d8f")
            ax.set_title("Top Features by Autoencoder Reconstruction Error (MAE)")
            ax.set_xlabel("Mean Absolute Reconstruction Error")
            ae_top_feature_path = save_figure(fig, viz_dir / "autoencoder_top_feature_reconstruction_error.png", dpi=300)
            plt.close(fig)

        vis_rows = min(len(reduced), len(ae_error), VIS_SAMPLE_ROWS)
        if vis_rows > 0:
            vis_reduced = reduced[:vis_rows]
            vis_error = ae_error[:vis_rows]
            q80, q95 = np.percentile(vis_error, [80, 95])
            levels = np.where(vis_error >= q95, 2, np.where(vis_error >= q80, 1, 0))
            fig, ax = plt.subplots(figsize=(9, 6))
            scatter = ax.scatter(
                vis_reduced[:, 0],
                vis_reduced[:, 1],
                c=levels,
                cmap="viridis",
                s=8,
                alpha=0.65,
            )
            ax.set_title("PCA Projection Colored by Autoencoder Anomaly Level")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            legend = ax.legend(*scatter.legend_elements(), title="Level")
            ax.add_artist(legend)
            ae_pca_overlay_path = save_figure(fig, viz_dir / "autoencoder_pca_anomaly_overlay.png", dpi=300)
            plt.close(fig)

        ae_loss_path = _save_ae_training_loss(ae_epoch_losses, viz_dir / "autoencoder_training_loss.png")
        if ae_loss_path:
            plt.close("all")

    if iforest_scores.size > 0 and ae_error.size > 0:
        corr_rows = min(len(ae_error), len(iforest_scores), VIS_SAMPLE_ROWS)
        if corr_rows > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(iforest_scores[:corr_rows], ae_error[:corr_rows], s=8, alpha=0.35, color="#264653")
            ax.set_title("Isolation Forest vs Autoencoder Anomaly Score")
            ax.set_xlabel("Isolation Forest Score")
            ax.set_ylabel("Autoencoder Reconstruction Error")
            score_correlation_path = save_figure(fig, viz_dir / "iforest_vs_autoencoder_score_scatter.png", dpi=300)
            plt.close(fig)

    anomaly_rate_payload = []
    if iforest_scores.size > 0 and iforest_threshold is not None:
        anomaly_rate_payload.append(("isolation_forest", float(np.mean(iforest_scores >= iforest_threshold))))
    if ocsvm_scores.size > 0 and ocsvm_threshold is not None:
        anomaly_rate_payload.append(("one_class_svm", float(np.mean(ocsvm_scores >= ocsvm_threshold))))
    if ae_error.size > 0 and ae_threshold is not None:
        anomaly_rate_payload.append(("autoencoder", float(np.mean(ae_error >= ae_threshold))))
    if anomaly_rate_payload:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [item[0] for item in anomaly_rate_payload]
        values = [item[1] for item in anomaly_rate_payload]
        ax.bar(labels, values, color=["#457b9d", "#8d99ae", "#e76f51"][: len(labels)])
        ax.set_ylim(0, max(0.1, max(values) * 1.2))
        ax.set_title("Anomaly Rate at Model-Specific 95th Percentile Threshold")
        ax.set_ylabel("Anomaly Rate")
        anomaly_rate_path = save_figure(fig, viz_dir / "anomaly_rate_comparison.png", dpi=300)
        plt.close(fig)

    artifact_paths: List[TopicArtifact] = [
        TopicArtifact(name="scaler", path=str(output_dir / "scaler.pkl"), kind="preprocessor"),
    ]
    if "isolation_forest" in selected and (output_dir / "isolation_forest.pkl").exists() and iforest is None:
        artifact_paths.append(
            TopicArtifact(name="isolation_forest", path=str(output_dir / "isolation_forest.pkl"), kind="model")
        )
    if "one_class_svm" in selected and (output_dir / "one_class_svm.pkl").exists() and ocsvm is None:
        artifact_paths.append(
            TopicArtifact(name="one_class_svm", path=str(output_dir / "one_class_svm.pkl"), kind="model")
        )
    if iforest is not None:
        save_joblib_artifact(iforest, output_dir / "isolation_forest.pkl")
        artifact_paths.append(
            TopicArtifact(name="isolation_forest", path=str(output_dir / "isolation_forest.pkl"), kind="model")
        )
    if ocsvm is not None:
        save_joblib_artifact(ocsvm, output_dir / "one_class_svm.pkl")
        artifact_paths.append(
            TopicArtifact(name="one_class_svm", path=str(output_dir / "one_class_svm.pkl"), kind="model")
        )
    if ae_path is not None:
        artifact_paths.append(TopicArtifact(name="autoencoder", path=str(ae_path), kind="model"))

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    if "isolation_forest" in selected and iforest is None:
        metrics["isolation_forest"] = {"status": "skipped_existing_artifact"}
    if iforest is not None:
        metrics["isolation_forest"] = {
            "score_mean": float(np.mean(iforest_scores)),
            "score_std": float(np.std(iforest_scores)),
            "threshold_95": iforest_threshold,
        }
    if "one_class_svm" in selected and ocsvm is None:
        metrics["one_class_svm"] = {"status": "skipped_existing_artifact"}
    if ocsvm is not None:
        metrics["one_class_svm"] = {
            "score_mean": float(np.mean(ocsvm_scores)),
            "score_std": float(np.std(ocsvm_scores)),
            "threshold_95": ocsvm_threshold,
        }
    if "autoencoder" in selected and ae_path is not None and ae_error.size == 0:
        metrics["autoencoder"] = {"status": "skipped_existing_artifact"}
    elif ae_path is not None:
        metrics["autoencoder"] = {
            "reconstruction_error_mean": float(np.mean(ae_error)),
            "reconstruction_error_std": float(np.std(ae_error)),
            "threshold_95": ae_threshold,
        }

    report = common_report(dataset_path, feature_names, artifact_paths, metrics)
    report["visualization_dir"] = str(viz_dir)
    report["visualizations"] = {
        "anomaly_pca_projection": scatter_path,
        "iforest_score_distribution": iforest_hist_path,
        "ocsvm_score_distribution": ocsvm_hist_path,
        "autoencoder_reconstruction_error": ae_hist_path,
        "autoencoder_error_percentile_curve": ae_percentile_path,
        "autoencoder_ranked_reconstruction_error": ae_ranked_path,
        "autoencoder_top_feature_reconstruction_error": ae_top_feature_path,
        "autoencoder_pca_anomaly_overlay": ae_pca_overlay_path,
        "autoencoder_training_loss": ae_loss_path,
        "iforest_vs_autoencoder_score_scatter": score_correlation_path,
        "anomaly_rate_comparison": anomaly_rate_path,
    }
    save_json(report, output_dir / "fraud_detection_report.json")
    log_message("fraud", f"Finished fraud pipeline in {elapsed_seconds(pipeline_start):.1f}s")
    return report


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Train fraud and anomaly models on the 5M synthetic dataset.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default="models/fraud_detection")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    run(dataset_path=args.dataset_path, output_root=args.output_root, seed=args.seed)


if __name__ == "__main__":
    main()