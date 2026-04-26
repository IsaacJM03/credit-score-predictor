from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn
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
    device = torch.device("cpu")
    model = Autoencoder(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    batch_size = min(2048, max(256, len(x_tensor) // 128))
    model.train()
    for _ in range(4):
        permutation = torch.randperm(len(x_tensor), device=device)
        for start in range(0, len(x_tensor), batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch = x_tensor[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        reconstruction = model(x_tensor).cpu().numpy()
    error = np.mean((x_train - reconstruction) ** 2, axis=1)
    artifact_path = output_dir / "autoencoder_state.pt"
    torch.save({"state_dict": model.state_dict(), "feature_count": x_train.shape[1]}, artifact_path)
    return model, error, artifact_path


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
    frame, feature_names = prepare_unsupervised_frame(dataset_path=dataset_path)
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
    ae_path = None

    if "isolation_forest" in selected:
        iforest = IsolationForest(
            n_estimators=300,
            contamination=0.05,
            random_state=seed,
            n_jobs=-1,
        )
        start = time.perf_counter()
        log_message("fraud", "Training model: isolation_forest")
        iforest.fit(fit_sample)
        iforest_scores = -iforest.score_samples(x_scaled)
        log_message("fraud", f"Completed isolation_forest in {elapsed_seconds(start):.1f}s")

    if "one_class_svm" in selected:
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        start = time.perf_counter()
        log_message("fraud", "Training model: one_class_svm")
        ocsvm.fit(fit_sample)
        ocsvm_scores = -ocsvm.decision_function(x_scaled)
        log_message("fraud", f"Completed one_class_svm in {elapsed_seconds(start):.1f}s")

    if "autoencoder" in selected:
        start = time.perf_counter()
        log_message("fraud", "Training model: autoencoder")
        _, ae_error, ae_path = _fit_autoencoder(fit_sample, seed, output_dir)
        log_message("fraud", f"Completed autoencoder in {elapsed_seconds(start):.1f}s")

    pca = PCA(n_components=2, random_state=seed)
    reduced = pca.fit_transform(x_scaled[:fit_rows])

    import matplotlib.pyplot as plt
    scatter_path = None
    iforest_hist_path = None
    ae_hist_path = None
    if iforest is not None:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(reduced[:, 0], reduced[:, 1], c=iforest.predict(fit_sample), cmap="coolwarm", s=8, alpha=0.55)
        ax.set_title("Fraud / Anomaly PCA Projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        scatter_path = save_figure(fig, viz_dir / "anomaly_pca_projection.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(iforest_scores, bins=80, color="#1f77b4", alpha=0.85)
        ax.set_title("Isolation Forest Anomaly Score Distribution")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        iforest_hist_path = save_figure(fig, viz_dir / "iforest_score_distribution.png", dpi=300)
        plt.close(fig)

    if ae_path is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(ae_error, bins=80, color="#d62728", alpha=0.85)
        ax.set_title("Autoencoder Reconstruction Error Distribution")
        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Count")
        ae_hist_path = save_figure(fig, viz_dir / "autoencoder_reconstruction_error.png", dpi=300)
        plt.close(fig)

    artifact_paths: List[TopicArtifact] = [
        TopicArtifact(name="scaler", path=str(output_dir / "scaler.pkl"), kind="preprocessor"),
    ]
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
    if iforest is not None:
        metrics["isolation_forest"] = {
            "score_mean": float(np.mean(iforest_scores)),
            "score_std": float(np.std(iforest_scores)),
        }
    if ocsvm is not None:
        metrics["one_class_svm"] = {
            "score_mean": float(np.mean(ocsvm_scores)),
            "score_std": float(np.std(ocsvm_scores)),
        }
    if ae_path is not None:
        metrics["autoencoder"] = {
            "reconstruction_error_mean": float(np.mean(ae_error)),
            "reconstruction_error_std": float(np.std(ae_error)),
        }

    report = common_report(dataset_path, feature_names, artifact_paths, metrics)
    report["visualization_dir"] = str(viz_dir)
    report["visualizations"] = {
        "anomaly_pca_projection": scatter_path,
        "iforest_score_distribution": iforest_hist_path,
        "autoencoder_reconstruction_error": ae_hist_path,
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