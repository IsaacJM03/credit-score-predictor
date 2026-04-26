from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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


def run(
    dataset_path: Path | str = DEFAULT_DATASET,
    output_root: Path | str = Path("models/borrower_segmentation"),
    seed: int = SEED,
    selected_models: Optional[Sequence[str]] = None,
):
    pipeline_start = time.perf_counter()
    all_model_names: Set[str] = {"kmeans", "dbscan", "agglomerative"}
    selected: Set[str] = set(selected_models or all_model_names)
    unknown = sorted(selected - all_model_names)
    if unknown:
        raise ValueError(f"Unknown segmentation models requested: {unknown}")

    log_message("segmentation", f"Loading dataset from {dataset_path}")
    frame, feature_names = prepare_unsupervised_frame(dataset_path=dataset_path)
    output_dir = make_topic_root("borrower_segmentation") if output_root is None else Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = make_topic_visualization_root("borrower_segmentation")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(frame)
    save_joblib_artifact(scaler, output_dir / "scaler.pkl")

    sample_rows = min(len(x_scaled), 200_000)
    sample = x_scaled[:sample_rows]

    kmeans = None
    dbscan = None
    agglomerative = None
    k_labels = None
    dbscan_labels = None
    agglomerative_labels = None
    agglomerative_sample = sample[: min(len(sample), 50_000)]

    if "kmeans" in selected:
        kmeans = MiniBatchKMeans(n_clusters=5, random_state=seed, batch_size=4096)
        start = time.perf_counter()
        log_message("segmentation", "Training model: kmeans")
        kmeans.fit(sample)
        k_labels = kmeans.predict(sample)
        log_message("segmentation", f"Completed kmeans in {elapsed_seconds(start):.1f}s")
        save_joblib_artifact(kmeans, output_dir / "kmeans.pkl")

    if "dbscan" in selected:
        dbscan = DBSCAN(eps=1.25, min_samples=25, n_jobs=-1)
        start = time.perf_counter()
        log_message("segmentation", "Training model: dbscan")
        dbscan_labels = dbscan.fit_predict(sample)
        log_message("segmentation", f"Completed dbscan in {elapsed_seconds(start):.1f}s")
        save_joblib_artifact(dbscan, output_dir / "dbscan.pkl")

    if "agglomerative" in selected:
        agglomerative = AgglomerativeClustering(n_clusters=5)
        start = time.perf_counter()
        log_message("segmentation", "Training model: agglomerative")
        agglomerative_labels = agglomerative.fit_predict(agglomerative_sample)
        log_message("segmentation", f"Completed agglomerative in {elapsed_seconds(start):.1f}s")
        save_joblib_artifact(agglomerative, output_dir / "agglomerative.pkl")

    artifact_paths: List[TopicArtifact] = [
        TopicArtifact(name="scaler", path=str(output_dir / "scaler.pkl"), kind="preprocessor"),
    ]
    if "kmeans" in selected:
        artifact_paths.append(TopicArtifact(name="kmeans", path=str(output_dir / "kmeans.pkl"), kind="model"))
    if "dbscan" in selected:
        artifact_paths.append(TopicArtifact(name="dbscan", path=str(output_dir / "dbscan.pkl"), kind="model"))
    if "agglomerative" in selected:
        artifact_paths.append(TopicArtifact(name="agglomerative", path=str(output_dir / "agglomerative.pkl"), kind="model"))

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    if k_labels is not None:
        metrics["kmeans"] = {"silhouette_score": float(silhouette_score(sample, k_labels))}
    if dbscan_labels is not None:
        metrics["dbscan"] = {
            "silhouette_score": float(silhouette_score(sample, dbscan_labels)) if len(set(dbscan_labels)) > 1 else None
        }
    if agglomerative_labels is not None:
        metrics["agglomerative"] = {
            "silhouette_score": float(silhouette_score(agglomerative_sample, agglomerative_labels))
            if len(set(agglomerative_labels)) > 1
            else None,
        }

    pca = PCA(n_components=2, random_state=seed)
    reduced = pca.fit_transform(sample)
    import matplotlib.pyplot as plt

    cluster_scatter = None
    cluster_sizes = None
    if k_labels is not None:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(reduced[:, 0], reduced[:, 1], c=k_labels, cmap="tab10", s=8, alpha=0.6)
        ax.set_title("Borrower Segmentation PCA Clusters")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        cluster_scatter = save_figure(fig, viz_dir / "kmeans_pca_clusters.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        counts = np.bincount(k_labels)
        ax.bar([str(i) for i in range(len(counts))], counts, color="#2ca02c")
        ax.set_title("K-Means Cluster Sizes")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")
        cluster_sizes = save_figure(fig, viz_dir / "kmeans_cluster_sizes.png", dpi=300)
        plt.close(fig)

    report = common_report(dataset_path, feature_names, artifact_paths, metrics)
    report["sample_rows"] = sample_rows
    report["visualization_dir"] = str(viz_dir)
    report["visualizations"] = {
        "kmeans_pca_clusters": cluster_scatter,
        "kmeans_cluster_sizes": cluster_sizes,
    }
    save_json(report, output_dir / "borrower_segmentation_report.json")
    log_message("segmentation", f"Finished segmentation pipeline in {elapsed_seconds(pipeline_start):.1f}s")
    return report


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Train segmentation models on the 5M synthetic dataset.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default="models/borrower_segmentation")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    run(dataset_path=args.dataset_path, output_root=args.output_root, seed=args.seed)


if __name__ == "__main__":
    main()