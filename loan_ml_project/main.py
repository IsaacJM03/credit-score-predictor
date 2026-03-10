"""
main.py
--------
End-to-end loan ML pipeline entry point.

This script orchestrates both the **Fraud Detection** and **Borrower
Segmentation** systems in a single run, from raw CSV to saved models and
visualisation outputs.

Usage
-----
    # From inside loan_ml_project/
    python main.py --csv ../comprehensive_loan_data.csv --out-dir ../output

    # Or from the repository root
    python loan_ml_project/main.py

Project Folder Structure
-------------------------
loan_ml_project/
├── data/
│   ├── raw/           (place comprehensive_loan_data.csv here for portability)
│   └── processed/     (pipeline writes scaled numpy arrays here)
├── notebooks/         (Jupyter exploration notebooks)
├── models/
│   ├── fraud_detection/
│   │   ├── autoencoder.py
│   │   ├── isolation_forest.py
│   │   └── one_class_svm.py
│   └── segmentation/
│       ├── kmeans.py
│       ├── dbscan.py
│       └── hierarchical.py
├── training/
│   └── train_autoencoder.py
├── utils/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── visualization.py
├── main.py
└── requirements.txt

Deployment Recommendations
---------------------------
1.  Containerise the pipeline with Docker; pin all package versions in
    requirements.txt.
2.  Expose the fraud detector via a REST API (FastAPI / Flask) so that new
    loan applications are scored in real-time.
3.  Store model artefacts (weights, scalers, thresholds) in a model registry
    (MLflow / W&B) with version tracking.
4.  Schedule periodic retraining (cron or Airflow) to handle concept drift:
    monitor reconstruction error distribution monthly and retrain if KS-test
    p-value drops below 0.05.
5.  Update segmentation clusters quarterly as borrower behaviour evolves.
    Use a sliding-window approach: fit on the last 12 months of applications.
6.  Log model predictions to a data warehouse for feedback loops and
    continuous evaluation.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Ensure loan_ml_project is on the path when running as a module
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from utils.preprocessing import preprocess_pipeline, scale_features, handle_missing_values, encode_categoricals
from utils.feature_engineering import add_engineered_features
from utils.visualization import (
    plot_anomaly_score_distribution,
    plot_reconstruction_error,
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_pca_clusters,
    plot_cluster_heatmap,
    plot_cluster_profiles,
)
from models.fraud_detection.isolation_forest import IsolationForestDetector
from models.fraud_detection.one_class_svm import OneClassSVMDetector
from models.fraud_detection.autoencoder import AutoencoderDetector
from models.segmentation.kmeans import KMeansSegmenter
from models.segmentation.dbscan import DBSCANSegmenter
from models.segmentation.hierarchical import HierarchicalSegmenter

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Loan Analytics ML Pipeline: Fraud Detection + Borrower Segmentation"
    )
    p.add_argument(
        "--csv",
        default=os.path.join(_THIS_DIR, "../comprehensive_loan_data.csv"),
        help="Path to comprehensive_loan_data.csv",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(_THIS_DIR, "output"),
        help="Directory for saved models and figures",
    )
    p.add_argument(
        "--n-clusters", type=int, default=4,
        help="Number of borrower segments for K-Means and Hierarchical",
    )
    p.add_argument(
        "--contamination", type=float, default=0.05,
        help="Expected fraction of anomalies (Isolation Forest / One-Class SVM nu)",
    )
    p.add_argument(
        "--ae-epochs", type=int, default=30,
        help="Autoencoder training epochs",
    )
    p.add_argument(
        "--skip-ae", action="store_true", default=False,
        help="Skip autoencoder training (faster for quick runs)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def load_and_engineer(csv_path: str):
    """
    Load the CSV, preprocess it, and add engineered features.

    Returns
    -------
    X            : np.ndarray  float32 feature matrix
    y            : np.ndarray  int labels (Repayment_Status) or None
    feature_names: list[str]
    """
    df_raw = pd.read_csv(csv_path)
    df = handle_missing_values(df_raw)
    df = encode_categoricals(df)

    if "Repayment_Status" in df.columns:
        y = df["Repayment_Status"].values
        df = df.drop(columns=["Repayment_Status"])
    else:
        y = None

    df = add_engineered_features(df)
    feature_names = df.columns.tolist()
    X = df.values.astype("float32")
    return X, y, feature_names


# ---------------------------------------------------------------------------
# System 1: Fraud Detection
# ---------------------------------------------------------------------------

def run_fraud_detection(
    X_train_sc: np.ndarray,
    X_test_sc:  np.ndarray,
    y_test:     np.ndarray | None,
    out_dir:    str,
    contamination: float,
    ae_epochs:  int,
    skip_ae:    bool,
):
    """
    Train and evaluate all three fraud detection models.

    Model Comparison Guide
    ----------------------
    Isolation Forest  – fast, scales well, good default choice.
    One-Class SVM     – better when normal class is compact; slow on large data.
    Autoencoder       – learns complex non-linear manifolds; best on large data.

    Evaluation strategy:
    * With labels (Repayment_Status): use precision, recall, F1, ROC-AUC.
    * Without labels: compare anomaly score distributions and flag overlap.
    """
    fraud_dir = os.path.join(out_dir, "fraud_detection")
    os.makedirs(fraud_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  SYSTEM 1: FRAUD DETECTION")
    print("="*60)

    # ---- 1a. Isolation Forest ----
    print("\n--- Isolation Forest ---")
    iforest = IsolationForestDetector(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
    )
    iforest.fit(X_train_sc)
    if_scores = iforest.decision_scores(X_test_sc)
    if y_test is not None:
        iforest.evaluate(X_test_sc, y_test, fraud_label=0)

    fig = plot_anomaly_score_distribution(
        if_scores,
        threshold=np.percentile(if_scores, contamination * 100),
        title="Isolation Forest – Anomaly Score Distribution",
    )
    fig.savefig(os.path.join(fraud_dir, "isolation_forest_scores.png"), dpi=150)
    plt.close(fig)
    iforest.save(os.path.join(fraud_dir, "isolation_forest.joblib"))

    # ---- 1b. One-Class SVM ----
    print("\n--- One-Class SVM ---")
    # Sub-sample for speed (OCSVM is O(n²))
    max_ocsvm = 3000
    if len(X_train_sc) > max_ocsvm:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train_sc), max_ocsvm, replace=False)
        X_ocsvm_train = X_train_sc[idx]
    else:
        X_ocsvm_train = X_train_sc

    ocsvm = OneClassSVMDetector(nu=contamination)
    ocsvm.fit(X_ocsvm_train)
    ocsvm_scores = ocsvm.decision_scores(X_test_sc)
    if y_test is not None:
        ocsvm.evaluate(X_test_sc, y_test, fraud_label=0)

    fig = plot_anomaly_score_distribution(
        ocsvm_scores,
        threshold=0.0,   # OCSVM: negative = anomaly
        title="One-Class SVM – Decision Function Distribution",
    )
    fig.savefig(os.path.join(fraud_dir, "ocsvm_scores.png"), dpi=150)
    plt.close(fig)
    ocsvm.save(os.path.join(fraud_dir, "one_class_svm.joblib"))

    # ---- 1c. Autoencoder ----
    if not skip_ae:
        print("\n--- Autoencoder (PyTorch) ---")
        ae = AutoencoderDetector(
            input_dim=X_train_sc.shape[1],
            latent_dim=8,
            epochs=ae_epochs,
            batch_size=64,
        )
        val_n = int(len(X_train_sc) * 0.15)
        ae.fit(X_train_sc[val_n:], X_val=X_train_sc[:val_n])
        ae_errors = ae.reconstruction_errors(X_test_sc)
        if y_test is not None:
            ae.evaluate(X_test_sc, y_test, fraud_label=0)

        fig = plot_reconstruction_error(
            ae_errors,
            threshold=ae.threshold,
            labels=y_test,
            title="Autoencoder – Reconstruction Error per Sample",
        )
        fig.savefig(os.path.join(fraud_dir, "autoencoder_errors.png"), dpi=150)
        plt.close(fig)
        ae.save(
            model_path=os.path.join(fraud_dir, "autoencoder_weights.pt"),
            meta_path=os.path.join(fraud_dir, "autoencoder_meta.joblib"),
        )

    print(f"\n[main] Fraud detection artefacts saved to '{fraud_dir}'")


# ---------------------------------------------------------------------------
# System 2: Borrower Segmentation
# ---------------------------------------------------------------------------

def run_segmentation(
    X_scaled:     np.ndarray,
    feature_names: list,
    out_dir:       str,
    n_clusters:    int,
):
    """
    Train and evaluate all three segmentation models, then visualise results.

    Updating clusters as new borrowers appear
    -------------------------------------------
    * K-Means: periodically re-fit with the latest data; map old labels to new
      centroids using Hungarian algorithm if consistent labelling is needed.
    * DBSCAN: re-fit on a sliding window of the last N borrowers.
    * Hierarchical: transductive – for new borrowers, assign to the nearest
      centroid of an existing cluster (train a kNN classifier on the labels).
    """
    seg_dir = os.path.join(out_dir, "segmentation")
    os.makedirs(seg_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  SYSTEM 2: BORROWER SEGMENTATION")
    print("="*60)

    # ---- 2a. K-Means (with Elbow analysis) ----
    print("\n--- K-Means: Elbow Analysis ---")
    k_range = range(2, 9)
    inertias, sil_scores = KMeansSegmenter.elbow_analysis(
        X_scaled, k_range=k_range, random_state=RANDOM_STATE
    )
    fig = plot_elbow_curve(inertias, k_range)
    fig.savefig(os.path.join(seg_dir, "kmeans_elbow.png"), dpi=150)
    plt.close(fig)

    fig = plot_silhouette_scores(sil_scores, k_range)
    fig.savefig(os.path.join(seg_dir, "kmeans_silhouette.png"), dpi=150)
    plt.close(fig)

    print(f"\n--- K-Means: Final fit with K={n_clusters} ---")
    kmeans = KMeansSegmenter(n_clusters=n_clusters, random_state=RANDOM_STATE)
    km_labels = kmeans.fit_predict(X_scaled)
    kmeans.evaluate(X_scaled)
    kmeans.cluster_profiles(X_scaled, feature_names)
    kmeans.save(os.path.join(seg_dir, "kmeans.joblib"))

    fig = plot_pca_clusters(X_scaled, km_labels, title="K-Means Clusters (PCA)")
    fig.savefig(os.path.join(seg_dir, "kmeans_pca.png"), dpi=150)
    plt.close(fig)

    df_km = pd.DataFrame(X_scaled, columns=feature_names)
    df_km["Cluster"] = km_labels
    fig = plot_cluster_heatmap(df_km, feature_names, title="K-Means Cluster Heatmap")
    fig.savefig(os.path.join(seg_dir, "kmeans_heatmap.png"), dpi=150)
    plt.close(fig)

    plot_cluster_profiles(df_km, feature_names)

    # ---- 2b. DBSCAN ----
    print("\n--- DBSCAN ---")
    suggested_eps = DBSCANSegmenter.suggest_eps(X_scaled)
    dbscan = DBSCANSegmenter(eps=suggested_eps, min_samples=10)
    dbscan.fit(X_scaled)
    db_labels = dbscan.labels()
    dbscan.evaluate(X_scaled)
    dbscan.cluster_profiles(X_scaled, feature_names, include_noise=False)
    dbscan.save(os.path.join(seg_dir, "dbscan.joblib"))

    fig = plot_pca_clusters(X_scaled, db_labels, title="DBSCAN Clusters (PCA)")
    fig.savefig(os.path.join(seg_dir, "dbscan_pca.png"), dpi=150)
    plt.close(fig)

    # ---- 2c. Hierarchical ----
    print("\n--- Hierarchical Clustering ---")
    # Limit to 2000 samples for memory efficiency with scipy linkage matrix
    max_hier = 2000
    if len(X_scaled) > max_hier:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_scaled), max_hier, replace=False)
        X_hier = X_scaled[idx]
    else:
        X_hier = X_scaled

    hier = HierarchicalSegmenter(n_clusters=n_clusters, linkage="ward")
    hier.fit(X_hier)
    hier_labels = hier.predict()
    hier.evaluate(X_hier)
    hier.cluster_profiles(X_hier, feature_names)
    hier.save(os.path.join(seg_dir, "hierarchical.joblib"))

    fig = hier.plot_dendrogram(title="Hierarchical Clustering Dendrogram")
    fig.savefig(os.path.join(seg_dir, "dendrogram.png"), dpi=150)
    plt.close(fig)

    fig = plot_pca_clusters(X_hier, hier_labels, title="Hierarchical Clusters (PCA)")
    fig.savefig(os.path.join(seg_dir, "hierarchical_pca.png"), dpi=150)
    plt.close(fig)

    print(f"\n[main] Segmentation artefacts saved to '{seg_dir}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  LOAN ANALYTICS ML PIPELINE")
    print("="*60)
    print(f"  CSV        : {args.csv}")
    print(f"  Output dir : {args.out_dir}")
    print(f"  Clusters   : {args.n_clusters}")
    print(f"  Contamination: {args.contamination}")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    X, y, feature_names = load_and_engineer(args.csv)
    print(f"\n[main] Feature matrix: {X.shape}, features: {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y if y is not None else np.zeros(len(X)),
        test_size=0.2, random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler, X_train_sc, X_test_sc = scale_features(X_train, X_test)

    # Use the same fitted scaler to scale all data for unsupervised segmentation
    # so that fraud-detection and segmentation features are in the same space.
    X_all_sc = scaler.transform(X)

    # ------------------------------------------------------------------
    # System 1: Fraud Detection
    # ------------------------------------------------------------------
    run_fraud_detection(
        X_train_sc=X_train_sc,
        X_test_sc=X_test_sc,
        y_test=y_test,
        out_dir=args.out_dir,
        contamination=args.contamination,
        ae_epochs=args.ae_epochs,
        skip_ae=args.skip_ae,
    )

    # ------------------------------------------------------------------
    # System 2: Borrower Segmentation
    # ------------------------------------------------------------------
    run_segmentation(
        X_scaled=X_all_sc,
        feature_names=feature_names,
        out_dir=args.out_dir,
        n_clusters=args.n_clusters,
    )

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  All outputs saved to: {args.out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
