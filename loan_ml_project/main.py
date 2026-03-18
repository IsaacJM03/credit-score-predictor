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
    python main.py --csv ../comprehensive_loan_data_45M.csv --out-dir ../output

    # Or from the repository root
    python loan_ml_project/main.py

Project Folder Structure
-------------------------
loan_ml_project/
├── data/
│   ├── raw/           (place comprehensive_loan_data_45M.csv here for portability)
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
import time
import joblib
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
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
        default=os.path.join(_THIS_DIR, "../comprehensive_loan_data_45M.csv"),
        help="Path to comprehensive_loan_data_45M.csv",
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
        "--ae-epochs", type=int, default=10,
        help="Autoencoder training epochs",
    )
    p.add_argument(
        "--skip-ae", action="store_true", default=False,
        help="Skip autoencoder training (faster for quick runs)",
    )
    p.add_argument(
        "--ae-max-samples", type=int, default=500_000,
        help="Max rows used to train the autoencoder (subsample for speed). 0 = use all.",
    )
    p.add_argument(
        "--clf-max-samples", type=int, default=2_000_000,
        help="Max rows for supervised classifier training. 0 = use all.",
    )
    p.add_argument(
        "--seg-max-samples", type=int, default=200_000,
        help="Max rows passed to segmentation models. 0 = use all (slow on large data).",
    )
    p.add_argument(
        "--dbscan-max-samples", type=int, default=50_000,
        help="Max rows for DBSCAN (separate tighter cap; DBSCAN is O(n*neighbours)).",
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
    X_train_sc:  np.ndarray,
    X_test_sc:   np.ndarray,
    y_train:     np.ndarray | None,
    y_test:      np.ndarray | None,
    out_dir:     str,
    contamination: float,
    ae_epochs:   int,
    skip_ae:     bool,
    ae_max_samples: int = 500_000,
    clf_max_samples: int = 2_000_000,
):
    """
    Train and evaluate fraud detection models.

    Unsupervised models (IF, OCSVM, AE) are trained on NORMAL-ONLY samples so
    they learn the normal manifold; anomalies are then poorly reconstructed /
    isolated. When labels are available a supervised HistGradientBoosting
    classifier is also trained, which is the recommended path to ≥95% accuracy.
    """
    fraud_dir = os.path.join(out_dir, "fraud_detection")
    os.makedirs(fraud_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  SYSTEM 1: FRAUD DETECTION")
    print("="*60)

    # Separate normal training samples for unsupervised models.
    # fraud_label=0 means Repayment_Status==0 is fraud; 1 is normal.
    if y_train is not None:
        normal_mask = y_train == 1
        X_train_normal = X_train_sc[normal_mask]
        print(f"[main] Normal-only train set: {X_train_normal.shape[0]:,} / "
              f"{X_train_sc.shape[0]:,} samples")
    else:
        X_train_normal = X_train_sc

    # ---- 1a. Isolation Forest ----
    print("\n--- Isolation Forest ---")
    iforest = IsolationForestDetector(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
    )
    iforest.fit(X_train_normal)
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
    # Sub-sample from NORMAL set for speed (OCSVM is O(n²))
    max_ocsvm = 3000
    if len(X_train_normal) > max_ocsvm:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train_normal), max_ocsvm, replace=False)
        X_ocsvm_train = X_train_normal[idx]
    else:
        X_ocsvm_train = X_train_normal

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
            max_train_samples=ae_max_samples if ae_max_samples > 0 else None,
        )
        # Train on normal-only; 15% of that held out as validation
        val_n = int(len(X_train_normal) * 0.15)
        ae.fit(X_train_normal[val_n:], X_val=X_train_normal[:val_n])
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

    # ---- 1d. Supervised: HistGradientBoosting Classifier ----
    if y_train is not None:
        print("\n--- Supervised Classifier (HistGradientBoosting) ---")
        X_clf, y_clf = X_train_sc, y_train
        if clf_max_samples > 0 and len(X_clf) > clf_max_samples:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(X_clf), clf_max_samples, replace=False)
            X_clf, y_clf = X_clf[idx], y_clf[idx]
            print(f"[Classifier] Subsampled to {clf_max_samples:,} rows.")

        # class_weight handled via sample_weight; HistGBT uses early stopping
        # with 15% internal validation split.
        clf = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=8,
            max_leaf_nodes=63,
            min_samples_leaf=50,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        clf.fit(X_clf, y_clf)
        y_pred  = clf.predict(X_test_sc)
        # predict_proba[:, 0] = P(fraud), [:, 1] = P(normal)
        y_proba_fraud = clf.predict_proba(X_test_sc)[:, 0]

        acc  = accuracy_score(y_test, y_pred)
        # For precision/recall/f1 treat fraud (0) as the positive class
        prec = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        rec  = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1   = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        auc  = roc_auc_score((y_test == 0).astype(int), y_proba_fraud)

        print("\n[Classifier] Evaluation results:")
        print(f"  {'accuracy':12s}: {acc:.4f}")
        print(f"  {'precision':12s}: {prec:.4f}")
        print(f"  {'recall':12s}: {rec:.4f}")
        print(f"  {'f1':12s}: {f1:.4f}")
        print(f"  {'roc_auc':12s}: {auc:.4f}")
        print()
        print(classification_report(
            y_test, y_pred,
            target_names=["Fraud (0)", "Normal (1)"], zero_division=0,
        ))
        joblib.dump(clf, os.path.join(fraud_dir, "hgbt_classifier.joblib"))
        print(f"[Classifier] Model saved.")

    print(f"\n[main] Fraud detection artefacts saved to '{fraud_dir}'")


# ---------------------------------------------------------------------------
# System 2: Borrower Segmentation
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_segmentation(
    X_scaled:        np.ndarray,
    feature_names:   list,
    out_dir:         str,
    n_clusters:      int,
    seg_max_samples: int = 200_000,
    dbscan_max_samples: int = 50_000,
):
    """
    Train and evaluate all three segmentation models, then visualise results.

    All models operate on a stratified subsample (seg_max_samples) of the full
    scaled dataset. This keeps each step to a few minutes on an M1 8 GB machine
    while still producing representative clusters.

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

    # ------------------------------------------------------------------ #
    # Subsample once – all segmentation models use this same smaller array
    # ------------------------------------------------------------------ #
    n_total = len(X_scaled)
    if seg_max_samples > 0 and n_total > seg_max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n_total, seg_max_samples, replace=False)
        X_seg = X_scaled[idx]
        _log(f"Segmentation subsample: {seg_max_samples:,} / {n_total:,} rows.")
    else:
        X_seg = X_scaled
        _log(f"Segmentation using all {n_total:,} rows.")

    # ---- 2a. K-Means (with Elbow analysis) ----
    _log(f"K-Means Elbow Analysis on {len(X_seg):,} rows (k=2..8) – "
         f"~30 s on M1 …")
    k_range = range(2, 9)
    inertias, sil_scores = KMeansSegmenter.elbow_analysis(
        X_seg, k_range=k_range, random_state=RANDOM_STATE
    )
    fig = plot_elbow_curve(inertias, k_range)
    fig.savefig(os.path.join(seg_dir, "kmeans_elbow.png"), dpi=150)
    plt.close(fig)

    fig = plot_silhouette_scores(sil_scores, k_range)
    fig.savefig(os.path.join(seg_dir, "kmeans_silhouette.png"), dpi=150)
    plt.close(fig)
    _log("Elbow / silhouette plots saved.")

    _log(f"K-Means final fit K={n_clusters} on {len(X_seg):,} rows …")
    kmeans = KMeansSegmenter(n_clusters=n_clusters, random_state=RANDOM_STATE)
    km_labels = kmeans.fit_predict(X_seg)
    kmeans.evaluate(X_seg)
    kmeans.cluster_profiles(X_seg, feature_names)
    kmeans.save(os.path.join(seg_dir, "kmeans.joblib"))

    fig = plot_pca_clusters(X_seg, km_labels, title="K-Means Clusters (PCA)")
    fig.savefig(os.path.join(seg_dir, "kmeans_pca.png"), dpi=150)
    plt.close(fig)

    df_km = pd.DataFrame(X_seg, columns=feature_names)
    df_km["Cluster"] = km_labels
    fig = plot_cluster_heatmap(df_km, feature_names, title="K-Means Cluster Heatmap")
    fig.savefig(os.path.join(seg_dir, "kmeans_heatmap.png"), dpi=150)
    plt.close(fig)

    plot_cluster_profiles(df_km, feature_names)
    _log("K-Means done.")

    # ---- 2b. DBSCAN ----
    # Use a tighter subsample to avoid O(n*neighbours) memory explosion
    if len(X_seg) > dbscan_max_samples:
        rng2 = np.random.default_rng(RANDOM_STATE + 1)
        db_idx = rng2.choice(len(X_seg), dbscan_max_samples, replace=False)
        X_db = X_seg[db_idx]
    else:
        X_db = X_seg
    _log(f"DBSCAN eps suggestion on {len(X_db):,} rows …")
    suggested_eps = DBSCANSegmenter.suggest_eps(X_db)
    _log(f"DBSCAN fitting with eps={suggested_eps:.4f} on {len(X_db):,} rows …")
    dbscan = DBSCANSegmenter(eps=suggested_eps, min_samples=10)
    dbscan.fit(X_db)
    db_labels = dbscan.labels()
    dbscan.evaluate(X_db)
    dbscan.cluster_profiles(X_db, feature_names, include_noise=False)
    dbscan.save(os.path.join(seg_dir, "dbscan.joblib"))

    fig = plot_pca_clusters(X_db, db_labels, title="DBSCAN Clusters (PCA)")
    fig.savefig(os.path.join(seg_dir, "dbscan_pca.png"), dpi=150)
    plt.close(fig)
    _log("DBSCAN done.")

    # ---- 2c. Hierarchical ----
    # scipy linkage is O(n²) memory – hard cap at 2000 rows
    max_hier = 2000
    if len(X_seg) > max_hier:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_seg), max_hier, replace=False)
        X_hier = X_seg[idx]
    else:
        X_hier = X_seg
    _log(f"Hierarchical clustering on {len(X_hier):,} rows …")

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
    _log("Hierarchical done.")

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
        y_train=y_train,
        y_test=y_test,
        out_dir=args.out_dir,
        contamination=args.contamination,
        ae_epochs=args.ae_epochs,
        skip_ae=args.skip_ae,
        ae_max_samples=args.ae_max_samples,
        clf_max_samples=args.clf_max_samples,
    )

    # ------------------------------------------------------------------
    # System 2: Borrower Segmentation
    # ------------------------------------------------------------------
    run_segmentation(
        X_scaled=X_all_sc,
        feature_names=feature_names,
        out_dir=args.out_dir,
        n_clusters=args.n_clusters,
        seg_max_samples=args.seg_max_samples,
        dbscan_max_samples=args.dbscan_max_samples,
    )

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  All outputs saved to: {args.out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
