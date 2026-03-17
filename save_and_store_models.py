"""
save_and_store_models.py
------------------------
Train and persist all loan-analytics models to the ``models/`` directory.

Two systems are built from ``comprehensive_loan_data.csv``:

System 1 – Fraud Detection
    Isolation Forest      → models/fraud_detection/isolation_forest.joblib
    One-Class SVM         → models/fraud_detection/one_class_svm.joblib
    Autoencoder (PyTorch) → models/fraud_detection/autoencoder_weights.pt
                            models/fraud_detection/autoencoder_meta.joblib

System 2 – Borrower Segmentation
    K-Means               → models/segmentation/kmeans.joblib
    DBSCAN                → models/segmentation/dbscan.joblib
    Hierarchical          → models/segmentation/hierarchical.joblib

Shared preprocessing artefacts
    models/scaler.joblib
    models/features_list.joblib

Usage
-----
    # From the repository root:
    python save_and_store_models.py

    # Override defaults:
    python save_and_store_models.py --csv comprehensive_loan_data.csv \\
                                    --models-dir models \\
                                    --n-clusters 4 \\
                                    --skip-ae
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Make sure the loan_ml_project package is importable regardless of where
# this script is invoked from.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOAN_ML_PROJECT_DIR = os.path.join(_REPO_ROOT, "loan_ml_project")
sys.path.insert(0, _LOAN_ML_PROJECT_DIR)

from utils.preprocessing import (
    handle_missing_values,
    encode_categoricals,
    scale_features,
)
from utils.feature_engineering import add_engineered_features
from models.fraud_detection.isolation_forest import IsolationForestDetector
from models.fraud_detection.one_class_svm import OneClassSVMDetector
from models.fraud_detection.autoencoder import AutoencoderDetector
from models.segmentation.kmeans import KMeansSegmenter
from models.segmentation.dbscan import DBSCANSegmenter
from models.segmentation.hierarchical import HierarchicalSegmenter

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        default=os.path.join(_REPO_ROOT, "comprehensive_loan_data.csv"),
        help="Path to the loan dataset CSV (default: comprehensive_loan_data.csv)",
    )
    p.add_argument(
        "--models-dir",
        default=os.path.join(_REPO_ROOT, "models"),
        help="Root directory for saved model artefacts (default: models/)",
    )
    p.add_argument(
        "--n-clusters", type=int, default=4,
        help="Number of borrower segments for K-Means and Hierarchical (default: 4)",
    )
    p.add_argument(
        "--contamination", type=float, default=0.05,
        help="Expected fraction of anomalies for Isolation Forest / One-Class SVM (default: 0.05)",
    )
    p.add_argument(
        "--ae-epochs", type=int, default=30,
        help="Autoencoder training epochs (default: 30)",
    )
    p.add_argument(
        "--skip-ae", action="store_true", default=False,
        help="Skip autoencoder training (useful for a quick smoke-test)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_and_prepare(csv_path: str):
    """
    Load the CSV, clean it, encode categoricals, add engineered features,
    and return a float32 feature matrix plus optional target labels.

    Returns
    -------
    X            : np.ndarray  (n_samples, n_features)  float32
    y            : np.ndarray  (n_samples,)  int  - Repayment_Status, or None
    feature_names: list[str]
    """
    print(f"\nLoading dataset from '{csv_path}' …")
    df_raw = pd.read_csv(csv_path)
    print(f"  {len(df_raw):,} rows, {df_raw.shape[1]} columns.")

    df = handle_missing_values(df_raw)
    df = encode_categoricals(df)

    if "Repayment_Status" in df.columns:
        y = df["Repayment_Status"].values.astype(int)
        df = df.drop(columns=["Repayment_Status"])
    else:
        y = None

    df = add_engineered_features(df)
    feature_names = df.columns.tolist()
    X = df.values.astype("float32")
    print(f"  Feature matrix: {X.shape}  |  features: {feature_names}")
    return X, y, feature_names


# ---------------------------------------------------------------------------
# System 1 – Fraud Detection
# ---------------------------------------------------------------------------

def _train_fraud_detection(
    X_train_sc: np.ndarray,
    X_test_sc:  np.ndarray,
    y_test:     np.ndarray | None,
    out_dir:    str,
    contamination: float,
    ae_epochs:  int,
    skip_ae:    bool,
    input_dim:  int,
):
    """Train and save all three fraud-detection models."""
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  SYSTEM 1: FRAUD DETECTION")
    print("=" * 60)

    # ── Isolation Forest ───────────────────────────────────────────────────
    print("\n[1/3] Isolation Forest")
    iforest = IsolationForestDetector(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
    )
    iforest.fit(X_train_sc)
    if y_test is not None:
        iforest.evaluate(X_test_sc, y_test, fraud_label=0)
    iforest.save(os.path.join(out_dir, "isolation_forest.joblib"))

    # ── One-Class SVM ──────────────────────────────────────────────────────
    print("\n[2/3] One-Class SVM")
    # Sub-sample: OCSVM scales as O(n^2), so we cap the training set
    max_ocsvm = 3_000
    if len(X_train_sc) > max_ocsvm:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train_sc), max_ocsvm, replace=False)
        X_ocsvm = X_train_sc[idx]
    else:
        X_ocsvm = X_train_sc
    ocsvm = OneClassSVMDetector(nu=contamination)
    ocsvm.fit(X_ocsvm)
    if y_test is not None:
        ocsvm.evaluate(X_test_sc, y_test, fraud_label=0)
    ocsvm.save(os.path.join(out_dir, "one_class_svm.joblib"))

    # ── Autoencoder (PyTorch) ──────────────────────────────────────────────
    if not skip_ae:
        print("\n[3/3] Autoencoder Neural Network (PyTorch)")
        ae = AutoencoderDetector(
            input_dim=input_dim,
            latent_dim=8,
            epochs=ae_epochs,
            batch_size=64,
        )
        val_n = max(1, int(len(X_train_sc) * 0.15))
        ae.fit(X_train_sc[val_n:], X_val=X_train_sc[:val_n])
        if y_test is not None:
            ae.evaluate(X_test_sc, y_test, fraud_label=0)
        ae.save(
            model_path=os.path.join(out_dir, "autoencoder_weights.pt"),
            meta_path=os.path.join(out_dir, "autoencoder_meta.joblib"),
        )
    else:
        print("\n[3/3] Autoencoder skipped (--skip-ae).")


# ---------------------------------------------------------------------------
# System 2 – Borrower Segmentation
# ---------------------------------------------------------------------------

def _train_segmentation(
    X_scaled: np.ndarray,
    feature_names: list,
    out_dir: str,
    n_clusters: int,
):
    """Train and save all three segmentation models."""
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  SYSTEM 2: BORROWER SEGMENTATION")
    print("=" * 60)

    # ── K-Means ────────────────────────────────────────────────────────────
    print(f"\n[1/3] K-Means  (K={n_clusters})")
    kmeans = KMeansSegmenter(n_clusters=n_clusters, random_state=RANDOM_STATE)
    kmeans.fit_predict(X_scaled)
    kmeans.evaluate(X_scaled)
    kmeans.cluster_profiles(X_scaled, feature_names)
    kmeans.save(os.path.join(out_dir, "kmeans.joblib"))

    # ── DBSCAN ─────────────────────────────────────────────────────────────
    print("\n[2/3] DBSCAN")
    suggested_eps = DBSCANSegmenter.suggest_eps(X_scaled)
    dbscan = DBSCANSegmenter(eps=suggested_eps, min_samples=10)
    dbscan.fit(X_scaled)
    dbscan.evaluate(X_scaled)
    dbscan.cluster_profiles(X_scaled, feature_names, include_noise=False)
    dbscan.save(os.path.join(out_dir, "dbscan.joblib"))

    # ── Hierarchical ───────────────────────────────────────────────────────
    print(f"\n[3/3] Hierarchical Clustering  (K={n_clusters}, linkage=ward)")
    # Cap at 2000 samples: scipy linkage matrix is O(n^2) in memory
    max_hier = 2_000
    X_hier = X_scaled
    if len(X_scaled) > max_hier:
        rng = np.random.default_rng(RANDOM_STATE)
        X_hier = X_scaled[rng.choice(len(X_scaled), max_hier, replace=False)]
    hier = HierarchicalSegmenter(n_clusters=n_clusters, linkage="ward")
    hier.fit(X_hier)
    hier.evaluate(X_hier)
    hier.cluster_profiles(X_hier, feature_names)
    hier.save(os.path.join(out_dir, "hierarchical.joblib"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    fraud_dir = os.path.join(args.models_dir, "fraud_detection")
    seg_dir   = os.path.join(args.models_dir, "segmentation")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    X, y, feature_names = _load_and_prepare(args.csv)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y if y is not None else np.zeros(len(X), dtype=int),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Fit scaler on train split only (prevents data leakage)
    scaler, X_train_sc, X_test_sc = scale_features(X_train, X_test)

    # Scale the full dataset with the same fitted scaler for segmentation
    X_all_sc = scaler.transform(X)

    # Persist shared preprocessing artefacts
    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(scaler,        os.path.join(args.models_dir, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(args.models_dir, "features_list.joblib"))
    print(f"\nSaved scaler and feature list to '{args.models_dir}'.")

    # ------------------------------------------------------------------
    # System 1: Fraud Detection
    # ------------------------------------------------------------------
    _train_fraud_detection(
        X_train_sc=X_train_sc,
        X_test_sc=X_test_sc,
        y_test=y_test,
        out_dir=fraud_dir,
        contamination=args.contamination,
        ae_epochs=args.ae_epochs,
        skip_ae=args.skip_ae,
        input_dim=X_train_sc.shape[1],
    )

    # ------------------------------------------------------------------
    # System 2: Borrower Segmentation
    # ------------------------------------------------------------------
    _train_segmentation(
        X_scaled=X_all_sc,
        feature_names=feature_names,
        out_dir=seg_dir,
        n_clusters=args.n_clusters,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ALL MODELS SAVED")
    print("=" * 60)
    for root, _, files in os.walk(args.models_dir):
        for fname in sorted(files):
            rel = os.path.relpath(os.path.join(root, fname), _REPO_ROOT)
            print(f"  {rel}")
    print()


if __name__ == "__main__":
    main()
