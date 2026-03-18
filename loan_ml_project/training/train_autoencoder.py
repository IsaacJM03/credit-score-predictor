"""
training/train_autoencoder.py
------------------------------
Standalone script to train the PyTorch autoencoder fraud detector and
persist the trained model.

Usage
-----
    python training/train_autoencoder.py \
        --csv  ../comprehensive_loan_data_45M.csv \
        --out  ../models/fraud_detection/autoencoder_weights.pt \
        --meta ../models/fraud_detection/autoencoder_meta.joblib

This script:
  1. Loads and preprocesses the loan dataset.
  2. Engineers additional features.
  3. Scales features with StandardScaler (fit on train split only).
  4. Trains the AutoencoderDetector.
  5. Evaluates against the test split (using Repayment_Status as a proxy
     label where 0 = default / potentially suspicious).
  6. Saves model weights and threshold metadata.

Practical notes on imbalanced data
------------------------------------
Fraud / default events are rare in real datasets.  Techniques to handle this:
  * Train the autoencoder on **normal samples only** (Repayment_Status == 1)
    so the reconstruction error threshold is calibrated on clean data.
  * If mixing classes during training, weight the anomalous class upwards
    in the loss function (not shown here as the autoencoder is unsupervised).
  * Evaluate with precision-recall AUC rather than ROC-AUC when classes are
    very imbalanced.

Detecting concept drift
------------------------
  * Periodically re-compute the reconstruction error distribution on a recent
    data window and compare it (e.g. KL-divergence or KS test) to the
    distribution seen at training time.
  * If the distribution shifts significantly, retrain the autoencoder.
  * Store a reference histogram of training errors alongside the model.
"""

import argparse
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd

# Allow running from the repo root or from within loan_ml_project/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.preprocessing import (
    preprocess_pipeline,
    scale_features,
    handle_missing_values,
    encode_categoricals,
)
from utils.feature_engineering import add_engineered_features
from models.fraud_detection.autoencoder import AutoencoderDetector


def parse_args():
    p = argparse.ArgumentParser(description="Train the fraud-detection autoencoder.")
    p.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "../../comprehensive_loan_data_45M.csv"),
        help="Path to comprehensive_loan_data_45M.csv",
    )
    p.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "../models/fraud_detection/autoencoder_weights.pt"),
        help="Output path for PyTorch model weights (.pt)",
    )
    p.add_argument(
        "--meta",
        default=os.path.join(os.path.dirname(__file__), "../models/fraud_detection/autoencoder_meta.joblib"),
        help="Output path for threshold metadata (.joblib)",
    )
    p.add_argument("--latent-dim",  type=int,   default=8,    help="Bottleneck size")
    p.add_argument("--epochs",      type=int,   default=50,   help="Max training epochs")
    p.add_argument("--lr",          type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch-size",  type=int,   default=64,   help="Batch size")
    p.add_argument("--test-size",   type=float, default=0.2,  help="Test fraction")
    p.add_argument(
        "--threshold-pct",
        type=float,
        default=95.0,
        help="Percentile of training errors to use as anomaly threshold",
    )
    p.add_argument(
        "--train-on-normal-only",
        action="store_true",
        default=False,
        help="If set, train only on Repayment_Status==1 (clean) samples.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load and preprocess
    # ------------------------------------------------------------------
    X_raw, y, feature_names = preprocess_pipeline(args.csv)

    # Feature engineering on the full DataFrame
    df_raw = pd.read_csv(args.csv)
    df = handle_missing_values(df_raw)
    df = encode_categoricals(df)
    if "Repayment_Status" in df.columns:
        y = df["Repayment_Status"].values
        df_feats = df.drop(columns=["Repayment_Status"])
    else:
        y = None
        df_feats = df

    df_eng = add_engineered_features(df_feats)
    feature_names = df_eng.columns.tolist()
    X = df_eng.values.astype("float32")

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test = train_test_split(X, test_size=args.test_size, random_state=42)
        y_train = y_test = None

    # ------------------------------------------------------------------
    # 3. Scale features (fit on train only)
    # ------------------------------------------------------------------
    scaler, X_train_sc, X_test_sc = scale_features(X_train, X_test)

    # Optionally train only on normal (non-default) samples
    if args.train_on_normal_only and y_train is not None:
        normal_mask = y_train == 1
        X_autoenc_train = X_train_sc[normal_mask]
        print(f"[train] Training on {normal_mask.sum():,} normal samples "
              f"(dropped {(~normal_mask).sum():,} default samples).")
    else:
        X_autoenc_train = X_train_sc

    # Use 15 % of autoencoder training set as validation
    val_split = int(len(X_autoenc_train) * 0.15)
    X_val_ae  = X_autoenc_train[:val_split]
    X_fit_ae  = X_autoenc_train[val_split:]

    # ------------------------------------------------------------------
    # 4. Train the autoencoder
    # ------------------------------------------------------------------
    detector = AutoencoderDetector(
        input_dim=X_fit_ae.shape[1],
        latent_dim=args.latent_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    history = detector.fit(
        X_fit_ae,
        X_val=X_val_ae,
        threshold_percentile=args.threshold_pct,
    )

    # ------------------------------------------------------------------
    # 5. Evaluate on test set (if labels available)
    # ------------------------------------------------------------------
    if y_test is not None:
        # Use Repayment_Status==0 (default) as a proxy for anomaly
        print("\n[train] Evaluating on test set (fraud proxy = Repayment_Status==0) …")
        detector.evaluate(X_test_sc, y_test, fraud_label=0)

    # ------------------------------------------------------------------
    # 6. Save model
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.meta), exist_ok=True)
    detector.save(model_path=args.out, meta_path=args.meta)
    print(f"\n[train] Done.  Autoencoder saved to '{args.out}'")


if __name__ == "__main__":
    main()
