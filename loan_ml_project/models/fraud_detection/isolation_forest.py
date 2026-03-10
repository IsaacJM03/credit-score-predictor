"""
models/fraud_detection/isolation_forest.py
-------------------------------------------
Fraud detection using scikit-learn's Isolation Forest.

How Isolation Forest detects anomalies
---------------------------------------
The algorithm builds an ensemble of random decision trees where each tree is
grown by:
  1. Randomly selecting a feature.
  2. Randomly selecting a split value between the feature's min and max.

The key insight is that *anomalous* samples (fraudulent applications) are
statistically rare and occupy sparse regions of the feature space, so they
require *fewer splits* to isolate than normal samples.  The anomaly score is
the average path length across all trees – short path → anomalous.

Why it works well for fraud detection
--------------------------------------
* Handles high-dimensional mixed numeric data without feature selection.
* No assumption about the data distribution (non-parametric).
* Scales linearly with dataset size (sub-sampling per tree keeps it fast).
* Works well on *unlabelled* data – no fraud labels required.
* Robust to irrelevant features; the random splits naturally down-weight them.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
import joblib


class IsolationForestDetector:
    """
    Thin wrapper around sklearn IsolationForest with helper methods for
    scoring, thresholding, and evaluation.

    Parameters
    ----------
    n_estimators   : int    Number of base estimators (trees).  100 is usually
                            sufficient; more → slightly better scores but slower.
    contamination  : float  Expected fraction of outliers in the dataset.
                            If unknown, start with 0.05 (5 %) and tune.
    random_state   : int
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.contamination = contamination
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Fit the isolation forest on the (ideally normal) training data.

        In a real deployment you would train on *clean* (non-fraudulent)
        applications only so the model learns the normal distribution.
        When labels are unavailable, train on all data and rely on
        *contamination* to set the threshold automatically.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)
        """
        print(f"[IsolationForest] Fitting on {X.shape[0]:,} samples …")
        self.model.fit(X)
        self._is_fitted = True
        print("[IsolationForest] Training complete.")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary predictions: +1 = normal, -1 = anomaly (sklearn convention).
        """
        self._check_fitted()
        return self.model.predict(X)

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores (lower / more negative = more anomalous).

        These can be used to build ROC curves or to set a custom threshold.
        """
        self._check_fitted()
        return self.model.decision_function(X)

    def flag_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask: True means the sample is flagged as anomalous.
        """
        preds = self.predict(X)
        return preds == -1   # -1 is the anomaly label in sklearn

    # ------------------------------------------------------------------
    # Evaluation (requires ground-truth labels)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        fraud_label: int = 1,
    ) -> dict:
        """
        Compute precision, recall, F1, and ROC-AUC against ground-truth labels.

        Parameters
        ----------
        X          : feature matrix
        y_true     : ground-truth labels (fraud_label = anomaly)
        fraud_label: value in y_true that represents fraud (default 1)

        Returns
        -------
        dict with metric names and values
        """
        self._check_fitted()
        sklearn_preds = self.predict(X)
        # Convert sklearn convention (-1/+1) to (1/0) for comparison
        y_pred_binary = (sklearn_preds == -1).astype(int)
        # Align fraud_label: if fraud = 0 in y_true, invert
        y_true_binary = (y_true == fraud_label).astype(int)

        scores = self.decision_scores(X)
        # Negate scores so that higher value = more anomalous (for AUC)
        auc = roc_auc_score(y_true_binary, -scores)

        metrics = {
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall":    recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "f1":        f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "roc_auc":   auc,
        }
        print("\n[IsolationForest] Evaluation results:")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
        print("\n" + classification_report(
            y_true_binary, y_pred_binary,
            target_names=["Normal", "Fraud"], zero_division=0,
        ))
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted model to disk with joblib."""
        joblib.dump(self.model, path)
        print(f"[IsolationForest] Model saved to '{path}'.")

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        """Load a previously saved model from disk."""
        detector = cls.__new__(cls)
        detector.model = joblib.load(path)
        detector._is_fitted = True
        print(f"[IsolationForest] Model loaded from '{path}'.")
        return detector

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted yet. Call .fit(X) before prediction."
            )
