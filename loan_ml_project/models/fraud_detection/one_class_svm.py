"""
models/fraud_detection/one_class_svm.py
----------------------------------------
Fraud detection using One-Class SVM (OCSVM).

How One-Class SVM models normal data
--------------------------------------
OCSVM learns a decision boundary (a hypersphere in kernel-transformed space)
that encloses the majority of *training* samples.  At prediction time any point
that falls *outside* this boundary is declared an anomaly.

Mathematically, the RBF kernel maps every sample into an infinite-dimensional
feature space.  In that space a hyperplane is found that:
  * Separates the training data from the origin with maximum margin.
  * Minimises the volume of the enclosing region (controlled by *nu*).

Points with a negative decision function value lie outside the boundary.

When OCSVM outperforms Isolation Forest
-----------------------------------------
* When the normal class has a *compact, well-defined* distribution in feature
  space and anomalies are truly "far away".
* When you have a relatively clean, large training set of normal samples
  (without contamination): OCSVM can model the normal manifold very precisely.
* When you can tune the RBF kernel width (*gamma*) and *nu* carefully –
  OCSVM can achieve tighter, non-convex decision boundaries that Isolation
  Forest (which uses axis-aligned splits) cannot replicate.
* For datasets where fraud patterns are *dense* relative to normal ones,
  OCSVM's kernel approach may capture the shape more accurately.

Limitations
-----------
* Scales quadratically with n_samples → use on ≤ ~10 k samples, or sub-sample.
* Sensitive to feature scale → always apply StandardScaler first.
* Requires careful *nu* and *gamma* tuning.
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
import joblib


class OneClassSVMDetector:
    """
    Wrapper around sklearn's OneClassSVM for fraud / anomaly detection.

    Parameters
    ----------
    kernel : str    Kernel type.  'rbf' is the standard choice for anomaly
                    detection as it allows non-linear decision boundaries.
    nu     : float  Upper bound on the fraction of training errors AND a lower
                    bound on the fraction of support vectors.  Values between
                    0.01 and 0.1 are typical when fraud is rare.
    gamma  : str|float  Kernel coefficient.  'scale' (1 / (n_features * X.var()))
                        is a good default; 'auto' uses 1/n_features.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str = "scale",
    ):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.nu = nu
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        """
        Fit the One-Class SVM on normal (or mixed, low-contamination) data.

        Always scale features (StandardScaler) before calling this method.
        OCSVM is highly sensitive to feature magnitude because it relies on
        Euclidean distances in the kernel-transformed space.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)  – must be scaled
        """
        print(f"[OneClassSVM] Fitting on {X.shape[0]:,} samples "
              f"(nu={self.nu}) …")
        self.model.fit(X)
        self._is_fitted = True
        print("[OneClassSVM] Training complete.")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return +1 (normal) or -1 (anomaly) for each sample."""
        self._check_fitted()
        return self.model.predict(X)

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw decision function values.
        Negative → anomalous, positive → normal.
        """
        self._check_fitted()
        return self.model.decision_function(X)

    def flag_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Return a boolean mask: True = anomaly."""
        return self.predict(X) == -1

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
        Evaluate predictions against ground-truth fraud labels.

        Parameters
        ----------
        X          : scaled feature matrix
        y_true     : ground-truth labels
        fraud_label: value in y_true that represents fraud

        Returns
        -------
        dict with precision, recall, f1, roc_auc
        """
        self._check_fitted()
        sklearn_preds = self.predict(X)
        y_pred_binary = (sklearn_preds == -1).astype(int)
        y_true_binary = (y_true == fraud_label).astype(int)

        scores = self.decision_scores(X)
        auc = roc_auc_score(y_true_binary, -scores)   # negate: lower score = more anomalous

        metrics = {
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall":    recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "f1":        f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "roc_auc":   auc,
        }
        print("\n[OneClassSVM] Evaluation results:")
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
        joblib.dump(self.model, path)
        print(f"[OneClassSVM] Model saved to '{path}'.")

    @classmethod
    def load(cls, path: str) -> "OneClassSVMDetector":
        detector = cls.__new__(cls)
        detector.model = joblib.load(path)
        detector._is_fitted = True
        print(f"[OneClassSVM] Model loaded from '{path}'.")
        return detector

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted yet. Call .fit(X) before prediction."
            )
