"""
models/segmentation/kmeans.py
-------------------------------
Borrower segmentation using K-Means clustering.

How K-Means determines clusters
---------------------------------
K-Means partitions n borrowers into K non-overlapping groups by minimising the
Within-Cluster Sum of Squares (WCSS / inertia):

    argmin Σ_k Σ_{x ∈ C_k} ||x - μ_k||²

Algorithm (Lloyd's iterations):
  1. Randomly initialise K cluster centres (centroids).
  2. Assign each sample to its nearest centroid (Euclidean distance).
  3. Recompute centroids as the mean of all assigned samples.
  4. Repeat steps 2–3 until assignments stabilise or max iterations reached.

Choosing optimal K
-------------------
* **Elbow Method**: plot inertia vs. K; the "elbow" where improvement slows
  is the heuristic optimal K.
* **Silhouette Score**: measures how similar a point is to its own cluster
  vs. neighbouring clusters.  Range: [-1, 1].  Higher is better.

Example cluster interpretations
---------------------------------
  Cluster 0 – High-income, low loan amounts, low risk (prime borrowers)
  Cluster 1 – Moderate income, high debt-to-income, medium risk
  Cluster 2 – Low income, frequent loans, high default risk
  Cluster 3 – Irregular repayment patterns (potential fraud signal)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib


class KMeansSegmenter:
    """
    K-Means clustering wrapper with Elbow / Silhouette selection helpers.

    Parameters
    ----------
    n_clusters  : int   Number of clusters K.
    random_state: int
    max_iter    : int   Maximum Lloyd iterations per run.
    n_init      : int   Number of random restarts (best result is kept).
                        Higher n_init → more robust but slower.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 10,
    ):
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
        )
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeansSegmenter":
        """
        Fit K-Means on the scaled feature matrix.

        K-Means uses Euclidean distances → always scale features first
        (StandardScaler) so that no single feature dominates due to its
        magnitude.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)
        """
        print(f"[KMeans] Fitting with K={self.n_clusters} on "
              f"{X.shape[0]:,} samples …")
        self.model.fit(X)
        self._is_fitted = True
        print(f"[KMeans] Inertia (WCSS): {self.model.inertia_:.2f}")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to its nearest cluster centroid."""
        self._check_fitted()
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    # ------------------------------------------------------------------
    # Optimal K selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def elbow_analysis(
        X: np.ndarray,
        k_range: range = range(2, 11),
        random_state: int = 42,
    ) -> tuple:
        """
        Compute inertia for each K in *k_range*.

        Returns
        -------
        (inertias, sil_scores)  both lists of floats, one per K value
        """
        inertias = []
        sil_scores = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            sil = silhouette_score(X, labels) if k > 1 else 0.0
            sil_scores.append(sil)
            print(f"  K={k:2d} | inertia={km.inertia_:,.1f} | silhouette={sil:.4f}")
        return inertias, sil_scores

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray) -> dict:
        """
        Compute clustering quality metrics.

        Returns
        -------
        dict with 'inertia', 'silhouette_score', 'davies_bouldin_score'
        """
        self._check_fitted()
        labels = self.model.labels_
        metrics = {
            "inertia":            self.model.inertia_,
            "silhouette_score":   silhouette_score(X, labels),
            "davies_bouldin_score": davies_bouldin_score(X, labels),
        }
        print("\n[KMeans] Clustering quality metrics:")
        for k, v in metrics.items():
            print(f"  {k:28s}: {v:.4f}")
        print("  (Silhouette: higher is better; Davies-Bouldin: lower is better)")
        return metrics

    def cluster_profiles(
        self,
        X: np.ndarray,
        feature_names: list,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with the mean feature value per cluster.

        Use this to label clusters semantically (e.g. 'High-income segment').
        """
        self._check_fitted()
        labels = self.model.labels_
        df = pd.DataFrame(X, columns=feature_names)
        df["Cluster"] = labels
        profile = df.groupby("Cluster").mean()
        print("\n[KMeans] Cluster profiles (mean feature values):")
        print(profile.to_string())
        return profile

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
        print(f"[KMeans] Model saved to '{path}'.")

    @classmethod
    def load(cls, path: str) -> "KMeansSegmenter":
        segmenter = cls.__new__(cls)
        segmenter.model = joblib.load(path)
        segmenter.n_clusters = segmenter.model.n_clusters
        segmenter._is_fitted = True
        print(f"[KMeans] Model loaded from '{path}'.")
        return segmenter

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit(X) before prediction.")
