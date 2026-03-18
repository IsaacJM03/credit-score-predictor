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
from sklearn.cluster import KMeans, MiniBatchKMeans
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
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        # model will be selected lazily in .fit() (KMeans or MiniBatchKMeans)
        self.model = None
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
        n_samples = X.shape[0]
        print(f"[KMeans] Fitting with K={self.n_clusters} on {n_samples:,} samples …")

        # For large datasets, prefer MiniBatchKMeans for speed and memory
        mini_batch_threshold = 50_000
        if n_samples > mini_batch_threshold:
            print(f"[KMeans] Large dataset detected (> {mini_batch_threshold:,}); using MiniBatchKMeans.")
            self.model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                max_iter=max(10, int(self.max_iter / 3)),
                batch_size=4096,
                n_init=max(1, int(self.n_init / 3)),
            )
        else:
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=self.n_init,
            )

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
        # For large X, compute elbow/silhouette on a representative subsample
        n_samples = X.shape[0]
        sample_size = 10_000
        if n_samples > sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n_samples, sample_size, replace=False)
            X_sample = X[idx]
            print(f"[KMeans] Using subsample of {sample_size:,} for elbow analysis (from {n_samples:,} samples)")
        else:
            X_sample = X

        inertias = []
        sil_scores = []
        for k in k_range:
            # use MiniBatchKMeans on the sample for speed
            km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3, batch_size=1024, max_iter=100)
            labels = km.fit_predict(X_sample)
            inertias.append(km.inertia_)
            sil = silhouette_score(X_sample, labels) if k > 1 else 0.0
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
        max_eval_samples = 50_000
        if len(X) > max_eval_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), max_eval_samples, replace=False)
            X_eval = X[idx]
            labels_eval = labels[idx]
            print(f"[KMeans] Evaluating on sample of {max_eval_samples:,} (from {len(X):,})")
        else:
            X_eval = X
            labels_eval = labels

        metrics = {
            "inertia":            self.model.inertia_,
            "silhouette_score":   silhouette_score(X_eval, labels_eval),
            "davies_bouldin_score": davies_bouldin_score(X_eval, labels_eval),
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
        max_profile_samples = 200_000
        if len(X) > max_profile_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), max_profile_samples, replace=False)
            X_profile = X[idx]
            labels_profile = labels[idx]
            print(f"[KMeans] Computing cluster profiles on sample of {max_profile_samples:,} (from {len(X):,})")
        else:
            X_profile = X
            labels_profile = labels

        df = pd.DataFrame(X_profile, columns=feature_names)
        df["Cluster"] = labels_profile
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
