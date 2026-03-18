"""
models/segmentation/dbscan.py
-------------------------------
Borrower segmentation using DBSCAN (Density-Based Spatial Clustering of
Applications with Noise).

How density-based clustering works
------------------------------------
DBSCAN groups points that are closely packed together (high density) while
marking points in low-density regions as *noise* (label = -1).

Two hyperparameters control density:
  eps    : maximum neighbourhood radius – two points are neighbours if their
           distance ≤ eps.
  min_samples: minimum number of points within eps to consider a point a
               *core point* (dense enough to start a cluster).

Algorithm:
  1. For each unvisited point, count its eps-neighbourhood.
  2. If count ≥ min_samples → core point; start a new cluster.
  3. Recursively expand the cluster to all density-reachable points.
  4. Points not reachable from any core point → noise (label -1).

Ability to detect outliers
----------------------------
Unlike K-Means, DBSCAN *explicitly* labels outliers as noise (-1).  In a
financial context these noise points are especially interesting: they represent
borrowers who do not fit any recognisable segment – a potential fraud or
high-risk signal.

Advantages over K-Means for financial datasets
------------------------------------------------
* Does **not** require specifying K in advance.
* Finds clusters of **arbitrary shape** (K-Means only finds convex/spherical clusters).
* Naturally identifies **outlier borrowers** (label -1) – valuable for risk.
* Robust to noise in financial data (income reporting errors, rounding artefacts).
* When borrower segments are irregular (e.g. a curved risk manifold) DBSCAN
  captures the true shape better.

Limitation: eps and min_samples must be tuned, typically using a k-NN distance
plot (elbow in sorted distances to k-th nearest neighbour).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import joblib


class DBSCANSegmenter:
    """
    DBSCAN clustering wrapper.

    Parameters
    ----------
    eps         : float  Neighbourhood radius.  Tune using kNN distance plot.
    min_samples : int    Minimum points to form a core point.  A common heuristic
                         is 2 × n_features.
    metric      : str    Distance metric.  'euclidean' (default) requires scaled
                         features.  'cosine' can be useful for sparse text data.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        n_jobs: int = -1,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            n_jobs=n_jobs,
        )
        self._labels: np.ndarray | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DBSCANSegmenter":
        """
        Fit DBSCAN on the scaled feature matrix.

        Always scale features (StandardScaler) before calling this method.
        DBSCAN relies on distance calculations, so feature magnitudes must
        be comparable.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)
        """
        print(f"[DBSCAN] Fitting with eps={self.eps}, "
              f"min_samples={self.min_samples} on {X.shape[0]:,} samples …")
        self._labels = self.model.fit_predict(X)
        n_clusters = len(set(self._labels)) - (1 if -1 in self._labels else 0)
        n_noise = (self._labels == -1).sum()
        self._is_fitted = True
        print(f"[DBSCAN] Found {n_clusters} clusters, {n_noise} noise points "
              f"({n_noise / len(self._labels) * 100:.1f}% of data).")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def labels(self) -> np.ndarray:
        """Return cluster label array from the last fit() call."""
        self._check_fitted()
        return self._labels

    def noise_mask(self) -> np.ndarray:
        """Boolean mask: True = noise / outlier point."""
        self._check_fitted()
        return self._labels == -1

    # ------------------------------------------------------------------
    # Hyperparameter tuning helper
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_eps(
        X: np.ndarray,
        k: int | None = None,
        percentile: float = 50.0,
        max_samples: int = 20_000,
        eps_cap: float = 1.0,
    ) -> float:
        """
        Suggest an eps value based on the k-NN distance distribution.

        Uses the median (50th percentile) of k-NN distances by default –
        tighter than the 90th percentile to avoid massive neighbourhoods in
        high-dimensional standardised space.
        Hard-caps eps at eps_cap (default 1.0) to prevent DBSCAN from
        building a single giant cluster that exhausts memory.
        Subsamples to max_samples rows for speed.
        """
        if k is None:
            k = max(2 * X.shape[1], 5)
        if len(X) > max_samples:
            rng = np.random.default_rng(42)
            X = X[rng.choice(len(X), max_samples, replace=False)]
            print(f"[DBSCAN] suggest_eps using {max_samples:,}-row subsample.")
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, _ = nbrs.kneighbors(X)
        kth_distances = np.sort(distances[:, -1])
        suggested = float(np.percentile(kth_distances, percentile))
        suggested = min(suggested, eps_cap)
        print(f"[DBSCAN] Suggested eps ({percentile}th percentile of "
              f"{k}-NN distances, capped at {eps_cap}): {suggested:.4f}")
        return suggested

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, sil_max_samples: int = 20_000) -> dict:
        """
        Compute silhouette and Davies-Bouldin scores on non-noise points.

        silhouette_score is capped at sil_max_samples to avoid an O(n²) hang.
        These metrics require at least 2 clusters; if DBSCAN finds fewer,
        they are set to NaN.
        """
        self._check_fitted()
        mask = self._labels != -1
        labels_clean = self._labels[mask]
        X_clean = X[mask]
        n_clusters = len(set(labels_clean))

        if n_clusters >= 2 and len(X_clean) > n_clusters:
            if len(X_clean) > sil_max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_clean), sil_max_samples, replace=False)
                sil = silhouette_score(X_clean[idx], labels_clean[idx])
                print(f"  [DBSCAN] silhouette computed on {sil_max_samples:,}-sample subset.")
            else:
                sil = silhouette_score(X_clean, labels_clean)
            db  = davies_bouldin_score(X_clean, labels_clean)
        else:
            sil = float("nan")
            db  = float("nan")

        metrics = {
            "n_clusters":           n_clusters,
            "n_noise":              int((self._labels == -1).sum()),
            "silhouette_score":     sil,
            "davies_bouldin_score": db,
        }
        print("\n[DBSCAN] Clustering quality metrics:")
        for k, v in metrics.items():
            print(f"  {k:28s}: {v}")
        return metrics

    def cluster_profiles(
        self,
        X: np.ndarray,
        feature_names: list,
        include_noise: bool = False,
        labels: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Mean feature values per cluster (noise label = -1 can be excluded).

        Parameters
        ----------
        X             : feature matrix (full or pre-filtered)
        feature_names : list of feature column names
        include_noise : if True, noise points are treated as 'Cluster -1'
        labels        : optional external labels array; if None uses self._labels
        """
        self._check_fitted()
        use_labels = labels if labels is not None else self._labels
        df = pd.DataFrame(X, columns=feature_names)
        # Only attach labels if lengths match; otherwise use a slice
        if len(use_labels) == len(df):
            df["Cluster"] = use_labels
        else:
            df["Cluster"] = use_labels[:len(df)]
        if not include_noise:
            df = df[df["Cluster"] != -1]
        profile = df.groupby("Cluster").mean()
        print("\n[DBSCAN] Cluster profiles (mean feature values):")
        print(profile.to_string())
        return profile

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "labels": self._labels}, path)
        print(f"[DBSCAN] Model saved to '{path}'.")

    @classmethod
    def load(cls, path: str) -> "DBSCANSegmenter":
        data = joblib.load(path)
        segmenter = cls.__new__(cls)
        segmenter.model = data["model"]
        segmenter._labels = data["labels"]
        segmenter.eps = segmenter.model.eps
        segmenter.min_samples = segmenter.model.min_samples
        segmenter._is_fitted = True
        print(f"[DBSCAN] Model loaded from '{path}'.")
        return segmenter

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit(X) before accessing results.")
