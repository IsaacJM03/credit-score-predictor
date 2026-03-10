"""
models/segmentation/hierarchical.py
--------------------------------------
Borrower segmentation using Agglomerative Hierarchical Clustering.

Dendrogram interpretation
---------------------------
A dendrogram is a tree diagram that shows the sequence of merges:
  * Leaf nodes = individual borrowers (or small groups).
  * Each internal node = a merge event at a given *linkage distance*.
  * The Y-axis shows how dissimilar two clusters are when they merge.
  * The horizontal cut-off level determines the number of clusters:
    cut at height h → the number of vertical lines crossed = K clusters.

To choose K, look for the **largest vertical gap** between successive merges
(a dendrogram 'elbow') and cut just below it.

When hierarchical clustering is useful
----------------------------------------
* When the optimal K is unknown and you want a visual tool (dendrogram) to
  select it – more interpretable than the K-Means elbow.
* When clusters may be nested (hierarchical risk sub-groups: prime > subprime
  > near-default > default).
* When the dataset is small-to-medium (≤ ~5 k borrowers): O(n² log n) time
  complexity makes it impractical for very large datasets.
* When you need a *deterministic* result (no random initialisation unlike
  K-Means).
* When different linkage criteria (ward, average, complete) help explore the
  data structure.

Linkage criteria
-----------------
  ward     : minimises total within-cluster variance (default, often best).
  average  : average distance between all pairs across two clusters.
  complete : maximum distance between any two points in the two clusters
             (tends to produce compact, roughly equal-sized clusters).
  single   : minimum distance (produces chained, elongated clusters).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib


class HierarchicalSegmenter:
    """
    Agglomerative hierarchical clustering wrapper.

    Parameters
    ----------
    n_clusters : int     Number of clusters.  Can also be set later via
                         cut_tree().
    linkage    : str     Linkage criterion: 'ward', 'average', 'complete',
                         'single'.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        linkage: str = "ward",
    ):
        self.n_clusters = n_clusters
        self.linkage_method = linkage
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
        self._labels: np.ndarray | None = None
        self._linkage_matrix: np.ndarray | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "HierarchicalSegmenter":
        """
        Fit agglomerative clustering and compute the full linkage matrix for
        dendrogram generation.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)  – should be scaled
        """
        print(f"[Hierarchical] Fitting with n_clusters={self.n_clusters}, "
              f"linkage='{self.linkage_method}' on {X.shape[0]:,} samples …")
        self._labels = self.model.fit_predict(X)
        # Compute full linkage matrix (needed for dendrogram)
        # scipy's linkage() accepts the same linkage methods
        self._linkage_matrix = linkage(X, method=self.linkage_method)
        self._is_fitted = True
        n_clusters_found = len(set(self._labels))
        print(f"[Hierarchical] Assigned {n_clusters_found} clusters.")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        """
        Return cluster labels from the last fit() call.

        Note: Agglomerative clustering is transductive (no .predict() for
        new data).  For production, use KMeans or train a classifier on the
        cluster labels.
        """
        self._check_fitted()
        return self._labels

    def cut_tree(self, n_clusters: int) -> np.ndarray:
        """
        Re-cut the dendrogram to a different number of clusters *without*
        re-fitting.

        Parameters
        ----------
        n_clusters : desired number of groups

        Returns
        -------
        np.ndarray  cluster labels (1-indexed from scipy's fcluster)
        """
        self._check_fitted()
        labels = fcluster(self._linkage_matrix, t=n_clusters, criterion="maxclust")
        return labels - 1   # zero-indexed for consistency

    # ------------------------------------------------------------------
    # Dendrogram
    # ------------------------------------------------------------------

    def plot_dendrogram(
        self,
        max_display_samples: int = 50,
        title: str = "Hierarchical Clustering Dendrogram",
        color_threshold: float | None = None,
    ) -> plt.Figure:
        """
        Plot a truncated dendrogram (last *max_display_samples* merges).

        Parameters
        ----------
        max_display_samples : number of merge levels to show at the bottom
        color_threshold     : y-axis height below which branches are coloured
                              (automatically set if None)
        """
        self._check_fitted()
        fig, ax = plt.subplots(figsize=(12, 6))

        # 'lastp' truncation shows the last p merge levels
        dendrogram(
            self._linkage_matrix,
            truncate_mode="lastp",
            p=max_display_samples,
            leaf_rotation=90.0,
            leaf_font_size=8.0,
            show_contracted=True,
            color_threshold=color_threshold,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Sample Index (or cluster size in brackets)")
        ax.set_ylabel("Linkage Distance")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray) -> dict:
        """
        Compute silhouette and Davies-Bouldin scores.
        """
        self._check_fitted()
        labels = self._labels
        n_unique = len(set(labels))

        if n_unique >= 2:
            sil = silhouette_score(X, labels)
            db  = davies_bouldin_score(X, labels)
        else:
            sil = float("nan")
            db  = float("nan")

        metrics = {
            "n_clusters":           n_unique,
            "silhouette_score":     sil,
            "davies_bouldin_score": db,
        }
        print("\n[Hierarchical] Clustering quality metrics:")
        for k, v in metrics.items():
            print(f"  {k:28s}: {v}")
        return metrics

    def cluster_profiles(
        self,
        X: np.ndarray,
        feature_names: list,
    ) -> pd.DataFrame:
        """Mean feature values per cluster."""
        self._check_fitted()
        df = pd.DataFrame(X, columns=feature_names)
        df["Cluster"] = self._labels
        profile = df.groupby("Cluster").mean()
        print("\n[Hierarchical] Cluster profiles (mean feature values):")
        print(profile.to_string())
        return profile

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "model":          self.model,
                "labels":         self._labels,
                "linkage_matrix": self._linkage_matrix,
            },
            path,
        )
        print(f"[Hierarchical] Model saved to '{path}'.")

    @classmethod
    def load(cls, path: str) -> "HierarchicalSegmenter":
        data = joblib.load(path)
        seg = cls.__new__(cls)
        seg.model = data["model"]
        seg._labels = data["labels"]
        seg._linkage_matrix = data["linkage_matrix"]
        seg.n_clusters = seg.model.n_clusters
        seg.linkage_method = seg.model.linkage
        seg._is_fitted = True
        print(f"[Hierarchical] Model loaded from '{path}'.")
        return seg

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit(X) before accessing results.")
