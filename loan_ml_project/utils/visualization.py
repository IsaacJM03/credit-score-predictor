"""
utils/visualization.py
-----------------------
Reusable plotting helpers for the fraud detection and borrower segmentation
systems.

All functions return the matplotlib Figure so callers can either display it
(plt.show()) or save it (fig.savefig(...)) without side-effects inside these
utilities.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend safe for scripts and CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Fraud Detection Visualisations
# ---------------------------------------------------------------------------

def plot_anomaly_score_distribution(
    scores: np.ndarray,
    threshold: float,
    title: str = "Anomaly Score Distribution",
) -> plt.Figure:
    """
    Histogram of anomaly scores with the decision threshold overlaid.

    For Isolation Forest the scores are the raw decision_function values
    (more negative → more anomalous).  For the autoencoder they are the
    per-sample reconstruction errors (higher → more anomalous).

    Parameters
    ----------
    scores    : 1-D array of floats
    threshold : decision boundary; samples beyond this are flagged
    title     : plot title
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(threshold, color="crimson", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reconstruction_error(
    errors: np.ndarray,
    threshold: float,
    labels: np.ndarray | None = None,
    title: str = "Autoencoder Reconstruction Error",
) -> plt.Figure:
    """
    Scatter plot of per-sample reconstruction errors.

    Points above the threshold are highlighted in red as suspected anomalies.
    If ground-truth *labels* are provided (0 = normal, 1 = fraud) the
    colours encode ground-truth for quick visual model assessment.

    Parameters
    ----------
    errors    : 1-D array of reconstruction errors per sample
    threshold : horizontal cut-off line
    labels    : optional ground-truth array (0/1)
    title     : plot title
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(errors))

    if labels is not None:
        colors = np.where(labels == 1, "crimson", "steelblue")
        legend_handles = [
            mpatches.Patch(color="steelblue", label="Normal"),
            mpatches.Patch(color="crimson", label="Fraud / Anomaly"),
        ]
        ax.scatter(x, errors, c=colors, s=5, alpha=0.6)
        ax.legend(handles=legend_handles)
    else:
        flagged = errors > threshold
        colors = np.where(flagged, "crimson", "steelblue")
        ax.scatter(x, errors, c=colors, s=5, alpha=0.6)
        legend_handles = [
            mpatches.Patch(color="steelblue", label="Normal"),
            mpatches.Patch(color="crimson", label="Flagged (above threshold)"),
        ]
        ax.legend(handles=legend_handles)

    ax.axhline(threshold, color="darkorange", linewidth=1.5, linestyle="--",
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Borrower Segmentation Visualisations
# ---------------------------------------------------------------------------

def plot_elbow_curve(
    inertias: list,
    k_range: range,
    title: str = "K-Means Elbow Curve",
) -> plt.Figure:
    """
    Plot within-cluster sum of squares (inertia) vs. number of clusters K.

    The 'elbow' – the point where the curve bends – is a heuristic for the
    optimal K.

    Parameters
    ----------
    inertias : list of inertia values for each K
    k_range  : range object or list of K values tested
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker="o", color="steelblue", linewidth=2)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_silhouette_scores(
    sil_scores: list,
    k_range: range,
    title: str = "Silhouette Scores vs K",
) -> plt.Figure:
    """
    Bar chart of silhouette scores for different K values.

    Silhouette score ranges from -1 (wrong cluster) to +1 (well-separated).
    Higher is better; the optimal K maximises this score.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(k_range), sil_scores, color="steelblue", edgecolor="white")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(title)
    ax.set_xticks(list(k_range))
    fig.tight_layout()
    return fig


def plot_pca_clusters(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "Borrower Clusters (PCA 2-D Projection)",
    noise_label: int = -1,
) -> plt.Figure:
    """
    Reduce features to 2 principal components and colour-code cluster labels.

    DBSCAN uses -1 for noise points, which are plotted in grey.

    Parameters
    ----------
    X_scaled       : scaled feature matrix  shape (n, p)
    cluster_labels : integer cluster IDs    shape (n,)
    noise_label    : cluster ID used for noise / outliers (DBSCAN default -1)
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    unique_labels = sorted(set(cluster_labels))
    palette = sns.color_palette("tab10", n_colors=max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, label in enumerate(unique_labels):
        mask = cluster_labels == label
        color = "lightgrey" if label == noise_label else palette[idx % len(palette)]
        marker = "x" if label == noise_label else "o"
        display_label = "Noise" if label == noise_label else f"Cluster {label}"
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[color], s=20, marker=marker, alpha=0.6, label=display_label,
        )

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.legend(markerscale=2, loc="best")
    fig.tight_layout()
    return fig


def plot_cluster_heatmap(
    df_with_clusters: pd.DataFrame,
    feature_cols: list,
    cluster_col: str = "Cluster",
    title: str = "Cluster Feature Heatmap (Mean Values)",
) -> plt.Figure:
    """
    Heatmap of mean feature values per cluster.

    This provides an at-a-glance profile of each borrower segment, making it
    easy to label clusters semantically (e.g. 'High-income, low-risk').

    Parameters
    ----------
    df_with_clusters : DataFrame containing feature columns and a cluster column
    feature_cols     : list of numeric feature column names to include
    cluster_col      : name of the column holding cluster IDs
    """
    cluster_means = (
        df_with_clusters.groupby(cluster_col)[feature_cols]
        .mean()
    )
    # Standardise rows for visual comparison (z-score per feature)
    cluster_means_z = (cluster_means - cluster_means.mean()) / (cluster_means.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(max(10, len(feature_cols) * 0.8), 5))
    sns.heatmap(
        cluster_means_z,
        annot=True, fmt=".2f", cmap="RdYlGn",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Z-score"},
    )
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return fig


def plot_cluster_profiles(
    df_with_clusters: pd.DataFrame,
    feature_cols: list,
    cluster_col: str = "Cluster",
    title: str = "Cluster Profile Summary",
) -> pd.DataFrame:
    """
    Compute and print a summary DataFrame with mean, median and std per cluster.

    The aggregated statistics are printed to stdout as a human-readable table
    and also returned as a DataFrame for further programmatic use.

    Returns
    -------
    pd.DataFrame – aggregated statistics (mean / median / std) per cluster
    """
    summary = df_with_clusters.groupby(cluster_col)[feature_cols].agg(
        ["mean", "median", "std"]
    )
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(summary.to_string())
    print(f"{'='*60}\n")
    return summary
