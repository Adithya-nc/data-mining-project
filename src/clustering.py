"""
MarineMine Pro — Marine Pollution Data Mining & ML Platform
===========================================================
Module: clustering.py
Description: Full KMeans clustering implementation for marine pollution datasets.
             Provides feature scaling, cluster fitting, elbow/silhouette diagnostics,
             per-cluster statistics, plain-language descriptions, and human-readable
             cluster labels.  All public functions are edge-case-safe (too few rows,
             all-NaN features, single cluster, etc.).
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*n_init.*")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MIN_ROWS_FOR_CLUSTERING = 2  # absolute floor
_CLUSTER_LABEL_PREFIX = "Cluster"


def _validate_and_clean(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], str | None]:
    """
    Validate inputs and return a cleaned copy of the feature sub-frame.

    Returns
    -------
    clean_df : pd.DataFrame
        Rows with no NaN across *all* feature columns (index preserved).
    valid_cols : list[str]
        Feature columns that are not entirely NaN.
    error : str | None
        Non-None error message when the data is unusable; callers should
        surface this to the user and return early.
    """
    # --- guard: df must be a DataFrame ----------------------------------
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(), [], "Input must be a pandas DataFrame."

    if df.empty:
        return pd.DataFrame(), [], "DataFrame is empty."

    # --- guard: feature_cols must be present ----------------------------
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return pd.DataFrame(), [], f"Feature columns not found in DataFrame: {missing}"

    sub = df[feature_cols].copy()

    # --- drop columns that are entirely NaN -----------------------------
    all_nan_cols = [c for c in feature_cols if sub[c].isna().all()]
    if all_nan_cols:
        logger.warning("Dropping all-NaN feature columns: %s", all_nan_cols)
    valid_cols = [c for c in feature_cols if c not in all_nan_cols]

    if not valid_cols:
        return pd.DataFrame(), [], "All feature columns are entirely NaN."

    # --- convert to numeric, coerce non-numeric to NaN -----------------
    sub[valid_cols] = sub[valid_cols].apply(pd.to_numeric, errors="coerce")

    # --- drop rows with any remaining NaN in the valid feature columns --
    clean = df.loc[sub[valid_cols].dropna().index].copy()

    if len(clean) < _MIN_ROWS_FOR_CLUSTERING:
        return (
            pd.DataFrame(),
            [],
            f"Too few usable rows after dropping NaNs "
            f"({len(clean)} < {_MIN_ROWS_FOR_CLUSTERING}).",
        )

    return clean, valid_cols, None


def _scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler on *df[feature_cols]* and return (X_scaled, scaler)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].astype(float))
    return X, scaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_kmeans(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Scale *feature_cols* and fit KMeans with *n_clusters* clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame (original; will not be mutated).
    feature_cols : list[str]
        Numeric feature columns to use for clustering.
    n_clusters : int
        Number of clusters (default 3).
    random_state : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    result_df : pd.DataFrame
        Copy of the *clean* rows of *df* with an extra integer ``"cluster"``
        column (0-based).
    labels : list[int]
        Cluster label for every row in *result_df* (same order).

    Raises
    ------
    ValueError
        If the data is unusable (surfaces the error message from validation).
    """
    clean, valid_cols, error = _validate_and_clean(df, feature_cols)
    if error:
        raise ValueError(f"run_kmeans: {error}")

    # Clamp n_clusters to the number of usable rows
    n_clusters = max(1, min(n_clusters, len(clean)))

    if n_clusters == 1:
        logger.warning(
            "run_kmeans: n_clusters clamped to 1 "
            "(fewer usable rows than requested clusters)."
        )
        result_df = clean.copy()
        result_df["cluster"] = 0
        return result_df, [0] * len(result_df)

    X, _ = _scale_features(clean, valid_cols)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels_arr = km.fit_predict(X)

    result_df = clean.copy()
    result_df["cluster"] = labels_arr.tolist()

    logger.info(
        "run_kmeans: fitted %d clusters on %d rows using features %s",
        n_clusters,
        len(result_df),
        valid_cols,
    )
    return result_df, labels_arr.tolist()


# ---------------------------------------------------------------------------


def compute_elbow(
    df: pd.DataFrame,
    feature_cols: List[str],
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Dict[int, float]:
    """
    Compute KMeans inertia for each *k* in *k_range* (elbow method).

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame.
    feature_cols : list[str]
        Numeric feature columns.
    k_range : range
        Values of *k* to evaluate (default ``range(2, 11)``).

    Returns
    -------
    dict[int, float]
        Mapping ``{k: inertia}``.  Keys are limited to values where
        ``k <= len(usable_rows)``.  Returns an empty dict on data errors.
    """
    clean, valid_cols, error = _validate_and_clean(df, feature_cols)
    if error:
        logger.warning("compute_elbow: %s — returning empty dict.", error)
        return {}

    X, _ = _scale_features(clean, valid_cols)
    n_rows = len(clean)

    elbow: Dict[int, float] = {}
    for k in k_range:
        if k < 2:
            continue
        if k > n_rows:
            logger.debug(
                "compute_elbow: skipping k=%d (only %d usable rows).", k, n_rows
            )
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X)
        elbow[k] = float(km.inertia_)
        logger.debug("compute_elbow: k=%d → inertia=%.4f", k, km.inertia_)

    return elbow


# ---------------------------------------------------------------------------


def compute_silhouette(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int,
    random_state: int = 42,
) -> Optional[float]:
    """
    Fit KMeans with *n_clusters* and return the average silhouette score.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame.
    feature_cols : list[str]
        Numeric feature columns.
    n_clusters : int
        Number of clusters to evaluate.

    Returns
    -------
    float | None
        Silhouette score in [-1, 1], or ``None`` when the score cannot be
        computed (e.g., only 1 cluster, too few rows, all same label).
    """
    clean, valid_cols, error = _validate_and_clean(df, feature_cols)
    if error:
        logger.warning("compute_silhouette: %s — returning None.", error)
        return None

    n_rows = len(clean)
    if n_clusters < 2:
        logger.warning(
            "compute_silhouette: n_clusters=%d < 2; silhouette undefined.",
            n_clusters,
        )
        return None

    if n_clusters >= n_rows:
        logger.warning(
            "compute_silhouette: n_clusters=%d ≥ n_rows=%d; score undefined.",
            n_clusters,
            n_rows,
        )
        return None

    X, _ = _scale_features(clean, valid_cols)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels_arr = km.fit_predict(X)

    # silhouette_score requires at least 2 distinct labels
    if len(np.unique(labels_arr)) < 2:
        logger.warning("compute_silhouette: only one distinct cluster label found.")
        return None

    score = float(silhouette_score(X, labels_arr))
    logger.info(
        "compute_silhouette: k=%d → silhouette=%.4f", n_clusters, score
    )
    return score


# ---------------------------------------------------------------------------


def get_cluster_stats(
    df: pd.DataFrame,
    feature_cols: List[str],
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """
    Compute per-cluster descriptive statistics (mean, std, size).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains a *cluster_col* column (e.g., the
        output of :func:`run_kmeans`).
    feature_cols : list[str]
        Numeric feature columns to aggregate.
    cluster_col : str
        Name of the cluster-label column (default ``"cluster"``).

    Returns
    -------
    pd.DataFrame
        Multi-level column DataFrame indexed by cluster id with columns
        ``(feature, stat)`` where *stat* ∈ {``"mean"``, ``"std"``, ``"size"``}.
        Returns an empty DataFrame on errors.
    """
    if cluster_col not in df.columns:
        logger.warning(
            "get_cluster_stats: cluster column '%s' not in DataFrame.", cluster_col
        )
        return pd.DataFrame()

    present_cols = [c for c in feature_cols if c in df.columns]
    if not present_cols:
        logger.warning("get_cluster_stats: none of feature_cols found in DataFrame.")
        return pd.DataFrame()

    sub = df[[cluster_col] + present_cols].copy()
    sub[present_cols] = sub[present_cols].apply(pd.to_numeric, errors="coerce")

    grouped = sub.groupby(cluster_col)[present_cols]

    mean_df = grouped.mean().rename(columns=lambda c: c)
    std_df = grouped.std(ddof=1).rename(columns=lambda c: c)
    size_series = grouped.size().rename("size")

    # Build a tidy multi-index column frame
    mean_df.columns = pd.MultiIndex.from_tuples(
        [(col, "mean") for col in mean_df.columns]
    )
    std_df.columns = pd.MultiIndex.from_tuples(
        [(col, "std") for col in std_df.columns]
    )
    size_df = size_series.to_frame()
    size_df.columns = pd.MultiIndex.from_tuples([("size", "count")])

    stats_df = pd.concat([mean_df, std_df, size_df], axis=1)
    stats_df.index.name = cluster_col
    stats_df = stats_df.sort_index()

    logger.info(
        "get_cluster_stats: computed stats for %d clusters × %d features.",
        len(stats_df),
        len(present_cols),
    )
    return stats_df


# ---------------------------------------------------------------------------


def describe_clusters(cluster_stats_df: pd.DataFrame) -> Dict[int, str]:
    """
    Generate plain-language descriptions for each cluster from its statistics.

    Uses the *mean* statistics to characterise each cluster relative to the
    others: features are tagged as "high", "moderate", or "low" based on
    z-score across cluster centroids.

    Parameters
    ----------
    cluster_stats_df : pd.DataFrame
        Output of :func:`get_cluster_stats` — a multi-level column DataFrame
        indexed by cluster id.

    Returns
    -------
    dict[int, str]
        Mapping ``{cluster_id: plain-language description}``.
        Returns an empty dict when the input is unusable.
    """
    if cluster_stats_df is None or cluster_stats_df.empty:
        logger.warning("describe_clusters: empty stats DataFrame — returning {}.")
        return {}

    # Extract mean sub-frame -----------------------------------------------
    try:
        means = cluster_stats_df.xs("mean", level=1, axis=1)
    except KeyError:
        logger.warning(
            "describe_clusters: 'mean' level not found in column index."
        )
        return {}

    if means.empty:
        return {}

    feature_cols = list(means.columns)
    cluster_ids = list(means.index)

    # Try to extract cluster sizes -----------------------------------------
    sizes: Dict[int, int] = {}
    try:
        sizes = cluster_stats_df[("size", "count")].to_dict()
    except KeyError:
        pass

    # z-score of cluster centroids across features -------------------------
    # shape: (n_clusters, n_features)
    centroid_array = means.values.astype(float)
    global_mean = np.nanmean(centroid_array, axis=0)
    global_std = np.nanstd(centroid_array, axis=0, ddof=0)
    # avoid div-by-zero when all clusters have same value for a feature
    global_std = np.where(global_std == 0, 1.0, global_std)
    z_scores = (centroid_array - global_mean) / global_std  # (n_clusters, n_features)

    def _tag(z: float) -> str:
        if z > 0.75:
            return "high"
        if z < -0.75:
            return "low"
        return "moderate"

    descriptions: Dict[int, str] = {}
    for i, cid in enumerate(cluster_ids):
        size_str = f" ({int(sizes[cid])} samples)" if cid in sizes else ""
        feature_parts: List[str] = []
        for j, feat in enumerate(feature_cols):
            mean_val = centroid_array[i, j]
            if np.isnan(mean_val):
                continue
            tag = _tag(z_scores[i, j])
            feature_parts.append(f"{tag} {feat} ({mean_val:.3g})")

        if feature_parts:
            body = ", ".join(feature_parts)
            desc = (
                f"Cluster {cid}{size_str}: characterised by {body}."
            )
        else:
            desc = f"Cluster {cid}{size_str}: no distinguishing features found."

        descriptions[int(cid)] = desc

    logger.info("describe_clusters: described %d clusters.", len(descriptions))
    return descriptions


# ---------------------------------------------------------------------------


def get_cluster_df(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """
    Append a human-readable ``"cluster_label"`` column to *df*.

    Maps integer cluster ids to labels of the form ``"Cluster 0"``,
    ``"Cluster 1"``, etc.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a numeric *cluster_col* column.
    cluster_col : str
        Name of the cluster-id column (default ``"cluster"``).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an additional ``"cluster_label"`` string column.
        Returns *df* unchanged (with a warning) when *cluster_col* is absent.
    """
    if cluster_col not in df.columns:
        logger.warning(
            "get_cluster_df: column '%s' not found — returning df unchanged.",
            cluster_col,
        )
        return df.copy()

    result = df.copy()
    result["cluster_label"] = (
        result[cluster_col]
        .astype(int)
        .map(lambda cid: f"{_CLUSTER_LABEL_PREFIX} {cid}")
    )
    return result


# ---------------------------------------------------------------------------
# Quick self-test  (python -m src.clustering)
# ---------------------------------------------------------------------------

def find_optimal_k(
    df: pd.DataFrame,
    feature_cols: List[str],
    k_range: range = range(2, 9),
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate k values using both inertia elbow curve and silhouette coefficients
    to recommend the optimal number of clusters k.
    """
    clean, valid_cols, err = _validate_and_clean(df, feature_cols)
    if err or clean.empty or len(valid_cols) < 2:
        return {"recommended_k": 3, "scores": [], "reason": "Insufficient features or data."}

    k_list = [k for k in k_range if k < len(clean)]
    if not k_list:
        return {"recommended_k": 2, "scores": [], "reason": "Data size too small."}

    elbow = compute_elbow(clean, valid_cols, k_range=k_list, random_state=random_state)
    scores = []
    best_sil = -2.0
    best_k = k_list[0]

    for k in k_list:
        sil = compute_silhouette(clean, valid_cols, n_clusters=k, random_state=random_state)
        scores.append({
            "k": k,
            "inertia": round(elbow.get(k, 0.0), 2),
            "silhouette": round(sil, 4) if sil is not None else 0.0,
        })
        if sil is not None and sil > best_sil:
            best_sil = sil
            best_k = k

    return {
        "recommended_k": best_k,
        "best_silhouette": round(best_sil, 3),
        "scores": scores,
        "features": valid_cols,
        "reason": f"Maximum Silhouette coefficient of {round(best_sil, 3)} achieved at k={best_k}.",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(0)
    _n = 90
    _demo = pd.DataFrame(
        {
            "pH": rng.normal(loc=[7, 8, 6], scale=0.3, size=(_n // 3, 3)).ravel(),
            "turbidity": rng.normal(
                loc=[2, 10, 20], scale=1.0, size=(_n // 3, 3)
            ).ravel(),
            "temperature": rng.normal(
                loc=[15, 25, 10], scale=2.0, size=(_n // 3, 3)
            ).ravel(),
        }
    )

    FEATS = ["pH", "turbidity", "temperature"]

    # run_kmeans
    result, labels = run_kmeans(_demo, FEATS, n_clusters=3)
    print(f"run_kmeans → shape={result.shape}, unique clusters={set(labels)}")

    # elbow
    elbow = compute_elbow(_demo, FEATS, k_range=range(2, 7))
    print(f"compute_elbow → {elbow}")

    # silhouette
    sil = compute_silhouette(_demo, FEATS, n_clusters=3)
    print(f"compute_silhouette(k=3) → {sil:.4f}")

    # cluster stats
    stats = get_cluster_stats(result, FEATS)
    print("get_cluster_stats →")
    print(stats)

    # describe
    descs = describe_clusters(stats)
    print("describe_clusters →")
    for cid, desc in descs.items():
        print(f"  {cid}: {desc}")

    # labelled df
    labelled = get_cluster_df(result)
    print("get_cluster_df (sample) →")
    print(labelled[["cluster", "cluster_label"]].value_counts())

    # --- edge cases ---
    print("\n--- Edge case: empty DataFrame ---")
    try:
        run_kmeans(pd.DataFrame(), FEATS, n_clusters=3)
    except ValueError as exc:
        print(f"  Caught: {exc}")

    print("--- Edge case: all-NaN feature ---")
    _nan_df = _demo.copy()
    _nan_df["pH"] = np.nan
    result2, _ = run_kmeans(_nan_df, FEATS, n_clusters=3)
    print(f"  shape={result2.shape}, cols used (NaN col dropped)=OK")

    print("--- Edge case: n_clusters > n_rows ---")
    result3, _ = run_kmeans(_demo.head(2), FEATS, n_clusters=10)
    print(f"  clamped clusters={result3['cluster'].nunique()}")

    print("--- Edge case: single cluster requested ---")
    result4, lbl4 = run_kmeans(_demo, FEATS, n_clusters=1)
    print(f"  clusters={set(lbl4)}")
