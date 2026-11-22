"""
MarineMine Pro — Hotspot Analysis Module
=========================================
Geographical hotspot analysis for marine pollution data.

Provides tools for:
  - Coordinate validation and cleaning
  - Grid-based pollution hotspot computation
  - Regional pollution summaries
  - Flexible filtering (type, date range, region)
  - Qualitative severity ranking of hotspots
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0

SEVERITY_HIGH_PCT = 0.75    # top 25 % of ranked hotspots → High
SEVERITY_LOW_PCT = 0.40     # bottom 40 % of ranked hotspots → Low
                             # middle 35 % → Medium


# ---------------------------------------------------------------------------
# 1. Coordinate Validation
# ---------------------------------------------------------------------------

def validate_coordinates(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
) -> Tuple[pd.DataFrame, int]:
    """
    Validate and clean geographic coordinates in *df*.

    Rules applied
    -------------
    - Coerce both columns to numeric; non-parseable values become NaN.
    - Drop rows where latitude or longitude is NaN.
    - Drop rows where latitude  ∉ [-90, 90].
    - Drop rows where longitude ∉ [-180, 180].

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that must contain *lat_col* and *lon_col*.
    lat_col : str
        Name of the latitude column.
    lon_col : str
        Name of the longitude column.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Copy of *df* retaining only rows with valid coordinates.
        The index is reset.
    error_count : int
        Number of rows that were removed due to invalid coordinates.

    Raises
    ------
    KeyError
        If *lat_col* or *lon_col* is not present in *df*.
    ValueError
        If *df* is empty.

    Examples
    --------
    >>> clean, errors = validate_coordinates(df, "latitude", "longitude")
    >>> print(f"Removed {errors} bad rows; {len(clean)} remain.")
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty — no geo data available to validate.")

    for col in (lat_col, lon_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    result = df.copy()
    original_len = len(result)

    # Coerce to numeric
    result[lat_col] = pd.to_numeric(result[lat_col], errors="coerce")
    result[lon_col] = pd.to_numeric(result[lon_col], errors="coerce")

    # Drop NaN coords
    result.dropna(subset=[lat_col, lon_col], inplace=True)

    # Drop out-of-range coords
    valid_mask = (
        result[lat_col].between(LAT_MIN, LAT_MAX) &
        result[lon_col].between(LON_MIN, LON_MAX)
    )
    result = result[valid_mask]

    error_count = original_len - len(result)

    if error_count > 0:
        logger.warning(
            "validate_coordinates: removed %d invalid coordinate row(s) "
            "out of %d total.",
            error_count,
            original_len,
        )

    if result.empty:
        logger.error(
            "validate_coordinates: ALL rows were removed. "
            "No valid geo data available. Check '%s' and '%s' columns.",
            lat_col,
            lon_col,
        )

    result.reset_index(drop=True, inplace=True)
    return result, error_count


# ---------------------------------------------------------------------------
# 2. Hotspot Computation
# ---------------------------------------------------------------------------

def compute_hotspots(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    value_col: str,
    grid_size: float = 1.0,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Aggregate pollution values onto a regular geographic grid and identify
    the top-N hotspot cells.

    Algorithm
    ---------
    Each observation is assigned to the nearest grid cell by flooring its
    latitude/longitude to the nearest multiple of *grid_size*:

        grid_lat = floor(lat / grid_size) * grid_size
        grid_lon = floor(lon / grid_size) * grid_size

    Cells are then ranked by *mean_value* (descending).

    Parameters
    ----------
    df : pd.DataFrame
        Input observations.  Must already have validated coordinates
        (use :func:`validate_coordinates` first).
    lat_col : str
        Name of the latitude column.
    lon_col : str
        Name of the longitude column.
    value_col : str
        Numeric pollution/concentration column to aggregate.
    grid_size : float, optional
        Cell size in degrees (default ``1.0``).  Smaller values produce
        finer grids; must be > 0.
    top_n : int, optional
        Number of top hotspot cells to return (default ``10``).

    Returns
    -------
    pd.DataFrame
        Columns: ``grid_lat``, ``grid_lon``, ``mean_value``, ``count``,
        ``max_value``, ``hotspot_rank``.
        Sorted by ``hotspot_rank`` ascending (rank 1 = worst hotspot).
        Returns an empty DataFrame with these columns when no valid data
        is available.

    Raises
    ------
    KeyError
        If any required column is absent.
    ValueError
        If *grid_size* ≤ 0 or *top_n* ≤ 0.

    Examples
    --------
    >>> hotspots = compute_hotspots(clean_df, "lat", "lon", "pollution_index",
    ...                             grid_size=0.5, top_n=5)
    """
    _RESULT_COLS = ["grid_lat", "grid_lon", "mean_value", "count", "max_value", "hotspot_rank"]

    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive; got {grid_size}.")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}.")

    for col in (lat_col, lon_col, value_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    if df.empty:
        logger.warning(
            "compute_hotspots: DataFrame is empty — no geo data available. "
            "Returning empty hotspot table."
        )
        return pd.DataFrame(columns=_RESULT_COLS)

    work = df[[lat_col, lon_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work.dropna(subset=[lat_col, lon_col, value_col], inplace=True)

    if work.empty:
        logger.warning(
            "compute_hotspots: All rows dropped after coercing '%s' to numeric. "
            "No geo data available for hotspot computation.",
            value_col,
        )
        return pd.DataFrame(columns=_RESULT_COLS)

    # Assign grid cells
    work["grid_lat"] = np.floor(work[lat_col] / grid_size) * grid_size
    work["grid_lon"] = np.floor(work[lon_col] / grid_size) * grid_size

    # Aggregate
    agg = (
        work.groupby(["grid_lat", "grid_lon"])[value_col]
        .agg(mean_value="mean", count="count", max_value="max")
        .reset_index()
    )

    # Rank by mean_value descending; limit to top_n
    agg.sort_values("mean_value", ascending=False, inplace=True)
    agg = agg.head(top_n).copy()
    agg["hotspot_rank"] = range(1, len(agg) + 1)

    agg.reset_index(drop=True, inplace=True)
    logger.info(
        "compute_hotspots: identified %d hotspot cell(s) with grid_size=%.2f°.",
        len(agg),
        grid_size,
    )
    return agg[_RESULT_COLS]


# ---------------------------------------------------------------------------
# 3. Regional Pollution Summary
# ---------------------------------------------------------------------------

def get_regional_pollution_summary(
    df: pd.DataFrame,
    region_col: str,
    value_col: str,
    type_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute per-region pollution statistics and, optionally, the dominant
    pollution type per region.

    Parameters
    ----------
    df : pd.DataFrame
        Input observations.
    region_col : str
        Column that identifies the geographic region (e.g. ocean basin,
        country, marine protected area name).
    value_col : str
        Numeric pollution measurement column.
    type_col : str or None, optional
        Categorical pollution-type column.  When supplied, the most frequent
        type per region is added as ``dominant_type``.

    Returns
    -------
    pd.DataFrame
        One row per region.  Columns:

        ``region``, ``mean_value``, ``median_value``, ``max_value``,
        ``min_value``, ``std_value``, ``count``
        [, ``dominant_type``  — only when *type_col* is provided]

        Sorted by ``mean_value`` descending.

    Raises
    ------
    KeyError
        If *region_col*, *value_col*, or (when given) *type_col* is absent.

    Examples
    --------
    >>> summary = get_regional_pollution_summary(
    ...     df, "ocean_region", "pollution_index", type_col="debris_type"
    ... )
    """
    for col in [region_col, value_col] + ([type_col] if type_col else []):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    if df.empty:
        logger.warning(
            "get_regional_pollution_summary: DataFrame is empty — "
            "no geo data available."
        )
        cols = ["region", "mean_value", "median_value", "max_value",
                "min_value", "std_value", "count"]
        if type_col:
            cols.append("dominant_type")
        return pd.DataFrame(columns=cols)

    work = df[[region_col, value_col] + ([type_col] if type_col else [])].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work.dropna(subset=[region_col, value_col], inplace=True)

    agg = (
        work.groupby(region_col)[value_col]
        .agg(
            mean_value="mean",
            median_value="median",
            max_value="max",
            min_value="min",
            std_value="std",
            count="count",
        )
        .reset_index()
        .rename(columns={region_col: "region"})
    )

    if type_col:
        def _dominant(series: pd.Series) -> str:
            counts = series.value_counts()
            return counts.index[0] if not counts.empty else "Unknown"

        dominant = (
            work.dropna(subset=[type_col])
            .groupby(region_col)[type_col]
            .apply(_dominant)
            .reset_index()
            .rename(columns={region_col: "region", type_col: "dominant_type"})
        )
        agg = agg.merge(dominant, on="region", how="left")

    agg.sort_values("mean_value", ascending=False, inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg


# ---------------------------------------------------------------------------
# 4. Filter Helpers
# ---------------------------------------------------------------------------

def filter_by_pollution_type(
    df: pd.DataFrame,
    type_col: str,
    selected_types: List[str],
) -> pd.DataFrame:
    """
    Keep only rows whose pollution type is in *selected_types*.

    Parameters
    ----------
    df : pd.DataFrame
        Input observations.
    type_col : str
        Column containing pollution type labels.
    selected_types : list of str
        Types to retain.  Matching is **case-insensitive**.

    Returns
    -------
    pd.DataFrame
        Filtered copy with reset index.  Returns empty DataFrame (same
        columns) if no rows match or *selected_types* is empty.

    Raises
    ------
    KeyError
        If *type_col* is absent.

    Examples
    --------
    >>> plastics_df = filter_by_pollution_type(
    ...     df, "debris_type", ["Microplastics", "Macroplastics"]
    ... )
    """
    if type_col not in df.columns:
        raise KeyError(
            f"Column '{type_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if not selected_types:
        logger.warning("filter_by_pollution_type: selected_types is empty; returning empty DataFrame.")
        return df.iloc[0:0].copy()

    normalised_types = {t.strip().lower() for t in selected_types}
    mask = df[type_col].astype(str).str.strip().str.lower().isin(normalised_types)
    result = df[mask].copy()
    result.reset_index(drop=True, inplace=True)

    logger.debug(
        "filter_by_pollution_type: kept %d / %d rows for types %s.",
        len(result),
        len(df),
        selected_types,
    )
    return result


def filter_by_date_range(
    df: pd.DataFrame,
    date_col: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Keep only rows whose date falls within [*start_date*, *end_date*]
    (both boundaries inclusive).

    Parameters
    ----------
    df : pd.DataFrame
        Input observations.
    date_col : str
        Column containing date/datetime values (will be coerced with
        ``pd.to_datetime``).
    start_date : str
        Lower bound, e.g. ``"2023-01-01"``.
    end_date : str
        Upper bound, e.g. ``"2023-12-31"``.

    Returns
    -------
    pd.DataFrame
        Filtered copy with reset index.

    Raises
    ------
    KeyError
        If *date_col* is absent.
    ValueError
        If *start_date* > *end_date* after parsing.

    Examples
    --------
    >>> q1 = filter_by_date_range(df, "sample_date", "2023-01-01", "2023-03-31")
    """
    if date_col not in df.columns:
        raise KeyError(
            f"Column '{date_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    if start > end:
        raise ValueError(
            f"start_date ({start_date}) must be ≤ end_date ({end_date})."
        )

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    mask = work[date_col].between(start, end)
    result = work[mask].copy()
    result.reset_index(drop=True, inplace=True)

    logger.debug(
        "filter_by_date_range: kept %d / %d rows between %s and %s.",
        len(result),
        len(df),
        start_date,
        end_date,
    )
    return result


def filter_by_region(
    df: pd.DataFrame,
    region_col: str,
    selected_regions: List[str],
) -> pd.DataFrame:
    """
    Keep only rows whose region is in *selected_regions*.

    Parameters
    ----------
    df : pd.DataFrame
        Input observations.
    region_col : str
        Column containing region labels.
    selected_regions : list of str
        Regions to retain.  Matching is **case-insensitive**.

    Returns
    -------
    pd.DataFrame
        Filtered copy with reset index.  Returns empty DataFrame (same
        columns) if no rows match.

    Raises
    ------
    KeyError
        If *region_col* is absent.

    Examples
    --------
    >>> pacific = filter_by_region(df, "ocean_region",
    ...                            ["North Pacific", "South Pacific"])
    """
    if region_col not in df.columns:
        raise KeyError(
            f"Column '{region_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if not selected_regions:
        logger.warning("filter_by_region: selected_regions is empty; returning empty DataFrame.")
        return df.iloc[0:0].copy()

    normalised = {r.strip().lower() for r in selected_regions}
    mask = df[region_col].astype(str).str.strip().str.lower().isin(normalised)
    result = df[mask].copy()
    result.reset_index(drop=True, inplace=True)

    logger.debug(
        "filter_by_region: kept %d / %d rows for regions %s.",
        len(result),
        len(df),
        selected_regions,
    )
    return result


# ---------------------------------------------------------------------------
# 5. Hotspot Severity Ranking
# ---------------------------------------------------------------------------

def rank_hotspots(hotspot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment a hotspot DataFrame (from :func:`compute_hotspots`) with a
    qualitative severity label.

    Severity thresholds (based on ``hotspot_rank`` percentile position):

    +---------+--------------------------------------+
    | Label   | Condition                            |
    +=========+======================================+
    | High    | top 25 % (lowest rank numbers)       |
    +---------+--------------------------------------+
    | Medium  | middle 35 %                          |
    +---------+--------------------------------------+
    | Low     | bottom 40 % (highest rank numbers)   |
    +---------+--------------------------------------+

    Parameters
    ----------
    hotspot_df : pd.DataFrame
        DataFrame returned by :func:`compute_hotspots`.  Must contain
        ``hotspot_rank`` and ``mean_value`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *hotspot_df* with an additional ``severity`` column
        (``"High"`` / ``"Medium"`` / ``"Low"``).
        Returns an empty DataFrame with the ``severity`` column appended
        when *hotspot_df* is empty.

    Raises
    ------
    KeyError
        If ``hotspot_rank`` or ``mean_value`` is not in *hotspot_df*.

    Examples
    --------
    >>> ranked = rank_hotspots(hotspots)
    >>> ranked[["hotspot_rank", "mean_value", "severity"]].head()
    """
    for col in ("hotspot_rank", "mean_value"):
        if col not in hotspot_df.columns:
            raise KeyError(
                f"Column '{col}' is required by rank_hotspots but was not found. "
                f"Available columns: {list(hotspot_df.columns)}"
            )

    result = hotspot_df.copy()

    if result.empty:
        result["severity"] = pd.Series(dtype="str")
        return result

    n = len(result)

    # Sort by rank ascending so rank 1 is first
    result.sort_values("hotspot_rank", ascending=True, inplace=True)
    result.reset_index(drop=True, inplace=True)

    # Compute percentile cut points on the rank position (1-indexed position)
    high_cut = max(1, int(np.ceil(n * (1.0 - SEVERITY_HIGH_PCT))))   # top 25 %
    low_cut  = max(1, int(np.floor(n * (1.0 - SEVERITY_LOW_PCT))))    # bottom 40 %

    def _label(pos: int) -> str:
        """pos is 0-indexed position after sort."""
        if pos < high_cut:
            return "High"
        if pos >= low_cut:
            return "Low"
        return "Medium"

    result["severity"] = [_label(i) for i in range(n)]

    logger.info(
        "rank_hotspots: assigned severity labels — "
        "High: %d, Medium: %d, Low: %d.",
        (result["severity"] == "High").sum(),
        (result["severity"] == "Medium").sum(),
        (result["severity"] == "Low").sum(),
    )
    return result


# ---------------------------------------------------------------------------
# Module-level convenience export
# ---------------------------------------------------------------------------

__all__ = [
    "validate_coordinates",
    "compute_hotspots",
    "get_regional_pollution_summary",
    "filter_by_pollution_type",
    "filter_by_date_range",
    "filter_by_region",
    "rank_hotspots",
]


# ---------------------------------------------------------------------------
# Quick smoke-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(name)s | %(message)s")

    # --- synthetic data ---
    rng = np.random.default_rng(42)
    N = 200

    sample_data = pd.DataFrame({
        "latitude":       rng.uniform(-80, 80, N).tolist() + [999, None, "bad"],
        "longitude":      rng.uniform(-170, 170, N).tolist() + [-999, 10, "x"],
        "pollution_index": rng.exponential(scale=5, size=N + 3).tolist(),
        "debris_type":    rng.choice(
            ["Microplastics", "Macroplastics", "Oil", "Chemical", "Biological"],
            size=N + 3,
        ).tolist(),
        "ocean_region":   rng.choice(
            ["North Pacific", "South Pacific", "Atlantic", "Indian Ocean"],
            size=N + 3,
        ).tolist(),
        "sample_date":    pd.date_range("2022-01-01", periods=N + 3, freq="D").tolist(),
    })

    print("\n=== validate_coordinates ===")
    clean_df, errors = validate_coordinates(sample_data, "latitude", "longitude")
    print(f"  Errors removed : {errors}")
    print(f"  Clean rows     : {len(clean_df)}")

    print("\n=== compute_hotspots ===")
    hotspots = compute_hotspots(clean_df, "latitude", "longitude", "pollution_index",
                                grid_size=10.0, top_n=5)
    print(hotspots.to_string(index=False))

    print("\n=== rank_hotspots ===")
    ranked = rank_hotspots(hotspots)
    print(ranked[["hotspot_rank", "grid_lat", "grid_lon", "mean_value", "severity"]].to_string(index=False))

    print("\n=== get_regional_pollution_summary ===")
    summary = get_regional_pollution_summary(
        clean_df, "ocean_region", "pollution_index", type_col="debris_type"
    )
    print(summary.to_string(index=False))

    print("\n=== filter_by_pollution_type ===")
    plastics = filter_by_pollution_type(clean_df, "debris_type", ["Microplastics", "Oil"])
    print(f"  Rows after type filter : {len(plastics)}")

    print("\n=== filter_by_date_range ===")
    q1 = filter_by_date_range(clean_df, "sample_date", "2022-01-01", "2022-03-31")
    print(f"  Rows in Q1 2022        : {len(q1)}")

    print("\n=== filter_by_region ===")
    pacific = filter_by_region(clean_df, "ocean_region", ["North Pacific", "South Pacific"])
    print(f"  Rows in Pacific regions: {len(pacific)}")
