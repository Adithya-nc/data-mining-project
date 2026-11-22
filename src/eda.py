"""
MarineMine Pro — Exploratory Data Analysis (EDA) Module
========================================================
Analytical routines for computing distributions, trends, correlations,
regional patterns, and key scientific findings from marine pollution data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return extended descriptive statistics for all numeric columns,
    including mean, median, std, min, max, IQR, skewness, and kurtosis.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()

    rows = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if s.empty:
            continue
        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        rows.append({
            "feature": col,
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4) if len(s) > 1 else 0.0,
            "min": round(float(s.min()), 4),
            "q25": round(q25, 4),
            "median": round(float(s.median()), 4),
            "q75": round(q75, 4),
            "max": round(float(s.max()), 4),
            "iqr": round(q75 - q25, 4),
            "skewness": round(float(s.skew()), 4) if len(s) > 2 else 0.0,
            "kurtosis": round(float(s.kurtosis()), 4) if len(s) > 3 else 0.0,
        })

    return pd.DataFrame(rows)


def get_categorical_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return summary dictionaries for all categorical and text columns.
    """
    if df is None or df.empty:
        return {}

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    summary = {}
    for col in cat_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        summary[col] = {
            "nunique": int(s.nunique()),
            "mode": s.mode().iloc[0] if not s.empty else None,
            "top_categories": s.value_counts().head(10).to_dict(),
        }
    return summary


def analyze_pollution_trends(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Resample time-series data to compute periodic trend statistics.
    freq: 'ME' (Monthly), 'QE' (Quarterly), 'YE' (Yearly), or 'W' (Weekly)
    """
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    sub = df[[date_col, value_col]].dropna().copy()
    try:
        sub[date_col] = pd.to_datetime(sub[date_col])
    except Exception as e:
        logger.warning(f"Could not convert {date_col} to datetime: {e}")
        return pd.DataFrame()

    sub = sub.set_index(date_col).sort_index()

    # pandas >= 2.2 uses ME instead of M
    freq_clean = freq
    if freq == "M":
        freq_clean = "ME"
    elif freq == "Q":
        freq_clean = "QE"
    elif freq == "Y":
        freq_clean = "YE"

    try:
        resampled = sub[value_col].resample(freq_clean).agg(
            mean_value="mean",
            median_value="median",
            max_value="max",
            min_value="min",
            count="count"
        ).reset_index()
    except Exception:
        # Fallback to simple monthly grouping if resample fails
        sub["year_month"] = sub.index.to_period("M").astype(str)
        resampled = sub.groupby("year_month")[value_col].agg(
            mean_value="mean",
            median_value="median",
            max_value="max",
            min_value="min",
            count="count"
        ).reset_index().rename(columns={"year_month": date_col})

    return resampled.round(3)


def analyze_regional_distribution(
    df: pd.DataFrame,
    region_col: str,
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Group by geographical region and compute record count and mean/median pollution value.
    """
    if df is None or df.empty or region_col not in df.columns:
        return pd.DataFrame()

    if value_col and value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
        grouped = df.groupby(region_col).agg(
            record_count=(region_col, "count"),
            mean_pollution=(value_col, "mean"),
            median_pollution=(value_col, "median"),
            max_pollution=(value_col, "max"),
            min_pollution=(value_col, "min"),
        ).reset_index()
        grouped = grouped.sort_values("mean_pollution", ascending=False).reset_index(drop=True)
    else:
        grouped = df[region_col].value_counts().reset_index()
        grouped.columns = [region_col, "record_count"]

    return grouped.round(3)


def analyze_pollution_type_distribution(
    df: pd.DataFrame,
    type_col: str,
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Group by pollution category and compute frequency and average values.
    """
    if df is None or df.empty or type_col not in df.columns:
        return pd.DataFrame()

    if value_col and value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
        grouped = df.groupby(type_col).agg(
            record_count=(type_col, "count"),
            mean_value=(value_col, "mean"),
            median_value=(value_col, "median"),
            max_value=(value_col, "max"),
            min_value=(value_col, "min"),
        ).reset_index()
        grouped["pct_records"] = round((grouped["record_count"] / len(df)) * 100, 2)
        grouped = grouped.sort_values("record_count", ascending=False).reset_index(drop=True)
    else:
        grouped = df[type_col].value_counts().reset_index()
        grouped.columns = [type_col, "record_count"]
        grouped["pct_records"] = round((grouped["record_count"] / len(df)) * 100, 2)

    return grouped.round(3)


def compute_correlations(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix for numerical features.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    target_cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
    valid_cols = [c for c in target_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if len(valid_cols) < 2:
        return pd.DataFrame()

    corr_df = df[valid_cols].corr(method=method)
    return corr_df.round(3)


def identify_top_polluted_regions(
    df: pd.DataFrame,
    region_col: str,
    value_col: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return top N regions with highest average pollution concentration.
    """
    reg_df = analyze_regional_distribution(df, region_col, value_col)
    if "mean_pollution" in reg_df.columns:
        return reg_df.head(top_n)
    return reg_df.head(top_n)


def get_seasonal_analysis(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> Dict[str, Any]:
    """
    Aggregate pollution measurements by season and month to observe cyclical variation.
    """
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        return {}

    sub = df[[date_col, value_col]].dropna().copy()
    try:
        dates = pd.to_datetime(sub[date_col])
        sub["month"] = dates.dt.month
        sub["month_name"] = dates.dt.month_name()

        # Define meteorological seasons
        season_map = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn"
        }
        sub["season"] = sub["month"].map(season_map)

        season_stats = sub.groupby("season")[value_col].agg(["mean", "median", "count"]).round(3).to_dict("index")
        month_stats = sub.groupby("month_name")[value_col].agg(["mean", "median", "count"]).round(3).to_dict("index")

        return {
            "by_season": season_stats,
            "by_month": month_stats,
        }
    except Exception as e:
        logger.warning(f"Seasonal analysis failed: {e}")
        return {}


def get_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compile a high-level summary of EDA findings for dashboards and reports.
    Automatically detects relevant columns.
    """
    if df is None or df.empty:
        return {}

    # Detect candidate columns
    cols_lower = {c.lower(): c for c in df.columns}

    # Value col
    val_col = None
    for cand in ["pollution_value", "concentration", "measurement", "value", "level"]:
        if cand in cols_lower and pd.api.types.is_numeric_dtype(df[cols_lower[cand]]):
            val_col = cols_lower[cand]
            break

    # Region col
    reg_col = None
    for cand in ["region", "ocean", "sea", "location", "area", "basin"]:
        if cand in cols_lower:
            reg_col = cols_lower[cand]
            break

    # Type col
    type_col = None
    for cand in ["pollution_type", "type", "category", "pollutant", "source"]:
        if cand in cols_lower:
            type_col = cols_lower[cand]
            break

    # Date col
    date_col = None
    for cand in ["observation_date", "date", "timestamp", "datetime", "survey_date"]:
        if cand in cols_lower:
            date_col = cols_lower[cand]
            break

    summary = {
        "detected_value_col": val_col,
        "detected_region_col": reg_col,
        "detected_type_col": type_col,
        "detected_date_col": date_col,
        "total_records": len(df),
        "total_features": len(df.columns),
    }

    if val_col:
        s = df[val_col].dropna()
        summary["pollution_stats"] = {
            "mean": round(float(s.mean()), 2) if not s.empty else 0.0,
            "median": round(float(s.median()), 2) if not s.empty else 0.0,
            "max": round(float(s.max()), 2) if not s.empty else 0.0,
            "min": round(float(s.min()), 2) if not s.empty else 0.0,
        }

    if type_col:
        summary["top_pollution_types"] = df[type_col].value_counts().head(5).to_dict()

    if reg_col and val_col:
        top_reg = identify_top_polluted_regions(df, reg_col, val_col, top_n=3)
        summary["top_polluted_regions"] = top_reg.to_dict("records")

    return summary
