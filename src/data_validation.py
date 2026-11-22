"""
MarineMine Pro — Data Validation & Quality Module
==================================================
Comprehensive data quality analysis, outlier detection, range verification,
and quality scoring for marine pollution datasets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse missing values per column and overall.

    Returns dict with:
      - total_cells: int
      - missing_cells: int
      - missing_pct: float (0..100)
      - columns_with_missing: int
      - summary_df: pd.DataFrame [column, missing_count, missing_pct, dtype]
    """
    if df is None or df.empty:
        return {
            "total_cells": 0,
            "missing_cells": 0,
            "missing_pct": 0.0,
            "columns_with_missing": 0,
            "summary_df": pd.DataFrame(columns=["column", "missing_count", "missing_pct", "dtype"]),
        }

    total_cells = int(df.size)
    missing_series = df.isna().sum()
    missing_cells = int(missing_series.sum())
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    summary_rows = []
    for col in df.columns:
        cnt = int(missing_series[col])
        pct = round((cnt / len(df)) * 100, 2)
        summary_rows.append({
            "column": col,
            "missing_count": cnt,
            "missing_pct": pct,
            "dtype": str(df[col].dtype),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("missing_count", ascending=False).reset_index(drop=True)
    cols_with_missing = int((missing_series > 0).sum())

    return {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "columns_with_missing": cols_with_missing,
        "summary_df": summary_df,
    }


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse duplicate rows in the dataset.

    Returns dict with:
      - duplicate_count: int
      - duplicate_pct: float
      - unique_rows: int
      - example_duplicates: pd.DataFrame (up to 10 rows)
    """
    if df is None or df.empty:
        return {
            "duplicate_count": 0,
            "duplicate_pct": 0.0,
            "unique_rows": 0,
            "example_duplicates": pd.DataFrame(),
        }

    dup_mask = df.duplicated(keep=False)
    dup_count = int(df.duplicated(keep="first").sum())
    dup_pct = round((dup_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
    examples = df[dup_mask].head(10).copy()

    return {
        "duplicate_count": dup_count,
        "duplicate_pct": dup_pct,
        "unique_rows": len(df) - dup_count,
        "example_duplicates": examples,
    }


def analyze_numeric_outliers(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Dict[str, Any]:
    """
    Detect outliers in numerical columns using IQR or Z-score method.

    Returns dict mapping column -> dict with:
      - count: int
      - percentage: float
      - lower_bound: float
      - upper_bound: float
      - outlier_indices: list[int]
    """
    if df is None or df.empty:
        return {}

    num_cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
    results = {}

    for col in num_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 4:
            continue

        if method == "iqr":
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower = q25 - threshold * iqr
            upper = q75 + threshold * iqr
            outliers = series[(series < lower) | (series > upper)]
        else:  # z-score
            mean = float(series.mean())
            std = float(series.std()) if float(series.std()) > 0 else 1.0
            z_scores = (series - mean).abs() / std
            lower = mean - threshold * std
            upper = mean + threshold * std
            outliers = series[z_scores > threshold]

        results[col] = {
            "count": int(len(outliers)),
            "percentage": round((len(outliers) / len(df)) * 100, 2),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_indices": outliers.index.tolist()[:50],
        }

    return results


def analyze_categorical_distributions(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    max_categories: int = 20,
) -> Dict[str, Any]:
    """
    Analyse frequency distributions and cardinality for categorical columns.
    """
    if df is None or df.empty:
        return {}

    cat_cols = cols or df.select_dtypes(include=["object", "category"]).columns.tolist()
    results = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        cardinality = int(series.nunique())
        top_counts = series.value_counts().head(max_categories).to_dict()
        rare_count = int((series.value_counts() == 1).sum())

        results[col] = {
            "cardinality": cardinality,
            "top_categories": top_counts,
            "rare_categories_count": rare_count,
            "mode": series.mode().iloc[0] if not series.empty else None,
        }

    return results


def check_value_ranges(
    df: pd.DataFrame,
    range_rules: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Check if numeric columns violate known physical/domain ranges.
    Default domain ranges for oceanographic parameters:
      - latitude: (-90, 90)
      - longitude: (-180, 180)
      - temperature_celsius: (-5, 45)
      - salinity_ppt: (0, 50)
      - ph_level: (5.0, 9.5)
      - dissolved_oxygen: (0, 20)
      - pollution_value: (0, inf)
    """
    default_rules = {
        "latitude": (-90.0, 90.0),
        "longitude": (-180.0, 180.0),
        "temperature_celsius": (-5.0, 45.0),
        "salinity_ppt": (0.0, 50.0),
        "ph_level": (5.0, 9.5),
        "dissolved_oxygen": (0.0, 25.0),
        "pollution_value": (0.0, 10000.0),
        "water_depth_m": (0.0, 11500.0),
    }
    rules = range_rules or default_rules
    violations = {}

    for col, (min_val, max_val) in rules.items():
        matched_col = None
        for c in df.columns:
            if c.lower() == col.lower() or col.lower() in c.lower():
                matched_col = c
                break

        if matched_col and pd.api.types.is_numeric_dtype(df[matched_col]):
            s = df[matched_col].dropna()
            violating = s[(s < min_val) | (s > max_val)]
            if len(violating) > 0:
                violations[matched_col] = {
                    "expected_range": (min_val, max_val),
                    "violation_count": int(len(violating)),
                    "violation_pct": round((len(violating) / len(df)) * 100, 2),
                    "min_observed": float(s.min()),
                    "max_observed": float(s.max()),
                }

    return violations


def compute_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute an overall Data Quality Index (0 to 100) based on:
      1. Completeness (missingness penalty): 40 pts
      2. Uniqueness (duplicate penalty): 20 pts
      3. Validity (range violations penalty): 20 pts
      4. Consistency (data type sanity): 20 pts

    Returns dict with overall score, grade, and component breakdowns.
    """
    if df is None or df.empty:
        return {"overall_score": 0.0, "grade": "F", "breakdown": {}}

    # 1. Completeness (40%)
    missing_info = analyze_missing_values(df)
    missing_rate = missing_info["missing_pct"] / 100.0
    completeness_score = max(0.0, 40.0 * (1.0 - missing_rate * 2.0))

    # 2. Uniqueness (20%)
    dup_info = analyze_duplicates(df)
    dup_rate = dup_info["duplicate_pct"] / 100.0
    uniqueness_score = max(0.0, 20.0 * (1.0 - dup_rate * 3.0))

    # 3. Validity (20%)
    range_violations = check_value_ranges(df)
    total_violations = sum(v["violation_count"] for v in range_violations.values())
    validity_rate = total_violations / (len(df) * max(1, len(range_violations))) if range_violations else 0.0
    validity_score = max(0.0, 20.0 * (1.0 - validity_rate * 5.0))

    # 4. Consistency & Types (20%)
    # Ratio of columns successfully typed without all-null or mixed objects
    clean_cols = sum(1 for c in df.columns if df[c].notna().sum() > 0)
    consistency_score = 20.0 * (clean_cols / max(1, len(df.columns)))

    overall = round(completeness_score + uniqueness_score + validity_score + consistency_score, 1)
    overall = min(100.0, max(0.0, overall))

    if overall >= 90:
        grade = "A (Excellent)"
    elif overall >= 75:
        grade = "B (Good)"
    elif overall >= 60:
        grade = "C (Fair)"
    elif overall >= 40:
        grade = "D (Needs Preprocessing)"
    else:
        grade = "F (Poor)"

    return {
        "overall_score": overall,
        "grade": grade,
        "breakdown": {
            "completeness": round(completeness_score, 1),
            "uniqueness": round(uniqueness_score, 1),
            "validity": round(validity_score, 1),
            "consistency": round(consistency_score, 1),
        },
    }


def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Produce a complete, unified Data Quality Report dictionary.
    """
    if df is None or df.empty:
        return {"error": "Dataset is empty or None."}

    missing = analyze_missing_values(df)
    dups = analyze_duplicates(df)
    outliers = analyze_numeric_outliers(df)
    cats = analyze_categorical_distributions(df)
    violations = check_value_ranges(df)
    score_info = compute_quality_score(df)

    # Basic shape
    rows, cols = df.shape

    return {
        "rows": rows,
        "columns": cols,
        "quality_score": score_info["overall_score"],
        "quality_grade": score_info["grade"],
        "score_breakdown": score_info["breakdown"],
        "missing_summary": missing,
        "duplicates": dups,
        "outliers": outliers,
        "categorical_profiles": cats,
        "range_violations": violations,
    }
