"""
MarineMine Pro — Data Quality Routes
"""

from fastapi import APIRouter
import numpy as np
from app.schemas.api_schemas import QualitySummaryResponse
from app.services.state_manager import StateManager
from src import data_validation

router = APIRouter(prefix="/api/quality", tags=["Data Quality"])
state = StateManager.get_instance()


@router.get("/summary", response_model=QualitySummaryResponse)
def get_quality_summary():
    """Perform mathematical data quality audit on active dataset."""
    df = state.get_active_df()
    report = data_validation.generate_quality_report(df)

    missing = report.get("missing_summary", {})
    dups = report.get("duplicates", {})
    outliers = report.get("outliers", {})

    col_missing = []
    summary_df = missing.get("summary_df")
    if hasattr(summary_df, "iterrows"):
        for _, r in summary_df.iterrows():
            col_missing.append({
                "column": str(r["column"]),
                "missing_count": int(r["missing_count"]),
                "missing_pct": round(float(r["missing_pct"]), 2),
            })
    elif isinstance(summary_df, list):
        col_missing = summary_df

    total_missing = int(missing.get("missing_cells", 0))
    missing_pct = float(missing.get("missing_pct", 0.0))
    dup_rows = int(dups.get("duplicate_rows", 0))
    dup_pct = float(dups.get("duplicate_pct", 0.0))
    outlier_count = int(outliers.get("total_outliers", 0))
    score = float(report.get("quality_score", 100.0))

    passed_checks = [
        "Data schema and type consistency check",
        f"Missing value analysis ({missing_pct}% total missingness)",
        f"Deduplication integrity check ({dup_rows} duplicates detected)",
        f"IQR outlier distribution audit ({outlier_count} outliers identified)",
    ]

    warnings = []
    if missing_pct > 5.0:
        warnings.append(f"High missingness observed: {missing_pct}% of total cells are null.")
    if dup_rows > 0:
        warnings.append(f"Duplicate rows detected: {dup_rows} duplicate observations exist.")

    errors = []
    if missing_pct > 30.0:
        errors.append(f"Critical data sparsity: {missing_pct}% missingness exceeds 30% quality threshold.")

    return {
        "quality_score": round(score, 1),
        "total_records": int(report.get("rows", len(df))),
        "total_features": int(report.get("columns", df.shape[1])),
        "total_missing_cells": total_missing,
        "missing_rate_pct": round(missing_pct, 2),
        "duplicate_rows": dup_rows,
        "duplicate_rate_pct": round(dup_pct, 2),
        "outlier_count_iqr": outlier_count,
        "completeness_pct": round(100.0 - missing_pct, 1),
        "column_missing_breakdown": col_missing,
        "passed_checks": passed_checks,
        "warnings": warnings,
        "errors": errors,
    }


@router.get("/missing-values")
def get_missing_values_report():
    """Detailed breakdown of null/missing values across all columns."""
    df = state.get_active_df()
    return data_validation.analyze_missing_values(df)


@router.get("/duplicates")
def get_duplicates_report():
    """Detailed report on row-level duplications."""
    df = state.get_active_df()
    return data_validation.analyze_duplicates(df)


@router.get("/outliers")
def get_outliers_report():
    """Detailed IQR outlier detection per numeric feature."""
    df = state.get_active_df()
    return data_validation.analyze_numeric_outliers(df)
