"""
data_loader.py
==============
MarineMine Pro — Marine Pollution Data Mining & ML Platform
Senior Python Developer implementation.

Provides robust CSV loading, validation, column-role inference,
dataset metadata, and comprehensive summary statistics.
"""

from __future__ import annotations

import io
import os
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Location of the bundled sample dataset, relative to this file's project root.
# Resolved at runtime so the module works regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_CSV = _PROJECT_ROOT / "data" / "sample" / "marine_pollution_sample.csv"

# Column-name patterns for automatic role detection (all matched case-insensitively)
_DATE_PATTERNS = re.compile(
    r"\b(date|time|timestamp|datetime|year|month|day|collected_at|sampled_at|recorded_at|"
    r"survey_date|sample_date|observation_date|report_date)\b",
    re.IGNORECASE,
)
_LAT_PATTERNS = re.compile(
    r"\b(lat|latitude|lat_deg|y_coord|northing)\b", re.IGNORECASE
)
_LON_PATTERNS = re.compile(
    r"\b(lon|long|longitude|lng|lon_deg|x_coord|easting)\b", re.IGNORECASE
)
_TARGET_PATTERNS = re.compile(
    r"\b(pollution|pollutant|contamination|toxicity|concentration|level|index|"
    r"score|label|class|target|outcome|category|severity|risk|status)\b",
    re.IGNORECASE,
)

# Columns that *should* be present for a meaningful marine pollution dataset
_RECOMMENDED_COLUMNS: list[str] = [
    "latitude", "longitude", "date",
    "pollutant", "concentration", "location",
]

# Maximum file size accepted for uploads (50 MB)
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Cardinality threshold: columns with <= this fraction of unique values are
# treated as categorical (provided they are not numeric/date).
_CAT_CARDINALITY_RATIO = 0.05
# Hard upper bound on unique values to still call something categorical
_CAT_MAX_UNIQUE = 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_sample_dataset() -> pd.DataFrame:
    """Load the bundled sample marine-pollution CSV.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame with best-effort dtype inference.

    Raises
    ------
    FileNotFoundError
        If the sample CSV is not found at the expected path.
    ValueError
        If the file is empty or cannot be parsed.
    """
    if not _SAMPLE_CSV.exists():
        raise FileNotFoundError(
            f"Sample dataset not found at '{_SAMPLE_CSV}'. "
            "Please ensure the data/sample/ directory is present in the project root."
        )

    try:
        df = pd.read_csv(
            _SAMPLE_CSV,
            parse_dates=True,       # let pandas attempt date inference on obvious cols
            infer_datetime_format=True,
            on_bad_lines="warn",
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Sample CSV is empty: {_SAMPLE_CSV}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to parse sample CSV: {exc}") from exc

    df = _coerce_dates(df)
    return df


def load_uploaded_dataset(uploaded_file: Any) -> tuple[pd.DataFrame | None, str | None]:
    """Validate and load a user-uploaded CSV file.

    Parameters
    ----------
    uploaded_file:
        A file-like object (e.g. ``st.file_uploader`` result), a file path
        ``str``/``Path``, or raw ``bytes``.

    Returns
    -------
    (df, error_msg) : tuple
        On success, ``(DataFrame, None)``.
        On failure, ``(None, human-readable error string)``.
    """
    try:
        raw, source_name = _resolve_upload_source(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not read uploaded file: {exc}"

    # Size check
    if len(raw) == 0:
        return None, "Uploaded file is empty."
    if len(raw) > _MAX_UPLOAD_BYTES:
        mb = len(raw) / (1024 * 1024)
        return None, (
            f"File is too large ({mb:.1f} MB). "
            f"Maximum allowed size is {_MAX_UPLOAD_BYTES // (1024*1024)} MB."
        )

    # Encoding sniff
    encoding = _detect_encoding(raw)

    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            encoding=encoding,
            parse_dates=True,
            infer_datetime_format=True,
            on_bad_lines="warn",
        )
    except UnicodeDecodeError:
        # Retry with latin-1 as a fallback
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                encoding="latin-1",
                parse_dates=True,
                on_bad_lines="warn",
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"Encoding error — could not decode file: {exc}"
    except pd.errors.EmptyDataError:
        return None, "Uploaded CSV is empty or has no columns."
    except pd.errors.ParserError as exc:
        return None, f"CSV parsing error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error loading file: {exc}"

    if df.empty:
        return None, "Uploaded file contains no data rows."

    df = _coerce_dates(df)
    return df, None


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate a DataFrame for marine-pollution analysis suitability.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys:
        - ``is_valid`` (bool): True if no hard errors.
        - ``errors`` (list[str]): Hard blockers.
        - ``warnings`` (list[str]): Soft recommendations.
        - ``passed_checks`` (list[str]): Things that look good.
    """
    errors: list[str] = []
    warnings: list[str] = []
    passed: list[str] = []

    # --- Hard checks ---
    if df is None or not isinstance(df, pd.DataFrame):
        return {
            "is_valid": False,
            "errors": ["Input is not a pandas DataFrame."],
            "warnings": [],
            "passed_checks": [],
        }

    if df.empty:
        errors.append("Dataset is completely empty (no rows and/or no columns).")
        return {"is_valid": False, "errors": errors, "warnings": warnings, "passed_checks": passed}

    if df.shape[0] < 1:
        errors.append("Dataset has no data rows.")
    else:
        passed.append(f"Dataset has {df.shape[0]:,} rows.")

    if df.shape[1] < 2:
        errors.append("Dataset has fewer than 2 columns — too little information.")
    else:
        passed.append(f"Dataset has {df.shape[1]} columns.")

    # Duplicate columns
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        errors.append(f"Duplicate column names detected: {dup_cols}. Please deduplicate.")

    # All-null columns
    all_null = [c for c in df.columns if df[c].isna().all()]
    if all_null:
        warnings.append(f"Completely empty columns (all NaN): {all_null}. Consider dropping them.")

    # --- Soft checks (recommended columns) ---
    lower_cols = {c.lower() for c in df.columns}
    missing_recommended = []
    for rec in _RECOMMENDED_COLUMNS:
        # fuzzy match: check if any column contains the keyword
        if not any(rec in c for c in lower_cols):
            missing_recommended.append(rec)
    if missing_recommended:
        warnings.append(
            f"Recommended column concept(s) not found: {missing_recommended}. "
            "The platform works best with latitude, longitude, date, and pollutant columns."
        )
    else:
        passed.append("All recommended column concepts are present.")

    # --- Numeric columns check ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        warnings.append("No numeric columns detected — ML features will be limited.")
    else:
        passed.append(f"{len(numeric_cols)} numeric column(s) found.")

    # --- Missing-value rate ---
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0
    if missing_pct > 50:
        warnings.append(
            f"High missing-value rate: {missing_pct:.1f}% of all cells are NaN. "
            "Consider imputation or filtering before modelling."
        )
    elif missing_pct > 20:
        warnings.append(f"Moderate missing-value rate: {missing_pct:.1f}%.")
    else:
        passed.append(f"Missing-value rate is acceptable: {missing_pct:.1f}%.")

    # --- Duplicate rows ---
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        pct = dup_rows / len(df) * 100
        warnings.append(f"{dup_rows:,} duplicate rows detected ({pct:.1f}%). Consider deduplication.")
    else:
        passed.append("No duplicate rows detected.")

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "passed_checks": passed,
    }


def get_dataset_info(df: pd.DataFrame, dataset_name: Optional[str] = None) -> dict[str, Any]:
    """Return comprehensive metadata about a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    dataset_name : Optional[str]

    Returns
    -------
    dict with keys:
        dataset_name, shape, n_rows, total_rows, n_cols, total_columns,
        dtypes, missing_counts, missing_per_col, missing_pct,
        memory_usage_bytes, memory_usage_human, memory_usage_mb, date_range,
        column_names, numeric_columns, categorical_columns, datetime_columns,
        duplicate_rows, constant_columns, high_cardinality_columns
    """
    if df is None or df.empty:
        return {"error": "DataFrame is None or empty."}

    n_rows, n_cols = df.shape

    # dtype map
    dtypes: dict[str, str] = {col: str(df[col].dtype) for col in df.columns}

    # Missing values
    missing_counts: dict[str, int] = df.isna().sum().to_dict()
    missing_pct: dict[str, float] = {
        col: round(cnt / n_rows * 100, 2) for col, cnt in missing_counts.items()
    }

    # Memory
    mem_bytes = int(df.memory_usage(deep=True).sum())
    mem_human = _human_bytes(mem_bytes)
    mem_mb = round(mem_bytes / (1024 * 1024), 2)

    # Date range from all datetime columns
    date_range = _compute_date_range(df)

    # Column categorisation
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Constant columns (single unique non-null value)
    constant_cols = [
        col for col in df.columns if df[col].nunique(dropna=True) <= 1
    ]

    # High cardinality: object cols where unique > 50% of rows
    high_card_cols = [
        col for col in categorical_cols
        if df[col].nunique(dropna=True) > max(50, n_rows * 0.5)
    ]

    # Duplicate rows
    dup_rows = int(df.duplicated().sum())

    return {
        "dataset_name": dataset_name or "dataset.csv",
        "shape": (n_rows, n_cols),
        "n_rows": n_rows,
        "total_rows": n_rows,
        "n_cols": n_cols,
        "total_columns": n_cols,
        "column_names": df.columns.tolist(),
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_per_col": missing_counts,
        "missing_pct": missing_pct,
        "memory_usage_bytes": mem_bytes,
        "memory_usage_human": mem_human,
        "memory_usage_mb": mem_mb,
        "date_range": date_range,
        "numeric_columns": numeric_cols,
        "datetime_columns": datetime_cols,
        "categorical_columns": categorical_cols,
        "duplicate_rows": dup_rows,
        "constant_columns": constant_cols,
        "high_cardinality_columns": high_card_cols,
    }


def infer_column_roles(df: pd.DataFrame) -> dict[str, list[str]]:
    """Categorise each column into a semantic role.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with lists of column names under these keys:
        date_cols, lat_cols, lon_cols, numeric_cols,
        categorical_cols, text_cols, target_candidates, unclassified
    """
    roles: dict[str, list[str]] = {
        "date_cols": [],
        "lat_cols": [],
        "lon_cols": [],
        "numeric_cols": [],
        "categorical_cols": [],
        "text_cols": [],
        "target_candidates": [],
        "unclassified": [],
    }

    assigned: set[str] = set()

    # --- Pass 1: datetime dtype columns ---
    for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        roles["date_cols"].append(col)
        assigned.add(col)

    # --- Pass 2: name-pattern matching ---
    for col in df.columns:
        if col in assigned:
            continue
        col_lower = col.strip()

        if _DATE_PATTERNS.search(col_lower):
            # Confirm it looks date-like in content
            if _column_looks_like_date(df[col]):
                roles["date_cols"].append(col)
                assigned.add(col)
                continue

        if _LAT_PATTERNS.search(col_lower):
            roles["lat_cols"].append(col)
            assigned.add(col)
            continue

        if _LON_PATTERNS.search(col_lower):
            roles["lon_cols"].append(col)
            assigned.add(col)
            continue

    # --- Pass 3: target candidates (by name, among numeric/low-cardinality) ---
    for col in df.columns:
        if col in assigned:
            continue
        if _TARGET_PATTERNS.search(col):
            roles["target_candidates"].append(col)
            assigned.add(col)
            continue

    # --- Pass 4: dtype-based classification for the remainder ---
    numeric_dtypes = df.select_dtypes(include="number").columns.tolist()
    for col in df.columns:
        if col in assigned:
            continue

        if col in numeric_dtypes:
            # Could it still be a lat/lon?
            if _column_looks_like_latitude(df[col]):
                roles["lat_cols"].append(col)
            elif _column_looks_like_longitude(df[col]):
                roles["lon_cols"].append(col)
            else:
                roles["numeric_cols"].append(col)
            assigned.add(col)
            continue

        # String-ish columns
        if df[col].dtype == object or str(df[col].dtype) == "string":
            # Try to cast to date first
            if _column_looks_like_date(df[col]):
                roles["date_cols"].append(col)
                assigned.add(col)
                continue

            n_unique = df[col].nunique(dropna=True)
            n_rows = len(df)
            avg_len = df[col].dropna().astype(str).str.len().mean() if n_rows > 0 else 0

            if avg_len > 50 or n_unique > min(_CAT_MAX_UNIQUE, n_rows * 0.8):
                # Long free-text
                roles["text_cols"].append(col)
            elif n_unique <= _CAT_MAX_UNIQUE or (n_rows > 0 and n_unique / n_rows <= _CAT_CARDINALITY_RATIO):
                roles["categorical_cols"].append(col)
            else:
                roles["text_cols"].append(col)
            assigned.add(col)
            continue

        # category dtype
        if hasattr(df[col], "cat"):
            roles["categorical_cols"].append(col)
            assigned.add(col)
            continue

    # Anything still unassigned
    for col in df.columns:
        if col not in assigned:
            roles["unclassified"].append(col)

    return roles


def get_summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Return comprehensive descriptive statistics for all column types.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys:
        numeric_stats   – dict keyed by column name, each holding full stats
        categorical_stats – dict keyed by column name
        datetime_stats  – dict keyed by column name
        overall         – dataset-level aggregate stats
    """
    if df is None or df.empty:
        return {"error": "DataFrame is None or empty."}

    result: dict[str, Any] = {
        "numeric_stats": {},
        "categorical_stats": {},
        "datetime_stats": {},
        "overall": {},
    }

    # --- Numeric ---
    num_df = df.select_dtypes(include="number")
    for col in num_df.columns:
        s = num_df[col].dropna()
        if s.empty:
            result["numeric_stats"][col] = {"error": "All values are NaN"}
            continue
        try:
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            skew = float(s.skew()) if len(s) > 2 else None
            kurt = float(s.kurt()) if len(s) > 3 else None
            result["numeric_stats"][col] = {
                "count": int(s.count()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "variance": float(s.var()),
                "min": float(s.min()),
                "max": float(s.max()),
                "range": float(s.max() - s.min()),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "skewness": skew,
                "kurtosis": kurt,
                "n_missing": int(num_df[col].isna().sum()),
                "n_zeros": int((s == 0).sum()),
                "n_negative": int((s < 0).sum()),
                "n_unique": int(s.nunique()),
                "pct_missing": round(num_df[col].isna().sum() / len(df) * 100, 2),
                # Outlier counts using IQR fence
                "n_outliers_iqr": int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum()),
            }
        except Exception as exc:  # noqa: BLE001
            result["numeric_stats"][col] = {"error": str(exc)}

    # --- Categorical / object ---
    cat_df = df.select_dtypes(include=["object", "category"])
    for col in cat_df.columns:
        s = cat_df[col]
        try:
            vc = s.value_counts(dropna=True)
            top_n = min(10, len(vc))
            result["categorical_stats"][col] = {
                "count": int(s.count()),
                "n_missing": int(s.isna().sum()),
                "pct_missing": round(s.isna().sum() / len(df) * 100, 2),
                "n_unique": int(s.nunique(dropna=True)),
                "top_values": vc.head(top_n).to_dict(),
                "mode": str(vc.index[0]) if len(vc) > 0 else None,
                "mode_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "mode_pct": round(vc.iloc[0] / s.count() * 100, 2) if len(vc) > 0 and s.count() > 0 else 0,
            }
        except Exception as exc:  # noqa: BLE001
            result["categorical_stats"][col] = {"error": str(exc)}

    # --- Datetime ---
    dt_df = df.select_dtypes(include=["datetime", "datetimetz"])
    for col in dt_df.columns:
        s = dt_df[col].dropna()
        try:
            if s.empty:
                result["datetime_stats"][col] = {"error": "All values are NaT"}
                continue
            result["datetime_stats"][col] = {
                "count": int(s.count()),
                "n_missing": int(dt_df[col].isna().sum()),
                "pct_missing": round(dt_df[col].isna().sum() / len(df) * 100, 2),
                "min": str(s.min()),
                "max": str(s.max()),
                "range_days": (s.max() - s.min()).days,
                "n_unique": int(s.nunique()),
            }
        except Exception as exc:  # noqa: BLE001
            result["datetime_stats"][col] = {"error": str(exc)}

    # --- Overall ---
    total_cells = df.shape[0] * df.shape[1]
    missing_total = int(df.isna().sum().sum())
    result["overall"] = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "total_cells": total_cells,
        "total_missing": missing_total,
        "pct_missing": round(missing_total / total_cells * 100, 2) if total_cells else 0,
        "n_numeric_cols": len(num_df.columns),
        "n_categorical_cols": len(cat_df.columns),
        "n_datetime_cols": len(dt_df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_human": _human_bytes(int(df.memory_usage(deep=True).sum())),
    }

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert object columns that look like dates to datetime."""
    for col in df.select_dtypes(include="object").columns:
        if _DATE_PATTERNS.search(col) or _column_looks_like_date(df[col], sample_size=200):
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                # Only replace if most values parsed successfully
                success_rate = converted.notna().sum() / max(len(converted), 1)
                if success_rate >= 0.6:
                    df[col] = converted
            except Exception:  # noqa: BLE001
                pass
    return df


def _column_looks_like_date(series: pd.Series, sample_size: int = 100) -> bool:
    """Return True if a sample of the series parses as datetimes."""
    sample = series.dropna().head(sample_size)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
        return parsed.notna().sum() / len(sample) >= 0.7
    except Exception:  # noqa: BLE001
        return False


def _column_looks_like_latitude(series: pd.Series) -> bool:
    """Return True if numeric values fall within valid latitude range [-90, 90]."""
    s = series.dropna()
    if s.empty or not np.issubdtype(s.dtype, np.number):
        return False
    return bool(s.between(-90, 90).mean() >= 0.95 and s.std() > 0)


def _column_looks_like_longitude(series: pd.Series) -> bool:
    """Return True if numeric values fall within valid longitude range [-180, 180]."""
    s = series.dropna()
    if s.empty or not np.issubdtype(s.dtype, np.number):
        return False
    return bool(s.between(-180, 180).mean() >= 0.95 and s.std() > 0)


def _compute_date_range(df: pd.DataFrame) -> dict[str, str | None]:
    """Return the min/max timestamps across all datetime columns."""
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if not dt_cols:
        return {"min": None, "max": None, "columns_used": []}
    all_min, all_max = [], []
    for col in dt_cols:
        s = df[col].dropna()
        if not s.empty:
            all_min.append(s.min())
            all_max.append(s.max())
    return {
        "min": str(min(all_min)) if all_min else None,
        "max": str(max(all_max)) if all_max else None,
        "columns_used": dt_cols,
    }


def _resolve_upload_source(uploaded_file: Any) -> tuple[bytes, str]:
    """Normalise various upload sources to raw bytes + a display name."""
    # Streamlit UploadedFile or generic file-like with read()
    if hasattr(uploaded_file, "read"):
        name = getattr(uploaded_file, "name", "uploaded_file")
        raw = uploaded_file.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return raw, name

    # Path or string path
    if isinstance(uploaded_file, (str, Path)):
        p = Path(uploaded_file)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p.read_bytes(), p.name

    # Raw bytes
    if isinstance(uploaded_file, (bytes, bytearray)):
        return bytes(uploaded_file), "uploaded_bytes"

    raise TypeError(
        f"Unsupported upload type: {type(uploaded_file).__name__}. "
        "Expected a file-like object, path, or bytes."
    )


def _detect_encoding(raw: bytes) -> str:
    """Best-effort encoding detection without chardet dependency."""
    # UTF-8 BOM
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    # UTF-16 LE/BE BOM
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16"
    # Try strict UTF-8
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    # Fallback
    return "latin-1"


def _human_bytes(n: int) -> str:
    """Return a human-readable string for a byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"
