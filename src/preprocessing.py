"""
MarineMine Pro — Marine Pollution Data Mining & ML Platform
===========================================================
Module: preprocessing.py
Description: Full-featured preprocessing pipeline for marine pollution datasets.
             Handles missing values, duplicates, date parsing, encoding, scaling,
             dtype correction, and change-tracking reporting.
"""

from __future__ import annotations

import copy
import warnings
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Season helper
# ---------------------------------------------------------------------------
_MONTH_TO_SEASON: Dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


def _month_to_season(month: int) -> str:
    """Return meteorological season name for a given month number (1-12)."""
    return _MONTH_TO_SEASON.get(int(month), "Unknown")


# ---------------------------------------------------------------------------
# Main Pipeline Class
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for marine pollution DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        The raw input DataFrame.  The original is **never** modified;
        all transformations work on an internal copy.

    Examples
    --------
    >>> pipeline = PreprocessingPipeline(raw_df)
    >>> pipeline.handle_missing_values()
    >>> pipeline.remove_duplicates()
    >>> pipeline.convert_dates(['sample_date'])
    >>> pipeline.encode_categoricals(['region', 'pollutant_type'])
    >>> pipeline.scale_numerics(['ph', 'dissolved_oxygen', 'turbidity'])
    >>> pipeline.correct_dtypes()
    >>> report = pipeline.get_preprocessing_report()
    >>> clean_df = pipeline.get_processed_df()
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}.")
        # Always preserve original; work on a deep copy
        self._original: pd.DataFrame = df  # reference only — never touched
        self._df: pd.DataFrame = df.copy(deep=True)
        self._report: Dict[str, object] = {
            "original_shape": df.shape,
            "steps": [],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log(self, step: str, detail: object) -> None:
        """Append a step entry to the internal report."""
        self._report["steps"].append({"step": step, "detail": detail})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_missing_values(
        self,
        strategy: Literal["mean", "median", "mode", "drop"] = "mean",
        numeric_strategy: Optional[str] = None,
        categorical_strategy: Optional[str] = None,
    ) -> "PreprocessingPipeline":
        """
        Handle missing values in the working DataFrame.

        Parameters
        ----------
        strategy : {'mean', 'median', 'mode', 'drop'}
            - ``'mean'``   – fill numeric NaNs with column mean;
                             fill categorical NaNs with column mode.
            - ``'median'`` – fill numeric NaNs with column median;
                             fill categorical NaNs with column mode.
            - ``'mode'``   – fill all NaNs with column mode (numeric & cat).
            - ``'drop'``   – drop every row that contains at least one NaN.
        numeric_strategy : optional alias for strategy
        categorical_strategy : optional categorical fill strategy

        Returns
        -------
        PreprocessingPipeline
            Returns *self* for optional method chaining.
        """
        if numeric_strategy and numeric_strategy != "none":
            strategy = numeric_strategy  # type: ignore
        before_nulls: int = int(self._df.isnull().sum().sum())
        filled: Dict[str, object] = {}

        if strategy == "drop":
            rows_before = len(self._df)
            self._df.dropna(inplace=True)
            self._df.reset_index(drop=True, inplace=True)
            rows_dropped = rows_before - len(self._df)
            filled["rows_dropped"] = rows_dropped
        else:
            numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = self._df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

            # Numeric fill
            for col in numeric_cols:
                n_missing = int(self._df[col].isnull().sum())
                if n_missing == 0:
                    continue
                if strategy == "mean":
                    fill_val = self._df[col].mean()
                elif strategy == "median":
                    fill_val = self._df[col].median()
                else:  # mode
                    mode_vals = self._df[col].mode()
                    fill_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan
                self._df[col].fillna(fill_val, inplace=True)
                filled[col] = {"missing": n_missing, "fill_value": float(fill_val) if pd.notna(fill_val) else None}

            # Categorical fill (always mode)
            for col in cat_cols:
                n_missing = int(self._df[col].isnull().sum())
                if n_missing == 0:
                    continue
                mode_vals = self._df[col].mode()
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                self._df[col].fillna(fill_val, inplace=True)
                filled[col] = {"missing": n_missing, "fill_value": str(fill_val)}

        after_nulls: int = int(self._df.isnull().sum().sum())
        self._log(
            "handle_missing_values",
            {
                "strategy": strategy,
                "nulls_before": before_nulls,
                "nulls_after": after_nulls,
                "nulls_resolved": before_nulls - after_nulls,
                "columns_affected": filled,
            },
        )
        return self

    # ------------------------------------------------------------------

    def remove_duplicates(
        self, subset: Optional[List[str]] = None, keep: Literal["first", "last", False] = "first"
    ) -> "PreprocessingPipeline":
        """
        Remove duplicate rows from the working DataFrame.

        Parameters
        ----------
        subset : list of str, optional
            Column labels to consider for identifying duplicates.
            Defaults to all columns.
        keep : {'first', 'last', False}
            Which duplicate occurrence to keep.  ``False`` drops all.

        Returns
        -------
        PreprocessingPipeline
        """
        rows_before = len(self._df)
        self._df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        rows_after = len(self._df)
        removed = rows_before - rows_after

        self._log(
            "remove_duplicates",
            {
                "rows_before": rows_before,
                "rows_after": rows_after,
                "duplicates_removed": removed,
                "subset": subset,
                "keep": keep,
            },
        )
        return self

    # ------------------------------------------------------------------

    def convert_dates(
        self,
        date_cols: List[str],
        dayfirst: bool = False,
        errors: Literal["raise", "coerce", "ignore"] = "coerce",
    ) -> "PreprocessingPipeline":
        """
        Parse date columns to ``datetime64`` and extract temporal features.

        For each column in *date_cols* the method:

        1. Converts the column to ``datetime64`` (original column updated in-place).
        2. Extracts ``<col>_year``, ``<col>_month``, ``<col>_day``,
           ``<col>_day_of_week``, ``<col>_quarter``, and ``<col>_season``.

        Parameters
        ----------
        date_cols : list of str
            Column names to process.
        dayfirst : bool
            Passed to ``pd.to_datetime``.
        errors : {'raise', 'coerce', 'ignore'}
            Passed to ``pd.to_datetime``.

        Returns
        -------
        PreprocessingPipeline
        """
        detail: Dict[str, object] = {}
        for col in date_cols:
            if col not in self._df.columns:
                detail[col] = "SKIPPED — column not found"
                continue

            original_dtype = str(self._df[col].dtype)
            self._df[col] = pd.to_datetime(
                self._df[col], dayfirst=dayfirst, errors=errors
            )
            parsed_ok = int(self._df[col].notna().sum())

            # Extract temporal features
            dt = self._df[col].dt
            self._df[f"{col}_year"] = dt.year.astype("Int64")
            self._df[f"{col}_month"] = dt.month.astype("Int64")
            self._df[f"{col}_day"] = dt.day.astype("Int64")
            self._df[f"{col}_day_of_week"] = dt.day_of_week.astype("Int64")
            self._df[f"{col}_quarter"] = dt.quarter.astype("Int64")
            self._df[f"{col}_season"] = (
                self._df[f"{col}_month"]
                .dropna()
                .apply(_month_to_season)
                .reindex(self._df.index)
            )

            detail[col] = {
                "original_dtype": original_dtype,
                "converted_dtype": str(self._df[col].dtype),
                "rows_parsed_ok": parsed_ok,
                "rows_failed": len(self._df) - parsed_ok,
                "new_columns": [
                    f"{col}_year", f"{col}_month", f"{col}_day",
                    f"{col}_day_of_week", f"{col}_quarter", f"{col}_season",
                ],
            }

        self._log("convert_dates", detail)
        return self

    # ------------------------------------------------------------------

    def encode_categoricals(
        self,
        cols: List[str],
        method: Literal["label", "onehot"] = "label",
    ) -> "PreprocessingPipeline":
        """
        Encode categorical columns as numeric representation.

        Parameters
        ----------
        cols : list of str
            Columns to encode.
        method : {'label', 'onehot'}
            - ``'label'``  – ``sklearn.LabelEncoder`` (in-place replacement).
            - ``'onehot'`` – ``pd.get_dummies`` (original column dropped,
                              dummy columns appended with prefix).

        Returns
        -------
        PreprocessingPipeline
        """
        detail: Dict[str, object] = {}

        for col in cols:
            if col not in self._df.columns:
                detail[col] = "SKIPPED — column not found"
                continue

            if method == "label":
                le = LabelEncoder()
                # Handle NaN: encode non-null rows only
                mask = self._df[col].notna()
                self._df.loc[mask, col] = le.fit_transform(
                    self._df.loc[mask, col].astype(str)
                )
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")
                detail[col] = {
                    "method": "label",
                    "classes": list(le.classes_),
                    "n_classes": len(le.classes_),
                }

            else:  # onehot
                dummies = pd.get_dummies(
                    self._df[col], prefix=col, dummy_na=False, dtype=int
                )
                self._df.drop(columns=[col], inplace=True)
                self._df = pd.concat([self._df, dummies], axis=1)
                detail[col] = {
                    "method": "onehot",
                    "new_columns": list(dummies.columns),
                    "n_dummies": len(dummies.columns),
                }

        self._log("encode_categoricals", {"method": method, "columns": detail})
        return self

    # ------------------------------------------------------------------

    def scale_numerics(
        self,
        cols: List[str],
        method: Literal["standard", "minmax"] = "standard",
    ) -> "PreprocessingPipeline":
        """
        Scale numeric columns using scikit-learn scalers.

        Parameters
        ----------
        cols : list of str
            Numeric columns to scale.
        method : {'standard', 'minmax'}
            - ``'standard'`` – zero-mean, unit-variance (``StandardScaler``).
            - ``'minmax'``   – scale to [0, 1] range (``MinMaxScaler``).

        Returns
        -------
        PreprocessingPipeline
        """
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        detail: Dict[str, object] = {"method": method, "columns": {}}

        valid_cols = [c for c in cols if c in self._df.columns]
        missing_cols = [c for c in cols if c not in self._df.columns]

        if missing_cols:
            detail["skipped_columns"] = missing_cols  # type: ignore[index]

        if valid_cols:
            # Fit & transform; NaN-safe by temporarily filling then restoring NaN mask
            subset = self._df[valid_cols].copy()
            nan_mask = subset.isnull()
            fill_tmp = subset.fillna(subset.mean())
            scaled_values = scaler.fit_transform(fill_tmp.values)
            scaled_df = pd.DataFrame(scaled_values, columns=valid_cols, index=self._df.index)
            scaled_df[nan_mask] = np.nan  # restore original NaN positions
            self._df[valid_cols] = scaled_df

            for col in valid_cols:
                col_detail: Dict[str, object] = {
                    "original_min": float(self._original[col].min()) if col in self._original.columns else "N/A",
                    "original_max": float(self._original[col].max()) if col in self._original.columns else "N/A",
                    "scaled_min": float(self._df[col].min()),
                    "scaled_max": float(self._df[col].max()),
                }
                if method == "standard" and hasattr(scaler, "mean_"):
                    idx = valid_cols.index(col)
                    col_detail["mean"] = float(scaler.mean_[idx])
                    col_detail["std"] = float(scaler.scale_[idx])
                detail["columns"][col] = col_detail  # type: ignore[index]

        self._log("scale_numerics", detail)
        return self

    # ------------------------------------------------------------------

    def handle_outliers(
        self,
        cols: Optional[List[str]] = None,
        method: Literal["winsorize", "drop"] = "winsorize",
        factor: float = 1.5,
    ) -> "PreprocessingPipeline":
        """
        Handle statistical outliers using the Interquartile Range (IQR) method.

        Parameters
        ----------
        cols : list of str, optional
            Numeric columns to process. If None, all numeric columns are selected.
        method : {'winsorize', 'drop'}
            - 'winsorize' : Clip values outside [Q1 - factor*IQR, Q3 + factor*IQR] to the boundary.
            - 'drop'      : Remove rows with values outside the boundary.
        factor : float, default 1.5
            Multiplier for the IQR range.
        """
        if cols is None:
            cols = self._df.select_dtypes(include=np.number).columns.tolist()

        valid_cols = [c for c in cols if c in self._df.columns and pd.api.types.is_numeric_dtype(self._df[c])]
        detail: Dict[str, object] = {"method": method, "factor": factor, "columns": {}}

        if method == "drop":
            keep_mask = pd.Series(True, index=self._df.index)
            for col in valid_cols:
                series = self._df[col].dropna()
                if series.empty:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                outliers_count = int(((self._df[col] < lower) | (self._df[col] > upper)).sum())
                keep_mask = keep_mask & ((self._df[col].isna()) | ((self._df[col] >= lower) & (self._df[col] <= upper)))
                detail["columns"][col] = {"lower": float(lower), "upper": float(upper), "outliers_dropped": outliers_count}
            dropped = len(self._df) - int(keep_mask.sum())
            self._df = self._df[keep_mask].reset_index(drop=True)
            detail["total_rows_dropped"] = dropped
        else:  # winsorize / clip
            for col in valid_cols:
                series = self._df[col].dropna()
                if series.empty:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = float(q1 - factor * iqr)
                upper = float(q3 + factor * iqr)
                clipped_count = int(((self._df[col] < lower) | (self._df[col] > upper)).sum())
                self._df[col] = self._df[col].clip(lower=lower, upper=upper)
                detail["columns"][col] = {"lower": lower, "upper": upper, "values_clipped": clipped_count}

        self._log("handle_outliers", detail)
        return self

    # ------------------------------------------------------------------

    def correct_dtypes(self) -> "PreprocessingPipeline":
        """
        Auto-detect and correct column data types.

        Rules applied:
        - Object columns that look like numbers → ``float64`` or ``int64``.
        - Object columns with ≤ 20 unique values → ``category``.
        - Object columns that parse cleanly as dates → ``datetime64``.
        - ``int64`` columns with no negatives that fit in ``uint32`` → ``uint32``.

        Returns
        -------
        PreprocessingPipeline
        """
        changes: Dict[str, Dict[str, str]] = {}

        for col in self._df.columns:
            original_dtype = str(self._df[col].dtype)

            # --- Object column analysis ---
            if self._df[col].dtype == object:
                # Try numeric conversion
                numeric_attempt = pd.to_numeric(self._df[col], errors="coerce")
                n_total = len(self._df[col].dropna())
                n_converted = int(numeric_attempt.notna().sum())

                if n_total > 0 and n_converted / n_total >= 0.9:
                    self._df[col] = numeric_attempt
                    # Downcast to int if no fractional parts
                    if self._df[col].dropna().apply(float.is_integer).all():  # type: ignore[arg-type]
                        self._df[col] = self._df[col].astype("Int64")
                    changes[col] = {"from": original_dtype, "to": str(self._df[col].dtype), "rule": "numeric"}
                    continue

                # Try datetime conversion
                try:
                    dt_attempt = pd.to_datetime(self._df[col], errors="coerce")
                    n_dt = int(dt_attempt.notna().sum())
                    if n_total > 0 and n_dt / n_total >= 0.9:
                        self._df[col] = dt_attempt
                        changes[col] = {"from": original_dtype, "to": str(self._df[col].dtype), "rule": "datetime"}
                        continue
                except Exception:
                    pass

                # Downcast to category if low cardinality
                n_unique = self._df[col].nunique(dropna=True)
                if n_unique <= 20:
                    self._df[col] = self._df[col].astype("category")
                    changes[col] = {"from": original_dtype, "to": "category", "rule": f"low_cardinality ({n_unique} unique)"}

            # --- Integer downcast ---
            elif self._df[col].dtype == np.int64:
                if self._df[col].min() >= 0 and self._df[col].max() <= np.iinfo(np.uint32).max:
                    self._df[col] = self._df[col].astype(np.uint32)
                    changes[col] = {"from": original_dtype, "to": "uint32", "rule": "non_negative_int"}

            # --- Float downcast ---
            elif self._df[col].dtype == np.float64:
                downcast = pd.to_numeric(self._df[col], downcast="float")
                if downcast.dtype != self._df[col].dtype:
                    self._df[col] = downcast
                    changes[col] = {"from": original_dtype, "to": str(self._df[col].dtype), "rule": "float_downcast"}

        self._log("correct_dtypes", {"dtype_changes": changes, "total_changed": len(changes)})
        return self

    # ------------------------------------------------------------------

    def get_preprocessing_report(self) -> Dict[str, object]:
        """
        Return a comprehensive summary dictionary of all changes applied.

        Returns
        -------
        dict
            Keys:
            - ``original_shape``   : (rows, cols) of the input DataFrame.
            - ``processed_shape``  : (rows, cols) of the current DataFrame.
            - ``rows_delta``       : rows lost/gained (negative = removed).
            - ``cols_delta``       : columns added/removed.
            - ``steps``            : ordered list of transformation dicts.
            - ``null_summary``     : per-column null count after processing.
            - ``dtypes_summary``   : per-column dtype after processing.
        """
        report = copy.deepcopy(self._report)
        report["processed_shape"] = self._df.shape
        orig_rows, orig_cols = self._report["original_shape"]  # type: ignore[misc]
        report["rows_delta"] = self._df.shape[0] - orig_rows
        report["cols_delta"] = self._df.shape[1] - orig_cols
        report["null_summary"] = self._df.isnull().sum().to_dict()
        report["dtypes_summary"] = {col: str(dt) for col, dt in self._df.dtypes.items()}
        return report

    # ------------------------------------------------------------------

    def get_processed_df(self) -> pd.DataFrame:
        """
        Return a copy of the cleaned/transformed DataFrame.

        Returns
        -------
        pd.DataFrame
            Deep copy of the internal working DataFrame.
        """
        return self._df.copy(deep=True)


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a sensible default preprocessing sequence to a DataFrame.

    Steps performed in order:
    1. Remove duplicates (keep first).
    2. Fill numeric NaNs with column mean; categorical NaNs with mode.
    3. Auto-correct data types.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame (not modified).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    pipeline = PreprocessingPipeline(df)
    pipeline.remove_duplicates()
    pipeline.handle_missing_values(strategy="mean")
    pipeline.correct_dtypes()
    return pipeline.get_processed_df()


def get_cleaning_stats(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Compute a side-by-side comparison of a DataFrame before and after cleaning.

    Parameters
    ----------
    original_df : pd.DataFrame
        The raw, unprocessed DataFrame.
    cleaned_df : pd.DataFrame
        The cleaned/preprocessed DataFrame.

    Returns
    -------
    dict
        Detailed statistics including shape changes, null counts, duplicate
        counts, dtype changes, memory usage, and per-column null deltas.
    """
    orig_nulls_per_col = original_df.isnull().sum()
    clean_nulls_per_col = cleaned_df.isnull().sum()

    # Per-column null delta (only columns present in both)
    common_cols = original_df.columns.intersection(cleaned_df.columns)
    null_delta_per_col: Dict[str, int] = {}
    for col in common_cols:
        delta = int(clean_nulls_per_col.get(col, 0)) - int(orig_nulls_per_col.get(col, 0))
        if delta != 0:
            null_delta_per_col[col] = delta

    # Dtype changes (columns in common)
    dtype_changes: Dict[str, Dict[str, str]] = {}
    for col in common_cols:
        orig_dtype = str(original_df[col].dtype)
        clean_dtype = str(cleaned_df[col].dtype)
        if orig_dtype != clean_dtype:
            dtype_changes[col] = {"before": orig_dtype, "after": clean_dtype}

    stats: Dict[str, object] = {
        # Shape
        "shape_before": original_df.shape,
        "shape_after": cleaned_df.shape,
        "rows_removed": original_df.shape[0] - cleaned_df.shape[0],
        "cols_added": cleaned_df.shape[1] - original_df.shape[1],

        # Nulls
        "total_nulls_before": int(original_df.isnull().sum().sum()),
        "total_nulls_after": int(cleaned_df.isnull().sum().sum()),
        "null_reduction_pct": round(
            (1 - cleaned_df.isnull().sum().sum() / max(original_df.isnull().sum().sum(), 1)) * 100,
            2,
        ),
        "null_delta_per_col": null_delta_per_col,

        # Duplicates
        "duplicates_before": int(original_df.duplicated().sum()),
        "duplicates_after": int(cleaned_df.duplicated().sum()),

        # Dtypes
        "dtype_changes": dtype_changes,

        # Memory
        "memory_before_kb": round(original_df.memory_usage(deep=True).sum() / 1024, 2),
        "memory_after_kb": round(cleaned_df.memory_usage(deep=True).sum() / 1024, 2),

        # New / dropped columns
        "new_columns": list(cleaned_df.columns.difference(original_df.columns)),
        "dropped_columns": list(original_df.columns.difference(cleaned_df.columns)),
    }
    return stats
