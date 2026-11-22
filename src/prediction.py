"""
MarineMine Pro — Marine Pollution Data Mining & ML Platform
prediction.py: Regression prediction module

Provides utilities to check data suitability, train and compare regressors,
make single predictions, and surface standard disclaimers.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. Suitability check
# ---------------------------------------------------------------------------

def check_prediction_suitability(df: pd.DataFrame, target_col: str) -> dict:
    """
    Check whether *df* and *target_col* are suitable for regression.

    Returns
    -------
    dict with keys:
        suitable       (bool)   – overall verdict
        is_numeric     (bool)   – target column is numeric
        has_variance   (bool)   – target has non-zero variance
        n_rows         (int)    – number of rows
        n_missing      (int)    – missing values in target
        target_min     (float)  – min value (or None)
        target_max     (float)  – max value (or None)
        target_std     (float)  – std  value (or None)
        issues         (list)   – human-readable problem descriptions
    """
    result: dict[str, Any] = {
        "suitable": False,
        "is_numeric": False,
        "has_variance": False,
        "n_rows": len(df),
        "n_missing": 0,
        "target_min": None,
        "target_max": None,
        "target_std": None,
        "issues": [],
    }

    # --- column existence ---
    if target_col not in df.columns:
        result["issues"].append(f"Column '{target_col}' not found in DataFrame.")
        return result

    series = df[target_col]
    result["n_missing"] = int(series.isna().sum())

    # --- numeric check ---
    if not pd.api.types.is_numeric_dtype(series):
        result["issues"].append(
            f"Target column '{target_col}' is not numeric (dtype={series.dtype})."
        )
        return result

    result["is_numeric"] = True

    # --- drop NaN for further stats ---
    clean = series.dropna()

    if len(clean) == 0:
        result["issues"].append("Target column contains only missing values.")
        return result

    result["target_min"] = float(clean.min())
    result["target_max"] = float(clean.max())
    result["target_std"] = float(clean.std())

    # --- variance check ---
    if clean.std() == 0.0:
        result["issues"].append(
            "Target column has zero variance (all values are identical)."
        )
        return result

    result["has_variance"] = True

    # --- minimum rows ---
    MIN_ROWS = 10
    usable_rows = len(df.dropna(subset=[target_col]))
    if usable_rows < MIN_ROWS:
        result["issues"].append(
            f"Too few usable rows ({usable_rows}); need at least {MIN_ROWS}."
        )
        return result

    result["suitable"] = True
    return result


# ---------------------------------------------------------------------------
# 2. Train regressors
# ---------------------------------------------------------------------------

def train_regressors(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    use_time_split: bool = False,
) -> dict[str, dict]:
    """
    Train LinearRegression (inside a StandardScaler Pipeline) and
    RandomForestRegressor on the provided data.

    Parameters
    ----------
    df            : Input DataFrame.
    feature_cols  : List of feature column names.
    target_col    : Name of the numeric target column.
    test_size     : Fraction of data for testing (ignored when use_time_split=True).
    random_state  : Random seed for reproducibility.
    use_time_split: If True, splits chronologically (first 80 % train, last 20 % test).

    Returns
    -------
    dict  model_name -> {
        "model"               : fitted estimator,
        "mae"                 : float,
        "rmse"                : float,
        "r2"                  : float,
        "y_test"              : np.ndarray,
        "y_pred"              : np.ndarray,
        "feature_importance"  : dict | None,
    }

    Raises
    ------
    ValueError  if the data is not suitable or columns are missing.
    """

    # --- validate suitability ---
    suitability = check_prediction_suitability(df, target_col)
    if not suitability["suitable"]:
        raise ValueError(
            "Data not suitable for regression: "
            + "; ".join(suitability["issues"])
        )

    # --- validate feature columns ---
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in DataFrame: {missing_features}")

    if not feature_cols:
        raise ValueError("feature_cols must contain at least one column.")

    # --- prepare clean subset ---
    cols_needed = feature_cols + [target_col]
    data = df[cols_needed].dropna()

    if data.empty:
        raise ValueError(
            "No rows remain after dropping NaN values in features + target."
        )

    X = data[feature_cols].values.astype(float)
    y = data[target_col].values.astype(float)

    # --- train / test split ---
    if use_time_split:
        split_idx = max(1, int(len(X) * (1 - test_size)))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Train or test split is empty. Provide more data or adjust test_size."
        )

    # --- model definitions ---
    models: dict[str, Any] = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=100,
            random_state=random_state,
            learning_rate=0.08,
            max_depth=4,
        ),
        "LinearRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        ),
    }

    results: dict[str, dict] = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae  = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2   = float(r2_score(y_test, y_pred))

            # --- feature importance ---
            feature_importance: dict | None = None

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_importance = dict(zip(feature_cols, importances.tolist()))

            elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("regressor", None), "coef_"):
                step = model.named_steps["regressor"]
                coefs = np.abs(step.coef_)
                total = np.sum(coefs) if np.sum(coefs) > 0 else 1.0
                normalized = (coefs / total).tolist()
                feature_importance = dict(zip(feature_cols, normalized))

            results[name] = {
                "model": model,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "y_test": y_test,
                "y_pred": y_pred,
                "feature_importance": feature_importance,
            }

        except Exception as exc:  # noqa: BLE001
            results[name] = {
                "model": None,
                "mae": None,
                "rmse": None,
                "r2": None,
                "y_test": None,
                "y_pred": None,
                "feature_importance": None,
                "error": str(exc),
            }

    return results


# ---------------------------------------------------------------------------
# 3. Compare regressors
# ---------------------------------------------------------------------------

def compare_regressors(results_dict: dict[str, dict]) -> pd.DataFrame:
    """
    Summarise regression metrics for all trained models.

    Parameters
    ----------
    results_dict : Output of :func:`train_regressors`.

    Returns
    -------
    pd.DataFrame with columns [Model, MAE, RMSE, R2], sorted by RMSE ascending.
    """
    if not results_dict:
        return pd.DataFrame(columns=["Model", "MAE", "RMSE", "R2"])

    rows = []
    for model_name, info in results_dict.items():
        rows.append(
            {
                "Model": model_name,
                "MAE":  info.get("mae"),
                "RMSE": info.get("rmse"),
                "R2":   info.get("r2"),
            }
        )

    comparison = pd.DataFrame(rows)

    # Sort by RMSE ascending (best first); models with errors (None) go last
    comparison = comparison.sort_values("RMSE", ascending=True, na_position="last")
    comparison = comparison.reset_index(drop=True)
    return comparison


# ---------------------------------------------------------------------------
# 4. Single-value prediction
# ---------------------------------------------------------------------------

def predict_value(
    model: Any,
    feature_cols: list[str],
    input_dict: dict[str, float],
) -> float:
    """
    Make a single prediction using a fitted model.

    Parameters
    ----------
    model        : A fitted scikit-learn estimator (or Pipeline).
    feature_cols : Ordered list of feature names the model was trained on.
    input_dict   : Mapping of feature_name -> value for the new observation.

    Returns
    -------
    float  – predicted target value.

    Raises
    ------
    ValueError  if model is None, feature_cols is empty, or required features
                are missing from input_dict.
    """
    if model is None:
        raise ValueError("Model is None — it may have failed during training.")

    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")

    missing_keys = [col for col in feature_cols if col not in input_dict]
    if missing_keys:
        raise ValueError(
            f"The following features are missing from input_dict: {missing_keys}"
        )

    # Build a single-row array in the correct column order
    try:
        row = np.array(
            [[float(input_dict[col]) for col in feature_cols]]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not convert input values to float: {exc}") from exc

    prediction = model.predict(row)
    return float(prediction[0])


# ---------------------------------------------------------------------------
# 5. Disclaimer
# ---------------------------------------------------------------------------

def get_prediction_disclaimer() -> str:
    """
    Return a standard disclaimer text for ML predictions in a marine context.
    """
    return (
        "DISCLAIMER: Predictions generated by MarineMine Pro are based on "
        "historical training data and statistical models. They are intended "
        "for research and decision-support purposes only and must NOT be used "
        "as the sole basis for environmental or regulatory decisions. Model "
        "accuracy depends on data quality, representativeness, and the "
        "assumptions embedded in each algorithm. Results should be validated "
        "by qualified marine scientists and domain experts before any "
        "operational or policy use. MarineMine Pro and its contributors "
        "accept no liability for decisions made solely on the basis of "
        "these predictions."
    )
