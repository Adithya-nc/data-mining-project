"""
MarineMine Pro — Model Evaluation Utilities
============================================
Provides a comprehensive suite of evaluation helpers for classification and
regression models used in marine pollution analysis pipelines.

Functions
---------
- classification_report_df      : Per-class + macro/weighted metrics as DataFrame
- regression_metrics            : MAE, MSE, RMSE, R², MAPE in a plain dict
- cross_validate_classifier     : k-fold CV accuracy (mean ± std)
- plot_learning_curve           : Train/val learning-curve matplotlib figure
- format_metrics_table          : Styled pandas DataFrame for inline display
- generate_model_summary        : Markdown summary card for a trained model
"""

from __future__ import annotations

import datetime
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, learning_curve

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, pd.Series, List]
MetricsDict = Dict[str, float]

# ---------------------------------------------------------------------------
# Colour palette used throughout the module
# ---------------------------------------------------------------------------
_PALETTE = {
    "ocean":    "#0077B6",
    "seafoam":  "#00B4D8",
    "coral":    "#F77F00",
    "sand":     "#F4A261",
    "algae":    "#2D6A4F",
    "light_bg": "#EAF4FB",
    "warn":     "#E63946",
}


# ===========================================================================
# 1. classification_report_df
# ===========================================================================

def classification_report_df(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return a per-class + summary classification report as a tidy DataFrame.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted class labels.
    labels : list of str, optional
        Display names for the classes (in the same order as the numeric/string
        classes).  When *None* the raw class values are used.

    Returns
    -------
    pd.DataFrame
        Columns: precision, recall, f1-score, support
        Index:   one row per class + accuracy / macro avg / weighted avg rows.
    """
    report: dict = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T

    # Coerce support to integer where sensible
    df["support"] = df["support"].apply(
        lambda v: int(v) if pd.notna(v) and v == int(v) else v
    )

    # Reorder columns for readability
    col_order = ["precision", "recall", "f1-score", "support"]
    df = df[[c for c in col_order if c in df.columns]]

    df.index.name = "class"
    return df


# ===========================================================================
# 2. regression_metrics
# ===========================================================================

def regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> MetricsDict:
    """Compute a standard suite of regression performance metrics.

    Parameters
    ----------
    y_true : array-like
        Observed target values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    dict
        Keys: ``mae``, ``mse``, ``rmse``, ``r2``, ``mape``
        All values are Python *float*.

    Notes
    -----
    MAPE is computed only over samples where ``y_true != 0`` to avoid
    division by zero.  If all true values are zero, ``mape`` is *NaN*.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae  = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse  = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true_arr, y_pred_arr))

    # MAPE — guard against zero denominators
    nonzero_mask = y_true_arr != 0
    if nonzero_mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(
            np.mean(
                np.abs(
                    (y_true_arr[nonzero_mask] - y_pred_arr[nonzero_mask])
                    / y_true_arr[nonzero_mask]
                )
            )
            * 100
        )

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}


# ===========================================================================
# 3. cross_validate_classifier
# ===========================================================================

def cross_validate_classifier(
    model: Any,
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
) -> Dict[str, float]:
    """Run stratified k-fold CV and return accuracy mean & standard deviation.

    Parameters
    ----------
    model :
        A scikit-learn compatible estimator with a ``fit`` / ``predict``
        interface.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    cv : int, default 5
        Number of cross-validation folds.

    Returns
    -------
    dict
        ``{"mean_accuracy": float, "std_accuracy": float, "cv_scores": list}``
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy":  float(scores.std()),
        "cv_scores":     scores.tolist(),
    }


# ===========================================================================
# 4. plot_learning_curve
# ===========================================================================

def plot_learning_curve(
    model: Any,
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
    n_jobs: int = -1,
    train_sizes: Optional[np.ndarray] = None,
    scoring: str = "accuracy",
    figsize: Tuple[int, int] = (10, 5),
) -> matplotlib.figure.Figure:
    """Plot training and cross-validation learning curves.

    Parameters
    ----------
    model :
        Scikit-learn compatible estimator.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    cv : int, default 5
        Cross-validation folds.
    n_jobs : int, default -1
        Parallel jobs for ``learning_curve``.
    train_sizes : ndarray, optional
        Fractions of training set sizes to evaluate.
        Defaults to ``np.linspace(0.1, 1.0, 10)``.
    scoring : str, default ``"accuracy"``
        Sklearn scoring string passed to ``learning_curve``.
    figsize : tuple, default ``(10, 5)``
        Figure size in inches ``(width, height)``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with two subplots: left = score curves, right = fit-time curve.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores, fit_times, _ = learning_curve(
        model,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring=scoring,
        return_times=True,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)
    fit_mean   = fit_times.mean(axis=1)
    fit_std    = fit_times.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(_PALETTE["light_bg"])

    # --- Left: score vs training size ----------------------------------------
    ax = axes[0]
    ax.set_facecolor(_PALETTE["light_bg"])
    ax.plot(
        train_sizes_abs, train_mean,
        "o-", color=_PALETTE["ocean"], label="Training score",
    )
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15, color=_PALETTE["ocean"],
    )
    ax.plot(
        train_sizes_abs, val_mean,
        "s--", color=_PALETTE["coral"], label="CV score",
    )
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15, color=_PALETTE["coral"],
    )
    ax.set_title("Learning Curve", fontsize=13, fontweight="bold",
                 color=_PALETTE["algae"])
    ax.set_xlabel("Training samples", fontsize=11)
    ax.set_ylabel(scoring.capitalize(), fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)

    # --- Right: fit time vs training size ------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor(_PALETTE["light_bg"])
    ax2.plot(
        train_sizes_abs, fit_mean,
        "^-", color=_PALETTE["seafoam"], label="Fit time",
    )
    ax2.fill_between(
        train_sizes_abs,
        fit_mean - fit_std,
        fit_mean + fit_std,
        alpha=0.15, color=_PALETTE["seafoam"],
    )
    ax2.set_title("Scalability", fontsize=13, fontweight="bold",
                  color=_PALETTE["algae"])
    ax2.set_xlabel("Training samples", fontsize=11)
    ax2.set_ylabel("Fit time (s)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "MarineMine Pro — Model Learning Analysis",
        fontsize=14, fontweight="bold", color=_PALETTE["ocean"], y=1.02,
    )
    fig.tight_layout()
    return fig


# ===========================================================================
# 5. format_metrics_table
# ===========================================================================

def format_metrics_table(
    metrics_dict: MetricsDict,
    highlight_best: bool = True,
) -> "pd.io.formats.style.Styler":
    """Return a visually styled pandas Styler from a flat metrics dictionary.

    Parameters
    ----------
    metrics_dict : dict
        ``{metric_name: value}`` mapping of numeric evaluation results.
    highlight_best : bool, default True
        When *True*, cells whose metric name ends with ``"accuracy"`` or
        ``"r2"`` are highlighted in ocean-blue (higher = better) and cells
        whose name ends with ``"error"`` / ``"loss"`` / ``"mse"`` /
        ``"mae"`` / ``"rmse"`` / ``"mape"`` are highlighted in coral
        (lower = better).

    Returns
    -------
    pandas.io.formats.style.Styler
        A styled DataFrame ready for Jupyter display.
    """
    # Build a tidy two-column table
    rows = [
        {
            "Metric": k.upper().replace("_", " "),
            "Value":  round(v, 6) if isinstance(v, float) else v,
        }
        for k, v in metrics_dict.items()
    ]
    df = pd.DataFrame(rows)

    _higher_better_suffixes = ("accuracy", "r2", "precision", "recall", "f1")
    _lower_better_suffixes  = ("error", "loss", "mse", "mae", "rmse", "mape")

    def _style_value(row: pd.Series) -> List[str]:
        """Return CSS for each cell in the row."""
        metric_key = row["Metric"].lower().replace(" ", "_")
        base = ["", ""]  # Metric col, Value col

        if highlight_best:
            if any(metric_key.endswith(s) for s in _higher_better_suffixes):
                base[1] = (
                    f"background-color: {_PALETTE['ocean']}22;"
                    f"color: {_PALETTE['ocean']}; font-weight: bold;"
                )
            elif any(metric_key.endswith(s) for s in _lower_better_suffixes):
                base[1] = (
                    f"background-color: {_PALETTE['coral']}22;"
                    f"color: {_PALETTE['coral']}; font-weight: bold;"
                )
        return base

    styler = (
        df.style
        .set_caption("📊 Model Evaluation Metrics — MarineMine Pro")
        .apply(_style_value, axis=1)
        .set_properties(
            **{
                "text-align": "left",
                "font-family": "monospace",
                "font-size":   "13px",
                "border":      "1px solid #dee2e6",
                "padding":     "6px 12px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "14px"),
                        ("font-weight", "bold"),
                        ("color", _PALETTE["ocean"]),
                        ("padding-bottom", "8px"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", _PALETTE["ocean"]),
                        ("color", "white"),
                        ("font-size", "13px"),
                        ("padding", "6px 12px"),
                    ],
                },
                {
                    "selector": "tr:nth-child(even)",
                    "props": [("background-color", _PALETTE["light_bg"])],
                },
            ]
        )
        .format({"Value": lambda v: f"{v:.6f}" if isinstance(v, float) else str(v)})
        .hide(axis="index")
    )
    return styler


# ===========================================================================
# 6. generate_model_summary
# ===========================================================================

def generate_model_summary(
    model_name: str,
    metrics_dict: MetricsDict,
    feature_cols: Sequence[str],
    target_col: str,
    extra_notes: Optional[str] = None,
) -> str:
    """Generate a Markdown model summary card.

    Parameters
    ----------
    model_name : str
        Human-readable name of the model (e.g. ``"Random Forest Classifier"``).
    metrics_dict : dict
        Flat ``{metric_name: value}`` mapping of evaluation results.
    feature_cols : sequence of str
        Names of input feature columns used during training.
    target_col : str
        Name of the target column.
    extra_notes : str, optional
        Any free-text notes to append in a *Notes* section.

    Returns
    -------
    str
        A multi-line Markdown string containing a formatted model summary.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build metrics table rows
    metric_rows = "\n".join(
        f"| {k.upper().replace('_', ' '):<25} | "
        f"{round(v, 6) if isinstance(v, float) else v!s:>12} |"
        for k, v in metrics_dict.items()
    )

    # Feature list (wrap at 80 chars)
    feature_list = ", ".join(f"`{f}`" for f in feature_cols)
    wrapped_features = textwrap.fill(feature_list, width=80, subsequent_indent="  ")

    notes_section = ""
    if extra_notes:
        notes_section = textwrap.dedent(
            f"""
            ## 📝 Notes

            {extra_notes.strip()}
            """
        )

    summary = textwrap.dedent(
        f"""\
        # 🌊 MarineMine Pro — Model Summary

        **Model:** `{model_name}`
        **Generated:** {timestamp}

        ---

        ## 🎯 Target

        | Key            | Value                     |
        |----------------|---------------------------|
        | Target column  | `{target_col}`            |
        | Feature count  | {len(feature_cols)}       |

        ---

        ## 📐 Features

        {wrapped_features}

        ---

        ## 📊 Evaluation Metrics

        | Metric                    |        Value |
        |---------------------------|--------------|
        {metric_rows}

        ---
        """
    )

    if notes_section:
        summary += notes_section

    summary += textwrap.dedent(
        """\

        ---

        > *Generated by MarineMine Pro Evaluation Module*
        """
    )

    return summary


# ===========================================================================
# Module-level convenience: quick sanity-check when run directly
# ===========================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    print("=" * 60)
    print("MarineMine Pro — Evaluation Module Self-Test")
    print("=" * 60)

    # --- Classification demo -------------------------------------------------
    X_c, y_c = make_classification(n_samples=300, n_features=10, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_c, y_c)
    y_pred_c = clf.predict(X_c)

    print("\n[1] classification_report_df:")
    cdf = classification_report_df(y_c, y_pred_c, labels=["Clean", "Polluted"])
    print(cdf.to_string())

    print("\n[2] cross_validate_classifier:")
    cv_res = cross_validate_classifier(clf, X_c, y_c, cv=3)
    print(f"  mean={cv_res['mean_accuracy']:.4f}  std={cv_res['std_accuracy']:.4f}")

    # --- Regression demo -----------------------------------------------------
    X_r, y_r = make_regression(n_samples=300, n_features=8, noise=10, random_state=42)
    reg = RandomForestRegressor(n_estimators=50, random_state=42)
    reg.fit(X_r, y_r)
    y_pred_r = reg.predict(X_r)

    print("\n[3] regression_metrics:")
    rm = regression_metrics(y_r, y_pred_r)
    for k, v in rm.items():
        print(f"  {k}: {v:.4f}")

    # --- generate_model_summary ----------------------------------------------
    print("\n[4] generate_model_summary:")
    summary = generate_model_summary(
        model_name="Random Forest Regressor",
        metrics_dict=rm,
        feature_cols=[f"sensor_{i}" for i in range(8)],
        target_col="pollution_index",
        extra_notes="Baseline model trained on synthetic marine sensor data.",
    )
    print(summary)

    # --- Learning curve (saves fig) ------------------------------------------
    print("[5] plot_learning_curve: generating figure …")
    fig = plot_learning_curve(
        RandomForestClassifier(n_estimators=20, random_state=42),
        X_c, y_c, cv=3,
    )
    out_path = "learning_curve_selftest.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Saved → {out_path}")

    print("\n✅ All self-tests passed.")
