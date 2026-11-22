"""
MarineMine Pro — Classification Module
=======================================
Senior ML Engineer: Full classification pipeline for marine pollution data.

Includes:
- Target suitability check
- Multi-classifier training with preprocessing pipeline (no data leakage)
- Feature importance extraction
- Model comparison
- Single-record prediction
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Target Suitability Check
# ---------------------------------------------------------------------------

def check_target_suitability(
    df: pd.DataFrame,
    target_col: str,
) -> dict[str, Any]:
    """
    Assess whether *target_col* is suitable for classification.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of the column to use as the classification target.

    Returns
    -------
    dict with keys:
        is_suitable : bool
        reason      : str  – human-readable explanation
        n_classes   : int  – number of unique classes (0 if not suitable)
        class_counts: dict – {class_label: count} (empty if not suitable)
    """
    result: dict[str, Any] = {
        "is_suitable": False,
        "reason": "",
        "n_classes": 0,
        "class_counts": {},
    }

    # --- Basic existence check -------------------------------------------------
    if target_col not in df.columns:
        result["reason"] = f"Column '{target_col}' does not exist in the dataframe."
        return result

    series = df[target_col].dropna()

    # --- Not enough rows -------------------------------------------------------
    if len(series) < 10:
        result["reason"] = (
            f"Column '{target_col}' has only {len(series)} non-null values; "
            "at least 10 are required."
        )
        return result

    # --- All values identical --------------------------------------------------
    unique_vals = series.unique()
    n_classes = len(unique_vals)

    if n_classes < 2:
        result["reason"] = (
            f"Column '{target_col}' has only {n_classes} unique value(s). "
            "Classification requires at least 2 classes."
        )
        return result

    # --- Too many classes (treat as regression target) -------------------------
    # Heuristic: if numeric and > 20 % of values are unique, flag it
    if pd.api.types.is_numeric_dtype(series) and (n_classes / len(series)) > 0.20:
        result["reason"] = (
            f"Column '{target_col}' appears to be continuous ({n_classes} unique "
            f"values out of {len(series)} rows). Consider regression instead."
        )
        return result

    # --- Excessive missing values ----------------------------------------------
    missing_pct = df[target_col].isna().mean()
    if missing_pct > 0.30:
        result["reason"] = (
            f"Column '{target_col}' has {missing_pct:.1%} missing values, "
            "which exceeds the 30 % threshold."
        )
        return result

    # --- All checks passed ----------------------------------------------------
    class_counts = series.value_counts().to_dict()
    result.update(
        {
            "is_suitable": True,
            "reason": (
                f"Column '{target_col}' is suitable for classification "
                f"({n_classes} classes, {len(series)} non-null rows, "
                f"{missing_pct:.1%} missing)."
            ),
            "n_classes": n_classes,
            "class_counts": class_counts,
        }
    )
    return result


# ---------------------------------------------------------------------------
# 2. Train Classifiers
# ---------------------------------------------------------------------------

def _build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Return a ColumnTransformer that handles numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_classifiers(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Train DecisionTree, RandomForest, and LogisticRegression classifiers.

    Data leakage is prevented by fitting all preprocessing steps exclusively
    on the training split inside a scikit-learn Pipeline.

    Parameters
    ----------
    df           : pd.DataFrame  – Full dataset.
    feature_cols : list[str]     – Feature column names.
    target_col   : str           – Target column name.
    test_size    : float         – Fraction of data for testing (default 0.2).
    random_state : int           – Random seed for reproducibility (default 42).

    Returns
    -------
    dict mapping model name → {
        "model"    : fitted Pipeline,
        "accuracy" : float,
        "precision": float,
        "recall"   : float,
        "f1"       : float,
        "cm"       : np.ndarray (confusion matrix),
        "report"   : str (classification_report),
        "y_test"   : np.ndarray,
        "y_pred"   : np.ndarray,
    }
    """
    suitability = check_target_suitability(df, target_col)
    if not suitability["is_suitable"]:
        raise ValueError(
            f"Target column is not suitable for classification: {suitability['reason']}"
        )

    # ---- Prepare X and y -------------------------------------------------------
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in dataframe: {missing_features}")

    X = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # Drop rows where the target is missing
    valid_mask = y_raw.notna()
    X, y_raw = X[valid_mask], y_raw[valid_mask]

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    # Identify numeric vs. categorical features
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # ---- Train / test split (before ANY fitting) --------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ---- Define classifiers -----------------------------------------------------
    n_classes = len(le.classes_)
    avg_method = "binary" if n_classes == 2 else "weighted"

    classifiers: dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=3,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state,
            learning_rate=0.08,
            max_depth=4,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=random_state,
            max_depth=10,
            min_samples_leaf=5,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1,
            C=1.0,
        ),
    }

    # ---- Build, fit, and evaluate each pipeline ---------------------------------
    results: dict[str, dict[str, Any]] = {}

    for name, clf in classifiers.items():
        preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        # Fit only on training data — no leakage
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall    = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1        = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        cm        = confusion_matrix(y_test, y_pred)
        report    = classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in le.classes_],
            zero_division=0,
        )

        results[name] = {
            "model":     pipeline,
            "label_encoder": le,
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "cm":        cm,
            "report":    report,
            "y_test":    y_test,
            "y_pred":    y_pred,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }

    return results


# ---------------------------------------------------------------------------
# 3. Feature Importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    trained_results: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """
    Extract feature importances for tree-based models in *trained_results*.

    Parameters
    ----------
    trained_results : dict
        Output of :func:`train_classifiers`.

    Returns
    -------
    dict mapping model_name → pd.DataFrame with columns
        ["feature", "importance"] sorted descending.
    Only DecisionTree and RandomForest are included (LogisticRegression is skipped).
    """
    importances: dict[str, pd.DataFrame] = {}

    tree_based = {"DecisionTree", "RandomForest", "GradientBoosting"}

    for name, info in trained_results.items():
        if name not in tree_based:
            continue

        pipeline: Pipeline = info["model"]
        preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        if not hasattr(classifier, "feature_importances_"):
            continue  # Safety guard

        # Reconstruct feature names after one-hot encoding
        feature_names: list[str] = []

        for trans_name, transformer, cols in preprocessor.transformers_:
            if trans_name == "num":
                feature_names.extend(cols)
            elif trans_name == "cat":
                ohe: OneHotEncoder = transformer.named_steps["onehot"]
                ohe_names = ohe.get_feature_names_out(cols).tolist()
                feature_names.extend(ohe_names)

        raw_importances = classifier.feature_importances_

        # Guard against length mismatch (e.g. if remainder kept extra cols)
        min_len = min(len(feature_names), len(raw_importances))
        df_imp = pd.DataFrame(
            {
                "feature":    feature_names[:min_len],
                "importance": raw_importances[:min_len],
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        importances[name] = df_imp

    return importances


# ---------------------------------------------------------------------------
# 4. Compare Models
# ---------------------------------------------------------------------------

def compare_models(
    trained_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a comparison DataFrame of key metrics for all trained models.

    Parameters
    ----------
    trained_results : dict
        Output of :func:`train_classifiers`.

    Returns
    -------
    pd.DataFrame with columns:
        Model | Accuracy | Precision | Recall | F1
    Sorted by F1 descending.
    """
    rows = []
    for name, info in trained_results.items():
        rows.append(
            {
                "Model":     name,
                "Accuracy":  round(info["accuracy"], 4),
                "Precision": round(info["precision"], 4),
                "Recall":    round(info["recall"], 4),
                "F1":        round(info["f1"], 4),
            }
        )

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )
    return comparison_df


# ---------------------------------------------------------------------------
# 5. Predict New Record
# ---------------------------------------------------------------------------

def predict_new(
    model: Pipeline,
    feature_cols: list[str],
    input_data_dict: dict[str, Any],
    label_encoder: LabelEncoder | None = None,
) -> dict[str, Any]:
    """
    Generate a prediction for a single new observation.

    Parameters
    ----------
    model           : sklearn Pipeline  – Fitted pipeline from train_classifiers.
    feature_cols    : list[str]         – Feature columns used during training.
    input_data_dict : dict              – {feature_name: value} for the new record.
    label_encoder   : LabelEncoder | None
                       – If provided, the numeric prediction is decoded back to
                         the original class label.

    Returns
    -------
    dict with keys:
        "prediction"       : int or str (decoded if label_encoder is provided)
        "prediction_index" : int        – raw numeric class index
        "probabilities"    : dict | None – {class_label: probability} if the
                             model supports predict_proba; None otherwise.
    """
    # Build a single-row DataFrame with all required columns
    # Missing columns are filled with NaN so the imputer handles them
    row: dict[str, Any] = {col: input_data_dict.get(col, np.nan) for col in feature_cols}
    input_df = pd.DataFrame([row], columns=feature_cols)

    prediction_index = int(model.predict(input_df)[0])
    prediction: int | str = prediction_index

    probabilities: dict[str, float] | None = None

    if label_encoder is not None:
        try:
            prediction = label_encoder.inverse_transform([prediction_index])[0]
        except Exception:
            pass  # Fall back to numeric index

        if hasattr(model, "predict_proba"):
            try:
                proba_array = model.predict_proba(input_df)[0]
                class_labels = [str(c) for c in label_encoder.classes_]
                probabilities = {
                    label: round(float(prob), 4)
                    for label, prob in zip(class_labels, proba_array)
                }
            except Exception:
                pass
    else:
        if hasattr(model, "predict_proba"):
            try:
                proba_array = model.predict_proba(input_df)[0]
                probabilities = {
                    str(i): round(float(p), 4) for i, p in enumerate(proba_array)
                }
            except Exception:
                pass

    return {
        "prediction":       prediction,
        "prediction_index": prediction_index,
        "probabilities":    probabilities,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test (run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # --- Synthetic marine pollution dataset -----------------------------------
    rng = np.random.default_rng(42)
    n = 300

    synthetic_df = pd.DataFrame(
        {
            "pH":              rng.uniform(6.5, 9.0, n),
            "dissolved_O2":    rng.uniform(2.0, 12.0, n),
            "turbidity":       rng.uniform(0.1, 50.0, n),
            "heavy_metal_ppm": rng.uniform(0.0, 5.0, n),
            "temperature_C":   rng.uniform(10.0, 35.0, n),
            "region":          rng.choice(["North", "South", "East", "West"], n),
            "season":          rng.choice(["Spring", "Summer", "Autumn", "Winter"], n),
            "pollution_level": rng.choice(["Low", "Medium", "High"], n),
        }
    )
    # Inject a few NaNs
    synthetic_df.loc[rng.choice(n, 15, replace=False), "pH"] = np.nan
    synthetic_df.loc[rng.choice(n, 10, replace=False), "region"] = np.nan

    target   = "pollution_level"
    features = [c for c in synthetic_df.columns if c != target]

    # 1. Suitability
    suit = check_target_suitability(synthetic_df, target)
    print("=== Suitability ===")
    print(suit, "\n")

    # 2. Train
    results = train_classifiers(synthetic_df, features, target)

    # 3. Feature importance
    fi = get_feature_importance(results)
    for m_name, imp_df in fi.items():
        print(f"=== Feature Importance: {m_name} ===")
        print(imp_df.head(5), "\n")

    # 4. Compare
    print("=== Model Comparison ===")
    print(compare_models(results), "\n")

    # 5. Predict new
    sample = {
        "pH": 7.4,
        "dissolved_O2": 8.1,
        "turbidity": 12.3,
        "heavy_metal_ppm": 1.2,
        "temperature_C": 22.0,
        "region": "North",
        "season": "Summer",
    }
    best_model_name = compare_models(results).iloc[0]["Model"]
    best_info = results[best_model_name]
    pred = predict_new(
        model=best_info["model"],
        feature_cols=features,
        input_data_dict=sample,
        label_encoder=best_info["label_encoder"],
    )
    print(f"=== Prediction (best model: {best_model_name}) ===")
    print(pred)

    sys.exit(0)
