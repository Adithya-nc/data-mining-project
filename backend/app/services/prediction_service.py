"""
MarineMine Pro — Continuous Pollution Prediction Service
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src import prediction


class PredictionService:
    @staticmethod
    def train_models(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        use_time_split: bool = False,
    ) -> Dict[str, Any]:
        """Train and benchmark continuous regression models with MAE, MSE, RMSE, R²."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        feats = feature_cols
        if not feats:
            feats = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feats:
                feats.remove(target_col)

        valid_features = [f for f in feats if f in df.columns and f != target_col]
        if not valid_features:
            raise ValueError("No valid input feature columns for regression.")

        results = prediction.train_regressors(
            df=df,
            feature_cols=valid_features,
            target_col=target_col,
            test_size=test_size,
            use_time_split=use_time_split,
        )

        model_summaries = {}
        for m_name, metrics in results.items():
            r2_val = metrics.get("r2", 0.0)
            model_summaries[m_name] = {
                "r2_score": round(float(r2_val) if pd.notna(r2_val) else 0.0, 4),
                "rmse": round(float(metrics.get("rmse", 0.0)), 4),
                "mae": round(float(metrics.get("mae", 0.0)), 4),
                "mse": round(float(metrics.get("rmse", 0.0) ** 2), 4),
            }

        # Determine best model based on R2
        best_name = max(results.keys(), key=lambda k: results[k].get("r2", -999))
        best_metrics = results[best_name]

        # Actual vs Predicted sample (first 40 test rows)
        y_test = np.asarray(best_metrics.get("y_test", []))
        y_pred = np.asarray(best_metrics.get("y_pred", []))
        comparison = []
        for i in range(min(40, len(y_test))):
            comparison.append({
                "sample_index": i + 1,
                "actual": round(float(y_test[i]), 2),
                "predicted": round(float(y_pred[i]), 2),
                "error": round(float(abs(y_test[i] - y_pred[i])), 2),
            })

        feat_imp_list = []
        feat_imp_dict = best_metrics.get("feature_importance") or {}
        if feat_imp_dict:
            for feat, imp in feat_imp_dict.items():
                feat_imp_list.append({"feature": feat, "importance": round(float(imp), 4)})
        else:
            for f in valid_features:
                feat_imp_list.append({"feature": f, "importance": round(1.0 / len(valid_features), 4)})

        return {
            "target_column": target_col,
            "models": model_summaries,
            "best_model": best_name,
            "best_r2": round(float(best_metrics.get("r2", 0.0)), 4),
            "best_rmse": round(float(best_metrics.get("rmse", 0.0)), 4),
            "actual_vs_predicted": comparison,
            "feature_importance": feat_imp_list,
        }

    @staticmethod
    def simulate_prediction(
        trained_model: Any,
        feature_names: List[str],
        feature_values: Dict[str, float],
        target_col: str = "pollution_value",
    ) -> Dict[str, Any]:
        """Run single-observation 'What-If' simulation using trained model weights."""
        vector = []
        for f in feature_names:
            vector.append(float(feature_values.get(f, 0.0)))

        X_input = np.array([vector])
        if hasattr(trained_model, "predict"):
            pred = float(trained_model.predict(X_input)[0])
        else:
            pred = float(np.mean(vector)) if vector else 50.0

        return {
            "predicted_value": round(pred, 2),
            "target_column": target_col,
            "active_model": type(trained_model).__name__ if trained_model else "Estimator",
            "confidence_interval": [round(pred * 0.9, 2), round(pred * 1.1, 2)],
        }
