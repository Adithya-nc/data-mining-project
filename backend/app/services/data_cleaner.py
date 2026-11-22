"""
MarineMine Pro — Data Cleaner & Preprocessing Service
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.preprocessing import PreprocessingPipeline


class DataCleanerService:
    @staticmethod
    def execute_pipeline(
        df: pd.DataFrame,
        numeric_strategy: str = "mean",
        categorical_strategy: str = "mode",
        remove_duplicates: bool = True,
        outlier_strategy: str = "winsorize",
        outlier_factor: float = 1.5,
        scale_features: bool = False,
        scale_method: str = "standard",
        encode_categoricals: bool = False,
        encode_method: str = "label",
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Run non-destructive preprocessing pipeline."""
        pipeline = PreprocessingPipeline(df)

        if remove_duplicates:
            pipeline.remove_duplicates()

        if outlier_strategy in ["winsorize", "drop"]:
            pipeline.handle_outliers(method=outlier_strategy, factor=outlier_factor)

        if numeric_strategy != "none" or categorical_strategy != "none":
            strategy = numeric_strategy if numeric_strategy in ["mean", "median", "mode", "drop"] else "mean"
            pipeline.handle_missing_values(strategy=strategy)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if scale_features and num_cols:
            pipeline.scale_numerics(cols=num_cols, method=scale_method)

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        safe_cat_cols = [
            c for c in cat_cols
            if "id" not in c.lower() and "date" not in c.lower() and "notes" not in c.lower()
        ]
        if encode_categoricals and safe_cat_cols:
            pipeline.encode_categoricals(cols=safe_cat_cols, method=encode_method)

        cleaned_df = pipeline.get_processed_df()
        report = pipeline.get_preprocessing_report()
        return cleaned_df, report

    @staticmethod
    def normalize_features(
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        method: str = "standard"
    ) -> pd.DataFrame:
        """Scale/Normalize numerical columns using MinMaxScaler or StandardScaler."""
        pipeline = PreprocessingPipeline(df)
        target_cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        if target_cols:
            pipeline.scale_numerics(cols=target_cols, method=method)
        return pipeline.get_processed_df()

