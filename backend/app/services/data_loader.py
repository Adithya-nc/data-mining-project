"""
MarineMine Pro — Data Loader Service
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from app.config import DEFAULT_SAMPLE_CSV
from src import data_loader, data_validation


class DataLoaderService:
    @staticmethod
    def load_dataset(file_path: Path) -> pd.DataFrame:
        """Load CSV dataset into a Pandas DataFrame."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        return df

    @staticmethod
    def load_default_sample() -> pd.DataFrame:
        """Load the standard marine pollution benchmark dataset."""
        if DEFAULT_SAMPLE_CSV.exists():
            return pd.read_csv(DEFAULT_SAMPLE_CSV, low_memory=False)
        return data_loader.load_sample_dataset()

    @staticmethod
    def inspect_dataset(df: pd.DataFrame, dataset_name: str = "dataset.csv") -> Dict[str, Any]:
        """Compute full structural overview and column schema."""
        info = data_loader.get_dataset_info(df, dataset_name=dataset_name)
        roles = data_loader.infer_column_roles(df)

        schema_list = []
        for col in df.columns:
            missing_cnt = int(df[col].isna().sum())
            non_null = len(df) - missing_cnt
            pct = round(missing_cnt / len(df) * 100, 2) if len(df) > 0 else 0.0
            schema_list.append({
                "name": str(col),
                "dtype": str(df[col].dtype),
                "non_null_count": non_null,
                "missing_count": missing_cnt,
                "missing_pct": pct,
                "role": roles.get(col, "feature"),
            })

        return {
            "dataset_name": dataset_name,
            "total_records": len(df),
            "total_columns": df.shape[1],
            "columns": df.columns.tolist(),
            "schema_info": schema_list,
            "file_size_human": info.get("memory_usage_human", "N/A"),
        }

    @staticmethod
    def validate_csv(df: pd.DataFrame) -> Dict[str, Any]:
        """Run structural validation against corruptions, empty columns, and duplicate rows."""
        audit = data_validation.audit_data_quality(df)
        return {
            "is_valid": audit.overall_score >= 40.0,
            "quality_score": audit.overall_score,
            "passed_checks": [c.name for c in audit.checks if c.passed],
            "warnings": [f"{c.name}: {c.message}" for c in audit.checks if not c.passed and c.severity == "warning"],
            "errors": [f"{c.name}: {c.message}" for c in audit.checks if not c.passed and c.severity == "error"],
        }

