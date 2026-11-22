"""
MarineMine Pro — Report Service (Markdown, CSV, Excel)
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from app.config import REPORTS_DIR
from src import report_generator, data_loader, eda


class ReportService:
    @staticmethod
    def generate_markdown(
        df: pd.DataFrame,
        dataset_name: str,
        preprocessing_report: Optional[Dict[str, Any]] = None,
        clustering_results: Optional[Dict[str, Any]] = None,
        classification_results: Optional[Dict[str, Any]] = None,
        prediction_results: Optional[Dict[str, Any]] = None,
        association_rules_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate structured analytical Markdown report."""
        dataset_info = data_loader.get_dataset_info(df, dataset_name=dataset_name)
        eda_sum = eda.get_eda_summary(df)

        return report_generator.generate_text_report(
            dataset_info=dataset_info,
            preprocessing_report=preprocessing_report,
            eda_summary=eda_sum,
            clustering_results=clustering_results,
            classification_results=classification_results,
            prediction_results=prediction_results,
            association_rules_summary=association_rules_summary,
        )

    @staticmethod
    def export_csv(df: pd.DataFrame, file_name: str = "cleaned_marine_pollution.csv") -> Path:
        """Export DataFrame to CSV file in reports directory."""
        target_path = REPORTS_DIR / file_name
        df.to_csv(target_path, index=False)
        return target_path

    @staticmethod
    def export_excel(df: pd.DataFrame, file_name: str = "marine_pollution_export.xlsx") -> Path:
        """Export DataFrame to Excel (.xlsx)."""
        target_path = REPORTS_DIR / file_name
        df.to_excel(target_path, index=False, engine="openpyxl")
        return target_path

