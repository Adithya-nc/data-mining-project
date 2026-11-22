"""
MarineMine Pro — Application State & Cache Manager
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from sqlalchemy.orm import Session
from app.config import DEFAULT_SAMPLE_CSV, UPLOADS_DIR
from app.models.db_models import DatasetModel, DatasetMetadataModel
from app.services.data_loader import DataLoaderService


class StateManager:
    _instance: Optional["StateManager"] = None

    def __init__(self):
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.dataset_name: str = "marine_pollution_sample.csv"
        self.dataset_id: Optional[int] = None
        self.view_mode: str = "raw"  # "raw" or "cleaned"
        self.column_roles: Dict[str, str] = {}
        self.preprocessing_report: Optional[Dict[str, Any]] = None
        self.clustering_results: Optional[Dict[str, Any]] = None
        self.classification_results: Optional[Dict[str, Any]] = None
        self.prediction_results: Optional[Dict[str, Any]] = None
        self.association_rules_results: Optional[Dict[str, Any]] = None
        self.trained_prediction_model: Any = None
        self.prediction_feature_names: list[str] = []

    @classmethod
    def get_instance(cls) -> "StateManager":
        if cls._instance is None:
            cls._instance = StateManager()
        return cls._instance

    def initialize_default(self, db: Optional[Session] = None):
        """Ensure sample benchmark dataset is loaded into state and registered in database."""
        if self.df_raw is None:
            df = DataLoaderService.load_default_sample()
            self.set_active_dataset(
                df=df,
                name="marine_pollution_sample.csv (Benchmark)",
                db=db,
                is_sample=True,
            )

    def set_active_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        db: Optional[Session] = None,
        is_sample: bool = False,
    ):
        """Set active dataset and persist metadata to SQLite."""
        self.df_raw = df.copy()
        self.df_processed = None
        self.dataset_name = name
        self.view_mode = "raw"
        self.preprocessing_report = None
        self.clustering_results = None
        self.classification_results = None
        self.prediction_results = None
        self.association_rules_results = None
        self.trained_prediction_model = None

        if db is not None:
            try:
                # Deactivate previously active datasets
                db.query(DatasetModel).update({DatasetModel.is_active: False})

                # Register or retrieve
                dataset_record = db.query(DatasetModel).filter(DatasetModel.filename == name).first()
                if not dataset_record:
                    file_path = str(DEFAULT_SAMPLE_CSV) if is_sample else str(UPLOADS_DIR / name)
                    dataset_record = DatasetModel(
                        filename=name,
                        original_name=name,
                        file_path=file_path,
                        file_size_bytes=int(df.memory_usage(deep=True).sum()),
                        total_rows=len(df),
                        total_columns=df.shape[1],
                        is_active=True,
                    )
                    db.add(dataset_record)
                    db.commit()
                    db.refresh(dataset_record)
                else:
                    dataset_record.is_active = True
                    dataset_record.total_rows = len(df)
                    dataset_record.total_columns = df.shape[1]
                    db.commit()

                self.dataset_id = dataset_record.id

                # Save metadata
                meta = db.query(DatasetMetadataModel).filter(DatasetMetadataModel.dataset_id == dataset_record.id).first()
                info = DataLoaderService.inspect_dataset(df, dataset_name=name)
                if not meta:
                    meta = DatasetMetadataModel(
                        dataset_id=dataset_record.id,
                        columns_json=json.dumps(df.columns.tolist()),
                        dtypes_json=json.dumps({c: str(df[c].dtype) for c in df.columns}),
                        summary_json=json.dumps(info),
                    )
                    db.add(meta)
                else:
                    meta.columns_json = json.dumps(df.columns.tolist())
                    meta.dtypes_json = json.dumps({c: str(df[c].dtype) for c in df.columns})
                    meta.summary_json = json.dumps(info)
                db.commit()
            except Exception as exc:
                db.rollback()
                print(f"Warning: SQLite sync error: {exc}")

    def has_dataset(self) -> bool:
        """Check if any valid dataset is active in memory."""
        return self.df_raw is not None or self.df_processed is not None

    def load_sample(self, sample_path: Optional[Path] = None, db: Optional[Session] = None):
        """Load benchmark dataset from disk."""
        df = DataLoaderService.load_default_sample()
        name = sample_path.name if sample_path else "marine_pollution_sample.csv (Benchmark)"
        self.set_active_dataset(df=df, name=name, db=db, is_sample=True)

    def get_active_df(self) -> pd.DataFrame:
        """Return the current working DataFrame based on active view mode."""
        if self.view_mode == "cleaned" and self.df_processed is not None:
            return self.df_processed
        if self.df_raw is not None:
            return self.df_raw
        if self.df_processed is not None:
            return self.df_processed
        self.initialize_default()
        return self.df_raw
