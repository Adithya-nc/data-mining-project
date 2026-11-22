"""
MarineMine Pro — SQLAlchemy Database Models
"""

from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from app.database import Base


class DatasetModel(Base):
    """Stores uploaded and active dataset records."""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    total_rows = Column(Integer, default=0)
    total_columns = Column(Integer, default=0)
    is_active = Column(Boolean, default=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    metadata_rel = relationship("DatasetMetadataModel", back_populates="dataset", uselist=False, cascade="all, delete-orphan")
    analyses = relationship("AnalysisHistoryModel", back_populates="dataset", cascade="all, delete-orphan")
    models = relationship("ModelResultModel", back_populates="dataset", cascade="all, delete-orphan")
    reports = relationship("GeneratedReportModel", back_populates="dataset", cascade="all, delete-orphan")


class DatasetMetadataModel(Base):
    """Stores schema, roles, and statistical metadata for a dataset."""
    __tablename__ = "dataset_metadata"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, unique=True)
    columns_json = Column(Text, nullable=False, default="[]")
    dtypes_json = Column(Text, nullable=False, default="{}")
    summary_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("DatasetModel", back_populates="metadata_rel")


class AnalysisHistoryModel(Base):
    """Tracks executed data mining and exploratory analytics."""
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    analysis_type = Column(String(64), nullable=False)  # eda, quality, arm, hotspots
    parameters_json = Column(Text, nullable=False, default="{}")
    metrics_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("DatasetModel", back_populates="analyses")


class ModelResultModel(Base):
    """Stores trained ML models (Clustering, Classification, Regression)."""
    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    model_type = Column(String(64), nullable=False)  # clustering, classification, prediction
    algorithm = Column(String(64), nullable=False)
    target_column = Column(String(128), nullable=True)
    features_json = Column(Text, nullable=False, default="[]")
    metrics_json = Column(Text, nullable=False, default="{}")
    model_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("DatasetModel", back_populates="models")


class GeneratedReportModel(Base):
    """Stores generated report files and export artifacts."""
    __tablename__ = "generated_reports"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    report_type = Column(String(64), nullable=False)  # full_audit, eda, ml, hotspots
    format = Column(String(32), nullable=False)       # csv, pdf, excel, markdown
    file_path = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("DatasetModel", back_populates="reports")

