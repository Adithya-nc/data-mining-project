"""
MarineMine Pro — Pydantic Request & Response Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dataset Schemas
# ---------------------------------------------------------------------------

class ColumnSchema(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    missing_count: int
    missing_pct: float
    role: Optional[str] = None


class DatasetInfoResponse(BaseModel):
    dataset_name: str
    name: Optional[str] = None
    total_records: int
    rows: Optional[int] = None
    total_columns: int
    columns: List[str]
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    schema_info: List[ColumnSchema]
    view_mode: str = "raw"
    is_benchmark: bool = False
    file_size_human: str = ""
    file_size: Optional[str] = ""


class DatasetPreviewResponse(BaseModel):
    columns: List[str]
    preview: List[Dict[str, Any]]
    total_rows: int


class ColumnMappingRequest(BaseModel):
    mappings: Optional[Dict[str, str]] = None
    mapping: Optional[Dict[str, str]] = None



# ---------------------------------------------------------------------------
# Data Quality Schemas
# ---------------------------------------------------------------------------

class QualitySummaryResponse(BaseModel):
    quality_score: float
    total_records: int
    total_features: int
    total_missing_cells: int
    missing_rate_pct: float
    duplicate_rows: int
    duplicate_rate_pct: float
    outlier_count_iqr: int
    completeness_pct: float
    column_missing_breakdown: List[Dict[str, Any]]
    passed_checks: List[str]
    warnings: List[str]
    errors: List[str]


# ---------------------------------------------------------------------------
# Data Preprocessing Schemas
# ---------------------------------------------------------------------------

class PreprocessCleanRequest(BaseModel):
    numeric_strategy: str = "mean"       # mean, median, mode, drop, none
    categorical_strategy: str = "mode"   # mode, unknown, drop, none
    remove_duplicates: bool = True
    outlier_strategy: str = "winsorize"  # winsorize, drop, none
    outlier_factor: float = 1.5
    scale_features: bool = False
    scale_method: str = "standard"       # standard, minmax
    encode_categoricals: bool = False
    encode_method: str = "label"         # label, onehot


class PreprocessNormalizeRequest(BaseModel):
    columns: Optional[List[str]] = None
    method: str = "standard"  # standard, minmax


class PreprocessResponse(BaseModel):
    message: str
    original_rows: int
    cleaned_rows: int
    original_missing: int
    cleaned_missing: int
    report: Dict[str, Any]


# ---------------------------------------------------------------------------
# EDA Schemas
# ---------------------------------------------------------------------------

class EDAStatisticsResponse(BaseModel):
    numerical_summary: List[Dict[str, Any]]
    categorical_summary: List[Dict[str, Any]]
    shape: List[int]


class EDACorrelationResponse(BaseModel):
    features: List[str]
    matrix: List[List[float]]


class EDADistributionResponse(BaseModel):
    feature: str
    bins: List[float]
    counts: List[int]
    mean: float
    std: float
    min: float
    max: float


class EDATrendsResponse(BaseModel):
    date_col: Optional[str] = None
    value_col: Optional[str] = None
    trends: List[Dict[str, Any]]
    seasonal: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Machine Learning Schemas
# ---------------------------------------------------------------------------

class ClusteringRequest(BaseModel):
    features: List[str]
    k: int = 3


class ClusteringResponse(BaseModel):
    k: int
    silhouette_score: float
    inertia: float
    cluster_counts: Dict[str, int]
    pca_data: List[Dict[str, Any]]
    cluster_profiles: List[Dict[str, Any]]
    feature_importances: Optional[Dict[str, float]] = None


class OptimalKResponse(BaseModel):
    k_range: List[int]
    inertias: List[float]
    silhouette_scores: List[float]
    recommended_k: int
    best_silhouette: float


class ClassificationRequest(BaseModel):
    target_col: str
    feature_cols: Optional[List[str]] = None
    test_size: float = 0.2


class ClassificationResponse(BaseModel):
    target_column: str
    classes: List[str]
    models: Dict[str, Any]
    best_model: str
    best_accuracy: float
    best_f1: float
    confusion_matrix: List[List[int]]
    feature_importance: List[Dict[str, Any]]
    elapsed_ms: int


class AssociationRequest(BaseModel):
    categorical_cols: Optional[List[str]] = None
    min_support: float = 0.05
    min_confidence: float = 0.4
    min_lift: float = 1.0


class AssociationResponse(BaseModel):
    rules_count: int
    rules: List[Dict[str, Any]]
    suitable: bool
    explanation: Optional[str] = None


class PredictionRequest(BaseModel):
    target_col: str
    feature_cols: Optional[List[str]] = None
    test_size: float = 0.2
    use_time_split: bool = False


class PredictionResponse(BaseModel):
    target_column: str
    models: Dict[str, Any]
    best_model: str
    best_r2: float
    best_rmse: float
    actual_vs_predicted: List[Dict[str, Any]]
    feature_importance: List[Dict[str, Any]]


class PredictionSimulateRequest(BaseModel):
    feature_values: Dict[str, float]


class PredictionSimulateResponse(BaseModel):
    predicted_value: float
    target_column: str
    active_model: str
    confidence_interval: List[float]


# ---------------------------------------------------------------------------
# Hotspot & Report Schemas
# ---------------------------------------------------------------------------

class HotspotPoint(BaseModel):
    lat: float
    lng: float
    val: float
    category: Optional[str] = None
    region: Optional[str] = None


class HotspotsResponse(BaseModel):
    lat_col: str
    lon_col: str
    val_col: str
    total_locations: int
    hotspots: List[Dict[str, Any]]
    points: List[HotspotPoint]


class ReportGenerateRequest(BaseModel):
    report_type: str = "full_audit"  # full_audit, eda, ml, hotspots
    format: str = "markdown"          # markdown, csv, excel
