"""
MarineMine Pro — Exploratory Data Analysis Routes
"""

from fastapi import APIRouter, HTTPException, Query
from app.schemas.api_schemas import (
    EDAStatisticsResponse,
    EDACorrelationResponse,
    EDADistributionResponse,
    EDATrendsResponse,
)
from app.services.data_analyzer import DataAnalyzerService
from app.services.state_manager import StateManager

router = APIRouter(prefix="/api/eda", tags=["Exploratory Data Analysis"])
state = StateManager.get_instance()


@router.get("/statistics", response_model=EDAStatisticsResponse)
def get_eda_statistics():
    """Compute statistical analysis: mean, median, mode, std, quartiles, IQR, unique counts."""
    df = state.get_active_df()
    stats = DataAnalyzerService.get_summary_statistics(df)
    return stats


@router.get("/correlation", response_model=EDACorrelationResponse)
def get_eda_correlation():
    """Compute full correlation matrix across numerical variables."""
    df = state.get_active_df()
    return DataAnalyzerService.compute_correlation_matrix(df)


@router.get("/distribution", response_model=EDADistributionResponse)
def get_feature_distribution(
    feature: str = Query(..., description="Numerical column to calculate distribution for"),
    bins: int = Query(15, ge=5, le=50)
):
    """Retrieve histogram density and statistical bounds for a feature."""
    df = state.get_active_df()
    try:
        dist = DataAnalyzerService.compute_feature_distribution(df, feature=feature, bins_count=bins)
        return dist
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/trends", response_model=EDATrendsResponse)
def get_eda_trends():
    """Calculate temporal trends, category distribution, and regional breakdown."""
    df = state.get_active_df()
    return DataAnalyzerService.compute_trends(df)

