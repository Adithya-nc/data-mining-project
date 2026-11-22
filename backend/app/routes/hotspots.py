"""
MarineMine Pro — Geographical Hotspot Analysis Routes
"""

from typing import Optional
from fastapi import APIRouter, Query
from app.schemas.api_schemas import HotspotsResponse
from app.services.data_analyzer import DataAnalyzerService
from app.services.state_manager import StateManager

router = APIRouter(prefix="/api", tags=["Geographical Hotspots"])
state = StateManager.get_instance()


@router.get("/hotspots", response_model=HotspotsResponse)
def get_pollution_hotspots(
    region: Optional[str] = Query(None, description="Region filter"),
    type: Optional[str] = Query(None, description="Pollution type filter"),
):
    """Extract spatial risk hotspots and coordinate markers."""
    df = state.get_active_df()
    return DataAnalyzerService.extract_hotspots(df, region_filter=region, type_filter=type)

