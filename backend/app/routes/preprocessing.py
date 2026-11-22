"""
MarineMine Pro — Data Preprocessing Routes
"""

from fastapi import APIRouter, HTTPException, Query, Response
import pandas as pd
from app.schemas.api_schemas import (
    PreprocessCleanRequest,
    PreprocessNormalizeRequest,
    PreprocessResponse,
)
from app.services.data_cleaner import DataCleanerService
from app.services.state_manager import StateManager

router = APIRouter(prefix="/api/preprocessing", tags=["Data Preprocessing"])
state = StateManager.get_instance()


@router.post("/clean", response_model=PreprocessResponse)
def clean_dataset(req: PreprocessCleanRequest):
    """Execute complete data cleaning, imputation, outlier clipping, and encoding pipeline."""
    if state.df_raw is None:
        state.initialize_default()

    cleaned_df, report = DataCleanerService.execute_pipeline(
        df=state.df_raw,
        numeric_strategy=req.numeric_strategy,
        categorical_strategy=req.categorical_strategy,
        remove_duplicates=req.remove_duplicates,
        outlier_strategy=req.outlier_strategy,
        outlier_factor=req.outlier_factor,
        scale_features=req.scale_features,
        scale_method=req.scale_method,
        encode_categoricals=req.encode_categoricals,
        encode_method=req.encode_method,
    )

    state.df_processed = cleaned_df
    state.preprocessing_report = report
    state.view_mode = "cleaned"

    return {
        "message": "Preprocessing pipeline executed successfully.",
        "original_rows": len(state.df_raw),
        "cleaned_rows": len(cleaned_df),
        "original_missing": int(state.df_raw.isna().sum().sum()),
        "cleaned_missing": int(cleaned_df.isna().sum().sum()),
        "report": report,
    }


@router.post("/normalize")
def normalize_features(req: PreprocessNormalizeRequest):
    """Normalize or standardize specific numerical columns."""
    df = state.get_active_df()
    scaled_df = DataCleanerService.normalize_features(
        df=df,
        cols=req.columns,
        method=req.method,
    )
    state.df_processed = scaled_df
    state.view_mode = "cleaned"
    return {
        "message": f"Features scaled successfully using {req.method} method.",
        "scaled_columns": req.columns or scaled_df.select_dtypes(include=["number"]).columns.tolist(),
        "shape": list(scaled_df.shape),
    }


@router.get("/preview")
def get_cleaned_preview(limit: int = Query(20, ge=1, le=100)):
    """Preview the cleaned dataset."""
    if state.df_processed is None:
        raise HTTPException(status_code=400, detail="No cleaned dataset generated yet. Run /clean first.")
    preview_df = state.df_processed.head(limit).replace({pd.NA: None, float("nan"): None})
    return {
        "columns": state.df_processed.columns.tolist(),
        "preview": preview_df.to_dict(orient="records"),
        "total_rows": len(state.df_processed),
    }


@router.get("/download")
def download_cleaned_dataset():
    """Download cleaned dataset as CSV."""
    if state.df_processed is None:
        raise HTTPException(status_code=400, detail="No cleaned dataset available to download.")
    csv_str = state.df_processed.to_csv(index=False)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_marine_pollution.csv"}
    )

