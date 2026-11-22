"""
MarineMine Pro — Dataset Management Routes
"""

import io
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
import pandas as pd
from sqlalchemy.orm import Session

from app.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, UPLOADS_DIR
from app.database import get_db
from app.models.db_models import DatasetModel
from app.schemas.api_schemas import (
    ColumnMappingRequest,
    DatasetInfoResponse,
    DatasetPreviewResponse,
)
from app.services.data_loader import DataLoaderService
from app.services.state_manager import StateManager

router = APIRouter(prefix="/api/dataset", tags=["Dataset Management"])
state = StateManager.get_instance()


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload and ingest a custom CSV dataset with validation."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file submitted.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Only CSV (.csv) files are permitted."
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is completely empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 100 MB limit.")

    # Save to disk in uploads directory
    safe_name = f"{Path(file.filename).stem}_{len(content)}{ext}"
    dest_path = UPLOADS_DIR / safe_name
    with open(dest_path, "wb") as f:
        f.write(content)

    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV format: {exc}")

    if df.empty or df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Dataset must have at least 1 row and 2 columns.")

    state.set_active_dataset(df=df, name=file.filename, db=db, is_sample=False)
    info = DataLoaderService.inspect_dataset(df, dataset_name=file.filename)
    return {
        "message": f"Dataset '{file.filename}' successfully uploaded and ingested.",
        "filename": file.filename,
        "rows": len(df),
        "columns": df.shape[1],
        "info": info,
    }


@router.get("/info", response_model=DatasetInfoResponse)
def get_dataset_info():
    """Retrieve full metadata, column types, and schema of the active dataset."""
    df = state.get_active_df()
    info = DataLoaderService.inspect_dataset(df, dataset_name=state.dataset_name)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return {
        "dataset_name": state.dataset_name,
        "name": state.dataset_name,
        "total_records": info["total_records"],
        "rows": info["total_records"],
        "total_columns": info["total_columns"],
        "columns": info["columns"],
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "schema_info": info["schema_info"],
        "view_mode": state.view_mode,
        "is_benchmark": "Benchmark" in state.dataset_name,
        "file_size_human": info["file_size_human"],
        "file_size": info["file_size_human"],
    }



@router.get("/preview", response_model=DatasetPreviewResponse)
def get_dataset_preview(limit: int = Query(20, ge=1, le=100)):
    """Return the first N preview rows of the active dataset."""
    df = state.get_active_df()
    preview_df = df.head(limit).replace({pd.NA: None, float("nan"): None})
    return {
        "columns": df.columns.tolist(),
        "preview": preview_df.to_dict(orient="records"),
        "total_rows": len(df),
    }


@router.get("/columns")
def get_columns():
    """Return list of column names, numerical columns, and categorical columns."""
    df = state.get_active_df()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return {
        "all_columns": df.columns.tolist(),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
    }


@router.post("/map-columns")
def map_columns(req: ColumnMappingRequest):
    """Rename or map CSV column names to standardized application fields."""
    df = state.get_active_df()
    mapping_dict = req.mappings or req.mapping or {}
    valid_mappings = {k: v for k, v in mapping_dict.items() if k in df.columns and v and v.strip()}
    if not valid_mappings:
        raise HTTPException(status_code=400, detail="No valid column mappings provided.")


    renamed_df = df.rename(columns=valid_mappings)
    if state.view_mode == "cleaned" and state.df_processed is not None:
        state.df_processed = renamed_df
    else:
        state.df_raw = renamed_df
    return {
        "message": f"Successfully mapped {len(valid_mappings)} columns.",
        "columns": renamed_df.columns.tolist(),
    }


@router.post("/view-mode")
def set_view_mode(mode: str = Query("raw", pattern="^(raw|cleaned)$")):
    """Toggle between raw and cleaned telemetry views."""
    if mode == "cleaned" and state.df_processed is None:
        raise HTTPException(status_code=400, detail="No cleaned dataset generated yet. Run preprocessing first.")
    state.view_mode = mode
    return {"message": f"Active pipeline view switched to {mode.upper()}.", "view_mode": state.view_mode}


@router.post("/load-sample")
def load_sample_dataset(db: Session = Depends(get_db)):
    """Reset the platform with the standard benchmark oceanographic dataset."""
    df = DataLoaderService.load_default_sample()
    state.set_active_dataset(df=df, name="marine_pollution_sample.csv (Benchmark)", db=db, is_sample=True)
    return {
        "message": "Benchmark marine pollution dataset loaded successfully.",
        "records": len(df),
        "columns": df.shape[1],
    }


@router.get("/download")
def download_dataset():
    """Download the currently active dataset as CSV."""
    df = state.get_active_df()
    csv_str = df.to_csv(index=False)
    filename = f"{state.dataset_name.split()[0]}.csv"
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.delete("")
def delete_dataset(db: Session = Depends(get_db)):
    """Reset platform to default sample dataset."""
    df = DataLoaderService.load_default_sample()
    state.set_active_dataset(df=df, name="marine_pollution_sample.csv (Benchmark)", db=db, is_sample=True)
    return {"message": "Custom dataset discarded; platform reverted to benchmark baseline."}
