"""
MarineMine Pro — FastAPI Application Entrypoint
Mining Patterns of Marine Pollution Using Data Mining & Machine Learning
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import (
    APP_NAME,
    APP_VERSION,
    DEBUG,
    CORS_ORIGINS,
    FRONTEND_DIR,
    REPORTS_DIR,
    UPLOADS_DIR,
)
from app.database import init_db
from app.routes import all_routers
from app.services.state_manager import StateManager

# Create FastAPI instance
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Mining Patterns of Marine Pollution Using Data Mining & Machine Learning REST API",
    debug=DEBUG,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routers
for r in all_routers:
    app.include_router(r)


@app.on_event("startup")
def on_startup():
    """Initialize SQLite database tables and load sample dataset if needed."""
    init_db()
    state = StateManager.get_instance()
    try:
        sample_path = Path("sample_data/marine_pollution_sample.csv")
        if not sample_path.exists():
            sample_path = Path("data/sample/marine_pollution_sample.csv")
        if sample_path.exists() and not state.has_dataset():
            state.load_sample(sample_path)
    except Exception as exc:
        print(f"[Startup Warning] Could not preload sample dataset: {exc}")


@app.get("/api/health")
def health_check():
    """Health status and telemetry."""
    state = StateManager.get_instance()
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "dataset_loaded": state.has_dataset(),
        "dataset_name": state.dataset_name if state.has_dataset() else None,
        "rows": len(state.get_active_df()) if state.has_dataset() else 0,
    }


# ---------------------------------------------------------------------------
# Backward Compatibility Route Aliases
# ---------------------------------------------------------------------------
from app.routes.dataset import get_dataset_info, load_sample_dataset
from app.routes.quality import get_quality_summary
from app.routes.preprocessing import clean_dataset
from app.routes.eda import get_eda_statistics
from app.routes.clustering import run_clustering
from app.routes.classification import train_classification_models
from app.routes.association import run_association_mining
from app.routes.prediction import train_prediction_models, simulate_prediction
from app.routes.reports import generate_analytical_report, download_generated_report
from app.schemas.api_schemas import (
    PreprocessCleanRequest,
    ClusteringRequest,
    ClassificationRequest,
    AssociationRequest,
    PredictionRequest,
    PredictionSimulateRequest,
    ReportGenerateRequest,
)
from fastapi.responses import PlainTextResponse, Response
from app.services.report_service import ReportService

@app.get("/api/dataset", include_in_schema=False)
def alias_dataset_info():
    return get_dataset_info()

@app.post("/api/dataset/load-benchmark", include_in_schema=False)
def alias_load_benchmark():
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        return load_sample_dataset(db=db)
    finally:
        db.close()


@app.get("/api/quality", include_in_schema=False)
def alias_quality():
    return get_quality_summary()

@app.post("/api/preprocess", include_in_schema=False)
def alias_preprocess(req: PreprocessCleanRequest):
    return clean_dataset(req)

@app.get("/api/eda", include_in_schema=False)
def alias_eda():
    return get_eda_statistics()

@app.post("/api/clustering", include_in_schema=False)
def alias_clustering(req: ClusteringRequest):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        return run_clustering(req, db=db)
    finally:
        db.close()

@app.post("/api/classification", include_in_schema=False)
def alias_classification(req: ClassificationRequest):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        return train_classification_models(req, db=db)
    finally:
        db.close()

@app.post("/api/association-rules", include_in_schema=False)
def alias_association(req: AssociationRequest):
    return run_association_mining(req)

@app.post("/api/prediction", include_in_schema=False)
def alias_prediction(req: PredictionRequest):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        return train_prediction_models(req, db=db)
    finally:
        db.close()

@app.get("/api/report/markdown", include_in_schema=False)
def alias_report_markdown():
    state = StateManager.get_instance()
    df = state.get_active_df()
    content = ReportService.generate_markdown(
        df=df,
        dataset_name=state.dataset_name,
        preprocessing_report=state.preprocessing_report,
        clustering_results=state.clustering_results,
        classification_results=state.classification_results,
        prediction_results=state.prediction_results,
        association_rules_summary=state.association_rules_results,
    )
    return PlainTextResponse(content)

@app.get("/api/report/csv", include_in_schema=False)
def alias_report_csv():
    state = StateManager.get_instance()
    df = state.get_active_df()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="Cleaned_{state.dataset_name}"'},
    )


# Mount frontend static directory if exists
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

