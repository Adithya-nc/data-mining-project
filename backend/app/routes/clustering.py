"""
MarineMine Pro — K-Means Clustering Routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.db_models import ModelResultModel
from app.schemas.api_schemas import ClusteringRequest, ClusteringResponse, OptimalKResponse
from app.services.ml_service import MLService
from app.services.state_manager import StateManager

router = APIRouter(prefix="/api/ml", tags=["Machine Learning - Clustering"])
state = StateManager.get_instance()


@router.post("/clustering", response_model=ClusteringResponse)
def run_clustering(req: ClusteringRequest, db: Session = Depends(get_db)):
    """Run K-Means clustering and save model telemetry to SQLite."""
    df = state.get_active_df()
    try:
        result = MLService.run_clustering(df, features=req.features, k=req.k)
        state.clustering_results = result

        # Record to database
        if state.dataset_id:
            try:
                rec = ModelResultModel(
                    dataset_id=state.dataset_id,
                    model_type="clustering",
                    algorithm="KMeans",
                    target_column=None,
                    features_json=str(req.features),
                    metrics_json=f'{{"k": {req.k}, "silhouette_score": {result["silhouette_score"]}}}',
                )
                db.add(rec)
                db.commit()
            except Exception:
                db.rollback()

        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/clustering/optimal-k", response_model=OptimalKResponse)
def find_optimal_clusters(req: ClusteringRequest):
    """Auto-detect optimal k via elbow inertia and silhouette coefficients."""
    df = state.get_active_df()
    try:
        opt_k = MLService.find_optimal_k(df, features=req.features)
        return opt_k
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

