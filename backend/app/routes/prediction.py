"""
MarineMine Pro — Continuous Pollution Prediction Routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.db_models import ModelResultModel
from app.schemas.api_schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionSimulateRequest,
    PredictionSimulateResponse,
)
from app.services.prediction_service import PredictionService
from app.services.state_manager import StateManager
from src import prediction as src_prediction

router = APIRouter(prefix="/api/ml", tags=["Machine Learning - Prediction"])
state = StateManager.get_instance()
trained_regressor = None
last_feature_names = []


@router.post("/prediction", response_model=PredictionResponse)
def train_prediction_models(req: PredictionRequest, db: Session = Depends(get_db)):
    """Train and evaluate regression models (Linear, RF, GB, Ridge) on pollution metrics."""
    global trained_regressor, last_feature_names
    df = state.get_active_df()
    try:
        result = PredictionService.train_models(
            df=df,
            target_col=req.target_col,
            feature_cols=req.feature_cols,
            test_size=req.test_size,
            use_time_split=req.use_time_split,
        )
        state.prediction_results = result

        # Cache trained estimator for What-If simulator
        feat_cols = req.feature_cols or [f["feature"] for f in result.get("feature_importance", [])]
        last_feature_names = feat_cols
        try:
            reg_dict = src_prediction.train_regressors(
                df=df,
                feature_cols=feat_cols,
                target_col=req.target_col,
                test_size=req.test_size,
                use_time_split=req.use_time_split,
            )
            trained_regressor = reg_dict.get("Random Forest", {}).get("model") or next(iter(reg_dict.values()))["model"]
        except Exception:
            trained_regressor = None


        # Record to SQLite database
        if state.dataset_id:
            try:
                rec = ModelResultModel(
                    dataset_id=state.dataset_id,
                    model_type="prediction",
                    algorithm=result.get("best_model", "Regressor"),
                    target_column=req.target_col,
                    features_json=str(feat_cols),
                    metrics_json=f'{{"r2": {result["best_r2"]}, "rmse": {result["best_rmse"]}}}',
                )
                db.add(rec)
                db.commit()
            except Exception:
                db.rollback()

        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/prediction/simulate", response_model=PredictionSimulateResponse)
def simulate_prediction(req: PredictionSimulateRequest):
    """Run interactive What-If scenario simulation."""
    global trained_regressor, last_feature_names
    features = list(req.feature_values.keys()) if not last_feature_names else last_feature_names
    try:
        sim = PredictionService.simulate_prediction(
            trained_model=trained_regressor,
            feature_names=features,
            feature_values=req.feature_values,
        )
        return sim
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

