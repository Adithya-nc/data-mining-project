"""
MarineMine Pro — Supervised Classification Routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.db_models import ModelResultModel
from app.schemas.api_schemas import ClassificationRequest, ClassificationResponse
from app.services.ml_service import MLService
from app.services.state_manager import StateManager

router = APIRouter(prefix="/api/ml", tags=["Machine Learning - Classification"])
state = StateManager.get_instance()


@router.post("/classification", response_model=ClassificationResponse)
def train_classification_models(req: ClassificationRequest, db: Session = Depends(get_db)):
    """Train and evaluate Decision Tree, Random Forest, Logistic Regression classifiers."""
    df = state.get_active_df()
    try:
        result = MLService.run_classification(
            df=df,
            target_col=req.target_col,
            feature_cols=req.feature_cols,
            test_size=req.test_size,
        )
        state.classification_results = result

        # Record to SQLite database
        if state.dataset_id:
            try:
                rec = ModelResultModel(
                    dataset_id=state.dataset_id,
                    model_type="classification",
                    algorithm=result.get("best_model", "Classifier"),
                    target_column=req.target_col,
                    features_json=str(req.feature_cols or []),
                    metrics_json=f'{{"accuracy": {result["best_accuracy"]}, "f1_score": {result["best_f1"]}}}',
                )
                db.add(rec)
                db.commit()
            except Exception:
                db.rollback()

        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

