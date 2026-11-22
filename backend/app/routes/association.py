"""
MarineMine Pro — Association Rule Mining Routes
"""

from fastapi import APIRouter, HTTPException
from app.schemas.api_schemas import AssociationRequest, AssociationResponse
from app.services.ml_service import MLService
from app.services.state_manager import StateManager

router = APIRouter(prefix="/api/ml", tags=["Machine Learning - Association"])
state = StateManager.get_instance()


@router.post("/association", response_model=AssociationResponse)
def run_association_mining(req: AssociationRequest):
    """Run Apriori frequent itemset and association rule mining."""
    df = state.get_active_df()
    try:
        result = MLService.run_association_rules(
            df=df,
            cat_cols=req.categorical_cols,
            min_support=req.min_support,
            min_confidence=req.min_confidence,
            min_lift=req.min_lift,
        )
        state.association_rules_summary = result
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

