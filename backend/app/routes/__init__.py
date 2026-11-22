"""
MarineMine Pro — Route Aggregator
"""

from app.routes.dataset import router as dataset_router
from app.routes.quality import router as quality_router
from app.routes.preprocessing import router as preprocessing_router
from app.routes.eda import router as eda_router
from app.routes.clustering import router as clustering_router
from app.routes.classification import router as classification_router
from app.routes.association import router as association_router
from app.routes.prediction import router as prediction_router
from app.routes.hotspots import router as hotspots_router
from app.routes.reports import router as reports_router

all_routers = [
    dataset_router,
    quality_router,
    preprocessing_router,
    eda_router,
    clustering_router,
    classification_router,
    association_router,
    prediction_router,
    hotspots_router,
    reports_router,
]

__all__ = ["all_routers"]

