"""
MarineMine Pro — FastAPI Integration & Unit Test Suite
Verifies all REST API endpoints, ML engines, data cleaning, and frontend static mounting.
"""

import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

# Ensure sys.path points to backend and root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import app
from app.services.state_manager import StateManager

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_platform():
    """Ensure platform has sample data initialized before tests."""
    state = StateManager.get_instance()
    state.initialize_default()


def test_health_check():
    """Verify system health telemetry."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["app"] == "MarineMine Pro"
    assert data["dataset_loaded"] is True
    assert data["rows"] > 0


def test_frontend_serving():
    """Verify that root URL serves the HTML5 ocean dashboard."""
    res = client.get("/")
    assert res.status_code == 200
    assert "MarineMine Pro" in res.text
    assert "section-dashboard" in res.text


def test_dataset_info_and_preview():
    """Verify dataset metadata and preview rows."""
    res = client.get("/api/dataset/info")
    assert res.status_code == 200
    data = res.json()
    assert data["total_records"] > 0
    assert len(data["columns"]) >= 5

    res_prev = client.get("/api/dataset/preview?limit=10")
    assert res_prev.status_code == 200
    prev_data = res_prev.json()
    assert len(prev_data["preview"]) == 10


def test_data_quality_summary():
    """Verify quality score calculation and telemetry."""
    res = client.get("/api/quality/summary")
    assert res.status_code == 200
    data = res.json()
    assert 0 <= data["quality_score"] <= 100
    assert data["total_records"] > 0
    assert "column_missing_breakdown" in data
    assert isinstance(data["passed_checks"], list)


def test_preprocessing_pipeline():
    """Verify missing value imputation, deduplication, and scaling."""
    payload = {
        "numeric_strategy": "mean",
        "categorical_strategy": "mode",
        "remove_duplicates": True,
        "outlier_strategy": "winsorize",
        "outlier_factor": 1.5,
        "scale_features": False,
        "scale_method": "standard",
        "encode_categoricals": False,
        "encode_method": "label",
    }
    res = client.post("/api/preprocessing/clean", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["cleaned_rows"] > 0
    assert data["cleaned_missing"] == 0  # Imputed completely


def test_eda_statistics_and_correlation():
    """Verify summary statistics and Pearson correlation matrix."""
    res_stats = client.get("/api/eda/statistics")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert len(stats["numerical_summary"]) > 0

    res_corr = client.get("/api/eda/correlation")
    assert res_corr.status_code == 200
    corr = res_corr.json()
    assert len(corr["features"]) > 0
    assert len(corr["matrix"]) == len(corr["features"])

    # Distribution for a feature
    feat = corr["features"][0]
    res_dist = client.get(f"/api/eda/distribution?feature={feat}")
    assert res_dist.status_code == 200
    dist = res_dist.json()
    assert len(dist["bins"]) > 0
    assert len(dist["counts"]) > 0


def test_kmeans_clustering():
    """Verify K-Means clustering, silhouette score, and PCA projection."""
    payload = {
        "features": ["temperature_celsius", "salinity_ppt", "ph_level"],
        "k": 3,
    }
    res = client.post("/api/ml/clustering", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["k"] == 3
    assert -1.0 <= data["silhouette_score"] <= 1.0
    assert len(data["cluster_profiles"]) == 3
    assert len(data["pca_data"]) > 0


def test_kmeans_optimal_k():
    """Verify optimal K detection curve."""
    payload = {
        "features": ["temperature_celsius", "salinity_ppt", "ph_level"],
        "k": 3,
    }
    res = client.post("/api/ml/clustering/optimal-k", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["k_range"]) >= 4
    assert 2 <= data["recommended_k"] <= 8


def test_supervised_classification():
    """Verify multi-model classification benchmark."""
    payload = {
        "target_col": "pollution_type",
        "feature_cols": ["temperature_celsius", "salinity_ppt", "ph_level", "dissolved_oxygen"],
        "test_size": 0.2,
    }
    res = client.post("/api/ml/classification", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert 0.0 <= data["best_accuracy"] <= 1.0
    assert len(data["confusion_matrix"]) > 0


def test_association_rule_mining():
    """Verify Apriori frequent itemset mining."""
    payload = {
        "categorical_cols": ["region", "pollution_type"],
        "min_support": 0.02,
        "min_confidence": 0.2,
        "min_lift": 1.0,
    }
    res = client.post("/api/ml/association", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "rules_count" in data
    assert "rules" in data


def test_pollution_prediction_and_simulator():
    """Verify continuous regression and interactive simulator."""
    payload = {
        "target_col": "pollution_value",
        "feature_cols": ["temperature_celsius", "salinity_ppt", "ph_level", "dissolved_oxygen"],
        "test_size": 0.2,
        "use_time_split": False,
    }
    res = client.post("/api/ml/prediction", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["best_r2"] is not None
    assert len(data["actual_vs_predicted"]) > 0

    # Test What-If Simulator
    sim_payload = {
        "feature_values": {
            "temperature_celsius": 24.5,
            "salinity_ppt": 35.0,
            "ph_level": 8.1,
            "dissolved_oxygen": 6.5,
        }
    }
    res_sim = client.post("/api/ml/prediction/simulate", json=sim_payload)
    assert res_sim.status_code == 200
    sim_data = res_sim.json()
    assert "predicted_value" in sim_data
    assert len(sim_data["confidence_interval"]) == 2


def test_geographical_hotspots():
    """Verify hotspot spatial risk ranking and coordinate points."""
    res = client.get("/api/hotspots")
    assert res.status_code == 200
    data = res.json()
    assert data["total_locations"] > 0
    assert len(data["points"]) > 0
    assert len(data["hotspots"]) > 0


def test_reports_generation():
    """Verify Markdown and CSV report generation."""
    # Markdown Report
    res_md = client.post("/api/reports/generate", json={"report_type": "full_audit", "format": "markdown"})
    assert res_md.status_code == 200
    data_md = res_md.json()
    assert data_md["format"] == "markdown"
    assert "preview" in data_md

    # CSV Export
    res_csv = client.post("/api/reports/generate", json={"report_type": "full_audit", "format": "csv"})
    assert res_csv.status_code == 200
    data_csv = res_csv.json()
    assert data_csv["format"] == "csv"
    assert "download_url" in data_csv
