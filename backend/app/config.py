"""
MarineMine Pro — Application Configuration
"""

import os
from pathlib import Path

# Paths
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

UPLOADS_DIR = BACKEND_DIR / "uploads"
REPORTS_DIR = BACKEND_DIR / "reports"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SAMPLE_CSV = SAMPLE_DATA_DIR / "marine_pollution_sample.csv"
if not DEFAULT_SAMPLE_CSV.exists():
    fallback_csv = PROJECT_ROOT / "data" / "sample" / "marine_pollution_sample.csv"
    if fallback_csv.exists():
        DEFAULT_SAMPLE_CSV = fallback_csv

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'marinemine.db'}")

# File Restrictions
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {".csv"}

# API Metadata
APP_NAME = "MarineMine Pro"
APP_VERSION = "2.0.0"
API_TITLE = "MarineMine Pro API"
API_DESCRIPTION = "Comprehensive Oceanographic Data Mining, Machine Learning & Geospatial REST Backend"
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1")
CORS_ORIGINS = ["*", "http://localhost:3000", "http://localhost:5173", "http://localhost:8000"]

