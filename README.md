# 🌊 MarineMine Pro: Mining Patterns of Marine Pollution Using Data Mining & Machine Learning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-0A3D62.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy_2.0-003B57.svg)](https://www.sqlalchemy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![MLxtend](https://img.shields.io/badge/MLxtend-0.23+-1E90FF.svg)](http://rasbt.github.io/mlxtend/)
[![Bootstrap 5](https://img.shields.io/badge/Bootstrap-5.3-7952B3.svg)](https://getbootstrap.com/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384.svg)](https://www.chartjs.org/)
[![Leaflet.js](https://img.shields.io/badge/Leaflet-1.9-199900.svg)](https://leafletjs.com/)

> **MarineMine Pro** is a modern, professional, oceanographic web application engineered to discover hidden patterns, spatial-temporal trends, and environmental risk concentrations in marine pollution datasets using advanced Data Mining, Machine Learning, and GIS analytics.

---

## 📌 Architecture & Design Overview

MarineMine Pro operates on a decoupled REST architecture:
- **Backend:** Python 3.11+ powered by **FastAPI** and **Uvicorn**, backed by an **SQLite** database using **SQLAlchemy 2.0** ORM for tracking datasets, metadata, model results, and generated reports.
- **Frontend:** Premium, ocean-themed dashboard constructed with **HTML5, CSS3, Vanilla JavaScript, Bootstrap 5, Chart.js**, and **Leaflet.js**, featuring deep navy backgrounds, teal and cyan accents, glassmorphic cards, and interactive controls.
- **Machine Learning Suite:** Scikit-learn pipelines with zero data leakage, automated Elbow/Silhouette K-Means clustering, PCA 2D projections, supervised classification benchmarking (Decision Tree, Random Forest, Logistic Regression), Apriori association rule mining via MLxtend, continuous regression forecasting, and geospatial hotspot clustering.

---

## 🚀 Quick Start & Execution

### 1. Prerequisites
Ensure you have Python 3.11+ installed. Activate the virtual environment:

```bash
# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Launch Application
Run the backend with FastAPI and Uvicorn:

```bash
# Direct FastAPI Execution
python -m uvicorn app.main:app --app-dir backend --port 8000 --reload

# Or via Root Server
python server.py
```

Open your browser and navigate to:
```
http://localhost:8000/
```
The complete interactive HTML5 ocean dashboard will load immediately with the preloaded marine pollution benchmark dataset.

---

## 🧭 Dashboard Modules (12-Section Navigation)

1. **Dashboard:** Interactive telemetry counters (Records, Regions, Categories, Missing Data, Quality Score, Hotspots), category breakdown doughnut chart, observations timeline line chart, regional bar chart, and environmental parameter trends.
2. **Dataset Management:** Drag-and-drop CSV ingestion, file validation, metadata summary, real-time 15-record preview, and interactive column-mapping mechanism.
3. **Data Quality Audit:** Comprehensive quality score (0–100), completeness rate, IQR outlier detection, duplicate row verification, missingness breakdown chart, and validation check logs.
4. **Data Preprocessing:** Configurable missing value imputation (Mean, Median, Mode, Drop), Winsorizing / removal of outliers, Z-score / MinMax scaling, categorical encoding, and before/after metric comparisons.
5. **Exploratory Data Analysis (EDA):** Numerical summary statistics table (Mean, Std, Min, 25%, Median, 75%, Max, IQR), Pearson correlation matrix heatmap, feature distribution histograms, and categorical frequency profiles.
6. **Clustering Analysis:** Standardized K-Means clustering, automated Elbow Method and Silhouette curve evaluation (K=2 to 8), interactive PCA 2D scatter plots, and cluster characteristic profiles.
7. **Supervised Classification:** Benchmarking of Decision Tree, Random Forest, and Logistic Regression models, accuracy/precision/recall/F1 comparison table, confusion matrix grid, and relative feature importance charts.
8. **Association Rule Mining:** Apriori algorithm via MLxtend discovering co-occurring environmental conditions, with tunable minimum support, confidence, and lift thresholds.
9. **Pollution Prediction:** Benchmarking of continuous regression models (Linear Regression, Random Forest Regressor), chronological time-split support, actual vs. predicted comparison curves, and an interactive **What-If Scenario Simulator**.
10. **Geographical Hotspot Analysis:** Leaflet.js interactive dark ocean map with color-coded intensity markers, coordinate popups, region/pollutant filters, and a ranked risk zone leaderboard.
11. **Report Generation:** One-click export of Executive Markdown (`.md`) audit reports, cleaned CSV datasets, and Excel analytics workbooks (`.xlsx`) with live interactive preview.
12. **About Project:** Architecture documentation, mathematical principles, data integrity standards, and platform specifications.

---

## 📁 Repository Directory Structure

```text
MarineMine-Pro/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application entrypoint & static mounting
│   │   ├── config.py            # Environment, paths, database, and CORS configs
│   │   ├── database.py          # SQLAlchemy 2.0 engine and session management
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── db_models.py     # SQLite schema definitions
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── api_schemas.py   # Pydantic request & response models
│   │   ├── routes/
│   │   │   ├── __init__.py      # Router aggregator
│   │   │   ├── dataset.py       # Ingestion, preview, mapping, download
│   │   │   ├── quality.py       # Completeness and quality audit
│   │   │   ├── preprocessing.py # Imputation, outlier handling, scaling
│   │   │   ├── eda.py           # Statistical metrics, heatmap, histograms
│   │   │   ├── clustering.py    # K-Means and optimal K curve
│   │   │   ├── classification.py# Supervised classification models
│   │   │   ├── association.py   # Apriori association rule mining
│   │   │   ├── prediction.py    # Continuous regression & What-If simulator
│   │   │   ├── hotspots.py      # Spatial risk clustering
│   │   │   └── reports.py       # Markdown, CSV, Excel export
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── state_manager.py # Thread-safe in-memory cache & DB sync
│   │       ├── data_loader.py   # CSV parsing and role inference
│   │       ├── data_cleaner.py  # Cleaning, scaling, and deduplication
│   │       ├── data_analyzer.py # Stats, correlation, and hotspot GIS
│   │       ├── ml_service.py    # Clustering, classification, Apriori
│   │       ├── prediction_service.py # Regression & simulation
│   │       └── report_service.py# Markdown, CSV, and Excel generator
│   ├── uploads/                 # Storage for user-uploaded CSV files
│   ├── reports/                 # Storage for generated report artifacts
│   ├── requirements.txt         # Backend Python dependencies
│   └── marinemine.db            # SQLite database
├── frontend/
│   ├── css/
│   │   ├── style.css            # Ocean design system, glassmorphism, responsive sidebar
│   │   └── dashboard.css        # Charts, dropzone, and map layout rules
│   ├── js/
│   │   ├── app.js               # Global state, router, and API fetch client
│   │   ├── dashboard.js         # KPI cards and Chart.js visualizations
│   │   ├── dataset.js           # File upload, preview table, column mapping
│   │   ├── quality.js           # Quality audit telemetry and checks
│   │   ├── eda.js               # Statistics table, heatmap, distribution charts
│   │   ├── ml.js                # Preprocessing, K-Means, Classifiers, Predictor
│   │   ├── maps.js              # Leaflet.js interactive ocean map
│   │   └── reports.js           # Report generator & live markdown previewer
│   └── index.html               # Complete 12-module ocean-themed dashboard
├── sample_data/
│   └── marine_pollution_sample.csv # 1,000-record oceanographic benchmark
├── tests/
│   ├── test_fastapi_backend.py  # Comprehensive integration test suite (13 tests)
│   ├── test_data_loader.py      # Dataset loading & schema validation tests
│   ├── test_models.py           # ML algorithmic tests
│   └── test_preprocessing.py    # Data cleaning & imputation tests
├── server.py                    # Root convenience entrypoint
└── README.md                    # Project documentation
```

---

## 🧪 Automated Testing

The platform includes a test suite covering all REST endpoints, ML engines, data cleaning pipelines, and static file serving:

```bash
# Run the complete test suite (28 tests)
pytest tests/ -v
```

All 28 tests pass with 100% success rate.

---

## ⚖️ Academic & Research Integrity

- **Zero Fabricated Metrics:** All statistics, correlations, clusters, confusion matrices, and risk rankings are calculated dynamically from the active dataset.
- **Prevention of Data Leakage:** All scaling and transformations in classification and regression pipelines are strictly fitted inside training folds.
- **Safe State Management:** The original user-uploaded dataset is never overwritten; all preprocessed datasets are held in a separate state and can be downloaded independently.
