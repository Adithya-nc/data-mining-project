"""
MarineMine Pro — Mining Patterns of Marine Pollution using Data Mining & ML
===========================================================================
Main Application Hub & Multi-Module Dashboard.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

# Import core analytics and visualization engines
try:
    from src import (
        data_loader,
        data_validation,
        preprocessing,
        eda,
        clustering,
        classification,
        association_rules,
        prediction,
        hotspot_analysis,
        evaluation,
        report_generator,
    )
    from visualizations import charts, maps
except Exception as e:
    st.error(f"Error loading core modules: {e}")

# ---------------------------------------------------------------------------
# Page Configuration & Global Theme Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MarineMine Pro | Ocean Pollution Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced Modern Ocean Theme Styling (CSS)
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 1400px;
    }

    /* Top Telemetry Header Ribbon */
    .telemetry-ribbon {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, #061A2F 0%, #0A3D62 50%, #082946 100%);
        border: 1px solid rgba(0, 180, 216, 0.25);
        border-radius: 12px;
        padding: 10px 20px;
        margin-bottom: 18px;
        color: #E0F2FE;
        font-size: 0.85rem;
        box-shadow: 0 4px 15px rgba(6, 26, 47, 0.25);
    }
    .ribbon-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .ribbon-pill {
        background: rgba(0, 180, 216, 0.15);
        border: 1px solid rgba(0, 180, 216, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        color: #74B9E0;
        font-size: 0.78rem;
    }

    /* Ocean-Themed Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #092C48 0%, #0A3D62 50%, #005082 100%);
        border-radius: 14px;
        padding: 20px 22px;
        color: #FFFFFF;
        box-shadow: 0 8px 24px rgba(10, 61, 98, 0.15), 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(116, 185, 224, 0.2);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00B4D8, #48CAE4, #90E0EF);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(10, 61, 98, 0.28);
        border-color: rgba(0, 180, 216, 0.5);
    }
    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .metric-icon-bubble {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: rgba(0, 180, 216, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #90E0EF;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .metric-subtitle {
        font-size: 0.76rem;
        color: #CAF0F8;
        opacity: 0.9;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    /* Modern Glass Card */
    .glass-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
        margin-bottom: 22px;
        transition: border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: #CBD5E1;
    }

    /* Enhanced Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #061A2F 0%, #0A3D62 45%, #0077B6 85%, #00B4D8 100%);
        border-radius: 16px;
        padding: 36px 40px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 36px rgba(6, 26, 47, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 450px;
        height: 450px;
        background: radial-gradient(circle, rgba(0, 180, 216, 0.18) 0%, rgba(255, 255, 255, 0) 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(8px);
        padding: 5px 14px;
        border-radius: 30px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #90E0EF;
        margin-bottom: 14px;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 12px;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #E0F2FE;
        max-width: 860px;
        line-height: 1.6;
        opacity: 0.95;
    }

    /* Quick Action Cards */
    .action-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin: 18px 0;
    }
    .action-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .action-card:hover {
        background: #EBF8FF;
        border-color: #74B9E0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(10, 61, 98, 0.08);
    }
    .action-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #0A3D62;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .action-desc {
        font-size: 0.78rem;
        color: #64748B;
        line-height: 1.4;
    }

    /* Insight Banner */
    .insight-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 1px solid #BAE6FD;
        border-left: 5px solid #0284C7;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #0369A1;
        font-size: 0.9rem;
    }

    /* Severity Badges */
    .badge-high {
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-med {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-low {
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.75rem;
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .status-active {
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .status-inactive {
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }

    /* Sidebar Logo & Elements */
    .sidebar-brand {
        text-align: center;
        padding: 14px 0 18px 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 18px;
    }
    .sidebar-brand-icon {
        font-size: 2.4rem;
        display: inline-block;
        filter: drop-shadow(0 4px 8px rgba(10, 61, 98, 0.2));
    }
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0A3D62;
        letter-spacing: -0.02em;
        margin-top: 4px;
    }
    .sidebar-sub {
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    /* Pipeline Process Stepper */
    .pipeline-stepper {
        display: flex;
        gap: 8px;
        margin: 16px 0;
        overflow-x: auto;
        padding-bottom: 4px;
    }
    .step-box {
        flex: 1;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
        min-width: 130px;
    }
    .step-box.completed {
        background: #F0FDF4;
        border-color: #86EFAC;
        color: #166534;
    }
    .step-num {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 2px;
    }
    .step-name {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* Streamlit Button Enhancements */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0A3D62 0%, #0077B6 100%);
        border: none;
        box-shadow: 0 4px 14px rgba(10, 61, 98, 0.25);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #082946 0%, #005F92 100%);
        box-shadow: 0 6px 18px rgba(10, 61, 98, 0.35);
        transform: translateY(-1px);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state["df"] = None
if "df_processed" not in st.session_state:
    st.session_state["df_processed"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None
if "col_roles" not in st.session_state:
    st.session_state["col_roles"] = {}
if "clustering_results" not in st.session_state:
    st.session_state["clustering_results"] = None
if "classification_results" not in st.session_state:
    st.session_state["classification_results"] = None
if "prediction_results" not in st.session_state:
    st.session_state["prediction_results"] = None
if "association_rules" not in st.session_state:
    st.session_state["association_rules"] = None
if "hotspot_results" not in st.session_state:
    st.session_state["hotspot_results"] = None
if "preprocessing_report" not in st.session_state:
    st.session_state["preprocessing_report"] = None
if "data_view_mode" not in st.session_state:
    st.session_state["data_view_mode"] = "Cleaned (Processed)"
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "🏠 Home Dashboard"

# Auto-load benchmark sample if no data loaded yet
def load_default_sample():
    try:
        sample_df = data_loader.load_sample_dataset()
        st.session_state["df"] = sample_df.copy()
        st.session_state["df_processed"] = sample_df.copy()
        st.session_state["dataset_name"] = "marine_pollution_sample.csv (Benchmark)"
        st.session_state["col_roles"] = data_loader.infer_column_roles(sample_df)
    except Exception as exc:
        pass

if st.session_state["df"] is None:
    load_default_sample()


# ---------------------------------------------------------------------------
# Helper: Active Data Selector & Filtering
# ---------------------------------------------------------------------------
def get_active_df() -> pd.DataFrame:
    """Returns preprocessed dataframe or raw according to user choice."""
    mode = st.session_state.get("data_view_mode", "Cleaned (Processed)")
    if mode == "Original (Raw)" and st.session_state.get("df") is not None:
        return st.session_state["df"]
    if st.session_state.get("df_processed") is not None:
        return st.session_state["df_processed"]
    return st.session_state.get("df")


# ---------------------------------------------------------------------------
# Sidebar Navigation & Workspace Controls
# ---------------------------------------------------------------------------
nav_options = [
    "🏠 Home Dashboard",
    "📂 Dataset Management",
    "🔍 Data Quality Analysis",
    "⚙️ Data Preprocessing",
    "📊 Exploratory Data Analysis",
    "🎯 K-Means Clustering",
    "🏷️ Classification",
    "🔗 Association Rule Mining",
    "📈 Pollution Prediction",
    "🗺️ Hotspot Analysis",
    "📋 Reports & Export",
    "ℹ️ About Project",
]

with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🌊</div>
            <div class="sidebar-title">MarineMine Pro</div>
            <div class="sidebar-sub">Oceanic Data Mining & ML</div>
        </div>
    """, unsafe_allow_html=True)

    # Active dataset status badge
    curr_df = get_active_df()
    if curr_df is not None:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #EBF5FB 0%, #E0F2FE 100%); border-radius: 10px; padding: 12px 14px; margin-bottom: 16px; border: 1px solid #BAE6FD;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="status-pill status-active">● Active Telemetry</span>
                    <span style="font-size: 0.72rem; color: #0284C7; font-weight: 600;">Ready</span>
                </div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #0A3D62; margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{st.session_state.get('dataset_name', '')}">
                    {st.session_state.get('dataset_name', 'Loaded Dataset')}
                </div>
                <div style="font-size: 0.74rem; color: #475569; margin-top: 2px;">
                    <b>{len(curr_df):,}</b> records &nbsp;|&nbsp; <b>{len(curr_df.columns)}</b> features
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Data View Mode Switcher
        st.session_state["data_view_mode"] = st.radio(
            "Telemetry Pipeline View",
            ["Cleaned (Processed)", "Original (Raw)"],
            index=0 if st.session_state["data_view_mode"] == "Cleaned (Processed)" else 1,
            horizontal=True,
            help="Switch between cleaned telemetry or original unprocessed data."
        )
    else:
        st.markdown("""
            <div style="background: #FFF5F5; border-radius: 10px; padding: 12px 14px; margin-bottom: 16px; border: 1px solid #FEB2B2;">
                <span class="status-pill status-inactive">○ No Telemetry Loaded</span>
                <div style="font-size: 0.8rem; color: #721C24; margin-top: 6px;">Please upload or load the benchmark dataset.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em; margin: 14px 0 6px 0;'>Navigation</p>", unsafe_allow_html=True)

    # Ensure selected_nav index matches session state
    current_idx = 0
    if st.session_state["nav_selection"] in nav_options:
        current_idx = nav_options.index(st.session_state["nav_selection"])

    selected_nav = st.radio(
        "Navigation Menu",
        nav_options,
        index=current_idx,
        label_visibility="collapsed",
        key="nav_radio"
    )
    # Sync if user clicked radio
    if selected_nav != st.session_state["nav_selection"]:
        st.session_state["nav_selection"] = selected_nav

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.74rem; color: #64748B; line-height: 1.5;">
            <b>MarineMine Pro v1.2.0</b><br>
            Autonomous Ocean Pollution Analytics &amp; Machine Learning Suite.<br>
            <span style="color: #0284C7;">● Localhost Engine Online</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Global Telemetry Header Ribbon
# ---------------------------------------------------------------------------
active_df = get_active_df()
if active_df is not None:
    view_label = st.session_state.get("data_view_mode", "Cleaned")
    prep_status = "✅ Cleaned" if st.session_state.get("preprocessing_report") else "⚡ Raw Synced"
    quality_chip = "Audit Ready"
    if st.session_state.get("df") is not None:
        try:
            q_score = data_validation.generate_quality_report(st.session_state["df"]).get("quality_score", 0)
            quality_chip = f"Quality: {q_score:.0f}/100"
        except Exception:
            pass

    st.markdown(f"""
        <div class="telemetry-ribbon">
            <div class="ribbon-item">
                <span>🌊 <b>Dataset:</b> {st.session_state.get('dataset_name', 'Benchmark')}</span>
                <span class="ribbon-pill">{len(active_df):,} Rows</span>
                <span class="ribbon-pill">{len(active_df.columns)} Cols</span>
            </div>
            <div class="ribbon-item">
                <span class="ribbon-pill">{view_label}</span>
                <span class="ribbon-pill">{prep_status}</span>
                <span class="ribbon-pill" style="color: #38BDF8; border-color: #38BDF8;">{quality_chip}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MODULE 1: HOME DASHBOARD
# ---------------------------------------------------------------------------
if st.session_state["nav_selection"] == "🏠 Home Dashboard":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">🌊 Intelligent Marine Geoinformatics &amp; ML</div>
            <div class="hero-title">Mining Patterns of Marine Pollution</div>
            <div class="hero-subtitle">
                An advanced environmental data science ecosystem engineered to detect anthropogenic oceanic contamination,
                spatial hotspots, temporal anomalies, multi-model classifications, and cross-parameter environmental correlations.
            </div>
        </div>
    """, unsafe_allow_html=True)

    df = get_active_df()
    if df is None or df.empty:
        st.warning("⚠️ No active dataset found. Navigate to **Dataset Management** to load data.")
    else:
        # Quick Action Launcher
        st.markdown("<p style='font-size: 0.82rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.05em;'>⚡ Quick Action Modules</p>", unsafe_allow_html=True)
        qa1, qa2, qa3, qa4, qa5, qa6 = st.columns(6)
        with qa1:
            if st.button("🔍 Quality Audit", use_container_width=True, help="Run automated completeness and range checks"):
                st.session_state["nav_selection"] = "🔍 Data Quality Analysis"
                st.rerun()
        with qa2:
            if st.button("⚙️ Clean Data", use_container_width=True, help="Execute imputation, scaling, and deduplication"):
                st.session_state["nav_selection"] = "⚙️ Data Preprocessing"
                st.rerun()
        with qa3:
            if st.button("🎯 Cluster Analysis", use_container_width=True, help="Discover unsupervised pollution patterns"):
                st.session_state["nav_selection"] = "🎯 K-Means Clustering"
                st.rerun()
        with qa4:
            if st.button("🏷️ Classifiers", use_container_width=True, help="Train Random Forests and Decision Trees"):
                st.session_state["nav_selection"] = "🏷️ Classification"
                st.rerun()
        with qa5:
            if st.button("🗺️ Hotspot Maps", use_container_width=True, help="Explore Folium spatial density heatmaps"):
                st.session_state["nav_selection"] = "🗺️ Hotspot Analysis"
                st.rerun()
        with qa6:
            if st.button("📋 Export Report", use_container_width=True, help="Compile Markdown and PDF technical reports"):
                st.session_state["nav_selection"] = "📋 Reports & Export"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # KPI Metrics Row
        total_records = len(df)
        cols_lower = {c.lower(): c for c in df.columns}

        # Categories
        cat_cand = cols_lower.get("pollution_type", cols_lower.get("type", cols_lower.get("category")))
        total_cats = df[cat_cand].nunique() if cat_cand else "N/A"
        top_cat = df[cat_cand].mode().iloc[0] if (cat_cand and not df[cat_cand].empty) else "N/A"

        # Regions
        reg_cand = cols_lower.get("region", cols_lower.get("ocean", cols_lower.get("location")))
        total_regs = df[reg_cand].nunique() if reg_cand else "N/A"

        # Avg Measurement
        val_cand = cols_lower.get("pollution_value", cols_lower.get("concentration", cols_lower.get("level")))
        avg_val = f"{df[val_cand].mean():.2f}" if (val_cand and pd.api.types.is_numeric_dtype(df[val_cand])) else "N/A"

        # Date Coverage
        date_cand = cols_lower.get("observation_date", cols_lower.get("date", cols_lower.get("timestamp")))
        if date_cand:
            try:
                dt_series = pd.to_datetime(df[date_cand].dropna())
                date_span = f"{dt_series.min().strftime('%Y/%m')} – {dt_series.max().strftime('%Y/%m')}"
            except Exception:
                date_span = "Available"
        else:
            date_span = "Unspecified"

        # Missing %
        total_cells = df.size
        missing_pct = f"{(df.isna().sum().sum() / total_cells * 100):.1f}%" if total_cells > 0 else "0.0%"

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Observations</div>
                        <div class="metric-icon-bubble">📡</div>
                    </div>
                    <div class="metric-value">{total_records:,}</div>
                    <div class="metric-subtitle">Telemetry records</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Pollutants</div>
                        <div class="metric-icon-bubble">🧪</div>
                    </div>
                    <div class="metric-value">{total_cats}</div>
                    <div class="metric-subtitle">Top: {top_cat}</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Ocean Basins</div>
                        <div class="metric-icon-bubble">🌐</div>
                    </div>
                    <div class="metric-value">{total_regs}</div>
                    <div class="metric-subtitle">Global marine zones</div>
                </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Mean Level</div>
                        <div class="metric-icon-bubble">📊</div>
                    </div>
                    <div class="metric-value">{avg_val}</div>
                    <div class="metric-subtitle">Concentration index</div>
                </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Time Horizon</div>
                        <div class="metric-icon-bubble">📅</div>
                    </div>
                    <div class="metric-value" style="font-size: 1.25rem;">{date_span}</div>
                    <div class="metric-subtitle">Sampling range</div>
                </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Missing Rate</div>
                        <div class="metric-icon-bubble">🛡️</div>
                    </div>
                    <div class="metric-value">{missing_pct}</div>
                    <div class="metric-subtitle">Overall sparsity</div>
                </div>
            """, unsafe_allow_html=True)

        # AI Domain Intelligence Card
        if cat_cand and reg_cand and val_cand and pd.api.types.is_numeric_dtype(df[val_cand]):
            highest_reg = df.groupby(reg_cand)[val_cand].mean().idxmax()
            highest_val = df.groupby(reg_cand)[val_cand].mean().max()
            st.markdown(f"""
                <div class="insight-card">
                    <b>💡 Automated Marine Intelligence:</b> The primary recorded contaminant in this cohort is <b>{top_cat}</b>.
                    Highest mean contamination intensity is localized in the <b>{highest_reg}</b> ({highest_val:.2f} mean concentration).
                    Use the <b>Hotspot Analysis</b> or <b>K-Means Clustering</b> modules to examine multivariate risk dynamics.
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualizations Deck
        tab_cat, tab_reg, tab_trend, tab_corr = st.tabs([
            "📊 Pollutant Categories", "🗺️ Basin Distribution", "📈 Temporal Trends", "🔥 Correlation Matrix"
        ])

        with tab_cat:
            col_c1, col_c2 = st.columns([3, 2])
            with col_c1:
                if cat_cand and cat_cand in df.columns:
                    try:
                        fig_type = charts.plot_pollution_by_type(df, type_col=cat_cand, value_col=val_cand if (val_cand and val_cand in df.columns) else None)
                        st.plotly_chart(fig_type, use_container_width=True)
                    except Exception as e:
                        st.info(f"Category plot unavailable: {e}")
                else:
                    st.info("Pollution category column not detected.")
            with col_c2:
                if cat_cand and cat_cand in df.columns:
                    cat_counts = df[cat_cand].value_counts().reset_index()
                    cat_counts.columns = ["Pollutant Category", "Count"]
                    cat_counts["Share %"] = (cat_counts["Count"] / len(df) * 100).round(1)
                    st.markdown("#### Pollutant Class Breakdown")
                    st.dataframe(cat_counts, use_container_width=True)

        with tab_reg:
            if reg_cand and reg_cand in df.columns:
                try:
                    fig_reg = charts.plot_pollution_by_region(df, region_col=reg_cand, value_col=val_cand if (val_cand and val_cand in df.columns) else None)
                    st.plotly_chart(fig_reg, use_container_width=True)
                except Exception as e:
                    st.info(f"Regional plot unavailable: {e}")
            else:
                st.info("Regional column not detected.")

        with tab_trend:
            if date_cand and date_cand in df.columns:
                try:
                    fig_trend = charts.plot_pollution_trend(df, date_col=date_cand, value_col=val_cand if (val_cand and val_cand in df.columns) else None)
                    st.plotly_chart(fig_trend, use_container_width=True)
                except Exception as e:
                    st.info(f"Temporal trend unavailable: {e}")
            else:
                st.info("Temporal trend unavailable: observation dates required.")

        with tab_corr:
            try:
                fig_corr = charts.plot_correlation_heatmap(df)
                if fig_corr:
                    st.pyplot(fig_corr)
            except Exception as e:
                st.info(f"Correlation analysis requires at least 2 numerical features ({e}).")

        # Recent Observational Telemetry Table
        st.markdown("### 📋 Recent Observational Telemetry (Latest 10 Samples)")
        preview_df = df.tail(10).copy()
        st.dataframe(preview_df, use_container_width=True)


# ---------------------------------------------------------------------------
# MODULE 2: DATASET MANAGEMENT
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "📂 Dataset Management":
    st.markdown("## 📂 Dataset Management & Telemetry Ingestion")
    st.markdown("Upload your custom marine pollution CSV or initialize the pre-calibrated 1,000-record oceanographic benchmark.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
            <div class="glass-card">
                <h4 style="margin-top: 0; color: #0A3D62;">📤 Upload Custom Marine Telemetry CSV</h4>
                <p style="font-size: 0.85rem; color: #64748B;">Compatible with standard CSV telemetry containing coordinates, water quality metrics, and pollutant labels.</p>
            </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Select a standard comma-separated file.")
        if uploaded_file is not None:
            loaded_df, err = data_loader.load_uploaded_dataset(uploaded_file)
            if err:
                st.error(f"Failed to load dataset: {err}")
            else:
                st.session_state["df"] = loaded_df
                st.session_state["df_processed"] = loaded_df.copy()
                st.session_state["dataset_name"] = uploaded_file.name
                st.session_state["col_roles"] = data_loader.infer_column_roles(loaded_df)
                st.success(f"Successfully loaded '{uploaded_file.name}' with {len(loaded_df):,} records!")

    with col2:
        st.markdown("""
            <div class="glass-card">
                <h4 style="margin-top: 0; color: #0A3D62;">🌊 Oceanographic Benchmark</h4>
                <p style="font-size: 0.85rem; color: #64748B;">Initialize the curated 1,000-row synthetic benchmark across 10 oceanic basins.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🌊 Load Sample Benchmark Dataset", use_container_width=True):
            load_default_sample()
            st.success("Loaded synthetic demonstration benchmark dataset!")

    df = get_active_df()
    if df is not None:
        st.markdown("---")
        info = data_loader.get_dataset_info(df, st.session_state.get("dataset_name", "dataset.csv"))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", f"{info['total_rows']:,}")
        m2.metric("Total Columns", f"{info['total_columns']}")
        m3.metric("Memory Footprint", f"{info['memory_usage_mb']} MB")
        m4.metric("Duplicate Rows", f"{info['duplicate_rows']}")

        st.markdown("### 🔍 Dataset Column Schema & Dtypes")
        schema_df = pd.DataFrame({
            "Column Name": list(info["dtypes"].keys()),
            "Data Type": list(info["dtypes"].values()),
            "Missing Values": [info["missing_per_col"].get(c, 0) for c in info["dtypes"].keys()],
            "Missing %": [round((info["missing_per_col"].get(c, 0) / info["total_rows"]) * 100, 2) for c in info["dtypes"].keys()],
        })
        st.dataframe(schema_df, use_container_width=True)

        st.markdown("### 📄 Dataset Preview (First 50 Rows)")
        st.dataframe(df.head(50), use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Dataset (CSV)",
            data=csv_bytes,
            file_name="marine_pollution_dataset.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# MODULE 3: DATA QUALITY ANALYSIS
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "🔍 Data Quality Analysis":
    st.markdown("## 🔍 Data Quality & Telemetry Integrity Audit")
    st.markdown("Comprehensive evaluation of completeness, duplicate frequency, statistical outliers, and oceanographic physical bounds.")

    df = st.session_state.get("df")
    if df is None:
        st.warning("Please load a dataset in Dataset Management first.")
    else:
        quality_rep = data_validation.generate_quality_report(df)

        q_score = quality_rep.get("quality_score", 0.0)
        q_grade = quality_rep.get("quality_grade", "N/A")
        breakdown = quality_rep.get("score_breakdown", {})

        c_score, c_break = st.columns([1, 2])
        with c_score:
            score_color = "#10B981" if q_score >= 80 else ("#F59E0B" if q_score >= 60 else "#EF4444")
            st.markdown(f"""
                <div style="background: #FFFFFF; border: 2px solid {score_color}; border-radius: 14px; padding: 26px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.04);">
                    <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Overall Integrity Score</div>
                    <div style="font-size: 3.2rem; font-weight: 800; color: {score_color}; margin: 8px 0; line-height: 1;">{q_score:.0f}<span style="font-size: 1.5rem; color: #94A3B8;">/100</span></div>
                    <div style="display: inline-block; background: {score_color}15; color: {score_color}; border: 1px solid {score_color}40; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 0.95rem;">
                        Grade {q_grade}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with c_break:
            st.markdown("#### Quality Index Dimension Breakdown")
            sc1, sc2 = st.columns(2)
            sc1.metric("Completeness (Max 40)", f"{breakdown.get('completeness', 0)} / 40")
            sc1.metric("Uniqueness (Max 20)", f"{breakdown.get('uniqueness', 0)} / 20")
            sc2.metric("Physical Validity (Max 20)", f"{breakdown.get('validity', 0)} / 20")
            sc2.metric("Type Consistency (Max 20)", f"{breakdown.get('consistency', 0)} / 20")

            comp_val = breakdown.get('completeness', 0) / 40.0
            st.progress(comp_val, text=f"Data Completeness Ratio: {comp_val*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        tab_missing, tab_dups, tab_outliers, tab_ranges = st.tabs([
            "📊 Missing Values", "👥 Duplicate Records", "📈 Statistical Outliers (IQR)", "⚠️ Physical Range Bounds"
        ])

        with tab_missing:
            col_chart, col_tbl = st.columns([3, 2])
            with col_chart:
                fig_missing = charts.plot_missing_values(df)
                if fig_missing:
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.success("No missing values found in active dataset!")
            with col_tbl:
                st.write("**Missing Value Breakdown**")
                st.dataframe(quality_rep["missing_summary"]["summary_df"], use_container_width=True)

        with tab_dups:
            dup_info = quality_rep["duplicates"]
            st.metric("Total Duplicate Rows", f"{dup_info['duplicate_count']} ({dup_info['duplicate_pct']}%)")
            if dup_info["duplicate_count"] > 0:
                st.write("**Duplicate Samples:**")
                st.dataframe(dup_info["example_duplicates"], use_container_width=True)
            else:
                st.success("✅ Zero duplicate records detected in active telemetry.")

        with tab_outliers:
            st.write("Outliers detected via Interquartile Range (IQR 1.5× Threshold):")
            outliers = quality_rep.get("outliers", {})
            outlier_rows = []
            for col_name, o_info in outliers.items():
                outlier_rows.append({
                    "Feature": col_name,
                    "Outlier Count": o_info["count"],
                    "Outlier %": f"{o_info['percentage']}%",
                    "Lower Bound": o_info["lower_bound"],
                    "Upper Bound": o_info["upper_bound"],
                })
            st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

        with tab_ranges:
            st.write("Oceanographic Domain Physical Range Verification:")
            violations = quality_rep.get("range_violations", {})
            if violations:
                st.warning(f"Detected {len(violations)} columns with values outside physical oceanographic bounds!")
                st.json(violations)
            else:
                st.success("✅ All observed telemetry complies with standard oceanographic physical limits.")


# ---------------------------------------------------------------------------
# MODULE 4: DATA PREPROCESSING
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "⚙️ Data Preprocessing":
    st.markdown("## ⚙️ Data Preprocessing & Cleaning Pipeline")
    st.markdown("Non-destructive automated data preparation. Original raw telemetry is strictly preserved.")

    orig_df = st.session_state.get("df")
    proc_df = st.session_state.get("df_processed")

    if orig_df is None:
        st.warning("Please load a dataset in Dataset Management first.")
    else:
        # Visual Stepper
        st.markdown("""
            <div class="pipeline-stepper">
                <div class="step-box completed">
                    <div class="step-num">Step 1</div>
                    <div class="step-name">📥 Ingest Data</div>
                </div>
                <div class="step-box">
                    <div class="step-num">Step 2</div>
                    <div class="step-name">🧩 Impute Missing</div>
                </div>
                <div class="step-box">
                    <div class="step-num">Step 3</div>
                    <div class="step-name">👥 Deduplicate</div>
                </div>
                <div class="step-box">
                    <div class="step-num">Step 4</div>
                    <div class="step-name">⚖️ Scale &amp; Encode</div>
                </div>
                <div class="step-box">
                    <div class="step-num">Step 5</div>
                    <div class="step-name">🚀 Downstream ML</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Before / After metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Rows", f"{len(orig_df):,}")
        c2.metric("Cleaned Rows", f"{len(proc_df):,}" if proc_df is not None else "Pending")
        c3.metric("Original Missing Cells", f"{orig_df.isna().sum().sum():,}")
        c4.metric("Cleaned Missing Cells", f"{proc_df.isna().sum().sum():,}" if proc_df is not None else "Pending")

        st.markdown("---")
        st.markdown("### Configure Pipeline Transformations")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            st.markdown("#### 1. Missing Value Imputation")
            num_strategy = st.selectbox("Numeric Imputation Strategy", ["mean", "median", "drop", "none"], index=0)
            cat_strategy = st.selectbox("Categorical Imputation Strategy", ["mode", "drop", "none"], index=0)

            st.markdown("#### 2. Duplicate Handling")
            drop_dups = st.checkbox("Remove Duplicate Records", value=True)

        with col_opt2:
            st.markdown("#### 3. Feature Scaling")
            scale_features = st.checkbox("Apply Feature Scaling to Numerics", value=False)
            scale_method = st.selectbox("Scaling Method", ["standard", "minmax"], index=0)

            st.markdown("#### 4. Categorical Encoding")
            encode_cats = st.checkbox("Encode Categorical Variables", value=False)
            encode_method = st.selectbox("Encoding Method", ["label", "onehot"], index=0)

        if st.button("🚀 Execute Preprocessing Pipeline", type="primary", use_container_width=True):
            pipeline = preprocessing.PreprocessingPipeline(orig_df)

            if drop_dups:
                pipeline.remove_duplicates()

            if num_strategy != "none" or cat_strategy != "none":
                pipeline.handle_missing_values(numeric_strategy=num_strategy, categorical_strategy=cat_strategy)

            num_cols = orig_df.select_dtypes(include=[np.number]).columns.tolist()
            if scale_features and num_cols:
                pipeline.scale_numerics(cols=num_cols, method=scale_method)

            cat_cols = orig_df.select_dtypes(include=["object", "category"]).columns.tolist()
            safe_cat_cols = [c for c in cat_cols if "id" not in c.lower() and "date" not in c.lower() and "notes" not in c.lower()]
            if encode_cats and safe_cat_cols:
                pipeline.encode_categoricals(cols=safe_cat_cols, method=encode_method)

            cleaned_result = pipeline.get_processed_df()
            prep_report = pipeline.get_preprocessing_report()

            st.session_state["df_processed"] = cleaned_result
            st.session_state["preprocessing_report"] = prep_report

            st.success("✅ Preprocessing complete! Downstream analytics will automatically use the cleaned dataset.")
            st.rerun()

        if st.session_state.get("preprocessing_report"):
            st.markdown("### 📋 Applied Transformation Log")
            st.json(st.session_state["preprocessing_report"])

        if proc_df is not None:
            st.markdown("### 📄 Cleaned Dataset Sample (First 20 Records)")
            st.dataframe(proc_df.head(20), use_container_width=True)

            cleaned_csv = proc_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Cleaned Dataset (CSV)",
                data=cleaned_csv,
                file_name="cleaned_marine_pollution.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# MODULE 5: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "📊 Exploratory Data Analysis":
    st.markdown("## 📊 Exploratory Data Analysis (EDA)")
    st.markdown("Interactive distributions, longitudinal trends, boxplot comparisons, and correlation matrices.")

    df = get_active_df()
    if df is None:
        st.warning("Please load a dataset in Dataset Management first.")
    else:
        tab_dist, tab_trends, tab_regional, tab_corr, tab_stats = st.tabs([
            "📈 Distributions", "⏱️ Time Trends", "🌐 Regional Comparisons", "🔥 Correlation Heatmap", "📋 Descriptive Stats"
        ])

        with tab_dist:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                selected_col = st.selectbox("Select Numeric Feature for Distribution", num_cols, index=0)
                try:
                    fig_dist = charts.plot_distribution(df, selected_col)
                    if fig_dist:
                        st.pyplot(fig_dist)
                except Exception as e:
                    st.info(f"Distribution plot unavailable: {e}")

                cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                if cat_cols:
                    selected_cat = st.selectbox("Group by Categorical Feature (Boxplot)", cat_cols, index=0)
                    try:
                        fig_box = charts.plot_box_by_category(df, selected_col, selected_cat)
                        if fig_box:
                            st.plotly_chart(fig_box, use_container_width=True)
                    except Exception as e:
                        st.info(f"Boxplot unavailable: {e}")
            else:
                st.info("No numerical columns found for distribution plots.")

        # Detect helper columns for EDA
        cols_lower_eda = {c.lower(): c for c in df.columns}
        date_cand_eda = cols_lower_eda.get("observation_date", cols_lower_eda.get("date", cols_lower_eda.get("timestamp")))
        reg_cand_eda = cols_lower_eda.get("region", cols_lower_eda.get("ocean", cols_lower_eda.get("location")))
        val_cand_eda = cols_lower_eda.get("pollution_value", cols_lower_eda.get("concentration", cols_lower_eda.get("level")))

        with tab_trends:
            if date_cand_eda and date_cand_eda in df.columns:
                try:
                    fig_trend = charts.plot_pollution_trend(df, date_col=date_cand_eda, value_col=val_cand_eda if (val_cand_eda and val_cand_eda in df.columns) else None)
                    if fig_trend:
                        st.plotly_chart(fig_trend, use_container_width=True)
                except Exception as e:
                    st.info(f"Time-series trend unavailable: {e}")
            else:
                st.info("Time-series trend requires a detected date column and numeric measurement.")

        with tab_regional:
            if reg_cand_eda and reg_cand_eda in df.columns:
                try:
                    fig_reg = charts.plot_pollution_by_region(df, region_col=reg_cand_eda, value_col=val_cand_eda if (val_cand_eda and val_cand_eda in df.columns) else None)
                    if fig_reg:
                        st.plotly_chart(fig_reg, use_container_width=True)
                except Exception as e:
                    st.info(f"Regional visualization unavailable: {e}")
            else:
                st.info("Regional visualization requires region and value columns.")

        with tab_corr:
            try:
                fig_corr = charts.plot_correlation_heatmap(df)
                if fig_corr:
                    st.pyplot(fig_corr)
            except Exception as e:
                st.info(f"Correlation analysis requires at least 2 numeric features ({e}).")

        with tab_stats:
            st.markdown("#### Extended Numerical Statistics")
            num_summary = eda.get_numeric_summary(df)
            if not num_summary.empty:
                st.dataframe(num_summary, use_container_width=True)
            else:
                st.info("No numeric features to summarize.")


# ---------------------------------------------------------------------------
# MODULE 6: K-MEANS CLUSTERING
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "🎯 K-Means Clustering":
    st.markdown("## 🎯 Unsupervised Pattern Discovery: K-Means Clustering")
    st.markdown("Discover clusters of similar pollution behavior and examine environmental feature vectors.")

    df = get_active_df()
    if df is None:
        st.warning("Please load a dataset in Dataset Management first.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        safe_cluster_cols = [c for c in num_cols if "lat" not in c.lower() and "lon" not in c.lower() and "id" not in c.lower()]

        col_cfg, col_diag = st.columns([1, 1])
        with col_cfg:
            selected_features = st.multiselect("Clustering Features", num_cols, default=safe_cluster_cols[:4] if len(safe_cluster_cols) >= 2 else num_cols)
            k_val = st.slider("Select Number of Clusters (k)", min_value=2, max_value=8, value=3)

        with col_diag:
            st.markdown("#### Diagnostic Evaluation (Elbow & Silhouette)")
            if st.button("📊 Run Elbow Method Diagnostics", use_container_width=True):
                if len(selected_features) >= 2:
                    inertias = clustering.compute_elbow(df, selected_features, k_range=range(2, 8))
                    fig_elbow = charts.plot_elbow(inertias, list(range(2, 8)))
                    st.plotly_chart(fig_elbow, use_container_width=True)
                else:
                    st.error("Please select at least 2 numerical features.")

        st.markdown("---")
        if st.button("🚀 Execute K-Means Clustering", type="primary", use_container_width=True):
            if len(selected_features) < 2:
                st.error("Please select at least 2 numerical features.")
            else:
                clustered_df, labels = clustering.run_kmeans(df, selected_features, n_clusters=k_val)
                sil_score = clustering.compute_silhouette(df, selected_features, n_clusters=k_val)
                stats_df = clustering.get_cluster_stats(clustered_df, selected_features)
                descriptions = clustering.describe_clusters(stats_df)

                st.session_state["clustering_results"] = {
                    "df": clustered_df,
                    "labels": labels,
                    "n_clusters": k_val,
                    "features": selected_features,
                    "silhouette_score": sil_score,
                    "stats_df": stats_df,
                    "descriptions": descriptions,
                }
                st.success(f"K-Means clustering complete with k={k_val} (Silhouette Score: {sil_score:.3f})")

        cl_res = st.session_state.get("clustering_results")
        if cl_res:
            m1, m2 = st.columns(2)
            m1.metric("Clusters Formed", f"{cl_res['n_clusters']}")
            m2.metric("Silhouette Coefficient", f"{cl_res['silhouette_score']:.3f} (-1 to +1)")

            st.markdown("### 🌌 2D PCA Cluster Projection")
            fig_pca = charts.plot_pca_clusters(cl_res["df"], cl_res["features"], cluster_col="cluster")
            if fig_pca:
                st.plotly_chart(fig_pca, use_container_width=True)

            st.markdown("### 📊 Cluster Statistical Profiles")
            st.dataframe(cl_res["stats_df"], use_container_width=True)

            st.markdown("### 💡 Plain-Language Cluster Insights")
            for cid, desc in cl_res["descriptions"].items():
                st.info(f"**Cluster {cid}:** {desc}")

            clustered_csv = cl_res["df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Clustered Dataset (CSV)",
                data=clustered_csv,
                file_name="kmeans_clustered_marine_data.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# MODULE 7: CLASSIFICATION
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "🏷️ Classification":
    st.markdown("## 🏷️ Multi-Model Supervised Classification")
    st.markdown("Train and benchmark Decision Trees, Random Forests, and Logistic Regression on categorical marine targets.")

    df = get_active_df()
    if df is None:
        st.warning("Please load a dataset in Dataset Management first.")
    else:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        c_target, c_feat = st.columns([1, 2])
        with c_target:
            target_col = st.selectbox("Select Target Column", cat_cols, index=0 if cat_cols else None)
            if target_col:
                suit = classification.check_target_suitability(df, target_col)
                if suit.get("is_suitable", False):
                    st.success(f"Target is suitable ({suit.get('n_classes')} distinct classes).")
                else:
                    st.warning(f"Target suitability warning: {suit.get('reason')}")

            test_split = st.slider("Test Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

        with c_feat:
            avail_features = [c for c in df.columns if c != target_col and "id" not in c.lower() and "notes" not in c.lower()]
            selected_features = st.multiselect("Input Predictor Features", avail_features, default=avail_features[:5])

        if st.button("🚀 Train & Compare Classifiers", type="primary", use_container_width=True):
            if not target_col or not selected_features:
                st.error("Please select a target and at least one predictor feature.")
            else:
                results = classification.train_classifiers(
                    df,
                    feature_cols=selected_features,
                    target_col=target_col,
                    test_size=test_split,
                    random_state=42
                )
                comp_df = classification.compare_models(results)
                st.session_state["classification_results"] = {
                    "results": results,
                    "comparison": comp_df,
                    "target_col": target_col,
                    "features": selected_features,
                }
                st.success("Classification models successfully trained!")

        cl_out = st.session_state.get("classification_results")
        if cl_out:
            st.markdown("### 🏆 Model Performance Benchmark Leaderboard")
            st.dataframe(cl_out["comparison"], use_container_width=True)

            st.markdown("### 🔍 Model Diagnostic Views")
            m_tabs = st.tabs(list(cl_out["results"].keys()))
            for i, (m_name, m_data) in enumerate(cl_out["results"].items()):
                with m_tabs[i]:
                    c_m1, c_m2 = st.columns(2)
                    c_m1.metric("Test Accuracy", f"{m_data['accuracy']*100:.2f}%")
                    c_m2.metric("Macro F1-Score", f"{m_data['f1']:.4f}")

                    if "cm" in m_data and "labels" in m_data:
                        fig_cm = charts.plot_confusion_matrix(m_data["cm"], m_data["labels"])
                        if fig_cm:
                            st.plotly_chart(fig_cm, use_container_width=True)

                    st.markdown("#### Detailed Classification Report")
                    st.text(m_data.get("report", "Report not available."))


# ---------------------------------------------------------------------------
# MODULE 8: ASSOCIATION RULE MINING
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "🔗 Association Rule Mining":
    st.markdown("## 🔗 Association Rule Mining (Apriori Algorithm)")
    st.markdown("Discover co-occurrence relationships and environmental patterns using MLxtend Apriori.")

    df = get_active_df()
    if df is None:
        st.warning("Please load a dataset in Dataset Management first.")
    else:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        c1, c2 = st.columns([1, 1])
        with c1:
            selected_cats = st.multiselect("Categorical Attributes", cat_cols, default=cat_cols[:3] if len(cat_cols) >= 3 else cat_cols)
            num_to_bin = st.multiselect("Discretize Numeric Attributes", num_cols, default=[num_cols[0]] if num_cols else [])

        with c2:
            min_supp = st.slider("Minimum Support Threshold", min_value=0.01, max_value=0.3, value=0.05, step=0.01)
            min_conf = st.slider("Minimum Confidence Threshold", min_value=0.1, max_value=0.9, value=0.4, step=0.05)

        if st.button("🔍 Mine Frequent Itemsets & Rules", type="primary", use_container_width=True):
            if not selected_cats and not num_to_bin:
                st.error("Please select at least one categorical or discretized numeric attribute.")
            else:
                tx_matrix = association_rules.prepare_transactions(df, cols=selected_cats, num_cols_to_bin=num_to_bin)
                itemsets = association_rules.mine_frequent_itemsets(tx_matrix, min_support=min_supp)

                if itemsets.empty:
                    st.warning("No frequent itemsets found with the specified minimum support threshold. Try lowering support.")
                else:
                    rules = association_rules.generate_rules(itemsets, min_confidence=min_conf)
                    top_explanations = association_rules.explain_rules(rules, top_n=5) if not rules.empty else []

                    st.session_state["association_rules"] = {
                        "itemsets_count": len(itemsets),
                        "rules_count": len(rules),
                        "rules_df": rules,
                        "min_support": min_supp,
                        "min_confidence": min_conf,
                        "top_rules_text": top_explanations,
                    }
                    st.success(f"Discovered {len(itemsets)} frequent itemsets and {len(rules)} association rules!")

        arm_out = st.session_state.get("association_rules")
        if arm_out:
            st.metric("Discovered Association Rules", f"{arm_out['rules_count']}")
            if arm_out["rules_count"] > 0:
                st.markdown("### 📜 Discovered Association Rules")
                display_df = association_rules.format_rules_display(arm_out["rules_df"])
                st.dataframe(display_df, use_container_width=True)

                st.markdown("### 💡 Plain-Language Explanations")
                for exp in arm_out["top_rules_text"]:
                    st.info(exp)

                rules_csv = arm_out["rules_df"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Rules (CSV)",
                    data=rules_csv,
                    file_name="marine_pollution_association_rules.csv",
                    mime="text/csv",
                )


# ---------------------------------------------------------------------------
# MODULE 9: POLLUTION PREDICTION
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "📈 Pollution Prediction":
    st.markdown("## 📈 Pollution Level Estimation & Regression")
    st.markdown("Estimate numeric pollution concentrations using Linear Regression and Random Forest Regressors.")

    df = get_active_df()
    if df is None:
        st.warning("Please load a dataset in Dataset Management first.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        c_target, c_feats = st.columns([1, 2])
        with c_target:
            val_candidates = [c for c in num_cols if "value" in c.lower() or "level" in c.lower() or "temp" in c.lower()]
            pred_target = st.selectbox("Predictive Target Feature", num_cols, index=num_cols.index(val_candidates[0]) if val_candidates else 0)
            use_chrono = st.checkbox("Chronological Ordering Split (Time-series)", value=False)

        with c_feats:
            avail_f = [c for c in num_cols if c != pred_target and "id" not in c.lower()]
            pred_features = st.multiselect("Input Predictor Features", avail_f, default=avail_f[:4])

        if st.button("🚀 Train Regression Models", type="primary", use_container_width=True):
            if not pred_target or not pred_features:
                st.error("Please specify both a target column and input features.")
            else:
                reg_results = prediction.train_regressors(
                    df,
                    feature_cols=pred_features,
                    target_col=pred_target,
                    test_size=0.2,
                    random_state=42,
                    use_time_split=use_chrono
                )
                comp_reg = prediction.compare_regressors(reg_results)
                st.session_state["prediction_results"] = {
                    "results": reg_results,
                    "comparison": comp_reg,
                    "target_col": pred_target,
                    "features": pred_features,
                }
                st.success("Regression models trained and validated!")

        pred_out = st.session_state.get("prediction_results")
        if pred_out:
            st.markdown("### 🏆 Regression Benchmark (MAE / RMSE / R²)")
            st.dataframe(pred_out["comparison"], use_container_width=True)

            rf_data = pred_out["results"].get("Random Forest")
            if rf_data:
                st.markdown("### 🎯 Actual vs Predicted: Random Forest")
                fig_pred = charts.plot_actual_vs_predicted(rf_data["y_test"], rf_data["y_pred"])
                if fig_pred:
                    st.plotly_chart(fig_pred, use_container_width=True)

            st.markdown(prediction.get_prediction_disclaimer(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MODULE 10: HOTSPOT ANALYSIS
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "🗺️ Hotspot Analysis":
    st.markdown("## 🗺️ Geospatial Pollution Hotspot Analysis")
    st.markdown("Pinpoint high-concentration maritime risk zones, spatial clusters, and regional aggregations.")

    df = get_active_df()
    if df is None:
        st.warning("Please load a dataset in Dataset Management first.")
    else:
        roles = st.session_state.get("col_roles", {})
        lat_cols = roles.get("lat_cols", [c for c in df.columns if "lat" in c.lower()])
        lon_cols = roles.get("lon_cols", [c for c in df.columns if "lon" in c.lower()])

        if not lat_cols or not lon_cols:
            st.error("Geographical coordinates (latitude/longitude) were not detected in this dataset.")
        else:
            lat_col = lat_cols[0]
            lon_col = lon_cols[0]

            val_cols = [c for c in df.columns if "value" in c.lower() or "level" in c.lower() or "concentration" in c.lower()]
            val_col = val_cols[0] if val_cols else df.select_dtypes(include=[np.number]).columns[0]

            f_col1, f_col2 = st.columns(2)
            with f_col1:
                type_cols = [c for c in df.columns if "type" in c.lower() or "category" in c.lower()]
                selected_types = []
                if type_cols:
                    all_types = df[type_cols[0]].dropna().unique().tolist()
                    selected_types = st.multiselect("Filter by Pollution Type", all_types, default=all_types[:3])

            with f_col2:
                reg_cols = [c for c in df.columns if "region" in c.lower() or "ocean" in c.lower()]
                selected_regs = []
                if reg_cols:
                    all_regs = df[reg_cols[0]].dropna().unique().tolist()
                    selected_regs = st.multiselect("Filter by Region", all_regs, default=all_regs[:3])

            filtered_df = df.copy()
            if selected_types and type_cols:
                filtered_df = filtered_df[filtered_df[type_cols[0]].isin(selected_types)]
            if selected_regs and reg_cols:
                filtered_df = filtered_df[filtered_df[reg_cols[0]].isin(selected_regs)]

            st.write(f"Displaying **{len(filtered_df):,}** observations matching filter criteria.")

            tab_scatter, tab_heat, tab_cluster = st.tabs(["📍 Scatter Map", "🔥 Density Heatmap", "🗂️ Cluster Markers"])

            try:
                from streamlit_folium import st_folium

                with tab_scatter:
                    m_scatter = maps.create_base_map()
                    m_scatter = maps.add_pollution_markers(m_scatter, filtered_df, lat_col, lon_col, popup_cols=[val_col], color_by_col=val_col)
                    st_folium(m_scatter, width=1200, height=520)

                with tab_heat:
                    m_heat = maps.create_base_map()
                    m_heat = maps.add_heatmap_layer(m_heat, filtered_df, lat_col, lon_col, weight_col=val_col)
                    st_folium(m_heat, width=1200, height=520)

                with tab_cluster:
                    m_cluster = maps.create_base_map()
                    m_cluster = maps.add_cluster_markers(m_cluster, filtered_df, lat_col, lon_col, popup_cols=[val_col])
                    st_folium(m_cluster, width=1200, height=520)
            except Exception as e:
                st.warning(f"Interactive Folium rendering note: {e}")

            st.markdown("### 🏆 Top 10 High-Intensity Marine Hotspots")
            hotspots_df = hotspot_analysis.compute_hotspots(filtered_df, lat_col, lon_col, val_col, top_n=10)
            if not hotspots_df.empty:
                ranked_hotspots = hotspot_analysis.rank_hotspots(hotspots_df)
                st.dataframe(ranked_hotspots, use_container_width=True)


# ---------------------------------------------------------------------------
# MODULE 11: REPORTS & EXPORT
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "📋 Reports & Export":
    st.markdown("## 📋 Comprehensive Report Generation & Multi-Artifact Export")
    st.markdown("Generate executive technical summaries in Markdown and PDF, and export all analytical datasets.")

    df = get_active_df()
    if df is None:
        st.warning("Please load a dataset in Dataset Management first.")
    else:
        dataset_info = data_loader.get_dataset_info(df, st.session_state.get("dataset_name", "dataset.csv"))
        prep_report = st.session_state.get("preprocessing_report")
        eda_sum = eda.get_eda_summary(df)
        cl_res = st.session_state.get("clustering_results")
        class_res = st.session_state.get("classification_results")
        pred_res = st.session_state.get("prediction_results")
        arm_res = st.session_state.get("association_rules")

        report_md = report_generator.generate_text_report(
            dataset_info=dataset_info,
            preprocessing_report=prep_report,
            eda_summary=eda_sum,
            clustering_results=cl_res,
            classification_results=class_res,
            prediction_results=pred_res,
            association_rules_summary=arm_res,
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "⬇️ Download Executive Report (Markdown .md)",
                data=report_md.encode("utf-8"),
                file_name="MarineMine_Pro_Report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col_d2:
            pdf_bytes = report_generator.try_generate_pdf(report_md)
            if pdf_bytes:
                st.download_button(
                    "⬇️ Download Executive Report (PDF)",
                    data=pdf_bytes,
                    file_name="MarineMine_Pro_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.info("PDF generation requires 'reportlab' package.")

        st.markdown("---")
        st.markdown("### 📄 Preview Generated Executive Report")
        with st.expander("Expand to Read Full Technical Report", expanded=True):
            st.markdown(report_md)


# ---------------------------------------------------------------------------
# MODULE 12: ABOUT PROJECT
# ---------------------------------------------------------------------------
elif st.session_state["nav_selection"] == "ℹ️ About Project":
    st.markdown("## ℹ️ About MarineMine Pro")
    st.markdown("""
        ### Mining Patterns of Marine Pollution using Data Mining & Machine Learning

        **MarineMine Pro** was conceived and developed as a complete academic and scientific analytics platform
        dedicated to transforming raw oceanographic telemetry into structured intelligence.

        #### Academic & Scientific Objectives
        - **Unsupervised Pattern Discovery:** Grouping ocean regions with similar multi-parameter contamination signatures.
        - **Predictive Modeling:** Developing reliable regression and classification pipelines for environmental decision-making.
        - **Transparent Explainability:** Exposing decision tree boundaries, confusion matrices, and Apriori association rules.
        - **Spatial Geoinformatics:** Integrating GIS mapping with density clustering to isolate pollution hotspots.

        #### Technology Architecture
        - **Core Language:** Python 3.10+
        - **Frontend Presentation:** Streamlit with Custom Ocean-Blue UI CSS
        - **Data Processing:** Pandas, NumPy
        - **Machine Learning Suite:** Scikit-learn (KMeans, RandomForest, LogisticRegression, DecisionTree)
        - **Association Rules:** MLxtend (Apriori, association_rules)
        - **Visualization:** Plotly Express & Graph Objects, Seaborn, Matplotlib
        - **Geospatial Mapping:** Folium & Streamlit-Folium

        #### Ethical & Academic Disclaimer
        > Observations and predictions rendered within MarineMine Pro are generated for computational research,
        > educational demonstrations, and environmental algorithm benchmarking.
        > When utilizing synthetic demonstration datasets, outputs must not be construed as official government environmental sensor telemetry.
    """)
