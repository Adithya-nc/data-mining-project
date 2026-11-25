"""
MarineMine Pro — Exploratory Data Analysis & Statistics Service
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src import eda, hotspot_analysis


class DataAnalyzerService:
    @staticmethod
    def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive statistical metrics for numerical and categorical features."""
        num_summary = []
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            s = df[c].dropna()
            if not s.empty:
                q25, q50, q75 = np.percentile(s, [25, 50, 75])
                iqr = q75 - q25
                num_summary.append({
                    "column": str(c),
                    "count": int(s.count()),
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
                    "min": round(float(s.min()), 2),
                    "q25": round(float(q25), 2),
                    "median": round(float(q50), 2),
                    "q75": round(float(q75), 2),
                    "max": round(float(s.max()), 2),
                    "iqr": round(float(iqr), 2),
                })

        cat_summary = []
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            s = df[c].dropna()
            if not s.empty:
                val_counts = s.value_counts()
                top_cat = str(val_counts.index[0]) if not val_counts.empty else "N/A"
                top_cnt = int(val_counts.iloc[0]) if not val_counts.empty else 0
                cat_summary.append({
                    "column": str(c),
                    "unique_values": int(s.nunique()),
                    "most_frequent": top_cat,
                    "frequency": top_cnt,
                    "frequency_pct": round(top_cnt / len(df) * 100, 1),
                })

        return {
            "numerical_summary": num_summary,
            "categorical_summary": cat_summary,
            "shape": list(df.shape),
        }

    @staticmethod
    def compute_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute Pearson correlation matrix across numerical features."""
        num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if num_df.shape[1] < 2:
            return {"features": [], "matrix": []}

        corr = num_df.corr().fillna(0.0).round(3)
        return {
            "features": corr.columns.tolist(),
            "matrix": corr.values.tolist(),
        }

    @staticmethod
    def compute_feature_distribution(df: pd.DataFrame, feature: str, bins_count: int = 15) -> Dict[str, Any]:
        """Compute histogram bins and density counts for a single numerical feature."""
        if feature not in df.columns or not np.issubdtype(df[feature].dtype, np.number):
            raise ValueError(f"Feature '{feature}' is not a valid numerical column.")

        clean_series = df[feature].dropna()
        if clean_series.empty:
            return {"feature": feature, "bins": [], "counts": [], "mean": 0, "std": 0, "min": 0, "max": 0}

        counts, bin_edges = np.histogram(clean_series, bins=bins_count)
        return {
            "feature": feature,
            "bins": [round(float(b), 2) for b in bin_edges],
            "counts": [int(c) for c in counts],
            "mean": round(float(clean_series.mean()), 2),
            "std": round(float(clean_series.std()), 2) if len(clean_series) > 1 else 0.0,
            "min": round(float(clean_series.min()), 2),
            "max": round(float(clean_series.max()), 2),
        }

    @staticmethod
    def compute_trends(df: pd.DataFrame) -> Dict[str, Any]:
        """Extract time-series trend progression and categorical breakdowns."""
        cols_lower = {c.lower(): c for c in df.columns}
        cat_col = cols_lower.get("pollution_type", cols_lower.get("type", cols_lower.get("category")))
        reg_col = cols_lower.get("region", cols_lower.get("ocean", cols_lower.get("location")))
        val_col = cols_lower.get("pollution_value", cols_lower.get("concentration", cols_lower.get("level")))
        date_col = cols_lower.get("observation_date", cols_lower.get("date", cols_lower.get("timestamp")))

        trends = []
        if date_col and val_col:
            try:
                temp_df = df[[date_col, val_col]].dropna().copy()
                temp_df["dt"] = pd.to_datetime(temp_df[date_col], errors="coerce")
                temp_df = temp_df.dropna(subset=["dt"]).sort_values("dt")
                grouped = temp_df.set_index("dt").resample("ME")[val_col].mean().reset_index()
                trends = [
                    {"date": row["dt"].strftime("%Y-%m"), "value": round(float(row[val_col]), 2)}
                    for _, row in grouped.iterrows() if pd.notna(row[val_col])
                ]
            except Exception:
                trends = []

        # Category Breakdown
        categories = []
        if cat_col and cat_col in df.columns:
            counts = df[cat_col].value_counts().reset_index()
            counts.columns = ["category", "count"]
            counts["percentage"] = (counts["count"] / len(df) * 100).round(1)
            categories = counts.to_dict(orient="records")

        # Regional Breakdown
        regions = []
        if reg_col and reg_col in df.columns:
            counts = df[reg_col].value_counts().reset_index()
            counts.columns = ["region", "count"]
            counts["percentage"] = (counts["count"] / len(df) * 100).round(1)
            regions = counts.to_dict(orient="records")

        # Bathymetric Depth Profile
        depth_profile = []
        depth_col = cols_lower.get("water_depth_m", cols_lower.get("depth_m", cols_lower.get("depth")))
        temp_col = cols_lower.get("temperature_celsius", cols_lower.get("temperature", cols_lower.get("temp")))
        do_col = cols_lower.get("dissolved_oxygen", cols_lower.get("do"))
        if depth_col and depth_col in df.columns and np.issubdtype(df[depth_col].dtype, np.number):
            try:
                sub = df.dropna(subset=[depth_col]).copy()
                if not sub.empty:
                    bins = [-float("inf"), 200, 1000, 2000, 3000, float("inf")]
                    labels = ["0-200m (Epipelagic)", "200-1,000m (Mesopelagic)", "1,000-2,000m (Bathypelagic)", "2,000-3,000m (Upper Abyssal)", "3,000m+ (Abyssal Floor)"]
                    sub["depth_zone"] = pd.cut(sub[depth_col], bins=bins, labels=labels)
                    for zone in labels:
                        z_df = sub[sub["depth_zone"] == zone]
                        if not z_df.empty:
                            entry: Dict[str, Any] = {"zone": zone, "count": int(len(z_df))}
                            if temp_col and temp_col in z_df.columns and np.issubdtype(z_df[temp_col].dtype, np.number):
                                entry["temperature"] = round(float(z_df[temp_col].mean()), 2)
                            if do_col and do_col in z_df.columns and np.issubdtype(z_df[do_col].dtype, np.number):
                                entry["dissolved_oxygen"] = round(float(z_df[do_col].mean()), 2)
                            if val_col and val_col in z_df.columns and np.issubdtype(z_df[val_col].dtype, np.number):
                                entry["pollution_value"] = round(float(z_df[val_col].mean()), 2)
                            depth_profile.append(entry)
            except Exception:
                depth_profile = []

        return {
            "date_col": date_col,
            "value_col": val_col,
            "trends": trends,
            "categories": categories,
            "regions": regions,
            "depth_profile": depth_profile,
        }

    @staticmethod
    def extract_hotspots(
        df: pd.DataFrame,
        region_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute ranked geographical risk hotspots and spatial coordinates."""
        cols_lower = {c.lower(): c for c in df.columns}
        lat_col = cols_lower.get("latitude", cols_lower.get("lat"))
        lon_col = cols_lower.get("longitude", cols_lower.get("lon", cols_lower.get("lng")))
        val_col = cols_lower.get("pollution_value", cols_lower.get("concentration", cols_lower.get("level", cols_lower.get("score"))))
        reg_col = cols_lower.get("region", cols_lower.get("location"))
        cat_col = cols_lower.get("pollution_type", cols_lower.get("type", cols_lower.get("category")))

        if not lat_col or not lon_col or not val_col:
            return {"lat_col": "", "lon_col": "", "val_col": "", "total_locations": 0, "hotspots": [], "points": []}

        filtered = df.dropna(subset=[lat_col, lon_col, val_col]).copy()
        if region_filter and region_filter != "All" and reg_col and reg_col in filtered.columns:
            filtered = filtered[filtered[reg_col].astype(str) == region_filter]
        if type_filter and type_filter != "All" and cat_col and cat_col in filtered.columns:
            filtered = filtered[filtered[cat_col].astype(str) == type_filter]

        try:
            ranked = hotspot_analysis.compute_hotspots(
                filtered,
                lat_col=lat_col,
                lon_col=lon_col,
                value_col=val_col,
                top_n=15,
            )
            hotspots_list = ranked.replace({np.nan: None}).to_dict(orient="records") if not ranked.empty else []
        except Exception:
            hotspots_list = []

        points = []
        for _, r in filtered.head(250).iterrows():
            points.append({
                "lat": float(r[lat_col]),
                "lng": float(r[lon_col]),
                "val": round(float(r[val_col]), 2),
                "region": str(r[reg_col]) if reg_col and pd.notna(r.get(reg_col)) else None,
                "category": str(r[cat_col]) if cat_col and pd.notna(r.get(cat_col)) else None,
            })

        return {
            "lat_col": lat_col,
            "lon_col": lon_col,
            "val_col": val_col,
            "total_locations": len(filtered),
            "hotspots": hotspots_list,
            "points": points,
        }
