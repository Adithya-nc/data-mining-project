"""
MarineMine Pro — Interactive Maps Module
=========================================
Folium-based geospatial visualization functions for marine pollution data.

All functions handle missing / invalid coordinates gracefully:
  - Rows with NaN lat/lon are silently dropped before rendering.
  - Values outside [-90, 90] (lat) or [-180, 180] (lon) are dropped.

Dependencies
------------
  pip install folium folium[plugins] pandas numpy
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "folium is required for maps.py. "
        "Install it with:  pip install folium"
    ) from exc

# ---------------------------------------------------------------------------
# Theme / palette constants (ocean-blue to match charts.py)
# ---------------------------------------------------------------------------
PRIMARY    = "#0A3D62"
ACCENT     = "#1E90FF"
BG_COLOR   = "#F0F8FF"

# Default colour ramp for CircleMarker colour-coding
_DEFAULT_COLORMAP = [
    "#1E90FF",  # low
    "#0A7DA6",
    "#0A5FA3",
    "#0A3D62",  # high
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_coords(df: pd.DataFrame,
                  lat_col: str,
                  lon_col: str) -> pd.DataFrame:
    """
    Return a copy of *df* with invalid or missing lat/lon rows removed.

    Valid ranges: latitude [-90, 90], longitude [-180, 180].
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(
            f"Coordinate columns '{lat_col}' / '{lon_col}' not found in DataFrame."
        )
    work = df.copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")

    n_before = len(work)
    work = work.dropna(subset=[lat_col, lon_col])
    work = work[
        work[lat_col].between(-90, 90) &
        work[lon_col].between(-180, 180)
    ]
    n_after = len(work)

    if n_after < n_before:
        warnings.warn(
            f"{n_before - n_after} row(s) dropped due to missing or invalid "
            f"coordinates in '{lat_col}'/'{lon_col}'.",
            stacklevel=3,
        )
    return work


def _value_to_color(value: float,
                    vmin: float,
                    vmax: float,
                    colormap: Optional[List[str]] = None) -> str:
    """
    Linearly interpolate *value* in [vmin, vmax] to a hex colour.

    Uses the internal ocean-blue ramp by default.
    """
    cmap = colormap or _DEFAULT_COLORMAP
    if vmax == vmin:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))

    # Map t to segment in cmap
    n = len(cmap) - 1
    seg = int(t * n)
    seg = min(seg, n - 1)
    local_t = (t * n) - seg

    def _hex_to_rgb(h: str):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(r, g, b):
        return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))

    r1, g1, b1 = _hex_to_rgb(cmap[seg])
    r2, g2, b2 = _hex_to_rgb(cmap[seg + 1])

    r = r1 + (r2 - r1) * local_t
    g = g1 + (g2 - g1) * local_t
    b = b1 + (b2 - b1) * local_t
    return _rgb_to_hex(r, g, b)


def _build_popup(row: pd.Series, popup_cols: Sequence[str]) -> str:
    """Build a simple HTML popup string from selected columns."""
    valid_cols = [c for c in popup_cols if c in row.index]
    if not valid_cols:
        return ""
    lines = "".join(
        f"<tr><td style='padding:2px 6px;font-weight:bold;color:{PRIMARY}'>"
        f"{col}</td><td style='padding:2px 6px'>{row[col]}</td></tr>"
        for col in valid_cols
    )
    return (
        f"<table style='font-family:Arial,sans-serif;font-size:12px;"
        f"border-collapse:collapse;background:{BG_COLOR}'>{lines}</table>"
    )


# ---------------------------------------------------------------------------
# 1. create_base_map
# ---------------------------------------------------------------------------

def create_base_map(center_lat: float = 0.0,
                    center_lon: float = 0.0,
                    zoom: int = 2) -> folium.Map:
    """
    Create a base Folium map with CartoDB Positron tiles.

    Parameters
    ----------
    center_lat : Latitude of the initial map centre.
    center_lon : Longitude of the initial map centre.
    zoom       : Initial zoom level (1–18).

    Returns
    -------
    folium.Map
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )

    # Branding title control
    title_html = (
        f"<div style='position:fixed;top:10px;left:50%;transform:translateX(-50%);"
        f"z-index:1000;background:{BG_COLOR};padding:6px 16px;"
        f"border:2px solid {PRIMARY};border-radius:6px;"
        f"font-family:Arial,sans-serif;font-size:15px;"
        f"font-weight:bold;color:{PRIMARY};box-shadow:2px 2px 6px rgba(0,0,0,0.2)'>"
        f"🌊 MarineMine Pro — Pollution Map</div>"
    )
    m.get_root().html.add_child(folium.Element(title_html))

    # Layer control (populated once layers are added)
    folium.LayerControl(position="topright").add_to(m)
    return m


# ---------------------------------------------------------------------------
# 2. add_pollution_markers
# ---------------------------------------------------------------------------

def add_pollution_markers(m: folium.Map,
                          df: pd.DataFrame,
                          lat_col: str = "latitude",
                          lon_col: str = "longitude",
                          popup_cols: Optional[List[str]] = None,
                          color_by_col: Optional[str] = None,
                          radius: float = 7.0,
                          layer_name: str = "Pollution Sites") -> folium.Map:
    """
    Add coloured CircleMarkers to a Folium map.

    Parameters
    ----------
    m            : Target folium.Map object.
    df           : Input DataFrame.
    lat_col      : Latitude column name.
    lon_col      : Longitude column name.
    popup_cols   : Columns to include in the popup tooltip.
    color_by_col : Numeric column used to colour markers (ocean-blue ramp).
                   If None or non-numeric, all markers use ACCENT (#1E90FF).
    radius       : Circle radius in pixels.
    layer_name   : Name of the FeatureGroup layer.

    Returns
    -------
    folium.Map  (the same map, mutated in-place)
    """
    popup_cols = popup_cols or []
    work = _clean_coords(df, lat_col, lon_col)
    if work.empty:
        warnings.warn("No valid coordinate rows — no markers added.", stacklevel=2)
        return m

    # Determine colour values
    use_color_scale = (
        color_by_col is not None
        and color_by_col in work.columns
        and pd.api.types.is_numeric_dtype(work[color_by_col])
    )
    if use_color_scale:
        vals = work[color_by_col].fillna(0)
        vmin, vmax = float(vals.min()), float(vals.max())
    else:
        vals = None
        vmin = vmax = 0.0

    fg = folium.FeatureGroup(name=layer_name, show=True)

    for _, row in work.iterrows():
        if use_color_scale:
            color = _value_to_color(float(row[color_by_col]), vmin, vmax)
        else:
            color = ACCENT

        popup_html = _build_popup(row, popup_cols)
        popup = folium.Popup(popup_html, max_width=300) if popup_html else None

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=radius,
            color=PRIMARY,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1.0,
            popup=popup,
            tooltip=popup_html or f"({row[lat_col]:.4f}, {row[lon_col]:.4f})",
        ).add_to(fg)

    fg.add_to(m)

    # Colour scale legend (bottom-right)
    if use_color_scale:
        legend_html = (
            f"<div style='position:fixed;bottom:30px;right:10px;"
            f"z-index:1000;background:{BG_COLOR};padding:8px 12px;"
            f"border:2px solid {PRIMARY};border-radius:6px;"
            f"font-family:Arial,sans-serif;font-size:11px;color:{PRIMARY}'>"
            f"<b>{color_by_col}</b><br>"
            f"<div style='width:120px;height:12px;"
            f"background:linear-gradient(to right,{_DEFAULT_COLORMAP[0]},{_DEFAULT_COLORMAP[-1]});"
            f"margin:4px 0;border-radius:3px'></div>"
            f"<span>{vmin:.2g}</span>"
            f"<span style='float:right'>{vmax:.2g}</span>"
            f"</div>"
        )
        m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ---------------------------------------------------------------------------
# 3. add_heatmap_layer
# ---------------------------------------------------------------------------

def add_heatmap_layer(m: folium.Map,
                      df: pd.DataFrame,
                      lat_col: str = "latitude",
                      lon_col: str = "longitude",
                      weight_col: Optional[str] = None,
                      radius: int = 20,
                      blur: int = 15,
                      min_opacity: float = 0.3,
                      max_zoom: int = 14,
                      layer_name: str = "Heat Map") -> folium.Map:
    """
    Add a HeatMap layer to a Folium map.

    Parameters
    ----------
    m          : Target folium.Map object.
    df         : Input DataFrame.
    lat_col    : Latitude column name.
    lon_col    : Longitude column name.
    weight_col : Optional numeric column to use as intensity weights.
                 If None, every point has equal weight.
    radius     : HeatMap radius parameter.
    blur       : HeatMap blur parameter.
    min_opacity: Minimum opacity of the heat layer.
    max_zoom   : Zoom level at which individual points begin to show.
    layer_name : Name of the layer for the layer control.

    Returns
    -------
    folium.Map  (the same map, mutated in-place)
    """
    work = _clean_coords(df, lat_col, lon_col)
    if work.empty:
        warnings.warn("No valid coordinate rows — heat map not added.", stacklevel=2)
        return m

    if weight_col and weight_col in work.columns:
        raw_weights = pd.to_numeric(work[weight_col], errors="coerce").fillna(0)
        w_max = raw_weights.max()
        weights = (raw_weights / w_max).tolist() if w_max > 0 else raw_weights.tolist()
        heat_data = [
            [row[lat_col], row[lon_col], w]
            for (_, row), w in zip(work.iterrows(), weights)
        ]
    else:
        heat_data = [[row[lat_col], row[lon_col]] for _, row in work.iterrows()]

    HeatMap(
        heat_data,
        name=layer_name,
        radius=radius,
        blur=blur,
        min_opacity=min_opacity,
        max_zoom=max_zoom,
        gradient={
            "0.0": "#F0F8FF",
            "0.4": "#74B9E0",
            "0.6": "#1E90FF",
            "0.8": "#0A5FA3",
            "1.0": "#0A3D62",
        },
        show=True,
    ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# 4. add_cluster_markers
# ---------------------------------------------------------------------------

def add_cluster_markers(m: folium.Map,
                        df: pd.DataFrame,
                        lat_col: str = "latitude",
                        lon_col: str = "longitude",
                        popup_cols: Optional[List[str]] = None,
                        layer_name: str = "Clustered Markers") -> folium.Map:
    """
    Add a MarkerCluster layer with popup information.

    Parameters
    ----------
    m          : Target folium.Map object.
    df         : Input DataFrame.
    lat_col    : Latitude column name.
    lon_col    : Longitude column name.
    popup_cols : Columns to include in the popup tooltip.
    layer_name : Name of the FeatureGroup layer.

    Returns
    -------
    folium.Map  (the same map, mutated in-place)
    """
    popup_cols = popup_cols or []
    work = _clean_coords(df, lat_col, lon_col)
    if work.empty:
        warnings.warn("No valid coordinate rows — cluster markers not added.",
                      stacklevel=2)
        return m

    fg = folium.FeatureGroup(name=layer_name, show=True)
    cluster = MarkerCluster(
        options={
            "maxClusterRadius": 60,
            "disableClusteringAtZoom": 12,
            "spiderfyOnMaxZoom": True,
        }
    )

    for _, row in work.iterrows():
        popup_html = _build_popup(row, popup_cols)
        popup = folium.Popup(popup_html, max_width=300) if popup_html else None
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=popup,
            tooltip=popup_html or f"({row[lat_col]:.4f}, {row[lon_col]:.4f})",
            icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
        ).add_to(cluster)

    cluster.add_to(fg)
    fg.add_to(m)
    return m


# ---------------------------------------------------------------------------
# 5. identify_hotspots
# ---------------------------------------------------------------------------

def identify_hotspots(df: pd.DataFrame,
                      lat_col: str = "latitude",
                      lon_col: str = "longitude",
                      value_col: str = "pollution_level",
                      top_n: int = 10,
                      grid_size: float = 1.0) -> pd.DataFrame:
    """
    Identify top-N pollution hotspot grid cells.

    The map is divided into cells of `grid_size` degrees. Rows falling
    into the same cell are aggregated (sum of *value_col*, count of rows,
    and centroid lat/lon).  The top-N cells by aggregated value are returned.

    Parameters
    ----------
    df         : Input DataFrame.
    lat_col    : Latitude column name.
    lon_col    : Longitude column name.
    value_col  : Numeric column to aggregate within each cell.
    top_n      : Number of top hotspot cells to return.
    grid_size  : Cell edge length in decimal degrees.

    Returns
    -------
    pd.DataFrame
        Columns: grid_lat, grid_lon, centre_lat, centre_lon,
                 total_{value_col}, count, mean_{value_col}
    """
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame.")

    work = _clean_coords(df, lat_col, lon_col)
    if work.empty:
        return pd.DataFrame()

    work = work.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])

    if work.empty:
        warnings.warn(f"No valid numeric rows in '{value_col}'.", stacklevel=2)
        return pd.DataFrame()

    # Snap lat/lon to grid cell origin
    work["_grid_lat"] = np.floor(work[lat_col] / grid_size) * grid_size
    work["_grid_lon"] = np.floor(work[lon_col] / grid_size) * grid_size

    agg = (
        work.groupby(["_grid_lat", "_grid_lon"])
        .agg(
            centre_lat=(lat_col, "mean"),
            centre_lon=(lon_col, "mean"),
            **{f"total_{value_col}": (value_col, "sum"),
               "count": (value_col, "count"),
               f"mean_{value_col}": (value_col, "mean")},
        )
        .reset_index()
        .rename(columns={"_grid_lat": "grid_lat", "_grid_lon": "grid_lon"})
        .sort_values(f"total_{value_col}", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return agg


# ---------------------------------------------------------------------------
# 6. get_regional_summary
# ---------------------------------------------------------------------------

def get_regional_summary(df: pd.DataFrame,
                         region_col: str = "region",
                         value_col: str = "pollution_level") -> pd.DataFrame:
    """
    Compute descriptive statistics of *value_col* grouped by *region_col*.

    Parameters
    ----------
    df         : Input DataFrame.
    region_col : Column holding region/zone labels.
    value_col  : Numeric column to summarise.

    Returns
    -------
    pd.DataFrame
        Indexed by region with columns:
        count, mean, median, std, min, max, total,
        missing_count, missing_pct
    """
    for col in (region_col, value_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    work = df[[region_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    # Base stats
    summary = (
        work.groupby(region_col)[value_col]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
            total="sum",
        )
        .round(4)
    )

    # Missing info
    missing = work.groupby(region_col)[value_col].apply(
        lambda s: s.isna().sum()
    ).rename("missing_count")
    total_rows = work.groupby(region_col)[value_col].apply(len).rename("total_rows")
    missing_pct = ((missing / total_rows) * 100).round(2).rename("missing_pct")

    summary = summary.join([missing, missing_pct])
    summary.index.name = region_col
    return summary.reset_index()
