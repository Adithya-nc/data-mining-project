"""
MarineMine Pro — Visualization Charts Module
=============================================
Plotly and Seaborn chart functions for marine pollution data analysis.

Ocean-Blue Theme
----------------
  Primary    : #0A3D62
  Accent     : #1E90FF
  Background : #F0F8FF
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
PRIMARY    = "#0A3D62"
ACCENT     = "#1E90FF"
BG_COLOR   = "#F0F8FF"
GRID_COLOR = "#C6DCF0"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=PRIMARY, family="Arial, sans-serif"),
        title=dict(font=dict(color=PRIMARY, size=18)),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        colorway=[ACCENT, "#2EACB4", "#0A7DA6", "#FF6B6B", "#F9CA24",
                  "#6AB04C", "#9B59B6", "#E67E22", "#1ABC9C", "#E74C3C"],
    )
)

OCEAN_COLORSCALE = [
    [0.0,  "#F0F8FF"],
    [0.25, "#74B9E0"],
    [0.5,  "#1E90FF"],
    [0.75, "#0A5FA3"],
    [1.0,  "#0A3D62"],
]


def _apply_plotly_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the MarineMine ocean-blue theme to a Plotly figure."""
    fig.update_layout(
        template=None,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=PRIMARY, family="Arial, sans-serif"),
        title=dict(text=title, font=dict(color=PRIMARY, size=18)),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                   linecolor=PRIMARY),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                   linecolor=PRIMARY),
        colorway=PLOTLY_TEMPLATE["layout"]["colorway"],
        legend=dict(bgcolor=BG_COLOR, bordercolor=GRID_COLOR),
        margin=dict(l=60, r=40, t=70, b=60),
    )
    return fig


def _mpl_theme() -> None:
    """Apply matplotlib ocean-blue style."""
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    BG_COLOR,
        "axes.edgecolor":    PRIMARY,
        "axes.labelcolor":   PRIMARY,
        "xtick.color":       PRIMARY,
        "ytick.color":       PRIMARY,
        "text.color":        PRIMARY,
        "axes.grid":         True,
        "grid.color":        GRID_COLOR,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.7,
        "font.family":       "sans-serif",
    })


# ---------------------------------------------------------------------------
# 1. plot_pollution_by_type
# ---------------------------------------------------------------------------

def plot_pollution_by_type(df: pd.DataFrame,
                           type_col: str = "pollution_type",
                           value_col: Optional[str] = None) -> go.Figure:
    """
    Bar chart of pollution counts (or totals) by type.

    Parameters
    ----------
    df        : Input DataFrame.
    type_col  : Column containing pollution type labels.
    value_col : Optional numeric column to sum per type.
                If None, counts rows per type.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if type_col not in df.columns:
        raise ValueError(f"Column '{type_col}' not found in DataFrame.")

    if value_col and value_col in df.columns:
        summary = df.groupby(type_col)[value_col].sum().reset_index()
        summary.columns = [type_col, "value"]
        y_label = f"Total {value_col}"
    else:
        summary = df[type_col].value_counts().reset_index()
        summary.columns = [type_col, "value"]
        y_label = "Count"

    summary = summary.sort_values("value", ascending=False)

    fig = go.Figure(
        go.Bar(
            x=summary[type_col],
            y=summary["value"],
            marker=dict(
                color=summary["value"],
                colorscale=OCEAN_COLORSCALE,
                line=dict(color=PRIMARY, width=0.8),
            ),
            hovertemplate="<b>%{x}</b><br>" + y_label + ": %{y:,.2f}<extra></extra>",
        )
    )
    fig = _apply_plotly_theme(fig, "Pollution by Type")
    fig.update_layout(
        xaxis_title="Pollution Type",
        yaxis_title=y_label,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. plot_pollution_by_region
# ---------------------------------------------------------------------------

def plot_pollution_by_region(df: pd.DataFrame,
                             region_col: str = "region",
                             value_col: Optional[str] = None) -> go.Figure:
    """
    Horizontal bar chart of regional pollution totals/counts.

    Parameters
    ----------
    df         : Input DataFrame.
    region_col : Column containing region labels.
    value_col  : Optional numeric column to sum per region.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if region_col not in df.columns:
        raise ValueError(f"Column '{region_col}' not found in DataFrame.")

    if value_col and value_col in df.columns:
        summary = df.groupby(region_col)[value_col].sum().reset_index()
        summary.columns = [region_col, "value"]
        x_label = f"Total {value_col}"
    else:
        summary = df[region_col].value_counts().reset_index()
        summary.columns = [region_col, "value"]
        x_label = "Count"

    summary = summary.sort_values("value", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=summary["value"],
            y=summary[region_col],
            orientation="h",
            marker=dict(
                color=summary["value"],
                colorscale=OCEAN_COLORSCALE,
                line=dict(color=PRIMARY, width=0.8),
            ),
            hovertemplate="<b>%{y}</b><br>" + x_label + ": %{x:,.2f}<extra></extra>",
        )
    )
    fig = _apply_plotly_theme(fig, "Pollution by Region")
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Region",
        showlegend=False,
        height=max(400, len(summary) * 30 + 150),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. plot_pollution_trend
# ---------------------------------------------------------------------------

def plot_pollution_trend(df: pd.DataFrame,
                         date_col: str = "date",
                         value_col: Optional[str] = None,
                         freq: str = "M",
                         color_col: Optional[str] = None) -> go.Figure:
    """
    Line chart of pollution over time, resampled by `freq`.

    Parameters
    ----------
    df        : Input DataFrame (must contain a parseable date column).
    date_col  : Name of the datetime column.
    value_col : Numeric column to aggregate. Counts rows if None.
    freq      : Pandas resample frequency string (default 'M' = monthly).
    color_col : Optional grouping column for multi-line chart.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])

    if work.empty:
        raise ValueError("No valid date rows remain after parsing.")

    if color_col and color_col in work.columns:
        groups = work.groupby(color_col)
        fig = go.Figure()
        palette = PLOTLY_TEMPLATE["layout"]["colorway"]
        for idx, (name, grp) in enumerate(groups):
            grp = grp.set_index(date_col).sort_index()
            if value_col and value_col in grp.columns:
                ts = grp[value_col].resample(freq).sum()
            else:
                ts = grp.resample(freq).size()
            ts.name = "value"
            color = palette[idx % len(palette)]
            fig.add_trace(go.Scatter(
                x=ts.index, y=ts.values,
                mode="lines+markers",
                name=str(name),
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:,.2f}}<extra></extra>",
            ))
    else:
        work = work.set_index(date_col).sort_index()
        if value_col and value_col in work.columns:
            ts = work[value_col].resample(freq).sum()
        else:
            ts = work.resample(freq).size()
        ts.name = "value"
        fig = go.Figure(
            go.Scatter(
                x=ts.index, y=ts.values,
                mode="lines+markers",
                fill="tozeroy",
                fillcolor=f"rgba(30,144,255,0.15)",
                line=dict(color=ACCENT, width=2.5),
                marker=dict(color=PRIMARY, size=5),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:,.2f}<extra></extra>",
            )
        )

    y_title = f"Total {value_col}" if value_col else "Count"
    fig = _apply_plotly_theme(fig, "Pollution Trend Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title=y_title)
    return fig


# ---------------------------------------------------------------------------
# 4. plot_distribution
# ---------------------------------------------------------------------------

def plot_distribution(df: pd.DataFrame,
                      col: str,
                      bins: int = 40,
                      kde: bool = True) -> matplotlib.figure.Figure:
    """
    Histogram + optional KDE for a numeric column.

    Parameters
    ----------
    df   : Input DataFrame.
    col  : Numeric column to plot.
    bins : Number of histogram bins.
    kde  : Whether to overlay a KDE curve.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    _mpl_theme()
    fig, ax = plt.subplots(figsize=(9, 5))

    series = df[col].dropna()
    ax.hist(series, bins=bins, color=ACCENT, alpha=0.65,
            edgecolor=PRIMARY, linewidth=0.6, density=kde, label="Histogram")

    if kde:
        try:
            from scipy.stats import gaussian_kde
            kde_func = gaussian_kde(series)
            x_range = np.linspace(series.min(), series.max(), 300)
            ax.plot(x_range, kde_func(x_range), color=PRIMARY, linewidth=2,
                    label="KDE")
        except ImportError:
            warnings.warn("scipy not installed — KDE skipped.")

    ax.set_title(f"Distribution of {col}", color=PRIMARY, fontsize=14, pad=12)
    ax.set_xlabel(col, color=PRIMARY)
    ax.set_ylabel("Density" if kde else "Frequency", color=PRIMARY)
    ax.legend(framealpha=0.7)
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. plot_correlation_heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame,
                             method: str = "pearson",
                             annot: bool = True) -> matplotlib.figure.Figure:
    """
    Seaborn heatmap of pairwise correlations for numeric columns.

    Parameters
    ----------
    df     : Input DataFrame.
    method : Correlation method: 'pearson', 'spearman', or 'kendall'.
    annot  : Whether to annotate cells with correlation values.

    Returns
    -------
    matplotlib.figure.Figure
    """
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("DataFrame has no numeric columns.")

    corr = numeric.corr(method=method)
    n = len(corr)
    size = max(6, n * 0.8)

    _mpl_theme()
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        linecolor=BG_COLOR,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(f"Correlation Heatmap ({method.capitalize()})",
                 color=PRIMARY, fontsize=14, pad=12)
    ax.tick_params(colors=PRIMARY)
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. plot_missing_values
# ---------------------------------------------------------------------------

def plot_missing_values(df: pd.DataFrame,
                        threshold: float = 0.0) -> go.Figure:
    """
    Bar chart showing percentage of missing values per column.

    Parameters
    ----------
    df        : Input DataFrame.
    threshold : Only show columns with missing % > threshold (0–100).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    missing = (df.isnull().mean() * 100).reset_index()
    missing.columns = ["column", "missing_pct"]
    missing = missing[missing["missing_pct"] > threshold]
    missing = missing.sort_values("missing_pct", ascending=False)

    if missing.empty:
        missing = pd.DataFrame({"column": ["(none)"], "missing_pct": [0.0]})

    colors = [ACCENT if pct < 50 else PRIMARY for pct in missing["missing_pct"]]

    fig = go.Figure(
        go.Bar(
            x=missing["column"],
            y=missing["missing_pct"],
            marker=dict(color=colors, line=dict(color=PRIMARY, width=0.8)),
            hovertemplate="<b>%{x}</b><br>Missing: %{y:.2f}%<extra></extra>",
        )
    )
    fig = _apply_plotly_theme(fig, "Missing Value Percentages")
    fig.update_layout(
        xaxis_title="Column",
        yaxis_title="Missing (%)",
        yaxis=dict(range=[0, 105]),
        showlegend=False,
    )
    fig.add_hline(y=50, line_dash="dash", line_color="#E74C3C",
                  annotation_text="50% threshold", annotation_font_color="#E74C3C")
    return fig


# ---------------------------------------------------------------------------
# 7. plot_box_by_category
# ---------------------------------------------------------------------------

def plot_box_by_category(df: pd.DataFrame,
                         num_col: str,
                         cat_col: str,
                         show_points: bool = True) -> go.Figure:
    """
    Box plot of a numeric column grouped by a categorical column.

    Parameters
    ----------
    df          : Input DataFrame.
    num_col     : Numeric column for the box values.
    cat_col     : Categorical column for grouping.
    show_points : Overlay individual data points (jitter).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    for col in (num_col, cat_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    work = df[[cat_col, num_col]].dropna()
    categories = work[cat_col].unique()
    palette = PLOTLY_TEMPLATE["layout"]["colorway"]

    fig = go.Figure()
    for idx, cat in enumerate(sorted(categories, key=str)):
        vals = work.loc[work[cat_col] == cat, num_col]
        color = palette[idx % len(palette)]
        fig.add_trace(go.Box(
            y=vals,
            name=str(cat),
            boxmean="sd",
            marker=dict(color=color, size=4, opacity=0.6),
            line=dict(color=color),
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.3)",
            boxpoints="all" if show_points else False,
            jitter=0.3,
            hovertemplate=f"<b>{cat}</b><br>{num_col}: %{{y:,.2f}}<extra></extra>",
        ))

    fig = _apply_plotly_theme(fig, f"{num_col} by {cat_col}")
    fig.update_layout(
        xaxis_title=cat_col,
        yaxis_title=num_col,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. plot_scatter
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame,
                 x: str,
                 y: str,
                 color: Optional[str] = None,
                 size: Optional[str] = None,
                 opacity: float = 0.75) -> go.Figure:
    """
    Scatter plot with optional color and size encoding.

    Parameters
    ----------
    df      : Input DataFrame.
    x       : Column for x-axis.
    y       : Column for y-axis.
    color   : Optional column to encode as color.
    size    : Optional numeric column to encode as marker size.
    opacity : Marker opacity (0–1).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    for col in (x, y):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    work = df[[c for c in [x, y, color, size] if c and c in df.columns]].dropna()

    kwargs: dict = dict(
        x=x, y=y,
        color=color if color and color in work.columns else None,
        size=size if size and size in work.columns else None,
        opacity=opacity,
        color_continuous_scale=OCEAN_COLORSCALE,
        color_discrete_sequence=PLOTLY_TEMPLATE["layout"]["colorway"],
    )
    fig = px.scatter(work, **{k: v for k, v in kwargs.items() if v is not None},
                     template=None)
    fig = _apply_plotly_theme(fig, f"{y} vs {x}")
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig


# ---------------------------------------------------------------------------
# 9. plot_cluster_scatter
# ---------------------------------------------------------------------------

def plot_cluster_scatter(df: pd.DataFrame,
                         x: str,
                         y: str,
                         cluster_col: str,
                         show_centroids: bool = True) -> go.Figure:
    """
    Scatter plot with data points color-coded by cluster label.

    Parameters
    ----------
    df           : Input DataFrame.
    x            : Column for x-axis.
    y            : Column for y-axis.
    cluster_col  : Column holding integer or string cluster IDs.
    show_centroids: Whether to overlay centroid markers.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    for col in (x, y, cluster_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    work = df[[x, y, cluster_col]].dropna()
    clusters = sorted(work[cluster_col].unique(), key=str)
    palette = PLOTLY_TEMPLATE["layout"]["colorway"]

    fig = go.Figure()
    for idx, cluster in enumerate(clusters):
        mask = work[cluster_col] == cluster
        pts = work[mask]
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(
            x=pts[x], y=pts[y],
            mode="markers",
            name=f"Cluster {cluster}",
            marker=dict(color=color, size=7, opacity=0.75,
                        line=dict(color="white", width=0.5)),
            hovertemplate=f"Cluster {cluster}<br>{x}: %{{x:,.2f}}<br>{y}: %{{y:,.2f}}<extra></extra>",
        ))
        if show_centroids:
            fig.add_trace(go.Scatter(
                x=[pts[x].mean()], y=[pts[y].mean()],
                mode="markers",
                name=f"Centroid {cluster}",
                marker=dict(symbol="x", size=14, color=color,
                            line=dict(color=PRIMARY, width=2)),
                showlegend=False,
                hovertemplate=f"Centroid {cluster}<br>{x}: %{{x:,.2f}}<br>{y}: %{{y:,.2f}}<extra></extra>",
            ))

    fig = _apply_plotly_theme(fig, f"Cluster Scatter: {y} vs {x}")
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig


# ---------------------------------------------------------------------------
# 10. plot_pca_clusters
# ---------------------------------------------------------------------------

def plot_pca_clusters(df: pd.DataFrame,
                      feature_cols: List[str],
                      cluster_col: str,
                      n_components: int = 2) -> go.Figure:
    """
    Reduce features via PCA, then render cluster scatter.

    Parameters
    ----------
    df           : Input DataFrame.
    feature_cols : List of numeric feature column names.
    cluster_col  : Column holding cluster labels.
    n_components : PCA components (2 for 2-D, 3 for 3-D).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    missing = [c for c in feature_cols + [cluster_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    work = df[feature_cols + [cluster_col]].dropna()
    X = work[feature_cols].values
    labels = work[cluster_col].values

    pca = PCA(n_components=min(n_components, X.shape[1]))
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    clusters = sorted(set(labels), key=str)
    palette = PLOTLY_TEMPLATE["layout"]["colorway"]

    if coords.shape[1] >= 3:
        fig = go.Figure()
        for idx, cl in enumerate(clusters):
            mask = labels == cl
            color = palette[idx % len(palette)]
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                mode="markers",
                name=f"Cluster {cl}",
                marker=dict(color=color, size=5, opacity=0.8),
            ))
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({var_exp[0]:.1f}%)",
                yaxis_title=f"PC2 ({var_exp[1]:.1f}%)",
                zaxis_title=f"PC3 ({var_exp[2]:.1f}%)",
            )
        )
    else:
        fig = go.Figure()
        for idx, cl in enumerate(clusters):
            mask = labels == cl
            color = palette[idx % len(palette)]
            fig.add_trace(go.Scatter(
                x=coords[mask, 0], y=coords[mask, 1],
                mode="markers",
                name=f"Cluster {cl}",
                marker=dict(color=color, size=7, opacity=0.8,
                            line=dict(color="white", width=0.5)),
                hovertemplate=f"Cluster {cl}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
            ))
        fig.update_layout(
            xaxis_title=f"PC1 ({var_exp[0]:.1f}% var)",
            yaxis_title=f"PC2 ({var_exp[1]:.1f}% var)",
        )

    fig = _apply_plotly_theme(fig, "PCA Cluster Visualization")
    return fig


# ---------------------------------------------------------------------------
# 11. plot_elbow
# ---------------------------------------------------------------------------

def plot_elbow(inertias: Sequence[float],
               k_range: Sequence[int]) -> go.Figure:
    """
    Elbow curve for K-Means cluster selection.

    Parameters
    ----------
    inertias : Sequence of inertia (WCSS) values.
    k_range  : Corresponding sequence of k values.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if len(inertias) != len(k_range):
        raise ValueError("inertias and k_range must have the same length.")

    # Identify elbow via maximum second-derivative
    best_k = None
    if len(inertias) >= 3:
        d2 = np.diff(np.diff(list(inertias)))
        best_k = list(k_range)[np.argmax(d2) + 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range), y=list(inertias),
        mode="lines+markers",
        name="Inertia",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(color=PRIMARY, size=8, symbol="circle"),
        hovertemplate="k=%{x}<br>Inertia: %{y:,.2f}<extra></extra>",
    ))

    if best_k is not None:
        best_inertia = list(inertias)[list(k_range).index(best_k)]
        fig.add_vline(x=best_k, line_dash="dash", line_color="#E74C3C")
        fig.add_annotation(
            x=best_k, y=best_inertia,
            text=f"  Elbow @ k={best_k}",
            font=dict(color="#E74C3C"),
            showarrow=False,
            xanchor="left",
        )

    fig = _apply_plotly_theme(fig, "Elbow Curve — Optimal k Selection")
    fig.update_layout(xaxis_title="Number of Clusters (k)",
                      yaxis_title="Inertia (WCSS)")
    return fig


# ---------------------------------------------------------------------------
# 12. plot_confusion_matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: Union[np.ndarray, List[List[int]]],
                           labels: Optional[List[str]] = None) -> go.Figure:
    """
    Plotly annotated heatmap of a confusion matrix.

    Parameters
    ----------
    cm     : 2-D confusion matrix array.
    labels : Class label names.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cm = np.array(cm)
    n = cm.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]

    # Normalize for color
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    text = [[f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})"
             for j in range(n)] for i in range(n)]

    fig = go.Figure(
        go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            colorscale=OCEAN_COLORSCALE,
            showscale=True,
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
        )
    )
    fig = _apply_plotly_theme(fig, "Confusion Matrix")
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# 13. plot_feature_importance
# ---------------------------------------------------------------------------

def plot_feature_importance(importances: Sequence[float],
                             feature_names: Sequence[str],
                             top_n: Optional[int] = None) -> go.Figure:
    """
    Horizontal bar chart of model feature importances.

    Parameters
    ----------
    importances   : Importance scores (same length as feature_names).
    feature_names : Feature name strings.
    top_n         : If set, show only the top-N features.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if len(importances) != len(feature_names):
        raise ValueError("importances and feature_names must have the same length.")

    df_fi = pd.DataFrame({"feature": list(feature_names),
                           "importance": list(importances)})
    df_fi = df_fi.sort_values("importance", ascending=True)

    if top_n:
        df_fi = df_fi.tail(top_n)

    fig = go.Figure(
        go.Bar(
            x=df_fi["importance"],
            y=df_fi["feature"],
            orientation="h",
            marker=dict(
                color=df_fi["importance"],
                colorscale=OCEAN_COLORSCALE,
                line=dict(color=PRIMARY, width=0.6),
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    title = f"Feature Importance (Top {top_n})" if top_n else "Feature Importance"
    fig = _apply_plotly_theme(fig, title)
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        showlegend=False,
        height=max(400, len(df_fi) * 28 + 150),
    )
    return fig


# ---------------------------------------------------------------------------
# 14. plot_actual_vs_predicted
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(y_true: Sequence[float],
                              y_pred: Sequence[float],
                              label: str = "Sample") -> go.Figure:
    """
    Scatter of actual vs predicted values with a perfect-fit diagonal.

    Parameters
    ----------
    y_true : Ground-truth target values.
    y_pred : Model-predicted values.
    label  : Hover label prefix for each point.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    residuals = y_pred - y_true
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    padding = (mx - mn) * 0.05

    fig = go.Figure()

    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[mn - padding, mx + padding],
        y=[mn - padding, mx + padding],
        mode="lines",
        name="Perfect Fit",
        line=dict(color="#E74C3C", dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    # Scatter points coloured by residual
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(
            color=residuals,
            colorscale=OCEAN_COLORSCALE,
            showscale=True,
            colorbar=dict(title="Residual"),
            size=7,
            opacity=0.8,
            line=dict(color="white", width=0.5),
        ),
        hovertemplate=(
            f"<b>{label}</b><br>"
            "Actual: %{x:,.3f}<br>"
            "Predicted: %{y:,.3f}<br>"
            "Residual: %{marker.color:,.3f}<extra></extra>"
        ),
    ))

    # Metrics annotation
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    rmse = np.sqrt(np.mean(residuals ** 2))

    fig.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text=f"R² = {r2:.4f}<br>RMSE = {rmse:.4f}",
        showarrow=False,
        bgcolor=BG_COLOR,
        bordercolor=PRIMARY,
        borderwidth=1,
        font=dict(color=PRIMARY, size=12),
        align="left",
    )

    fig = _apply_plotly_theme(fig, "Actual vs Predicted")
    fig.update_layout(
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        xaxis=dict(range=[mn - padding, mx + padding]),
        yaxis=dict(range=[mn - padding, mx + padding]),
    )
    return fig
