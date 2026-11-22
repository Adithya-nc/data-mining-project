"""MarineMine Pro — Visualizations package."""

from .charts import (
    plot_pollution_by_type,
    plot_pollution_by_region,
    plot_pollution_trend,
    plot_distribution,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_box_by_category,
    plot_scatter,
    plot_cluster_scatter,
    plot_pca_clusters,
    plot_elbow,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_actual_vs_predicted,
)
from .maps import (
    create_base_map,
    add_pollution_markers,
    add_heatmap_layer,
    add_cluster_markers,
    identify_hotspots,
    get_regional_summary,
)

__all__ = [
    # charts
    "plot_pollution_by_type",
    "plot_pollution_by_region",
    "plot_pollution_trend",
    "plot_distribution",
    "plot_correlation_heatmap",
    "plot_missing_values",
    "plot_box_by_category",
    "plot_scatter",
    "plot_cluster_scatter",
    "plot_pca_clusters",
    "plot_elbow",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_actual_vs_predicted",
    # maps
    "create_base_map",
    "add_pollution_markers",
    "add_heatmap_layer",
    "add_cluster_markers",
    "identify_hotspots",
    "get_regional_summary",
]
