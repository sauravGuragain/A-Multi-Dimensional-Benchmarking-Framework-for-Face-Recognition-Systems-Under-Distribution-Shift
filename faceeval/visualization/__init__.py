"""
faceeval.visualization
=======================
Publication-quality figure generation for FaceEval-X.
"""

from faceeval.visualization.style import (
    apply_style, figure, multi_figure, model_color,
    OKABE_ITO, TRADITIONAL_COLOR, DL_COLOR, FIGURE_SIZES,
)
from faceeval.visualization.exporter import FigureExporter
from faceeval.visualization.roc import (
    plot_roc_curves, plot_adc_curves, plot_adc_grid,
)
from faceeval.visualization.calibration_plot import (
    plot_reliability_diagram, plot_ece_vs_severity,
)
from faceeval.visualization.radar import (
    plot_fingerprint_radar, plot_paradigm_comparison_radar,
)
from faceeval.visualization.heatmap import (
    plot_heatmap, plot_fingerprint_distance_heatmap,
    plot_failure_overlap_heatmap, plot_subgroup_accuracy_heatmap,
    plot_confusion_matrix,
)
from faceeval.visualization.deployment_chart import (
    plot_deployment_score_bars, plot_cross_scenario_ranking,
    plot_component_comparison, plot_sensitivity_waterfall,
)

__all__ = [
    "apply_style", "figure", "multi_figure", "model_color",
    "OKABE_ITO", "TRADITIONAL_COLOR", "DL_COLOR", "FIGURE_SIZES",
    "FigureExporter",
    "plot_roc_curves", "plot_adc_curves", "plot_adc_grid",
    "plot_reliability_diagram", "plot_ece_vs_severity",
    "plot_fingerprint_radar", "plot_paradigm_comparison_radar",
    "plot_heatmap", "plot_fingerprint_distance_heatmap",
    "plot_failure_overlap_heatmap", "plot_subgroup_accuracy_heatmap",
    "plot_confusion_matrix",
    "plot_deployment_score_bars", "plot_cross_scenario_ranking",
    "plot_component_comparison", "plot_sensitivity_waterfall",
]
