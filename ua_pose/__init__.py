"""Uncertainty-Aware Lightweight Pose Estimation Extension for MMPose."""
from . import compat  # noqa: F401  (must run before mmpose is imported)
from .backbones import (GlobalContextBlock, SOCAPRNet,
                        SpatiallyOrientedContextualAttention)
from .evaluation import (UncertaintyPoseMetric, compute_auroc, compute_ece,
                         compute_risk_coverage_and_ause)
from .heads import CompositionalTokenMixer, UAHeatmapHead, UASimCCHead
from .losses import HeteroscedasticHeatmapLoss, HeteroscedasticKLLoss

__all__ = [
    'SOCAPRNet',
    'GlobalContextBlock',
    'SpatiallyOrientedContextualAttention',
    'UASimCCHead',
    'UAHeatmapHead',
    'CompositionalTokenMixer',
    'HeteroscedasticKLLoss',
    'HeteroscedasticHeatmapLoss',
    'UncertaintyPoseMetric',
    'compute_auroc',
    'compute_ece',
    'compute_risk_coverage_and_ause',
]
