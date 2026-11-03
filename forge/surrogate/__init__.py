"""Machine learning and neural surrogate modeling engine for FORGE."""
from .model import DuctFlowSurrogate, extract_feature_tensor
from .active_learning import screen_candidates, rank_candidates_by_acquisition
from .fno_field import (
    FNOFlowFieldSurrogate,
    SpatialFlowFieldPrediction,
    SpectralConv2d,
    predict_flow_field,
)

__all__ = [
    "DuctFlowSurrogate",
    "extract_feature_tensor",
    "screen_candidates",
    "rank_candidates_by_acquisition",
    "FNOFlowFieldSurrogate",
    "SpatialFlowFieldPrediction",
    "SpectralConv2d",
    "predict_flow_field",
]
