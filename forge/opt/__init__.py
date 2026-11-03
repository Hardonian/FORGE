"""Inverse design optimization and generative search engine for FORGE."""
from .search_space import DuctSearchSpace, ParamRange, sample_latin_hypercube
from .optimizer import (
    CandidateEvaluation,
    evaluate_spec,
    fast_non_dominated_sort,
    compute_crowding_distances,
    run_nsga2_optimization,
)
from .report import generate_html_report, generate_pareto_svg
from .bezier_diffuser import compute_multi_station_diffuser_loss, optimize_bezier_diffuser

__all__ = [
    "DuctSearchSpace",
    "ParamRange",
    "sample_latin_hypercube",
    "CandidateEvaluation",
    "evaluate_spec",
    "fast_non_dominated_sort",
    "compute_crowding_distances",
    "run_nsga2_optimization",
    "generate_html_report",
    "generate_pareto_svg",
    "compute_multi_station_diffuser_loss",
    "optimize_bezier_diffuser",
]
