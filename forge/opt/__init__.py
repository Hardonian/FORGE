"""Inverse design optimization and generative search engine for FORGE."""
from .search_space import DuctSearchSpace, ParamRange, sample_latin_hypercube
from .optimizer import CandidateEvaluation, evaluate_spec, run_nsga2_optimization
from .report import generate_html_report, generate_pareto_svg

__all__ = [
    "DuctSearchSpace",
    "ParamRange",
    "sample_latin_hypercube",
    "CandidateEvaluation",
    "evaluate_spec",
    "run_nsga2_optimization",
    "generate_html_report",
    "generate_pareto_svg",
]

