"""Physics simulation and automated CFD generation engine for FORGE."""
from .solver_1d import solve_duct_physics, solve_fan_operating_point, SERVER_FAN_CURVES
from .fluid_domain import extract_fluid_domain
from .openfoam import generate_openfoam_case
from .su2 import generate_su2_config

__all__ = [
    "solve_duct_physics",
    "solve_fan_operating_point",
    "SERVER_FAN_CURVES",
    "extract_fluid_domain",
    "generate_openfoam_case",
    "generate_su2_config",
]
