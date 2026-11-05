"""FORGE: Physics-informed inverse design platform for AI accelerator thermal management."""
from .duct import Duct, triangles, stl_text, metrics, write_candidate
from .core.models import DuctSpec, PhysicsConditions, PhysicsMetrics, DFMReport
from .cad.mesh import build_duct_mesh
from .physics.solver_1d import solve_duct_physics

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "Duct",
    "triangles",
    "stl_text",
    "metrics",
    "write_candidate",
    "DuctSpec",
    "PhysicsConditions",
    "PhysicsMetrics",
    "DFMReport",
    "build_duct_mesh",
    "solve_duct_physics",
]
