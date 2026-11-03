"""Core models and mathematical primitives for FORGE."""
from .geometry import Vec3, Triangle, calculate_mesh_volume, calculate_mesh_surface_area
from .hashes import canonical_json, sha256_hash
from .models import (
    CrossSectionShape,
    ServerFanStandard,
    DuctMaterial,
    MATERIAL_PROPERTIES,
    DuctSpec,
    PhysicsConditions,
    PhysicsMetrics,
    DFMReport,
)

__all__ = [
    "Vec3",
    "Triangle",
    "calculate_mesh_volume",
    "calculate_mesh_surface_area",
    "canonical_json",
    "sha256_hash",
    "CrossSectionShape",
    "ServerFanStandard",
    "DuctMaterial",
    "MATERIAL_PROPERTIES",
    "DuctSpec",
    "PhysicsConditions",
    "PhysicsMetrics",
    "DFMReport",
]

