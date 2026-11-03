"""Parametric CAD and mesh generation engine for FORGE."""
from .mesh import (
    build_duct_mesh,
    generate_profile_ring,
    compute_normal_wall_compensation,
    split_duct_for_printing,
    mesh_to_WebGL_buffers,
)
from .flanges import generate_flange_mesh, FAN_STANDARDS, FanFlangeDimensions
from .exporters import (
    export_ascii_stl,
    export_binary_stl,
    export_3mf,
    export_obj,
    export_step,
    export_gltf,
    export_glb,
)
from .dfm import analyze_dfm, PRINTER_BED_LIMITS
from .manifold import ManifoldBranch, ManifoldSpec, branch_to_duct_spec, build_manifold_mesh
from .honeycomb import (
    HoneycombStatorSpec,
    generate_honeycomb_stator_mesh,
    evaluate_honeycomb_hydraulics,
)

__all__ = [
    "build_duct_mesh",
    "generate_profile_ring",
    "compute_normal_wall_compensation",
    "split_duct_for_printing",
    "mesh_to_WebGL_buffers",
    "generate_flange_mesh",
    "FAN_STANDARDS",
    "FanFlangeDimensions",
    "export_ascii_stl",
    "export_binary_stl",
    "export_3mf",
    "export_obj",
    "export_step",
    "export_gltf",
    "export_glb",
    "analyze_dfm",
    "PRINTER_BED_LIMITS",
    "ManifoldBranch",
    "ManifoldSpec",
    "branch_to_duct_spec",
    "build_manifold_mesh",
    "HoneycombStatorSpec",
    "generate_honeycomb_stator_mesh",
    "evaluate_honeycomb_hydraulics",
]
