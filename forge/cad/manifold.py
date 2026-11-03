"""Multi-branch Y-splitter and 1-to-N manifold parametric CAD generator."""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, model_validator
from forge.core.geometry import Triangle
from forge.core.models import CrossSectionShape, DuctSpec
from forge.cad.mesh import build_duct_mesh


class ManifoldBranch(BaseModel):
    """Specification for a single downstream branch of a 1-to-N cooling manifold."""
    name: str = Field(..., description="Identifier for the target device (e.g., GPU_0, GPU_1)")
    outlet_width: float = Field(..., gt=4.0, description="Branch outlet width in mm")
    outlet_height: float = Field(..., gt=4.0, description="Branch outlet height in mm")
    offset_x: float = Field(0.0, description="Lateral X offset from manifold centerline in mm")
    offset_y: float = Field(0.0, description="Vertical Y offset from manifold centerline in mm")
    length: float = Field(90.0, gt=10.0, description="Branch transition length in mm")
    outlet_shape: CrossSectionShape = Field(CrossSectionShape.RECTANGULAR)
    corner_radius: float = Field(2.0, ge=0.0)
    heat_load_watts: float = Field(350.0, ge=0.0, description="Downstream accelerator heat load in Watts")
    downstream_k_loss: float = Field(1.5, ge=0.0, description="Downstream impedance loss coefficient K")


class ManifoldSpec(BaseModel):
    """Parametric specification for a 1-to-N splitter manifold (e.g. 1x120mm fan to 2xPCIe GPUs)."""
    inlet_width: float = Field(120.0, gt=10.0, description="Plenum inlet width in mm")
    inlet_height: float = Field(120.0, gt=10.0, description="Plenum inlet height in mm")
    inlet_shape: CrossSectionShape = Field(CrossSectionShape.CIRCULAR)
    plenum_length: float = Field(35.0, gt=5.0, description="Shared inlet plenum length before branch split in mm")
    wall: float = Field(2.0, gt=0.4, description="Wall thickness in mm")
    branches: List[ManifoldBranch] = Field(..., min_length=2, description="List of downstream branches (N >= 2)")

    @model_validator(mode="after")
    def validate_branches(self) -> "ManifoldSpec":
        if len(self.branches) < 2:
            raise ValueError("ManifoldSpec requires at least 2 branches")
        return self


def branch_to_duct_spec(spec: ManifoldSpec, branch_idx: int) -> DuctSpec:
    """Convert the i-th branch of a manifold into an equivalent DuctSpec starting from its plenum slice."""
    n_branches = len(spec.branches)
    branch = spec.branches[branch_idx]
    slice_w = max(spec.wall * 2.5, spec.inlet_width / float(n_branches))
    slice_h = spec.inlet_height
    safe_r = min(branch.corner_radius, min(slice_w, slice_h, branch.outlet_width, branch.outlet_height) / 2.0 - 0.5)
    return DuctSpec(
        inlet_width=round(slice_w, 3),
        inlet_height=round(slice_h, 3),
        outlet_width=round(branch.outlet_width, 3),
        outlet_height=round(branch.outlet_height, 3),
        length=round(branch.length, 3),
        wall=spec.wall,
        offset_x=round(branch.offset_x, 3),
        offset_y=round(branch.offset_y, 3),
        corner_radius=max(0.0, round(safe_r, 3)),
        inlet_shape=CrossSectionShape.RECTANGULAR,
        outlet_shape=branch.outlet_shape,
    )


def build_manifold_mesh(spec: ManifoldSpec, segments: int = 8) -> List[Triangle]:
    """Build a complete 3D triangular mesh for the 1-to-N splitter manifold (plenum + all N branches)."""
    triangles: List[Triangle] = []

    # 1. Shared inlet plenum morphing from inlet_shape to rectangular split plane at z = plenum_length
    plenum_spec = DuctSpec(
        inlet_width=spec.inlet_width,
        inlet_height=spec.inlet_height,
        outlet_width=spec.inlet_width,
        outlet_height=spec.inlet_height,
        length=spec.plenum_length,
        wall=spec.wall,
        inlet_shape=spec.inlet_shape,
        outlet_shape=CrossSectionShape.RECTANGULAR,
        guide_vanes=max(0, min(5, len(spec.branches) - 1)),
    )
    plenum_tris = build_duct_mesh(plenum_spec, num_profile_points=16, segments=max(4, segments // 2))
    triangles.extend(plenum_tris)

    # 2. Generate each branch translated to its slice position at z = plenum_length
    n_branches = len(spec.branches)
    slice_w = spec.inlet_width / float(n_branches)

    for idx, _branch in enumerate(spec.branches):
        b_duct = branch_to_duct_spec(spec, idx)
        b_tris = build_duct_mesh(b_duct, num_profile_points=16, segments=segments)
        # X-center of the i-th plenum slice relative to manifold center
        x_shift = -0.5 * spec.inlet_width + (idx + 0.5) * slice_w
        z_shift = spec.plenum_length

        for tri in b_tris:
            triangles.append(
                Triangle(
                    a=tri.a.add(type(tri.a)(x_shift, 0.0, z_shift)),
                    b=tri.b.add(type(tri.b)(x_shift, 0.0, z_shift)),
                    c=tri.c.add(type(tri.c)(x_shift, 0.0, z_shift)),
                )
            )

    return triangles
