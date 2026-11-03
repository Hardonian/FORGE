"""Negative fluid domain extraction and boundary patch tagging for CFD."""
from __future__ import annotations
from typing import Dict, List
from forge.core.geometry import Triangle, Vec3
from forge.core.models import DuctSpec
from forge.cad.mesh import generate_profile_ring, CrossSectionShape


def extract_fluid_domain(spec: DuctSpec, num_profile_points: int = 16, segments: int = 4) -> Dict[str, List[Triangle]]:
    """Extract negative internal fluid volume and tag CFD boundary patches."""
    is_simple = (
        spec.inlet_shape == CrossSectionShape.RECTANGULAR
        and spec.outlet_shape == CrossSectionShape.RECTANGULAR
        and spec.corner_radius == 0.0
        and spec.offset_x == 0.0
        and spec.offset_y == 0.0
    )
    n_pts = 4 if is_simple else max(16, num_profile_points)
    n_segs = 1 if is_simple else max(2, segments)

    inner_rings: List[List[Vec3]] = []

    for s in range(n_segs + 1):
        t = s / float(n_segs)
        smooth_t = t * t * (3.0 - 2.0 * t)
        cz = t * spec.length
        cx = smooth_t * spec.offset_x
        cy = smooth_t * spec.offset_y
        center = Vec3(cx, cy, cz)

        w = spec.inlet_width + t * (spec.outlet_width - spec.inlet_width)
        h = spec.inlet_height + t * (spec.outlet_height - spec.inlet_height)
        cur_shape = spec.inlet_shape if t < 0.5 else spec.outlet_shape
        cur_corner_r = (1.0 - t) * spec.corner_radius

        inner_w = max(0.1, w - 2.0 * spec.wall)
        inner_h = max(0.1, h - 2.0 * spec.wall)
        inner_r = max(0.0, cur_corner_r - spec.wall)
        ring = generate_profile_ring(inner_w, inner_h, center, cur_shape, inner_r, n_pts)
        inner_rings.append(ring)

    patches: Dict[str, List[Triangle]] = {
        "inlet": [],
        "outlet": [],
        "walls": [],
        "vanes": [],
    }

    def quad(patch_name: str, p: Vec3, q: Vec3, r: Vec3, s: Vec3):
        patches[patch_name].append(Triangle(p, q, r))
        patches[patch_name].append(Triangle(p, r, s))

    # 1. Internal sidewalls (facing inwards into fluid)
    for s in range(n_segs):
        r1 = inner_rings[s]
        r2 = inner_rings[s + 1]
        for i in range(n_pts):
            j = (i + 1) % n_pts
            # Fluid walls face inward
            quad("walls", r1[i], r1[j], r2[j], r2[i])

    # 2. Inlet cap at z = 0 (fan entrance)
    in_ring = inner_rings[0]
    in_center = Vec3(0, 0, 0)
    for i in range(n_pts):
        j = (i + 1) % n_pts
        # Normal pointing into the duct (+Z)
        patches["inlet"].append(Triangle(in_center, in_ring[j], in_ring[i]))

    # 3. Outlet cap at z = length (GPU heatsink interface)
    out_ring = inner_rings[-1]
    out_center = Vec3(spec.offset_x, spec.offset_y, spec.length)
    for i in range(n_pts):
        j = (i + 1) % n_pts
        # Normal pointing out of the duct (+Z)
        patches["outlet"].append(Triangle(out_center, out_ring[i], out_ring[j]))

    return patches

