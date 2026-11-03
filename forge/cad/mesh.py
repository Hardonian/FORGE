"""Parametric mesh generation for complex cooling ducts, lofts, fillets, and vanes."""
from __future__ import annotations
import math
from typing import List
from forge.core.geometry import Vec3, Triangle
from forge.core.models import CrossSectionShape, DuctSpec


def generate_profile_ring(
    width: float,
    height: float,
    center: Vec3,
    shape: CrossSectionShape = CrossSectionShape.RECTANGULAR,
    corner_radius: float = 0.0,
    num_points: int = 16,
) -> List[Vec3]:
    """Generate a closed loop of exactly `num_points` 3D vertices representing a cross-sectional profile."""
    cx, cy, cz = center.x, center.y, center.z

    # 1. Circular shape
    if shape == CrossSectionShape.CIRCULAR:
        radius = min(width, height) / 2.0
        ring: List[Vec3] = []
        for i in range(num_points):
            angle = 2.0 * math.pi * i / float(num_points)
            ring.append(Vec3(cx + radius * math.cos(angle), cy + radius * math.sin(angle), cz))
        return ring

    w_half = width / 2.0
    h_half = height / 2.0
    pts_per_corner = max(1, num_points // 4)

    # 2. Rounded rectangle with positive corner radius
    if corner_radius > 0:
        r = min(corner_radius, w_half - 1e-4, h_half - 1e-4)
        ring = []
        # Centers of the 4 corner arcs
        centers = [
            (-w_half + r, -h_half + r, math.pi),            # Bottom-Left: pi to 3pi/2
            (w_half - r, -h_half + r, 3.0 * math.pi / 2.0), # Bottom-Right: 3pi/2 to 2pi
            (w_half - r, h_half - r, 0.0),                  # Top-Right: 0 to pi/2
            (-w_half + r, h_half - r, math.pi / 2.0),       # Top-Left: pi/2 to pi
        ]
        for bx, by, start_angle in centers:
            for p in range(pts_per_corner):
                th = start_angle + (math.pi / 2.0) * (p / float(pts_per_corner))
                ring.append(Vec3(cx + bx + r * math.cos(th), cy + by + r * math.sin(th), cz))
        return ring

    # 3. Sharp rectangle with exactly `num_points` vertices (pts_per_corner per side)
    ring = []
    corners = [
        Vec3(-w_half, -h_half, 0.0), # Bottom-Left
        Vec3(w_half, -h_half, 0.0),  # Bottom-Right
        Vec3(w_half, h_half, 0.0),   # Top-Right
        Vec3(-w_half, h_half, 0.0),  # Top-Left
    ]
    for i in range(4):
        c1 = corners[i]
        c2 = corners[(i + 1) % 4]
        for p in range(pts_per_corner):
            frac = p / float(pts_per_corner)
            vx = c1.x + frac * (c2.x - c1.x)
            vy = c1.y + frac * (c2.y - c1.y)
            ring.append(Vec3(cx + vx, cy + vy, cz))

    return ring


def build_duct_mesh(spec: DuctSpec, num_profile_points: int = 16, segments: int = 4) -> List[Triangle]:
    """Generate a watertight 2-manifold triangular mesh for the given DuctSpec."""
    is_simple = (
        spec.inlet_shape == CrossSectionShape.RECTANGULAR
        and spec.outlet_shape == CrossSectionShape.RECTANGULAR
        and spec.corner_radius == 0.0
        and spec.offset_x == 0.0
        and spec.offset_y == 0.0
        and spec.guide_vanes == 0
    )

    n_pts = 4 if is_simple else max(16, num_profile_points)
    n_segs = 1 if is_simple else max(2, segments)

    triangles: List[Triangle] = []

    def quad(p: Vec3, q: Vec3, r: Vec3, s: Vec3):
        triangles.append(Triangle(p, q, r))
        triangles.append(Triangle(p, r, s))

    outer_rings: List[List[Vec3]] = []
    inner_rings: List[List[Vec3]] = []

    for s in range(n_segs + 1):
        t = s / float(n_segs)
        # S-curve Hermite smoothstep interpolation for centerline
        smooth_t = t * t * (3.0 - 2.0 * t)
        cz = t * spec.length
        cx = smooth_t * spec.offset_x
        cy = smooth_t * spec.offset_y
        center = Vec3(cx, cy, cz)

        # Cross-sectional dimensions at fraction t
        w = spec.inlet_width + t * (spec.outlet_width - spec.inlet_width)
        h = spec.inlet_height + t * (spec.outlet_height - spec.inlet_height)

        # Shape transition
        cur_shape = spec.inlet_shape if t < 0.5 else spec.outlet_shape
        cur_corner_r = (1.0 - t) * spec.corner_radius

        # Outer ring
        outer_ring = generate_profile_ring(w, h, center, cur_shape, cur_corner_r, n_pts)
        outer_rings.append(outer_ring)

        # Inner ring with true wall thickness
        inner_w = max(0.1, w - 2.0 * spec.wall)
        inner_h = max(0.1, h - 2.0 * spec.wall)
        inner_r = max(0.0, cur_corner_r - spec.wall)
        inner_ring = generate_profile_ring(inner_w, inner_h, center, cur_shape, inner_r, n_pts)
        inner_rings.append(inner_ring)

    # 1. Connect outside walls along segments
    for s in range(n_segs):
        r1 = outer_rings[s]
        r2 = outer_rings[s + 1]
        for i in range(n_pts):
            j = (i + 1) % n_pts
            quad(r1[i], r1[j], r2[j], r2[i])

    # 2. Connect inside walls along segments (inward facing, reversed vertex winding)
    for s in range(n_segs):
        r1 = inner_rings[s]
        r2 = inner_rings[s + 1]
        for i in range(n_pts):
            j = (i + 1) % n_pts
            quad(r1[j], r1[i], r2[i], r2[j])

    # 3. Inlet rim face (connects outer[0] to inner[0])
    r_out_0 = outer_rings[0]
    r_in_0 = inner_rings[0]
    for i in range(n_pts):
        j = (i + 1) % n_pts
        quad(r_out_0[j], r_out_0[i], r_in_0[i], r_in_0[j])

    # 4. Outlet rim face (connects outer[-1] to inner[-1])
    r_out_end = outer_rings[-1]
    r_in_end = inner_rings[-1]
    for i in range(n_pts):
        j = (i + 1) % n_pts
        quad(r_out_end[i], r_out_end[j], r_in_end[j], r_in_end[i])

    # 5. Internal splitter guide vanes (aerodynamic flow separators)
    if spec.guide_vanes > 0:
        vane_thickness = min(spec.wall, 1.2)
        for v in range(1, spec.guide_vanes + 1):
            frac = v / (spec.guide_vanes + 1.0)  # e.g. 1/3, 2/3
            # Build vertical vane plate along centerline from inlet to outlet
            vane_rings = []
            for s in range(n_segs + 1):
                in_ring = inner_rings[s]
                min_x = min(p.x for p in in_ring)
                max_x = max(p.x for p in in_ring)
                min_y = min(p.y for p in in_ring)
                max_y = max(p.y for p in in_ring)
                vx = min_x + frac * (max_x - min_x)
                cz = in_ring[0].z
                vane_rings.append([
                    Vec3(vx - vane_thickness / 2.0, min_y, cz),
                    Vec3(vx + vane_thickness / 2.0, min_y, cz),
                    Vec3(vx + vane_thickness / 2.0, max_y, cz),
                    Vec3(vx - vane_thickness / 2.0, max_y, cz),
                ])
            for s in range(n_segs):
                vr1 = vane_rings[s]
                vr2 = vane_rings[s + 1]
                for i in range(4):
                    j = (i + 1) % 4
                    quad(vr1[i], vr1[j], vr2[j], vr2[i])
            # Cap the vane at inlet (s=0) and outlet (s=n_segs)
            vr0 = vane_rings[0]
            quad(vr0[3], vr0[2], vr0[1], vr0[0])
            vrend = vane_rings[-1]
            quad(vrend[0], vrend[1], vrend[2], vrend[3])

    return triangles
