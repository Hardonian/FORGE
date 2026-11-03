"""Parametric mesh generation for complex cooling ducts, lofts, fillets, flanges, and vanes."""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
from forge.core.geometry import Vec3, Triangle
from forge.core.models import CrossSectionShape, DuctSpec
from forge.cad.flanges import FAN_STANDARDS, generate_flange_mesh


def generate_profile_ring(
    width: float,
    height: float,
    center: Vec3,
    shape: CrossSectionShape = CrossSectionShape.RECTANGULAR,
    corner_radius: float = 0.0,
    num_points: int = 16,
) -> List[Vec3]:
    """Generate a closed loop of exactly `num_points` 3D vertices representing a cross-sectional profile.

    All profile types start at the bottom-left quadrant (-3*pi/4) and proceed counter-clockwise
    so lofts between circular, rounded-rectangle, and rectangular profiles have zero torsional twist.
    """
    cx, cy, cz = center.x, center.y, center.z
    pts_per_corner = max(1, num_points // 4)
    actual_pts = pts_per_corner * 4

    # 1. Circular / Elliptical shape aligned to -3*pi/4 start angle
    if shape == CrossSectionShape.CIRCULAR:
        rx = width / 2.0
        ry = height / 2.0
        start_angle = -3.0 * math.pi / 4.0
        ring: List[Vec3] = []
        for i in range(actual_pts):
            angle = start_angle + 2.0 * math.pi * i / float(actual_pts)
            ring.append(Vec3(cx + rx * math.cos(angle), cy + ry * math.sin(angle), cz))
        return ring

    w_half = width / 2.0
    h_half = height / 2.0

    # 2. Rounded rectangle with positive corner radius
    if corner_radius > 0:
        r = min(corner_radius, w_half - 1e-4, h_half - 1e-4)
        ring = []
        # Centers of the 4 corner arcs proceeding counter-clockwise from Bottom-Left
        centers = [
            (-w_half + r, -h_half + r, math.pi),             # Bottom-Left: pi to 3pi/2
            (w_half - r, -h_half + r, 3.0 * math.pi / 2.0),  # Bottom-Right: 3pi/2 to 2pi
            (w_half - r, h_half - r, 0.0),                   # Top-Right: 0 to pi/2
            (-w_half + r, h_half - r, math.pi / 2.0),        # Top-Left: pi/2 to pi
        ]
        for bx, by, start_angle in centers:
            for p in range(pts_per_corner):
                th = start_angle + (math.pi / 2.0) * (p / float(pts_per_corner))
                ring.append(Vec3(cx + bx + r * math.cos(th), cy + by + r * math.sin(th), cz))
        return ring

    # 3. Sharp rectangle with `actual_pts` vertices (pts_per_corner per side)
    ring = []
    corners = [
        Vec3(-w_half, -h_half, 0.0),  # Bottom-Left
        Vec3(w_half, -h_half, 0.0),   # Bottom-Right
        Vec3(w_half, h_half, 0.0),    # Top-Right
        Vec3(-w_half, h_half, 0.0),   # Top-Left
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


def compute_normal_wall_compensation(spec: DuctSpec) -> Tuple[float, float, float]:
    """Compute effective planar wall thicknesses (wall_x, wall_y) and minimum normal thickness.

    On a sloped transition wall of angle theta, a purely horizontal XY cut of width `w`
    produces a thinner perpendicular wall `w * cos(theta)`. Returning `w / cos(theta)`
    guarantees the true 3D normal thickness equals `spec.wall`.
    """
    tan_x = (abs(spec.outlet_width - spec.inlet_width) / 2.0 + abs(spec.offset_x)) / max(1e-6, spec.length)
    tan_y = (abs(spec.outlet_height - spec.inlet_height) / 2.0 + abs(spec.offset_y)) / max(1e-6, spec.length)
    sec_x = math.sqrt(1.0 + tan_x * tan_x)
    sec_y = math.sqrt(1.0 + tan_y * tan_y)

    if spec.preserve_normal_thickness:
        return (spec.wall * sec_x, spec.wall * sec_y, spec.wall)
    else:
        min_normal = spec.wall / max(sec_x, sec_y)
        return (spec.wall, spec.wall, min_normal)


def interpolate_morphed_ring(
    width: float,
    height: float,
    center: Vec3,
    inlet_shape: CrossSectionShape,
    outlet_shape: CrossSectionShape,
    corner_radius: float,
    blend_t: float,
    num_points: int,
) -> List[Vec3]:
    """Smoothly morph between inlet_shape and outlet_shape across axial blend_t in [0, 1]."""
    if inlet_shape == outlet_shape:
        return generate_profile_ring(width, height, center, inlet_shape, corner_radius, num_points)

    ring_in = generate_profile_ring(width, height, center, inlet_shape, corner_radius, num_points)
    ring_out = generate_profile_ring(width, height, center, outlet_shape, corner_radius, num_points)
    return [
        Vec3(
            (1.0 - blend_t) * a.x + blend_t * b.x,
            (1.0 - blend_t) * a.y + blend_t * b.y,
            center.z,
        )
        for a, b in zip(ring_in, ring_out)
    ]


def evaluate_bezier_scalar(start_val: float, interior_ctrl: List[float], end_val: float, t: float) -> float:
    """Evaluate a 1D Bernstein-Bezier curve at t in [0, 1] with given interior control points."""
    if not interior_ctrl:
        return start_val + t * (end_val - start_val)
    pts = [start_val] + list(interior_ctrl) + [end_val]
    n = len(pts) - 1
    val = 0.0
    for i, p_val in enumerate(pts):
        binom = math.comb(n, i)
        basis = binom * ((1.0 - t) ** (n - i)) * (t ** i)
        val += basis * p_val
    return val


def build_duct_mesh(spec: DuctSpec, num_profile_points: int = 16, segments: int = 4) -> List[Triangle]:
    """Generate a watertight 2-manifold triangular mesh for the given DuctSpec."""
    is_simple = (
        spec.inlet_shape == CrossSectionShape.RECTANGULAR
        and spec.outlet_shape == CrossSectionShape.RECTANGULAR
        and spec.corner_radius == 0.0
        and spec.offset_x == 0.0
        and spec.offset_y == 0.0
        and spec.guide_vanes == 0
        and len(spec.width_profile_ctrl) == 0
        and len(spec.height_profile_ctrl) == 0
        and not spec.preserve_normal_thickness
        and spec.flange_thickness == 0.0
    )

    n_pts = 4 if is_simple else max(16, (num_profile_points // 4) * 4)
    n_segs = 1 if is_simple else max(8 if (spec.width_profile_ctrl or spec.height_profile_ctrl) else 2, segments)

    wall_x, wall_y, _ = compute_normal_wall_compensation(spec)
    wall_r = 0.5 * (wall_x + wall_y)

    triangles: List[Triangle] = []

    def quad(p: Vec3, q: Vec3, r: Vec3, s: Vec3):
        triangles.append(Triangle(p, q, r))
        triangles.append(Triangle(p, r, s))

    outer_rings: List[List[Vec3]] = []
    inner_rings: List[List[Vec3]] = []

    for s in range(n_segs + 1):
        t = s / float(n_segs)
        # S-curve Hermite smoothstep interpolation for centerline and shape morphing
        smooth_t = t * t * (3.0 - 2.0 * t)
        cz = t * spec.length
        cx = smooth_t * spec.offset_x
        cy = smooth_t * spec.offset_y
        center = Vec3(cx, cy, cz)

        # Cross-sectional dimensions at fraction t (supports multi-point Bezier expansion laws)
        w = evaluate_bezier_scalar(spec.inlet_width, spec.width_profile_ctrl, spec.outlet_width, t)
        h = evaluate_bezier_scalar(spec.inlet_height, spec.height_profile_ctrl, spec.outlet_height, t)
        cur_corner_r = (1.0 - t) * spec.corner_radius + t * spec.corner_radius

        # Outer ring (supports continuous morphing between circular and rectangular ends)
        outer_ring = interpolate_morphed_ring(
            w, h, center, spec.inlet_shape, spec.outlet_shape, cur_corner_r, smooth_t, n_pts
        )
        outer_rings.append(outer_ring)

        # Inner ring with normal-compensated wall thickness
        inner_w = max(0.2, w - 2.0 * wall_x)
        inner_h = max(0.2, h - 2.0 * wall_y)
        inner_r = max(0.0, cur_corner_r - wall_r)
        inner_ring = interpolate_morphed_ring(
            inner_w, inner_h, center, spec.inlet_shape, spec.outlet_shape, inner_r, smooth_t, n_pts
        )
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
            frac = v / (spec.guide_vanes + 1.0)
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

    # 6. Optional Server Fan Mounting Flange Collar
    if spec.flange_thickness > 0.0:
        if spec.fan_standard and spec.fan_standard.value in FAN_STANDARDS:
            f_dim = FAN_STANDARDS[spec.fan_standard.value]
            fw = max(spec.inlet_width + 12.0, f_dim.frame_size_mm + 8.0)
            fh = max(spec.inlet_height + 12.0, f_dim.frame_size_mm + 8.0)
        else:
            fw = spec.inlet_width + 16.0
            fh = spec.inlet_height + 16.0
        flange_tris = generate_flange_mesh(
            inner_width=spec.inlet_width,
            inner_height=spec.inlet_height,
            flange_width=fw,
            flange_height=fh,
            flange_thickness=spec.flange_thickness,
            z_start=0.0,
        )
        triangles.extend(flange_tris)

    return triangles


def split_duct_for_printing(spec: DuctSpec, split_ratio: float = 0.5) -> Tuple[DuctSpec, DuctSpec]:
    """Split an oversized duct into two printable axial segments at `split_ratio` in (0.1, 0.9)."""
    if not (0.1 <= split_ratio <= 0.9):
        raise ValueError("split_ratio must be between 0.1 and 0.9")

    t = split_ratio
    smooth_t = t * t * (3.0 - 2.0 * t)
    mid_w = round(spec.inlet_width + t * (spec.outlet_width - spec.inlet_width), 3)
    mid_h = round(spec.inlet_height + t * (spec.outlet_height - spec.inlet_height), 3)
    mid_ox = round(smooth_t * spec.offset_x, 3)
    mid_oy = round(smooth_t * spec.offset_y, 3)
    mid_shape = spec.inlet_shape if t < 0.5 else spec.outlet_shape

    part_a = DuctSpec(
        inlet_width=spec.inlet_width,
        inlet_height=spec.inlet_height,
        outlet_width=mid_w,
        outlet_height=mid_h,
        length=round(spec.length * t, 3),
        wall=spec.wall,
        offset_x=mid_ox,
        offset_y=mid_oy,
        corner_radius=spec.corner_radius,
        inlet_shape=spec.inlet_shape,
        outlet_shape=mid_shape,
        preserve_normal_thickness=spec.preserve_normal_thickness,
        guide_vanes=spec.guide_vanes,
        fan_standard=spec.fan_standard,
        flange_thickness=spec.flange_thickness,
        material=spec.material,
    )

    part_b = DuctSpec(
        inlet_width=mid_w,
        inlet_height=mid_h,
        outlet_width=spec.outlet_width,
        outlet_height=spec.outlet_height,
        length=round(spec.length * (1.0 - t), 3),
        wall=spec.wall,
        offset_x=round(spec.offset_x - mid_ox, 3),
        offset_y=round(spec.offset_y - mid_oy, 3),
        corner_radius=spec.corner_radius,
        inlet_shape=mid_shape,
        outlet_shape=spec.outlet_shape,
        preserve_normal_thickness=spec.preserve_normal_thickness,
        guide_vanes=spec.guide_vanes,
        fan_standard=None,
        flange_thickness=0.0,
        material=spec.material,
    )

    return (part_a, part_b)


def mesh_to_WebGL_buffers(triangles: List[Triangle]) -> Dict[str, List[float]]:
    """Serialize triangles into flat position and normal float arrays for Three.js BufferGeometry."""
    positions: List[float] = []
    normals: List[float] = []
    for tri in triangles:
        n = tri.normal
        for v in (tri.a, tri.b, tri.c):
            positions.extend([round(v.x, 4), round(v.y, 4), round(v.z, 4)])
            normals.extend([round(n.x, 4), round(n.y, 4), round(n.z, 4)])
    return {"positions": positions, "normals": normals, "triangle_count": len(triangles)}
