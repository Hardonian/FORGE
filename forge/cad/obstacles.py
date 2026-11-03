"""3D obstacle collision detection and automatic Bezier centerline path routing."""
from __future__ import annotations
import math
from typing import List, Tuple
from pydantic import BaseModel
from forge.core.geometry import Triangle, Vec3
from forge.core.models import DuctSpec, ObstacleBox
from forge.cad.mesh import build_duct_mesh, evaluate_bezier_scalar


class ObstacleCollisionDetail(BaseModel):
    """Clearance evaluation against a single 3D obstacle bounding box."""
    name: str
    colliding: bool
    min_distance_mm: float
    required_clearance_mm: float
    margin_mm: float


class ObstacleClearanceReport(BaseModel):
    """Aggregate 3D obstacle clearance report for a duct mesh."""
    all_clear: bool
    min_margin_mm: float
    details: List[ObstacleCollisionDetail]


def _point_to_aabb_signed_distance(p: Vec3, box: ObstacleBox) -> float:
    """Compute signed Euclidean distance from 3D point p to axis-aligned ObstacleBox (negative if inside)."""
    dx = max(box.min_x - p.x, 0.0, p.x - box.max_x)
    dy = max(box.min_y - p.y, 0.0, p.y - box.max_y)
    dz = max(box.min_z - p.z, 0.0, p.z - box.max_z)
    outside_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if outside_dist > 0.0:
        return outside_dist
    # Point is strictly inside or on the boundary of the AABB
    in_x = min(p.x - box.min_x, box.max_x - p.x)
    in_y = min(p.y - box.min_y, box.max_y - p.y)
    in_z = min(p.z - box.min_z, box.max_z - p.z)
    return -min(in_x, in_y, in_z)


def _slice_aabb_distance(
    cx: float,
    cy: float,
    cz: float,
    half_w: float,
    half_h: float,
    box: ObstacleBox,
) -> float:
    """Compute signed distance between a duct cross-sectional slice at z=cz and an ObstacleBox."""
    if cz < box.min_z or cz > box.max_z:
        dz = min(abs(cz - box.min_z), abs(cz - box.max_z))
    else:
        dz = 0.0

    # Interval overlap in X: [cx - half_w, cx + half_w] vs [box.min_x, box.max_x]
    dx = max(box.min_x - (cx + half_w), (cx - half_w) - box.max_x, 0.0)
    dy = max(box.min_y - (cy + half_h), (cy - half_h) - box.max_y, 0.0)

    if dx > 0.0 or dy > 0.0 or dz > 0.0:
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    # Overlapping in X, Y, and Z -> negative penetration depth
    pen_x = min((cx + half_w) - box.min_x, box.max_x - (cx - half_w))
    pen_y = min((cy + half_h) - box.min_y, box.max_y - (cy - half_h))
    return -min(pen_x, pen_y)


def check_obstacle_clearances(
    spec: DuctSpec,
    triangles: List[Triangle] | None = None,
    num_slices: int = 32,
) -> ObstacleClearanceReport:
    """Evaluate 3D clearance between the duct geometry and all obstacles in `spec.obstacles`."""
    if not spec.obstacles:
        return ObstacleClearanceReport(all_clear=True, min_margin_mm=999.0, details=[])

    tris = triangles if triangles is not None else build_duct_mesh(spec, segments=12)
    sample_points: List[Vec3] = []
    for tri in tris:
        sample_points.append(tri.a)
        sample_points.append(tri.b)
        sample_points.append(tri.c)
        sample_points.append(
            Vec3(
                (tri.a.x + tri.b.x + tri.c.x) / 3.0,
                (tri.a.y + tri.b.y + tri.c.y) / 3.0,
                (tri.a.z + tri.b.z + tri.c.z) / 3.0,
            )
        )

    details: List[ObstacleCollisionDetail] = []
    min_overall_margin = 999.0

    for obs in spec.obstacles:
        min_d = min(_point_to_aabb_signed_distance(pt, obs) for pt in sample_points)
        # Also check continuous axial slice bounding boxes so coarse triangle spans cannot straddle an obstacle
        for s in range(num_slices + 1):
            t = s / float(num_slices)
            smooth_t = t * t * (3.0 - 2.0 * t)
            cz = t * spec.length
            cx = (
                evaluate_bezier_scalar(0.0, spec.centerline_x_ctrl, spec.offset_x, t)
                if spec.centerline_x_ctrl
                else smooth_t * spec.offset_x
            )
            cy = (
                evaluate_bezier_scalar(0.0, spec.centerline_y_ctrl, spec.offset_y, t)
                if spec.centerline_y_ctrl
                else smooth_t * spec.offset_y
            )
            w = evaluate_bezier_scalar(spec.inlet_width, spec.width_profile_ctrl, spec.outlet_width, t)
            h = evaluate_bezier_scalar(spec.inlet_height, spec.height_profile_ctrl, spec.outlet_height, t)
            d_slice = _slice_aabb_distance(cx, cy, cz, 0.5 * w, 0.5 * h, obs)
            min_d = min(min_d, d_slice)

        margin = min_d - obs.clearance_mm
        colliding = margin < 0.0
        min_overall_margin = min(min_overall_margin, margin)
        details.append(
            ObstacleCollisionDetail(
                name=obs.name,
                colliding=colliding,
                min_distance_mm=round(min_d, 2),
                required_clearance_mm=round(obs.clearance_mm, 2),
                margin_mm=round(margin, 2),
            )
        )

    all_clear = all(not d.colliding for d in details)
    return ObstacleClearanceReport(
        all_clear=all_clear,
        min_margin_mm=round(min_overall_margin, 2),
        details=details,
    )


def route_duct_around_obstacles(spec: DuctSpec) -> Tuple[DuctSpec, ObstacleClearanceReport]:
    """Automatically synthesize Bezier centerline control points to bow around 3D obstacles."""
    initial_report = check_obstacle_clearances(spec)
    if initial_report.all_clear or not spec.obstacles:
        return (spec, initial_report)

    req_shift_x = 0.0
    req_shift_y = 0.0

    for obs in spec.obstacles:
        t_mid = max(0.1, min(0.9, 0.5 * (obs.min_z + obs.max_z) / max(1.0, spec.length)))
        w_mid = evaluate_bezier_scalar(spec.inlet_width, spec.width_profile_ctrl, spec.outlet_width, t_mid)
        h_mid = evaluate_bezier_scalar(spec.inlet_height, spec.height_profile_ctrl, spec.outlet_height, t_mid)
        cx_nom = t_mid * spec.offset_x
        cy_nom = t_mid * spec.offset_y

        d_cur = _slice_aabb_distance(cx_nom, cy_nom, t_mid * spec.length, 0.5 * w_mid, 0.5 * h_mid, obs)
        if d_cur < obs.clearance_mm:
            pad = obs.clearance_mm + 3.5
            # Evaluate 4 candidate bypass directions: +X, -X, +Y, -Y and pick the shortest excursion
            shift_pos_x = (obs.max_x + 0.5 * w_mid + pad) - cx_nom
            shift_neg_x = (obs.min_x - 0.5 * w_mid - pad) - cx_nom
            shift_pos_y = (obs.max_y + 0.5 * h_mid + pad) - cy_nom
            shift_neg_y = (obs.min_y - 0.5 * h_mid - pad) - cy_nom

            candidates = [
                (abs(shift_pos_x), shift_pos_x, 0.0),
                (abs(shift_neg_x), shift_neg_x, 0.0),
                (abs(shift_pos_y), 0.0, shift_pos_y),
                (abs(shift_neg_y), 0.0, shift_neg_y),
            ]
            candidates.sort(key=lambda item: item[0])
            _, best_dx, best_dy = candidates[0]
            if abs(best_dx) > abs(req_shift_x):
                req_shift_x = best_dx
            if abs(best_dy) > abs(req_shift_y):
                req_shift_y = best_dy

    # For a cubic Bezier curve with P0=0, P1=C, P2=C, P3=offset, value at t=0.5 is 0.75*C + 0.125*offset.
    # Scaling by 1.45 guarantees the mid-span apex reaches the full required clearance excursion.
    ctrl_x = list(spec.centerline_x_ctrl)
    ctrl_y = list(spec.centerline_y_ctrl)
    if abs(req_shift_x) > 1e-3:
        apex_x = round(0.5 * spec.offset_x + 1.45 * req_shift_x, 2)
        ctrl_x = [apex_x, apex_x]
    if abs(req_shift_y) > 1e-3:
        apex_y = round(0.5 * spec.offset_y + 1.45 * req_shift_y, 2)
        ctrl_y = [apex_y, apex_y]

    routed_spec = spec.model_copy(update={
        "centerline_x_ctrl": ctrl_x,
        "centerline_y_ctrl": ctrl_y,
    })
    final_report = check_obstacle_clearances(routed_spec)
    return (routed_spec, final_report)
