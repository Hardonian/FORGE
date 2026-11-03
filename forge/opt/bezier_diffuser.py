"""Multi-station Bezier diffuser shape optimization (Stratford / Bell-Mouth pressure recovery)."""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
from forge.core.models import DuctSpec, PhysicsConditions, PhysicsMetrics
from forge.cad.mesh import compute_normal_wall_compensation, evaluate_bezier_scalar
from forge.physics.solver_1d import _profile_area_and_perimeter, solve_duct_physics


def compute_multi_station_diffuser_loss(
    spec: DuctSpec,
    conditions: PhysicsConditions,
    flow_cfm: float,
    num_stations: int = 16,
) -> Tuple[float, float, float]:
    """March along `num_stations` axial segments of a Bezier-profiled duct.

    Returns:
        (dp_friction_pa, dp_expansion_pa, peak_half_angle_deg)
    """
    q_m3_s = flow_cfm * 0.000471947443
    wall_x, wall_y, _ = compute_normal_wall_compensation(spec)
    length_m = max(0.001, spec.length / 1000.0)
    dz = length_m / float(num_stations)
    inner_r_m = max(0.0, (spec.corner_radius - 0.5 * (wall_x + wall_y)) / 1000.0)

    rho = conditions.air_density_kg_m3
    mu = conditions.air_viscosity_pa_s
    roughness = max(1e-7, conditions.surface_roughness_mm / 1000.0)

    dp_friction_total = 0.0
    dp_expansion_total = 0.0
    weighted_angle_deg = 0.0

    for s in range(num_stations):
        t0 = s / float(num_stations)
        t1 = (s + 1) / float(num_stations)

        w0 = max(0.001, (evaluate_bezier_scalar(spec.inlet_width, spec.width_profile_ctrl, spec.outlet_width, t0) - 2.0 * wall_x) / 1000.0)
        h0 = max(0.001, (evaluate_bezier_scalar(spec.inlet_height, spec.height_profile_ctrl, spec.outlet_height, t0) - 2.0 * wall_y) / 1000.0)
        w1 = max(0.001, (evaluate_bezier_scalar(spec.inlet_width, spec.width_profile_ctrl, spec.outlet_width, t1) - 2.0 * wall_x) / 1000.0)
        h1 = max(0.001, (evaluate_bezier_scalar(spec.inlet_height, spec.height_profile_ctrl, spec.outlet_height, t1) - 2.0 * wall_y) / 1000.0)

        shape_s = spec.inlet_shape if t0 < 0.5 else spec.outlet_shape
        a0, p0 = _profile_area_and_perimeter(w0, h0, shape_s, inner_r_m)
        a1, p1 = _profile_area_and_perimeter(w1, h1, shape_s, inner_r_m)

        v0 = q_m3_s / a0
        v1 = q_m3_s / a1
        v_seg = 0.5 * (v0 + v1)
        dh_seg = 0.5 * (4.0 * a0 / p0 + 4.0 * a1 / p1)

        re_seg = (rho * v_seg * dh_seg) / mu
        if re_seg < 2300.0:
            f = 64.0 / max(1.0, re_seg)
        else:
            f = 0.25 / ((math.log10(roughness / (3.7 * dh_seg) + 5.74 / (re_seg ** 0.9))) ** 2)
        f = max(0.010, min(0.12, f))
        dp_friction_total += f * (dz / dh_seg) * (0.5 * rho * (v_seg ** 2))

        dw = abs(w1 - w0)
        dh = abs(h1 - h0)
        local_angle_rad = math.atan(max(dw, dh) / (2.0 * dz))
        local_angle_deg = math.degrees(local_angle_rad)

        # Stratford boundary-layer separation sensitivity: high kinetic energy near inlet (t0 -> 0)
        # tolerates steeper divergence than depleted boundary layer near outlet (t0 -> 1)
        bl_fatigue_weight = 0.70 + 0.60 * (t0 ** 1.2)
        weighted_angle_deg += (local_angle_deg * bl_fatigue_weight) / float(num_stations)

        if a1 > a0:
            phi = (
                2.6 * math.sin(local_angle_rad * bl_fatigue_weight)
                if local_angle_deg <= 20.0
                else 3.5 * (math.sin(min(math.pi / 2.0, local_angle_rad * bl_fatigue_weight)) ** 0.8)
            )
            vane_suppression = 1.0 / (1.0 + 0.45 * spec.guide_vanes)
            # Local Borda-Carnot step loss scaled for continuous multi-station integration
            dv = max(0.0, v0 - v1)
            dp_step = 0.5 * rho * (dv ** 2) * phi * vane_suppression * (num_stations ** 0.35)
            dp_expansion_total += dp_step
        elif a1 < a0:
            dp_expansion_total += 0.45 * (1.0 - a1 / a0) * (0.5 * rho * (v0 ** 2)) / float(num_stations)

    # S-duct lateral offset bend loss
    offset_dist = math.sqrt(spec.offset_x ** 2 + spec.offset_y ** 2) / 1000.0
    if offset_dist > 0.0:
        v_in_total = q_m3_s / _profile_area_and_perimeter(
            max(0.001, (spec.inlet_width - 2.0 * wall_x) / 1000.0),
            max(0.001, (spec.inlet_height - 2.0 * wall_y) / 1000.0),
            spec.inlet_shape,
            inner_r_m,
        )[0]
        dp_expansion_total += 0.8 * (offset_dist / length_m) * (0.5 * rho * (v_in_total ** 2))

    max_possible_r = max(1e-3, min(spec.inlet_width, spec.inlet_height, spec.outlet_width, spec.outlet_height) / 2.0)
    fillet_ratio = min(1.0, spec.corner_radius / max_possible_r)
    fillet_factor = 1.0 - 0.15 * fillet_ratio
    dp_expansion_total *= fillet_factor * conditions.minor_loss_multiplier

    return (dp_friction_total, dp_expansion_total, weighted_angle_deg)


def optimize_bezier_diffuser(
    baseline_spec: DuctSpec,
    conditions: PhysicsConditions,
    num_control_points: int = 2,
    iterations: int = 35,
    learning_rate: float = 1.5,
) -> Tuple[DuctSpec, PhysicsMetrics, Dict[str, float]]:
    """Optimize interior Bezier control points (width_profile_ctrl, height_profile_ctrl) via finite-difference Adam/GD.

    Discovers non-linear Stratford / bell-mouth diffuser profiles that expand rapidly where the
    inlet boundary layer is energized and gently decelerate near the exit to suppress separation.
    """
    n_ctrl = max(1, min(4, num_control_points))
    # Initialize at linear interpolation positions
    w_ctrl: List[float] = [
        baseline_spec.inlet_width + ((i + 1) / float(n_ctrl + 1)) * (baseline_spec.outlet_width - baseline_spec.inlet_width)
        for i in range(n_ctrl)
    ]
    h_ctrl: List[float] = [
        baseline_spec.inlet_height + ((i + 1) / float(n_ctrl + 1)) * (baseline_spec.outlet_height - baseline_spec.inlet_height)
        for i in range(n_ctrl)
    ]

    baseline_metrics = solve_duct_physics(baseline_spec, conditions)

    w_min = min(baseline_spec.inlet_width, baseline_spec.outlet_width)
    w_max = max(baseline_spec.inlet_width, baseline_spec.outlet_width)
    h_min = min(baseline_spec.inlet_height, baseline_spec.outlet_height)
    h_max = max(baseline_spec.inlet_height, baseline_spec.outlet_height)

    def _objective(wc: List[float], hc: List[float]) -> float:
        candidate = baseline_spec.model_copy(update={
            "width_profile_ctrl": [round(x, 3) for x in wc],
            "height_profile_ctrl": [round(y, 3) for y in hc],
        })
        m = solve_duct_physics(candidate, conditions)
        # Minimize total pressure drop and maximize uniformity
        return m.total_pressure_drop_pa + 60.0 * (1.0 - m.flow_uniformity_index)

    best_w = list(w_ctrl)
    best_h = list(h_ctrl)
    best_loss = _objective(best_w, best_h)

    eps = 0.25
    for step in range(iterations):
        lr = learning_rate * (0.95 ** step)
        grad_w = [0.0] * n_ctrl
        grad_h = [0.0] * n_ctrl
        base_val = _objective(w_ctrl, h_ctrl)
        if base_val < best_loss:
            best_loss = base_val
            best_w = list(w_ctrl)
            best_h = list(h_ctrl)

        for i in range(n_ctrl):
            if abs(w_max - w_min) > 1e-3:
                w_plus = list(w_ctrl)
                w_plus[i] = min(w_max, max(w_min, w_plus[i] + eps))
                grad_w[i] = (_objective(w_plus, h_ctrl) - base_val) / eps
            if abs(h_max - h_min) > 1e-3:
                h_plus = list(h_ctrl)
                h_plus[i] = min(h_max, max(h_min, h_plus[i] + eps))
                grad_h[i] = (_objective(w_ctrl, h_plus) - base_val) / eps

        for i in range(n_ctrl):
            w_ctrl[i] = min(w_max, max(w_min, w_ctrl[i] - lr * grad_w[i]))
            h_ctrl[i] = min(h_max, max(h_min, h_ctrl[i] - lr * grad_h[i]))

    final_loss = _objective(w_ctrl, h_ctrl)
    if final_loss < best_loss:
        best_w = list(w_ctrl)
        best_h = list(h_ctrl)

    opt_spec = baseline_spec.model_copy(update={
        "width_profile_ctrl": [round(x, 3) for x in best_w],
        "height_profile_ctrl": [round(y, 3) for y in best_h],
    })
    opt_metrics = solve_duct_physics(opt_spec, conditions)

    dp_reduction_pct = 0.0
    if baseline_metrics.total_pressure_drop_pa > 0:
        dp_reduction_pct = round(
            100.0 * (baseline_metrics.total_pressure_drop_pa - opt_metrics.total_pressure_drop_pa)
            / baseline_metrics.total_pressure_drop_pa,
            2,
        )

    summary = {
        "baseline_dp_pa": baseline_metrics.total_pressure_drop_pa,
        "optimized_dp_pa": opt_metrics.total_pressure_drop_pa,
        "dp_reduction_pct": dp_reduction_pct,
        "baseline_uniformity": baseline_metrics.flow_uniformity_index,
        "optimized_uniformity": opt_metrics.flow_uniformity_index,
    }
    return (opt_spec, opt_metrics, summary)
