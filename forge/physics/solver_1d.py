"""Aerodynamic loss, pressure drop, thermal convection, aeroacoustics, and structural solver."""
from __future__ import annotations
import math
from typing import Dict, Tuple
from forge.core.models import (
    CrossSectionShape,
    DuctSpec,
    MATERIAL_PROPERTIES,
    PhysicsConditions,
    PhysicsMetrics,
)
from forge.cad.mesh import compute_normal_wall_compensation


# Industrial Server Fan P-Q Curve Parameters: (max_flow_cfm, max_static_pressure_pa)
SERVER_FAN_CURVES: Dict[str, Tuple[float, float]] = {
    "40mm": (28.0, 450.0),     # High-RPM 1U Server Fan (e.g., San Ace 40)
    "60mm": (58.0, 380.0),     # 2U Server Fan
    "80mm": (110.0, 320.0),    # Delta 80mm High-Speed
    "92mm": (135.0, 260.0),    # 92mm Rackmount Fan
    "120mm": (190.0, 240.0),   # Industrial 120mm 3000-5000 RPM Fan
    "140mm": (240.0, 210.0),   # Industrial 140mm Fan
    "custom": (160.0, 250.0),
}


def _profile_area_and_perimeter(
    width_m: float,
    height_m: float,
    shape: CrossSectionShape,
    corner_radius_m: float,
) -> Tuple[float, float]:
    """Compute exact internal cross-sectional area (m^2) and wetted perimeter (m)."""
    if shape == CrossSectionShape.CIRCULAR:
        a = width_m / 2.0
        b = height_m / 2.0
        area = math.pi * a * b
        # Ramanujan's approximation for ellipse perimeter
        h_val = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
        perimeter = math.pi * (a + b) * (1.0 + (3.0 * h_val) / (10.0 + math.sqrt(4.0 - 3.0 * h_val)))
        return (max(1e-6, area), max(1e-4, perimeter))

    r = min(corner_radius_m, min(width_m, height_m) / 2.0 - 1e-5) if corner_radius_m > 0 else 0.0
    if r > 0.0:
        area = width_m * height_m - (4.0 - math.pi) * (r * r)
        perimeter = 2.0 * (width_m + height_m) - 2.0 * (4.0 - math.pi) * r
        return (max(1e-6, area), max(1e-4, perimeter))

    area = width_m * height_m
    perimeter = 2.0 * (width_m + height_m)
    return (max(1e-6, area), max(1e-4, perimeter))


def _compute_duct_dp_for_cfm(
    spec: DuctSpec,
    conditions: PhysicsConditions,
    flow_cfm: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """Internal helper returning (dp_friction, dp_expansion, total_dp, v_in, v_out, reynolds, dh_mean)."""
    q_m3_s = flow_cfm * 0.000471947443

    wall_x, wall_y, _ = compute_normal_wall_compensation(spec)
    in_w = max(0.001, (spec.inlet_width - 2.0 * wall_x) / 1000.0)
    in_h = max(0.001, (spec.inlet_height - 2.0 * wall_y) / 1000.0)
    out_w = max(0.001, (spec.outlet_width - 2.0 * wall_x) / 1000.0)
    out_h = max(0.001, (spec.outlet_height - 2.0 * wall_y) / 1000.0)
    length_m = max(0.001, spec.length / 1000.0)
    inner_r_m = max(0.0, (spec.corner_radius - 0.5 * (wall_x + wall_y)) / 1000.0)

    a_in, p_in = _profile_area_and_perimeter(in_w, in_h, spec.inlet_shape, inner_r_m)
    a_out, p_out = _profile_area_and_perimeter(out_w, out_h, spec.outlet_shape, inner_r_m)

    v_in = q_m3_s / a_in
    v_out = q_m3_s / a_out
    v_mean = 0.5 * (v_in + v_out)

    dh_in = 4.0 * a_in / p_in
    dh_out = 4.0 * a_out / p_out
    dh_mean = 0.5 * (dh_in + dh_out)

    rho = conditions.air_density_kg_m3
    mu = conditions.air_viscosity_pa_s
    reynolds = (rho * v_mean * dh_mean) / mu

    roughness = max(1e-7, conditions.surface_roughness_mm / 1000.0)

    if reynolds < 2300.0:
        f = 64.0 / max(1.0, reynolds)
    else:
        term1 = roughness / (3.7 * dh_mean)
        term2 = 5.74 / (reynolds ** 0.9)
        f = 0.25 / ((math.log10(term1 + term2)) ** 2)

    f = max(0.010, min(0.12, f))
    dp_friction = f * (length_m / dh_mean) * (0.5 * rho * (v_mean ** 2))

    delta_w = abs(out_w - in_w)
    delta_h = abs(out_h - in_h)
    side_angle_rad = math.atan(max(delta_w, delta_h) / (2.0 * length_m))
    side_angle_deg = math.degrees(side_angle_rad)

    area_ratio = a_out / a_in
    dynamic_head = 0.5 * rho * (v_in ** 2)

    k_minor = 0.0
    if area_ratio > 1.0:
        phi = 2.6 * math.sin(side_angle_rad) if side_angle_deg <= 20 else 3.5 * (math.sin(side_angle_rad) ** 0.8)
        vane_suppression = 1.0 / (1.0 + 0.45 * spec.guide_vanes)
        k_minor += ((1.0 - 1.0 / area_ratio) ** 2) * phi * vane_suppression
    else:
        k_minor += 0.45 * (1.0 - area_ratio)

    # S-duct lateral offset bend loss
    offset_dist = math.sqrt(spec.offset_x ** 2 + spec.offset_y ** 2) / 1000.0
    if offset_dist > 0.0:
        k_minor += 0.8 * (offset_dist / length_m)

    # Corner fillet secondary vortex reduction (up to 15% minor loss reduction for smooth rounded corners)
    max_possible_r = max(1e-3, min(spec.inlet_width, spec.inlet_height, spec.outlet_width, spec.outlet_height) / 2.0)
    fillet_ratio = min(1.0, spec.corner_radius / max_possible_r)
    fillet_factor = 1.0 - 0.15 * fillet_ratio

    dp_expansion = k_minor * fillet_factor * conditions.minor_loss_multiplier * dynamic_head
    total_dp = dp_friction + dp_expansion
    return (dp_friction, dp_expansion, total_dp, v_in, v_out, reynolds, dh_mean)


def solve_fan_operating_point(spec: DuctSpec, conditions: PhysicsConditions) -> float:
    """Compute equilibrium airflow (CFM) where the fan P-Q curve intersects duct impedance."""
    fan_key = spec.fan_standard.value if spec.fan_standard else "120mm"
    q_max, dp_max = SERVER_FAN_CURVES.get(fan_key, (190.0, 240.0))

    lo, hi = 1.0, q_max
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        fan_avail_dp = dp_max * max(0.0, 1.0 - (mid / q_max) ** 2)
        _, _, duct_req_dp, _, _, _, _ = _compute_duct_dp_for_cfm(spec, conditions, mid)
        if duct_req_dp < fan_avail_dp:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 2)


def solve_duct_physics(spec: DuctSpec, conditions: PhysicsConditions) -> PhysicsMetrics:
    """Calculate aerodynamic flow metrics, pressure loss, thermal dissipation, acoustics, and structural deflection."""
    q_m3_s = conditions.volumetric_flow_cfm * 0.000471947443
    dp_friction, dp_expansion, total_dp, v_in, v_out, reynolds, dh_mean = _compute_duct_dp_for_cfm(
        spec, conditions, conditions.volumetric_flow_cfm
    )

    length_m = max(0.001, spec.length / 1000.0)
    delta_w = abs(spec.outlet_width - spec.inlet_width) / 1000.0
    delta_h = abs(spec.outlet_height - spec.inlet_height) / 1000.0
    side_angle_deg = math.degrees(math.atan(max(delta_w, delta_h) / (2.0 * length_m)))

    # Corner fillet bonus to flow uniformity (suppresses corner boundary layer stagnation)
    max_possible_r = max(1e-3, min(spec.inlet_width, spec.inlet_height, spec.outlet_width, spec.outlet_height) / 2.0)
    fillet_ratio = min(1.0, spec.corner_radius / max_possible_r)
    fillet_uniformity_bonus = 0.03 * fillet_ratio

    raw_uniformity = 1.0 - (0.018 * side_angle_deg * (1.0 - 0.25 * spec.guide_vanes)) + fillet_uniformity_bonus
    uniformity = max(0.35, min(0.99, raw_uniformity))

    # Thermal heat removal: Q = m_dot * Cp * delta_T
    rho = conditions.air_density_kg_m3
    m_dot = rho * q_m3_s
    cp_air = 1005.0
    temp_rise_c = conditions.heat_dissipation_watts / max(1e-4, (m_dot * cp_air))

    # Aeroacoustic noise estimation (Curle dipole aerodynamic duct noise model in dBA)
    v_peak = max(1.0, v_in, v_out)
    noise_dba = max(20.0, min(95.0, 18.0 + 22.0 * math.log10(v_peak) + 3.0 * math.log10(1.0 + dp_expansion)))

    # Structural plate bending deflection of widest duct wall under static pressure dP (Roark flat plate formula)
    # delta_max = 0.0284 * dP * b^4 / (E * t^3)
    max_span_m = max(spec.inlet_width, spec.inlet_height, spec.outlet_width, spec.outlet_height) / 1000.0
    _, _, min_normal_wall_mm = compute_normal_wall_compensation(spec)
    wall_m = max(0.0005, min_normal_wall_mm / 1000.0)
    mat_props = MATERIAL_PROPERTIES.get(spec.material.value, {"youngs_modulus_mpa": 2100.0})
    e_pa = mat_props.get("youngs_modulus_mpa", 2100.0) * 1e6
    deflection_m = (0.0284 * total_dp * (max_span_m ** 4)) / max(1.0, e_pa * (wall_m ** 3))
    deflection_um = deflection_m * 1e6

    fan_op_cfm = solve_fan_operating_point(spec, conditions)

    return PhysicsMetrics(
        flow_rate_m3_s=round(q_m3_s, 6),
        inlet_velocity_m_s=round(v_in, 2),
        outlet_velocity_m_s=round(v_out, 2),
        reynolds_number=round(reynolds, 1),
        hydraulic_diameter_mm=round(dh_mean * 1000.0, 2),
        friction_pressure_drop_pa=round(dp_friction, 2),
        expansion_loss_pa=round(dp_expansion, 2),
        total_pressure_drop_pa=round(total_dp, 2),
        flow_uniformity_index=round(uniformity, 3),
        estimated_temp_rise_c=round(temp_rise_c, 2),
        acoustic_noise_dba=round(noise_dba, 1),
        max_wall_deflection_um=round(deflection_um, 3),
        fan_operating_cfm=fan_op_cfm,
    )
