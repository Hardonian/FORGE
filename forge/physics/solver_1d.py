"""Aerodynamic loss, pressure drop, and thermal convection solver for cooling ducts."""
from __future__ import annotations
import math
from forge.core.models import DuctSpec, PhysicsConditions, PhysicsMetrics


def solve_duct_physics(spec: DuctSpec, conditions: PhysicsConditions) -> PhysicsMetrics:
    """Calculate aerodynamic flow metrics, pressure loss, and thermal dissipation."""
    # Convert volumetric flow rate from CFM to m^3/s
    # 1 CFM = 0.000471947443 m^3/s
    q_m3_s = conditions.volumetric_flow_cfm * 0.000471947443

    # Internal opening dimensions (converted mm to meters)
    in_w = max(0.001, (spec.inlet_width - 2.0 * spec.wall) / 1000.0)
    in_h = max(0.001, (spec.inlet_height - 2.0 * spec.wall) / 1000.0)
    out_w = max(0.001, (spec.outlet_width - 2.0 * spec.wall) / 1000.0)
    out_h = max(0.001, (spec.outlet_height - 2.0 * spec.wall) / 1000.0)
    length_m = max(0.001, spec.length / 1000.0)

    # Cross-sectional areas in m^2
    a_in = in_w * in_h
    a_out = out_w * out_h

    # Velocities in m/s
    v_in = q_m3_s / a_in
    v_out = q_m3_s / a_out
    v_mean = 0.5 * (v_in + v_out)

    # Hydraulic diameters: Dh = 4 * Area / Perimeter
    p_in = 2.0 * (in_w + in_h)
    p_out = 2.0 * (out_w + out_h)
    dh_in = 4.0 * a_in / p_in
    dh_out = 4.0 * a_out / p_out
    dh_mean = 0.5 * (dh_in + dh_out)

    # Fluid properties
    rho = conditions.air_density_kg_m3
    mu = conditions.air_viscosity_pa_s
    reynolds = (rho * v_mean * dh_mean) / mu

    # 3D printed surface roughness epsilon (approx 0.05 mm for FDM, 5e-5 m)
    roughness = 0.00005

    # Swamee-Jain formula for Darcy friction factor f
    if reynolds < 2300:
        # Laminar flow
        f = 64.0 / max(1.0, reynolds)
    else:
        # Turbulent flow
        term1 = roughness / (3.7 * dh_mean)
        term2 = 5.74 / (reynolds ** 0.9)
        f = 0.25 / ((math.log10(term1 + term2)) ** 2)

    f = max(0.015, min(0.08, f))

    # Friction pressure loss: dP_friction = f * (L / Dh) * 0.5 * rho * v^2
    dp_friction = f * (length_m / dh_mean) * (0.5 * rho * (v_mean ** 2))

    # Minor losses (expansions, contractions, and curvature)
    # Angle calculation
    delta_w = abs(out_w - in_w)
    delta_h = abs(out_h - in_h)
    side_angle_rad = math.atan(max(delta_w, delta_h) / (2.0 * length_m))
    side_angle_deg = math.degrees(side_angle_rad)

    # Area ratio
    area_ratio = a_out / a_in
    dynamic_head = 0.5 * rho * (v_in ** 2)

    k_minor = 0.0
    if area_ratio > 1.0:
        # Diffuser expansion
        # Idelchik diffuser coefficient: k = (1 - 1/AR)^2 * phi(theta)
        # phi(theta) rises dramatically above 15-20 deg due to flow separation
        phi = 2.6 * math.sin(side_angle_rad) if side_angle_deg <= 20 else 3.5 * math.sin(side_angle_rad) ** 0.8
        # Guide vanes split the angle and suppress separation
        vane_suppression = 1.0 / (1.0 + 0.45 * spec.guide_vanes)
        k_minor += ((1.0 - 1.0 / area_ratio) ** 2) * phi * vane_suppression
    else:
        # Nozzle contraction
        k_minor += 0.45 * (1.0 - area_ratio)

    # S-duct lateral offset loss
    offset_dist = math.sqrt(spec.offset_x ** 2 + spec.offset_y ** 2) / 1000.0
    if offset_dist > 0.0:
        k_minor += 0.8 * (offset_dist / length_m)

    dp_expansion = k_minor * dynamic_head
    total_dp = dp_friction + dp_expansion

    # Velocity uniformity index gamma at outlet (1.0 = ideal plug flow)
    # Severe side angles decrease uniformity; guide vanes restore uniformity
    raw_uniformity = 1.0 - (0.018 * side_angle_deg * (1.0 - 0.25 * spec.guide_vanes))
    uniformity = max(0.35, min(0.99, raw_uniformity))

    # Thermal heat removal: Q = m_dot * Cp * delta_T
    # Cp of air approx 1005 J/(kg*K)
    m_dot = rho * q_m3_s
    cp_air = 1005.0
    temp_rise_c = conditions.heat_dissipation_watts / max(1e-4, (m_dot * cp_air))

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
    )
