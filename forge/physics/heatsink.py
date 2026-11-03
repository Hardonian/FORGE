"""Conjugate Heat Transfer (CHT) finned heatsink and coupled fan-duct-heatsink system solver."""
from __future__ import annotations
import math
from typing import Dict
from pydantic import BaseModel, Field
from forge.core.models import DuctSpec, PhysicsConditions
from forge.physics.solver_1d import SERVER_FAN_CURVES, _compute_duct_dp_for_cfm, solve_duct_physics


HEATSINK_CONDUCTIVITY_W_MK: Dict[str, float] = {
    "COPPER": 398.0,
    "ALUMINUM_6063": 205.0,
    "ALUMINUM_6061": 167.0,
}


class HeatsinkSpec(BaseModel):
    """Parametric specification for a skived or stacked-fin GPU/TPU heatsink downstream of the duct."""
    width_mm: float = Field(110.0, gt=5.0, description="Total heatsink fin-pack width in mm")
    fin_height_mm: float = Field(38.0, gt=2.0, description="Active fin channel height in mm")
    length_mm: float = Field(180.0, gt=10.0, description="Axial flow length of fins in mm")
    base_thickness_mm: float = Field(4.0, gt=0.5, description="Baseplate / vapor-chamber thickness in mm")
    fin_count: int = Field(52, ge=4, le=300, description="Number of parallel cooling fins")
    fin_thickness_mm: float = Field(0.5, gt=0.1, description="Individual fin thickness in mm")
    material: str = Field("COPPER", description="Fin & base material (COPPER, ALUMINUM_6063, ALUMINUM_6061)")
    tim_resistance_k_w: float = Field(0.022, ge=0.0, description="Die-to-baseplate TIM + spreading thermal resistance (K/W)")
    max_junction_temp_c: float = Field(85.0, gt=30.0, description="Maximum safe accelerator junction temperature in C")


class CoupledHeatsinkReport(BaseModel):
    """End-to-end conjugate fluid-thermal performance report for Fan + Duct + Heatsink."""
    equilibrium_flow_cfm: float
    duct_pressure_drop_pa: float
    heatsink_pressure_drop_pa: float
    total_system_pressure_drop_pa: float
    channel_velocity_m_s: float
    channel_reynolds: float
    nusselt_number: float
    convective_h_w_m2k: float
    fin_efficiency: float
    heatsink_thermal_resistance_k_w: float
    mean_die_temp_c: float
    hotspot_die_temp_c: float
    thermal_margin_c: float
    throttling_risk: bool


def evaluate_heatsink_at_cfm(
    heatsink: HeatsinkSpec,
    conditions: PhysicsConditions,
    flow_cfm: float,
    flow_uniformity_index: float = 0.90,
) -> CoupledHeatsinkReport:
    """Compute fin-array hydraulic impedance and conjugate heat transfer at a specified airflow (CFM)."""
    q_m3_s = max(1e-5, flow_cfm * 0.000471947443)
    rho = conditions.air_density_kg_m3
    mu = conditions.air_viscosity_pa_s
    k_air = 0.0263  # Thermal conductivity of air at ~300K (W/m*K)
    pr = 0.707      # Prandtl number of air
    cp_air = 1005.0

    w_m = heatsink.width_mm / 1000.0
    h_m = heatsink.fin_height_mm / 1000.0
    l_m = heatsink.length_mm / 1000.0
    t_fin_m = heatsink.fin_thickness_mm / 1000.0
    n_fins = heatsink.fin_count
    n_channels = max(1, n_fins - 1)

    # Fin spacing (gap between adjacent fins)
    total_fin_metal_w = n_fins * t_fin_m
    gap_m = max(0.0003, (w_m - total_fin_metal_w) / float(n_channels))

    # Open flow area and porosity sigma
    a_open = n_channels * gap_m * h_m
    a_frontal = w_m * h_m
    sigma = max(0.15, min(0.95, a_open / max(1e-6, a_frontal)))

    v_channel = q_m3_s / max(1e-6, a_open)
    dh_channel = (2.0 * gap_m * h_m) / (gap_m + h_m)
    re_dh = (rho * v_channel * dh_channel) / mu

    # Developing channel flow friction factor (Shah & London rectangular channel correlation)
    aspect = min(gap_m, h_m) / max(gap_m, h_m)
    f_re_fully_dev = 96.0 * (
        1.0 - 1.3553 * aspect + 1.9467 * (aspect ** 2) - 1.7012 * (aspect ** 3) + 0.9564 * (aspect ** 4) - 0.2537 * (aspect ** 5)
    )
    x_plus = l_m / max(1e-6, dh_channel * re_dh)
    if re_dh < 2300.0:
        f_app = (f_re_fully_dev + 3.44 / math.sqrt(max(1e-5, x_plus))) / max(1.0, re_dh)
    else:
        f_app = 0.3164 / (re_dh ** 0.25)

    # Kays & London entrance contraction (Kc) and exit expansion (Ke) loss coefficients
    kc = 0.42 * (1.0 - sigma ** 2)
    ke = (1.0 - sigma) ** 2
    dp_heatsink = (f_app * (l_m / dh_channel) + kc + ke) * (0.5 * rho * (v_channel ** 2))

    # Convective Nusselt number (Muzychka & Yovanovich / Gnielinski transition)
    z_star = l_m / max(1e-6, dh_channel * re_dh * pr)
    if re_dh < 2300.0:
        nu_fd = 7.541 * (
            1.0 - 2.610 * aspect + 4.970 * (aspect ** 2) - 5.119 * (aspect ** 3) + 2.702 * (aspect ** 4) - 0.548 * (aspect ** 5)
        )
        nu_dev = 1.86 * ((1.0 / max(1e-5, z_star)) ** (1.0 / 3.0))
        nu = (nu_fd ** 3 + nu_dev ** 3) ** (1.0 / 3.0)
    else:
        f_gn = (0.79 * math.log(re_dh) - 1.64) ** (-2)
        nu = ((f_gn / 8.0) * (re_dh - 1000.0) * pr) / (1.0 + 12.7 * math.sqrt(f_gn / 8.0) * ((pr ** (2.0 / 3.0)) - 1.0))
        nu = max(7.54, nu)

    h_conv = (nu * k_air) / dh_channel

    # Fin efficiency eta_f = tanh(m * Lc) / (m * Lc)
    k_solid = HEATSINK_CONDUCTIVITY_W_MK.get(heatsink.material.upper(), 398.0)
    m_param = math.sqrt((2.0 * h_conv) / max(1e-6, k_solid * t_fin_m))
    lc = h_m + 0.5 * t_fin_m
    m_lc = max(1e-6, m_param * lc)
    eta_fin = math.tanh(m_lc) / m_lc

    # Total heat transfer surface area (fins + exposed base between fins)
    a_fins = 2.0 * n_fins * h_m * l_m
    a_base_exposed = n_channels * gap_m * l_m
    a_total = a_fins + a_base_exposed
    eta_overall = 1.0 - (a_fins / a_total) * (1.0 - eta_fin)

    r_conv = 1.0 / max(1e-5, eta_overall * h_conv * a_total)
    r_base_cond = (heatsink.base_thickness_mm / 1000.0) / max(1e-6, k_solid * w_m * l_m)
    r_hs_total = r_conv + r_base_cond + heatsink.tim_resistance_k_w

    # Mean air caloric rise + convective wall rise
    m_dot = rho * q_m3_s
    dt_air_mean = 0.5 * (conditions.heat_dissipation_watts / max(1e-4, m_dot * cp_air))
    mean_die_temp = conditions.air_temp_c + dt_air_mean + conditions.heat_dissipation_watts * r_hs_total

    # Velocity maldistribution hotspot factor: lower uniformity gamma starves outer channels
    maldistribution_factor = 1.0 + 0.65 * max(0.0, 1.0 - flow_uniformity_index)
    hotspot_die_temp = conditions.air_temp_c + dt_air_mean * maldistribution_factor + (
        conditions.heat_dissipation_watts * r_hs_total * maldistribution_factor
    )
    margin = heatsink.max_junction_temp_c - hotspot_die_temp

    return CoupledHeatsinkReport(
        equilibrium_flow_cfm=round(flow_cfm, 2),
        duct_pressure_drop_pa=0.0,
        heatsink_pressure_drop_pa=round(dp_heatsink, 2),
        total_system_pressure_drop_pa=round(dp_heatsink, 2),
        channel_velocity_m_s=round(v_channel, 2),
        channel_reynolds=round(re_dh, 1),
        nusselt_number=round(nu, 2),
        convective_h_w_m2k=round(h_conv, 2),
        fin_efficiency=round(eta_fin, 4),
        heatsink_thermal_resistance_k_w=round(r_hs_total, 4),
        mean_die_temp_c=round(mean_die_temp, 2),
        hotspot_die_temp_c=round(hotspot_die_temp, 2),
        thermal_margin_c=round(margin, 2),
        throttling_risk=bool(hotspot_die_temp > heatsink.max_junction_temp_c),
    )


def solve_coupled_duct_heatsink_system(
    duct: DuctSpec,
    heatsink: HeatsinkSpec,
    conditions: PhysicsConditions,
) -> CoupledHeatsinkReport:
    """Solve the coupled non-linear system where Fan P-Q Curve = dP_duct(Q) + dP_heatsink(Q)."""
    fan_key = duct.fan_standard.value if duct.fan_standard else "120mm"
    q_max, dp_max = SERVER_FAN_CURVES.get(fan_key, (190.0, 240.0))

    # First obtain duct flow uniformity index
    duct_metrics_nominal = solve_duct_physics(duct, conditions)
    gamma = duct_metrics_nominal.flow_uniformity_index

    lo, hi = 1.0, q_max
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        fan_avail_dp = dp_max * max(0.0, 1.0 - (mid / q_max) ** 2)
        _, _, dp_duct, _, _, _, _ = _compute_duct_dp_for_cfm(duct, conditions, mid)
        hs_rep = evaluate_heatsink_at_cfm(heatsink, conditions, mid, gamma)
        total_sys_dp = dp_duct + hs_rep.heatsink_pressure_drop_pa
        if total_sys_dp < fan_avail_dp:
            lo = mid
        else:
            hi = mid

    eq_cfm = 0.5 * (lo + hi)
    _, _, dp_duct_eq, _, _, _, _ = _compute_duct_dp_for_cfm(duct, conditions, eq_cfm)
    final_rep = evaluate_heatsink_at_cfm(heatsink, conditions, eq_cfm, gamma)

    return final_rep.model_copy(update={
        "equilibrium_flow_cfm": round(eq_cfm, 2),
        "duct_pressure_drop_pa": round(dp_duct_eq, 2),
        "total_system_pressure_drop_pa": round(dp_duct_eq + final_rep.heatsink_pressure_drop_pa, 2),
    })

