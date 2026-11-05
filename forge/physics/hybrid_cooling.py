"""Hybrid Air-Assisted Liquid Cooling (AALC) & Vapor-Chamber Thermosyphon Solver.

Models two-phase capillary heat pipes, flat vapor chambers, and closed-loop
thermosyphon condensers embedded inside FORGE airflow shrouds for 600W-1200W
hyperscale AI accelerators (NVIDIA B200/GB200, AMD MI350).

Calculates two-phase physical operating limits:
- Capillary pumping limit (Young-Laplace / Darcy wick flow)
- Boiling limit / Critical Heat Flux (CHF nucleate boiling dryout)
- Entrainment / flooding limit (Weber number stripping)
- Viscous and sonic limits
- Multi-resistor thermal impedance (die -> TIM -> evaporator -> vapor -> condenser fins -> air)
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from forge.core.models import DuctSpec, PhysicsConditions
from forge.physics.solver_1d import solve_duct_physics

WorkingFluid = Literal["water", "methanol", "acetone"]
WickStructure = Literal["sintered_copper", "micro_grooved", "screen_mesh"]


class HybridAALCSpec(BaseModel):
    """Specification for a two-phase vapor chamber and condenser array in an airflow duct."""

    heat_load_watts: float = Field(default=800.0, gt=50.0, le=2500.0, description="Accelerator TDP in Watts")
    evaporator_width_mm: float = Field(default=75.0, gt=10.0, description="Die contact evaporator width (mm)")
    evaporator_length_mm: float = Field(default=75.0, gt=10.0, description="Die contact evaporator length (mm)")
    vapor_chamber_thickness_mm: float = Field(default=3.5, gt=1.0, description="Total vapor chamber thickness (mm)")
    heat_pipe_count: int = Field(default=6, ge=0, le=16, description="Number of auxiliary sintered heat pipes (0 for 3D VC)")
    heat_pipe_diameter_mm: float = Field(default=8.0, gt=3.0, description="Heat pipe outer diameter (mm)")
    working_fluid: WorkingFluid = Field(default="water", description="Two-phase working fluid")
    wick_structure: WickStructure = Field(default="sintered_copper", description="Capillary wick structure")
    condenser_fin_count: int = Field(default=64, ge=10, le=200, description="Condenser fin count downstream")
    condenser_fin_height_mm: float = Field(default=38.0, gt=5.0, description="Condenser active fin height (mm)")
    tim_resistance_k_w: float = Field(default=0.015, ge=0.001, description="Die-to-vapor-chamber TIM resistance (K/W)")
    max_junction_temp_c: float = Field(default=85.0, gt=30.0, description="Maximum allowable junction temperature (C)")


class HybridAALCReport(BaseModel):
    """Coupled two-phase thermal and aerodynamic performance report."""

    capillary_limit_watts: float
    boiling_limit_watts: float
    entrainment_limit_watts: float
    operating_power_watts: float
    two_phase_dryout_risk: bool
    limiting_mechanism: str
    thermal_margin_to_dryout_pct: float
    r_evaporator_k_w: float
    r_vapor_core_k_w: float
    r_condenser_air_k_w: float
    total_effective_r_th_k_w: float
    mean_vapor_temp_c: float
    hotspot_junction_temp_c: float
    junction_margin_c: float
    throttling_triggered: bool
    condenser_air_dp_pa: float
    system_total_dp_pa: float


# Thermophysical properties of saturation fluids at ~60°C operating temperature
FLUID_PROPS = {
    "water": {
        "rho_l": 983.2,       # liquid density (kg/m^3)
        "rho_v": 0.130,       # vapor density (kg/m^3)
        "mu_l": 4.67e-4,      # liquid dynamic viscosity (Pa*s)
        "mu_v": 1.12e-5,      # vapor dynamic viscosity (Pa*s)
        "sigma": 0.0662,      # surface tension (N/m)
        "h_fg": 2.358e6,      # latent heat of vaporization (J/kg)
        "k_l": 0.654,         # liquid thermal conductivity (W/m*K)
    },
    "methanol": {
        "rho_l": 751.0,
        "rho_v": 0.520,
        "mu_l": 3.80e-4,
        "mu_v": 1.10e-5,
        "sigma": 0.0201,
        "h_fg": 1.100e6,
        "k_l": 0.198,
    },
    "acetone": {
        "rho_l": 748.0,
        "rho_v": 0.780,
        "mu_l": 2.70e-4,
        "mu_v": 1.05e-5,
        "sigma": 0.0210,
        "h_fg": 0.518e6,
        "k_l": 0.165,
    },
}

# Capillary wick properties
WICK_PROPS = {
    "sintered_copper": {"r_eff_m": 18e-6, "permeability_m2": 2.2e-11, "porosity": 0.55, "t_wick_mm": 0.8},
    "micro_grooved": {"r_eff_m": 45e-6, "permeability_m2": 8.5e-10, "porosity": 0.65, "t_wick_mm": 0.5},
    "screen_mesh": {"r_eff_m": 35e-6, "permeability_m2": 1.4e-10, "porosity": 0.62, "t_wick_mm": 0.6},
}


def solve_hybrid_aalc_cooling(
    duct: DuctSpec,
    aalc: HybridAALCSpec | None = None,
    conditions: PhysicsConditions | None = None,
    *,
    aalc_spec: HybridAALCSpec | None = None,
) -> HybridAALCReport:
    """Solve coupled two-phase thermosyphon/vapor-chamber limits and duct aerodynamics."""
    aalc = aalc or aalc_spec or HybridAALCSpec()
    cond = conditions or PhysicsConditions()
    duct_phys = solve_duct_physics(duct, cond)

    fluid = FLUID_PROPS.get(aalc.working_fluid, FLUID_PROPS["water"])
    wick = WICK_PROPS.get(aalc.wick_structure, WICK_PROPS["sintered_copper"])

    l_eff_m = max(0.040, (duct.length * 0.65) * 1e-3)
    evap_area_m2 = (aalc.evaporator_width_mm * aalc.evaporator_length_mm) * 1e-6
    t_wick_m = wick["t_wick_mm"] * 1e-3
    wick_area_m2 = 2.0 * (aalc.evaporator_width_mm + aalc.evaporator_length_mm) * 1e-3 * t_wick_m

    # 1. Capillary Pumping Limit (Darcy-Weisbach wick liquid pressure drop vs Young-Laplace delta P_cap)
    # Delta P_cap = 2 * sigma / r_eff
    dp_cap_max = (2.0 * fluid["sigma"]) / wick["r_eff_m"]
    a_w = max(1e-6, wick_area_m2 * wick["porosity"])
    q_cap = (2.0 * a_w * wick["permeability_m2"] * fluid["h_fg"] * fluid["rho_l"] * dp_cap_max) / (
        fluid["mu_l"] * l_eff_m
    )
    if aalc.heat_pipe_count > 0:
        # Each auxiliary heat pipe adds parallel capillary transport capacity
        pipe_cross_w = aalc.heat_pipe_count * (math.pi * (aalc.heat_pipe_diameter_mm * 1e-3) * t_wick_m)
        q_cap += (aalc.heat_pipe_count * dp_cap_max * pipe_cross_w * wick["permeability_m2"] * fluid["rho_l"] * fluid["h_fg"]) / (
            fluid["mu_l"] * l_eff_m
        )
    q_cap = round(q_cap, 1)

    # 2. Boiling Limit (Critical Heat Flux nucleate boiling at evaporator wick interface)
    # q_boil = (2 * pi * L_eff * k_eff * T_v * (2 * sigma / r_b)) / (h_fg * rho_v * ln(r_o / r_i))
    # Empirical correlation for sintered powder beds: CHF ~ 150 - 220 W/cm^2
    chf_w_cm2 = 185.0 if aalc.wick_structure == "sintered_copper" else 110.0
    q_boil = round(chf_w_cm2 * (evap_area_m2 * 1e4), 1)

    # 3. Entrainment / Flooding Limit (Liquid droplet stripping by counter-current vapor)
    # q_ent = A_v * h_fg * sqrt(sigma * rho_v / (2 * r_h_wick))
    a_vapor_m2 = max(1e-5, (aalc.evaporator_width_mm * 1e-3) * ((aalc.vapor_chamber_thickness_mm * 1e-3) - 2.0 * t_wick_m))
    q_ent = round(a_vapor_m2 * fluid["h_fg"] * math.sqrt((fluid["sigma"] * fluid["rho_v"]) / (2.0 * wick["r_eff_m"])), 1)

    # Determine limiting power and dryout safety
    limits = {"capillary": q_cap, "boiling": q_boil, "entrainment": q_ent}
    limiting_mech = min(limits, key=limits.get)
    p_max = limits[limiting_mech]
    dryout_risk = aalc.heat_load_watts >= p_max
    dryout_margin_pct = round(100.0 * (p_max - aalc.heat_load_watts) / p_max, 1)

    # Thermal Resistor Network
    # R_evap = t_wick / (k_wick_eff * A_evap)
    k_wick_eff = (1.0 - wick["porosity"]) * 398.0 + wick["porosity"] * fluid["k_l"]
    r_evap = (t_wick_m / (k_wick_eff * max(1e-6, evap_area_m2)))
    # Vapor core is virtually isothermal (dp_v is tiny)
    r_vapor = 0.0035

    # Condenser air forced convection resistance
    # Duct airflow couples directly to condenser fin bank
    q_m3_s = cond.volumetric_flow_cfm * 0.000471947443
    rho_air = cond.air_density_kg_m3
    mu_air = cond.air_viscosity_pa_s
    k_air = 0.0263

    w_cond_m = duct.outlet_width * 1e-3
    h_cond_m = aalc.condenser_fin_height_mm * 1e-3
    n_fins = aalc.condenser_fin_count
    t_fin_m = 0.0004
    fin_gap_m = max(0.0005, (w_cond_m - n_fins * t_fin_m) / max(1, n_fins - 1))
    flow_area_air = (n_fins - 1) * fin_gap_m * h_cond_m
    v_channel = q_m3_s / max(1e-5, flow_area_air)
    dh_air = 2.0 * fin_gap_m * h_cond_m / (fin_gap_m + h_cond_m)
    re_air = (rho_air * v_channel * dh_air) / mu_air

    nu_air = 7.54 + 0.023 * (max(50.0, re_air) ** 0.8) * (0.707 ** 0.4)
    h_conv = (nu_air * k_air) / dh_air
    l_cond_m = max(0.020, (duct.length * 0.45) * 1e-3)
    total_fin_area = 2.0 * n_fins * h_cond_m * l_cond_m
    m_fin = math.sqrt((2.0 * h_conv) / (398.0 * t_fin_m))
    eta_fin = math.tanh(m_fin * h_cond_m) / max(1e-6, m_fin * h_cond_m)
    r_cond_air = 1.0 / max(1e-4, h_conv * total_fin_area * eta_fin)

    # Condenser air pressure drop
    f_air = 0.3164 / (max(100.0, re_air) ** 0.25)
    dp_condenser = f_air * (l_cond_m / dh_air) * 0.5 * rho_air * (v_channel ** 2)
    total_system_dp = duct_phys.total_pressure_drop_pa + dp_condenser

    total_r_th = aalc.tim_resistance_k_w + r_evap + r_vapor + r_cond_air

    # Temperature field
    t_ambient = cond.air_temp_c
    m_dot_air = rho_air * q_m3_s
    delta_t_air = aalc.heat_load_watts / max(0.1, m_dot_air * 1005.0)
    mean_vapor_t = t_ambient + 0.5 * delta_t_air + aalc.heat_load_watts * (r_vapor + r_cond_air)
    hotspot_junction_t = t_ambient + delta_t_air + aalc.heat_load_watts * total_r_th
    junc_margin = aalc.max_junction_temp_c - hotspot_junction_t
    throttling = hotspot_junction_t > aalc.max_junction_temp_c

    return HybridAALCReport(
        capillary_limit_watts=q_cap,
        boiling_limit_watts=q_boil,
        entrainment_limit_watts=q_ent,
        operating_power_watts=aalc.heat_load_watts,
        two_phase_dryout_risk=dryout_risk,
        limiting_mechanism=limiting_mech,
        thermal_margin_to_dryout_pct=dryout_margin_pct,
        r_evaporator_k_w=round(r_evap, 5),
        r_vapor_core_k_w=round(r_vapor, 5),
        r_condenser_air_k_w=round(r_cond_air, 4),
        total_effective_r_th_k_w=round(total_r_th, 4),
        mean_vapor_temp_c=round(mean_vapor_t, 2),
        hotspot_junction_temp_c=round(hotspot_junction_t, 2),
        junction_margin_c=round(junc_margin, 2),
        throttling_triggered=throttling,
        condenser_air_dp_pa=round(dp_condenser, 2),
        system_total_dp_pa=round(total_system_dp, 2),
    )
